"""Third experiment: fresh selection/confirmation and causal collaborative context."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import lightgbm as lgb
from .benchmark import Catalog
from .optimization_data import load_features,build_features,FEATURE_NAMES
from .view_features import build as build_views,VIEW_NAMES
from .recbole_model import ndcg_by_query

ROOT=Path('data/iteration3')
OUT=Path('results/iteration3')
EXTRA=['collab_last','collab_max','collab_mean','collab_recency_weighted','collab_item_frequency',
       'known_user','user_has_profile','user_prior_exact','user_prior_recency','user_collab_max',
       'user_collab_mean','user_content_max','older_session_has_profile','older_session_exact','older_session_collab_max']
NAMES=FEATURE_NAMES+VIEW_NAMES+EXTRA


def hash_ids(values):return hashlib.sha256(','.join(map(str,sorted(values))).encode()).hexdigest()


def reserve_and_prepare():
    ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet('data/processed/queries_metadata.parquet')
    earlier=pd.read_parquet('data/processed/validation.parquet')
    pool=meta[meta.eventdate.between('2016-05-01','2016-05-14')&~meta['is.test']&~meta.sessionId.isin(earlier.sessionId)]
    selection=pool.sample(n=8000,random_state=20261029).sort_values(['eventdate','sessionId','timeframe','queryId'])
    protocol_path=Path('config/iteration3_protocol.json')
    protocol=json.loads(protocol_path.read_text())
    old=pd.read_parquet('data/processed/test.parquet')
    earlier_confirmation=pd.read_parquet('data/processed_v2/confirmation_ids.parquet')
    excluded=set(old.sessionId)|set(earlier_confirmation.sessionId)
    remaining=meta[(meta.eventdate>='2016-05-16')&~meta['is.test']&~meta.sessionId.isin(excluded)]
    confirmation=remaining.sample(n=10000,random_state=20261028).sort_values(['eventdate','sessionId','timeframe','queryId'])
    assert hash_ids(confirmation.queryId)==protocol['confirmation_query_ids_sha256']
    reservation=ROOT/'confirmation_ids.parquet'
    if reservation.exists():assert confirmation.queryId.tolist()==pd.read_parquet(reservation).queryId.tolist()
    else:confirmation.to_parquet(reservation,index=False)
    if 'selection_query_ids_sha256' in protocol:
        assert protocol['selection_query_ids_sha256']==hash_ids(selection.queryId)
    else:
        protocol.update(selection_queries=8000,selection_sampling_seed=20261029,
                        selection_query_ids_sha256=hash_ids(selection.queryId),
                        selection_rule='Fresh same-date development query sample excluding original validation session IDs. Open once after candidate fitting; select highest mean NDCG and freeze before new confirmation.')
        protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    ids={'selection':set(selection.queryId),'confirmation':set(confirmation.queryId)}
    if not all((ROOT/f'{name}_queries.parquet').exists() for name in ids):
        found={name:[] for name in ids}
        for chunk in pd.read_csv('data/raw/train-queries.csv',sep=';',chunksize=50000,dtype={'items':str,'searchstring.tokens':str}):
            for name,wanted in ids.items():found[name].append(chunk[chunk.queryId.isin(wanted)].copy())
        for name,frames in found.items():
            q=pd.concat(frames).sort_values(['eventdate','sessionId','timeframe','queryId'])
            assert hash_ids(q.queryId)==hash_ids(ids[name])
            q.to_parquet(ROOT/f'{name}_queries.parquet',index=False)
    print('Reserved fresh selection and confirmation queries; outcome metrics remain unopened.',flush=True)


def load_data(split,variant='base'):
    if split in ['train','validation']:
        data=load_features(split,'views')
    else:data=load_features(split,'views',root=ROOT)
    if variant=='collab':data['X']=np.load(ROOT/f'{split}_collab.npy',mmap_mode='r')
    elif variant=='no_user':
        data['X']=np.load(ROOT/f'{split}_collab.npy',mmap_mode='r')
    return data


def build_split(split):
    if split=='confirmation' and not (OUT/'frozen_selection.json').exists():raise RuntimeError('Freeze before confirmation access')
    if split in ['selection','confirmation']:
        if not (ROOT/split/'manifest.json').exists():
            build_features(split,root=ROOT,protocol_path='config/iteration3_protocol.json',freeze_path=OUT/'frozen_selection.json')
        if not (ROOT/split/'features_views.npy').exists():
            build_views(split,root=ROOT,output=OUT,freeze_path=OUT/'frozen_selection.json')
    build_context(split)


def context_cutoff(split,day):
    if split in ['validation','selection']:return '2016-04-29'
    if split=='confirmation':return '2016-05-14'
    if day<='2016-04-07':return '2016-03-30'
    if day<='2016-04-14':return '2016-04-06'
    if day<='2016-04-21':return '2016-04-13'
    return '2016-04-20'


def profile_map(frame,key):
    result={}
    for ident,group in frame.dropna(subset=[key]).groupby(key,sort=False):
        result[str(int(ident))]=list(dict.fromkeys(reversed(group.itemId.tolist())))[:20]
    return result


def build_snapshot(cutoff,catalog):
    folder=ROOT/'graphs';folder.mkdir(exist_ok=True)
    graph_path=folder/f'{cutoff}.npz';profiles_path=folder/f'{cutoff}_profiles.json'
    if graph_path.exists() and profiles_path.exists():
        return sparse.load_npz(graph_path).tocsr(),json.loads(profiles_path.read_text())
    clicks=pd.read_parquet('data/processed/clicks.parquet')
    source=clicks[clicks.eventdate<=cutoff].sort_values(['eventdate','timeframe','queryId'])
    outside_catalog=int((~source.itemId.isin(catalog.row)).sum())
    source=source[source.itemId.isin(catalog.row)]
    meta=pd.read_parquet('data/processed/queries_metadata.parquet')[['queryId','userId']]
    source=source.merge(meta,on='queryId',validate='many_to_one')
    distinct=source.drop_duplicates(['sessionId','eventdate','itemId'],keep='last')
    group=distinct.groupby(['sessionId','eventdate'],sort=False).ngroup().to_numpy()
    row=np.array([catalog.row[int(item)] for item in distinct.itemId],dtype=np.int32)
    incidence=sparse.csr_matrix((np.ones(len(row),dtype=np.float32),(group,row)),shape=(int(group.max())+1,len(catalog.ids)))
    frequency=np.asarray(incidence.sum(axis=0)).ravel()
    graph=(incidence.T@incidence).tocsr()
    graph.setdiag(0);graph.eliminate_zeros()
    left=np.repeat(frequency,np.diff(graph.indptr))
    graph.data/=np.sqrt(left*frequency[graph.indices])+10.
    profiles=dict(users=profile_map(source,'userId'),sessions=profile_map(source,'sessionId'),
                  frequency=frequency.astype(int).tolist(),cutoff=cutoff,source_queries=int(source.queryId.nunique()),
                  graph_nonzeros=int(graph.nnz),historical_clicks_outside_catalog=outside_catalog)
    sparse.save_npz(graph_path,graph)
    profiles_path.write_text(json.dumps(profiles))
    print(f'Built causal graph through {cutoff}: {graph.nnz:,} edges',flush=True)
    return graph,profiles


def collaborative_scores(graph,items,history,catalog):
    if not history:return np.zeros((len(items),1),dtype=np.float32)
    return graph[[catalog.row[x] for x in history]][:,[catalog.row[x] for x in items]].toarray().T


def build_context(split):
    path=ROOT/f'{split}_collab.npy'
    if path.exists():return
    data=load_data(split)
    catalog=Catalog()
    if split in ['train','validation']:q=pd.read_parquet(f'data/processed_v2/{split}_queries.parquet').set_index('queryId')
    else:q=pd.read_parquet(ROOT/f'{split}_queries.parquet').set_index('queryId')
    temporary=path.with_suffix('.npy.part')
    X=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(data['X']),len(NAMES)))
    X[:,:48]=data['X']
    active=None;graph=profiles=None
    coverage=defaultdict(int)
    for j,row in enumerate(data['meta'].itertuples(index=False)):
        cutoff=context_cutoff(split,row.date)
        if cutoff!=active:graph,profiles=build_snapshot(cutoff,catalog);active=cutoff
        a,b=data['offsets'][j:j+2]
        items=data['items'][a:b].tolist()
        history=data['prefixes'][j][data['prefixes'][j]>0].tolist()[::-1]
        signal=collaborative_scores(graph,items,history,catalog)
        uid=q.loc[row.query_id,'userId']
        known=not pd.isna(uid)
        profile=profiles['users'].get(str(int(uid)),[]) if known else []
        user_signal=collaborative_scores(graph,items,profile,catalog)
        user_content=catalog.similarity(items,profile).max(axis=1) if profile else np.zeros(len(items))
        older=profiles['sessions'].get(str(int(row.session_id)),[])
        older_signal=collaborative_scores(graph,items,older,catalog)
        extra=np.column_stack([signal[:,0],signal.max(axis=1),signal.mean(axis=1),
            (signal/(np.arange(signal.shape[1])+1)).sum(axis=1),
            np.log1p([profiles['frequency'][catalog.row[item]] for item in items]),
            np.full(len(items),known),np.full(len(items),bool(profile)),
            [item in profile for item in items],
            [1/(profile.index(item)+1) if item in profile else 0 for item in items],
            user_signal.max(axis=1),user_signal.mean(axis=1),user_content,
            np.full(len(items),bool(older)),[item in older for item in items],older_signal.max(axis=1)])
        X[a:b,48:]=extra
        coverage['queries']+=1;coverage['known_users']+=known
        coverage['users_with_past_profile']+=bool(profile);coverage['sessions_with_older_context']+=bool(older)
        if (j+1)%10000==0:print(f'{split}: contextual features for {j+1:,} queries',flush=True)
    X.flush();del X;temporary.replace(path)
    (OUT/f'{split}_context_audit.json').write_text(json.dumps(dict(coverage),indent=2)+'\n')


def train_tree(name,leaves=31,no_user=False):
    if (OUT/'frozen_selection.json').exists() or (OUT/'selection_summary.csv').exists():raise RuntimeError('Selection frozen')
    if (OUT/f'{name}.txt').exists():raise RuntimeError('Model already recorded')
    train=load_data('train','collab');valid=load_data('validation','collab')
    columns=list(range(len(NAMES)))
    if no_user:columns=[i for i in columns if not 53<=i<=59]
    model=lgb.LGBMRanker(objective='lambdarank',n_estimators=700,learning_rate=.04,num_leaves=leaves,
        min_child_samples=150,reg_lambda=5,n_jobs=4,random_state=4285,deterministic=True,
        force_col_wise=True,verbosity=-1,max_bin=127,subsample=.85,subsample_freq=1,
        colsample_bytree=.9,lambdarank_truncation_level=20)
    print(f'Training {name}, {len(train["meta"]):,} queries, {len(columns)} features',flush=True)
    X=train['X'] if len(columns)==len(NAMES) else train['X'][:,columns]
    VX=valid['X'] if len(columns)==len(NAMES) else valid['X'][:,columns]
    model.fit(X,train['y'],group=np.diff(train['offsets']),
        feature_name=[NAMES[i] for i in columns],categorical_feature=['item_category'],
        eval_set=[(VX,valid['y'])],eval_group=[np.diff(valid['offsets'])],
        eval_at=[10],callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(25)])
    model.booster_.save_model(str(OUT/f'{name}.txt'))
    score=model.predict(VX);np.save(OUT/f'{name}_validation.npy',score)
    info=dict(name=name,columns=columns,leaves=leaves,no_user=no_user,best_iteration=model.best_iteration_,
              validation_ndcg10=float(ndcg_by_query(score,valid['y'],valid['offsets']).mean()))
    (OUT/f'{name}.json').write_text(json.dumps(info,indent=2)+'\n')
    pd.DataFrame({'feature':[NAMES[i] for i in columns],'gain':model.booster_.feature_importance('gain')}).to_csv(OUT/f'{name}_importance.csv',index=False)
    print(json.dumps(info,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','context','train','run'])
    parser.add_argument('--split',default='train',choices=['train','validation','selection','confirmation'])
    parser.add_argument('--name',default='collab31');parser.add_argument('--leaves',type=int,default=31)
    parser.add_argument('--no-user',action='store_true')
    args=parser.parse_args()
    if args.action=='run':run_all()
    elif args.action=='prepare':reserve_and_prepare()
    elif args.action=='context':build_split(args.split)
    else:train_tree(args.name,args.leaves,args.no_user)


def run_all():
    """Run from a fresh checkout or resume completed training stages safely."""
    def run(*arguments):subprocess.run([sys.executable,*arguments],check=True)
    if not Path('data/processed_v2/train/manifest.json').exists():
        run('-m','session_rerank.run_experiment','prepare')
    if not (ROOT/'confirmation_ids.parquet').exists():
        reserve_and_prepare()
    elif not (ROOT/'selection_queries.parquet').exists():reserve_and_prepare()
    if (OUT/'frozen_selection.json').exists():
        if not (OUT/'confirmation_per_query.npz').exists():run('-m','session_rerank.iteration3_report','confirm')
        return
    for name,arguments in [('views_60k',['--features','views']),('views_60k_63',['--features','views','--leaves','63'])]:
        if not Path(f'results/optimization/{name}.txt').exists():
            run('-m','session_rerank.optimize_ranker','--name',name,*arguments)
    if not Path('results/optimization/recbole_gru4rec.pth').exists():run('-m','session_rerank.recbole_model','train')
    for split in ['validation','train']:build_split(split)
    for name,leaves,anonymous in [('collab31',31,False),('collab63',63,False),('anonymous31',31,True)]:
        if not (OUT/f'{name}.json').exists():train_tree(name,leaves,anonymous)
    for variant in ['sas_ce','gru_ce']:
        if not (OUT/f'{variant}.json').exists():run('-m','session_rerank.neural_search','train','--variant',variant)
    run('-m','session_rerank.iteration3_report','select')
    run('-m','session_rerank.iteration3_report','confirm')


if __name__=='__main__':main()
