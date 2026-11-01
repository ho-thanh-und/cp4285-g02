"""Common-query check of the previously frozen collaborative tree/hybrid.

No new fitting, hyperparameter search or policy selection. This comparator
retains its original daily context and pre-confirmation archive, so it is a
practical pipeline comparison rather than a matched feature ablation.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from .benchmark import Catalog
from .iteration3 import build_snapshot, collaborative_scores, hash_ids, NAMES
from .optimization_data import build_features, load_features
from .view_features import build as build_views
from .benchmark_data import digest, write_json
from .benchmark_experiment import score_queries
from .policies import cluster_bootstrap

ROOT=Path('data/collaborative_comparison')
OUT=Path('results/submission/collaborative_comparison')


def extra_features(items,history,uid,sid,catalog,graph,profiles):
    signal=collaborative_scores(graph,items,history,catalog)
    known=not pd.isna(uid)
    profile=profiles['users'].get(str(int(uid)),[]) if known else []
    us=collaborative_scores(graph,items,profile,catalog)
    content=catalog.similarity(items,profile).max(axis=1) if profile else np.zeros(len(items))
    older=profiles['sessions'].get(str(int(sid)),[])
    old=collaborative_scores(graph,items,older,catalog)
    return np.column_stack([signal[:,0],signal.max(axis=1),signal.mean(axis=1),
        (signal/(np.arange(signal.shape[1])+1)).sum(axis=1),
        np.log1p([profiles['frequency'][catalog.row[item]] for item in items]),
        np.full(len(items),known),np.full(len(items),bool(profile)),
        [item in profile for item in items],[1/(profile.index(item)+1) if item in profile else 0 for item in items],
        us.max(axis=1),us.mean(axis=1),content,np.full(len(items),bool(older)),
        [item in older for item in items],old.max(axis=1)])


def check_adapter(catalog,graph,profiles):
    """Compare against the original saved feature matrix on 200 old queries."""
    from .iteration3 import load_data
    old=load_data('confirmation','collab')
    queries=pd.read_parquet('data/iteration3/confirmation_queries.parquet').set_index('queryId')
    maximum=0.
    for j in np.random.default_rng(4285).choice(len(old['meta']),200,replace=False):
        m=old['meta'].iloc[j];a,b=old['offsets'][j:j+2]
        history=old['prefixes'][j][old['prefixes'][j]>0].tolist()[::-1]
        result=extra_features(old['items'][a:b].tolist(),history,queries.loc[m.query_id,'userId'],m.session_id,catalog,graph,profiles).astype(np.float32)
        maximum=max(maximum,float(np.max(np.abs(result-old['X'][a:b,48:]))))
    assert maximum<1e-6
    return maximum


def run():
    ROOT.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet('results/session_benchmark/confirmation_metadata.parquet')
    selected=meta[meta.positives>0];wanted=set(selected.query_id)
    original=json.loads(Path('results/iteration3/frozen_selection.json').read_text())
    assert original['selected']==dict(kind='fusion',tree='collab31',neural='gru_ce',weight=.9)
    paths=['results/iteration3/collab31.txt','results/iteration3/gru_ce.pth','results/iteration3/frozen_selection.json']
    for p in paths[:2]:assert digest(p)==original['hashes'][p]
    protocol=dict(status='Post-confirmation diagnostic. Fixed earlier-selected tree and .9 hybrid; no test-based tuning.',
        query_ids_sha256=hash_ids(wanted),confirmation_query_ids_sha256=hash_ids(wanted),
        model_hashes={p:digest(p) for p in paths},
        protocol_comparison='Original daily distinct-click/view prefixes and causal May14 statistics/graph versus fixed pre-March31 archive and 20m prefixes. Different training sample/feature schema.',
        earlier_selection=original['selected'])
    protocol_path=ROOT/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
    else:write_json(protocol_path,protocol)
    query_path=ROOT/'confirmation_queries.parquet'
    if not query_path.exists():
        chunks=[]
        for c in pd.read_csv('data/raw/train-queries.csv',sep=';',chunksize=50000,dtype={'items':str,'searchstring.tokens':str}):chunks.append(c[c.queryId.isin(wanted)])
        queries=pd.concat(chunks).set_index('queryId').loc[selected.query_id].reset_index()
        queries.to_parquet(query_path,index=False)
    if not (ROOT/'confirmation/manifest.json').exists():
        build_features('confirmation',root=ROOT,protocol_path=protocol_path,freeze_path='results/iteration3/frozen_selection.json')
    if not (ROOT/'confirmation/features_views.npy').exists():
        build_views('confirmation',root=ROOT,output=OUT,freeze_path='results/iteration3/frozen_selection.json')
    data=load_features('confirmation','views',root=ROOT)
    assert np.array_equal(data['meta'].query_id,selected.query_id)
    assert (data['meta'].date>'2016-05-14').all()
    offsets=np.load('data/session_benchmark/test/offsets.npy');test_meta=pd.read_parquet('data/session_benchmark/test/meta.parquet')
    index={q:j for j,q in enumerate(test_meta.query_id)}
    items=np.load('data/session_benchmark/test/items.npy',mmap_mode='r');labels=np.load('data/session_benchmark/test/labels.npy',mmap_mode='r')
    for j,q in enumerate(selected.query_id):
        a,b=data['offsets'][j:j+2];u,v=offsets[index[q]:index[q]+2]
        assert np.array_equal(data['items'][a:b],items[u:v])
        assert np.array_equal(data['y'][a:b],labels[u:v])
    catalog=Catalog();graph,profiles=build_snapshot('2016-05-14',catalog)
    parity=check_adapter(catalog,graph,profiles)
    features_path=ROOT/'collaborative.npy'
    if not features_path.exists():
        queries=pd.read_parquet(query_path).set_index('queryId')
        temporary=features_path.with_suffix('.part.npy')
        features=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(data['X']),len(NAMES)))
        features[:,:48]=data['X']
        for j,m in enumerate(data['meta'].itertuples(index=False)):
            a,b=data['offsets'][j:j+2];history=data['prefixes'][j][data['prefixes'][j]>0].tolist()[::-1]
            features[a:b,48:]=extra_features(data['items'][a:b].tolist(),history,queries.loc[m.query_id,'userId'],m.session_id,catalog,graph,profiles)
            if (j+1)%5000==0:print('Collaborative context',j+1,'/',len(selected),flush=True)
        features.flush();del features;temporary.replace(features_path)
    pred_path=OUT/'collab31.npy'
    if not pred_path.exists():
        model=lgb.Booster(model_file=paths[0]);prediction=model.predict(np.load(features_path,mmap_mode='r'),num_threads=1);np.save(pred_path,prediction)
    neural_path=OUT/'gru_ce.npy'
    if not neural_path.exists():
        import torch
        from .recbole_model import candidate_scores
        from .neural_search import configuration
        from recbole.utils import get_model
        torch.set_num_threads(1)
        checkpoint=torch.load(paths[1],map_location='cpu',weights_only=False)
        class Shape:
            def num(self,field):return checkpoint['n_items']
        model=get_model(checkpoint['model_name'])(configuration('gru_ce','cpu'),Shape())
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        np.save(neural_path,candidate_scores(model,checkpoint['mapping'],data,batch_size=256))
    from .optimization_report import hybrid_scores
    tree=np.load(pred_path);neural=np.load(neural_path)
    scores=dict(collaborative_tree=tree,original_hybrid=hybrid_scores(tree,neural,data,.9))
    cache=np.load('results/session_benchmark/confirmation_metrics.npz');k=cache['names'].tolist().index('relevance')
    values={'benchmark_tree':cache['values'][meta.positives.to_numpy()>0,k,:2].astype(np.float64)}
    adapted=dict(labels=data['y'],offsets=data['offsets'],group=np.diff(data['offsets']))
    values.update({name:score_queries(pred,adapted)[:,:2] for name,pred in scores.items()})
    array=np.stack(list(values.values()),axis=1);means=array.mean(axis=0)
    boot=cluster_bootstrap(array,selected.session_id.to_numpy(),1000);rows=[]
    for j,name in enumerate(values):
        change=100*(boot[:,j,0]/boot[:,0,0]-1)
        rows.append(dict(model=name,queries=len(selected),ndcg10=means[j,0],calibrated_recall10=means[j,1],
            ndcg_relative_to_benchmark=100*(means[j,0]/means[0,0]-1),relative_low=np.quantile(change,.025),relative_high=np.quantile(change,.975)))
    pd.DataFrame(rows).to_csv(OUT/'common_query_comparison.csv',index=False)
    increments=[]
    for k,metric in enumerate(['ndcg10','calibrated_recall10']):
        change=100*(boot[:,2,k]/boot[:,1,k]-1)
        increments.append(dict(method='original_hybrid',baseline='collaborative_tree',metric=metric,
            absolute_delta=means[2,k]-means[1,k],relative_percent=100*(means[2,k]/means[1,k]-1),
            relative_low=np.quantile(change,.025),relative_high=np.quantile(change,.975)))
    pd.DataFrame(increments).to_csv(OUT/'hybrid_increment.csv',index=False)
    np.savez_compressed(OUT/'per_query_metrics.npz',values=array,names=list(values),query_ids=selected.query_id.to_numpy())
    write_json(OUT/'receipt.json',dict(protocol_sha256=digest(protocol_path),queries=len(selected),
        original_feature_parity_queries=200,original_feature_max_error=parity,candidates_and_labels_match=True,
        archive_end='2016-05-14',all_queries_after_archive=True,
        metrics_sha256=digest(OUT/'per_query_metrics.npz'),
        model_hashes=protocol['model_hashes'],interpretation=protocol['status']))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':run()
