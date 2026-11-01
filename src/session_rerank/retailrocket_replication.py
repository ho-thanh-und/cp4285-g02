"""Prespecified independent Retailrocket replication; no injected test targets."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import sparse
import lightgbm as lgb
from .protocol_core import calibrated_metrics, floor_order
from .benchmark_data import write_json,digest
from .serving_benchmark import candidate_features
from .benchmark_experiment import feature_view
from .policies import cluster_bootstrap

RAW=Path('data/retailrocket');ROOT=RAW/'replication_v2';OUT=Path('results/submission/replication_v2')
PROTOCOL=Path('config/submission_replication_v2_protocol.json')
GRID=np.array([0,.25,.5,.75,.8,.85,.9,.95,.975,.99,1.])


def encode_categories(raw_categories):
    """Reserve zero for unknown; Retailrocket raw category zero is valid."""
    return {int(item):int(category)+1 for item,category in raw_categories.items()}


def candidate_pool(last_item,categories,pools,global_pool):
    category=categories.get(last_item,0) if last_item is not None else 0
    return category,pools.get(category,global_pool)


class PropertyCatalog:
    def __init__(self,mapping,categories):
        self.ids=np.array(sorted(mapping),dtype=np.int32)
        self.row={int(item):j for j,item in enumerate(self.ids)}
        self.item_tokens=mapping;self.category=categories
        self.price=np.zeros(len(self.ids),dtype=np.float32)
        vocabulary={};indices=[];indptr=[0]
        for item in self.ids:
            for token in sorted(mapping[int(item)]):
                if token not in vocabulary:vocabulary[token]=len(vocabulary)
                indices.append(vocabulary[token])
            indptr.append(len(indices))
        self.matrix=sparse.csr_matrix((np.ones(len(indices),dtype=np.float32),np.array(indices,dtype=np.int32),np.array(indptr,dtype=np.int64)),shape=(len(self.ids),len(vocabulary)))
        self.sizes=np.diff(self.matrix.indptr).astype(np.float32)

    def similarity(self,items,other=None):
        other=items if other is None else other
        a=np.array([self.row[int(i)] for i in items]);b=np.array([self.row[int(i)] for i in other])
        inter=(self.matrix[a]@self.matrix[b].T).toarray()
        union=self.sizes[a,None]+self.sizes[b][None,:]-inter
        return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)


def context():
    protocol=json.loads(PROTOCOL.read_text())
    e=pd.read_csv(RAW/'events.csv');assert len(e)==2_756_101
    e['day']=pd.to_datetime(e.timestamp,unit='ms',utc=True).dt.strftime('%Y-%m-%d')
    e=e.sort_values(['visitorid','timestamp','itemid','event']).reset_index(drop=True)
    boundary=e.visitorid.ne(e.visitorid.shift())|e.day.ne(e.day.shift())|e.timestamp.diff().gt(1_200_000)
    e['episode']=boundary.cumsum().astype('int32')
    e['kind']=e.event.map({'view':1.,'addtocart':2.,'transaction':4.})
    cutoff=int(pd.Timestamp(protocol['archive_end_exclusive'],tz='UTC').timestamp()*1000)
    cache=ROOT/'property_snapshot.json'
    if cache.exists():
        payload=json.loads(cache.read_text());mapping={int(k):set(v) for k,v in payload['tokens'].items()};categories={int(k):int(v) for k,v in payload['categories'].items()}
    else:
        wanted=set(e.itemid);latest={}
        for file in ['item_properties_part1.csv','item_properties_part2.csv']:
            for chunk in pd.read_csv(RAW/file,chunksize=250000,dtype={'property':str,'value':str}):
                chunk=chunk[(chunk.timestamp<cutoff)&chunk.itemid.isin(wanted)&(chunk.property!='available')]
                chunk=chunk.sort_values('timestamp').drop_duplicates(['itemid','property'],keep='last')
                for ts,item,prop,value in chunk[['timestamp','itemid','property','value']].itertuples(index=False,name=None):
                    key=(int(item),prop)
                    if key not in latest or ts>latest[key][0]:latest[key]=(int(ts),value)
            print('Read causal property snapshots:',file,len(latest),flush=True)
        mapping={};categories={}
        for (item,prop),(ts,value) in latest.items():
            if not isinstance(value,str):continue
            mapping.setdefault(item,set()).update(prop+':'+v for v in value.split())
            if prop=='categoryid':
                try:categories[item]=int(value)
                except ValueError:pass
        write_json(cache,dict(cutoff_exclusive=protocol['archive_end_exclusive'],tokens={str(k):sorted(v) for k,v in mapping.items()},categories=categories))
    categories=encode_categories(categories)
    catalog=PropertyCatalog(mapping,categories)
    past=e[(e.timestamp<cutoff)&e.itemid.isin(catalog.row)]
    popularity=past.groupby('itemid').kind.sum().to_dict();cat_total=Counter()
    for item,value in popularity.items():cat_total[categories.get(item,-1)]+=value
    distinct=past[['episode','itemid']].drop_duplicates();rows,unique=pd.factorize(distinct.episode)
    cols=np.array([catalog.row[int(i)] for i in distinct.itemid])
    incidence=sparse.csr_matrix((np.ones(len(rows),dtype=np.float32),(rows,cols)),shape=(len(unique),len(catalog.ids)))
    sizes=np.sqrt(np.asarray(incidence.sum(axis=1)).ravel())
    users={int(uid):list(dict.fromkeys(reversed(g.itemid.tolist())))[:50] for uid,g in past.groupby('visitorid',sort=False)}
    pools={};global_pool=sorted(popularity,key=lambda i:(-popularity[i],i))[:200]
    category_items={}
    for item in popularity:
        if item in categories:category_items.setdefault(categories[item],[]).append(item)
    for category,items in category_items.items():pools[category]=sorted(items,key=lambda i:(-popularity[i],i))[:200]
    return e,catalog,(popularity,cat_total,incidence,sizes,users),pools,global_pool


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'complete.json').exists():return
    e,catalog,archive,pools,global_pool=context();protocol=json.loads(PROTOCOL.read_text())
    groups=e.groupby('episode',sort=False).indices
    e['view_event']=e.event.eq('view')
    available=e.groupby('episode').agg(day=('day','first'),rows=('event','size'),views=('view_event','sum'),visitor=('visitorid','first'))
    for split,maximum in [('train',6000),('validation',1500),('confirmation',10000)]:
        folder=ROOT/split;folder.mkdir(exist_ok=True)
        if (folder/'manifest.json').exists():continue
        lo,hi=protocol[split];eligible=available[available.day.between(lo,hi)&(available.rows>=2)&(available.views>=1)]
        if split=='confirmation':
            previous=pd.read_parquet('results/submission/replication/confirmation_metadata.parquet')
            eligible=eligible[~eligible.visitor.isin(set(previous.session_id))]
        chosen=eligible.sample(n=min(maximum,len(eligible)),random_state=20261031 if split=='confirmation' else 4285).sort_index()
        Xs=[];items_out=[];labels=[];published=[];metadata=[];offsets=[0]
        for episode in chosen.index:
            group=e.iloc[groups[episode]]
            timestamps=group.timestamp.to_numpy();ids=group.itemid.to_numpy();kinds=group.kind.to_numpy()
            for j,row in enumerate(group.itertuples(index=False)):
                if row.event!='view' or j==0:continue
                right=int(np.searchsorted(timestamps,row.timestamp,side='left'))
                if not right:continue
                request_time=int(timestamps[right-1]);left=int(np.searchsorted(timestamps,request_time-1_200_000,side='left'))
                context_rows=[k for k in range(left,right) if int(ids[k]) in catalog.row]
                history=[int(ids[k]) for k in context_rows];weights=kinds[context_rows]
                category,candidates=candidate_pool(history[-1] if history else None,catalog.category,pools,global_pool)
                if not candidates:continue
                qt=catalog.item_tokens[history[-1]] if history else set()
                X,description=candidate_features(candidates,history,weights,qt,category,int(row.visitorid),catalog,archive)
                y=np.array([int(i)==int(row.itemid) for i in candidates],dtype=np.int8)
                # Eq.11 from the open Gharahighehi/Vens news paper, transferred
                # from cosine news embeddings to property-token Jaccard distance.
                dissimilarity=1-catalog.similarity(candidates,list(dict.fromkeys(history))).mean(axis=1) if history else np.ones(len(candidates))
                Xs.append(X);items_out.extend(candidates);labels.extend(y);published.extend(X[:,16]*dissimilarity)
                offsets.append(offsets[-1]+len(candidates))
                metadata.append(dict(query_id=len(metadata),session_id=int(row.visitorid),episode=int(episode),date=row.day,
                    history_count=len(history),history_diversity=description['history_diversity'],contradiction=description['contradiction'],
                    known=True,has_user_profile=int(row.visitorid) in archive[-1],positives=int(y.sum()),candidate_count=len(candidates),
                    category_count=len({catalog.category.get(i,-1) for i in candidates}),target=int(row.itemid),
                    target_in_catalog=int(row.itemid) in catalog.row,request_time=request_time,target_time=int(row.timestamp)))
            if len(metadata)%1000<5:print(split,len(metadata),'queries',flush=True)
        np.save(folder/'features.npy',np.concatenate(Xs));np.save(folder/'items.npy',np.array(items_out,dtype=np.int32))
        np.save(folder/'labels.npy',np.array(labels,dtype=np.int8));np.save(folder/'published.npy',np.array(published,dtype=np.float32))
        np.save(folder/'offsets.npy',np.array(offsets,dtype=np.int64));pd.DataFrame(metadata).to_parquet(folder/'meta.parquet',index=False)
        write_json(folder/'manifest.json',dict(episodes=len(chosen),queries=len(metadata),candidates=len(labels),
            retrieved_targets=int(sum(m['positives'] for m in metadata)),no_target_injection=True,
            query_ids_sha256=__import__('hashlib').sha256(pd.DataFrame(metadata)[['episode','target_time','target']].to_numpy().tobytes()).hexdigest()))
    # Keep only the metadata snapshot and matrix, never future property changes.
    sparse.save_npz(ROOT/'property_matrix.npz',catalog.matrix)
    np.save(ROOT/'catalog_ids.npy',catalog.ids);np.save(ROOT/'catalog_sizes.npy',catalog.sizes)
    write_json(ROOT/'complete.json',dict(protocol_sha256=digest(PROTOCOL),source_manifest_sha256=digest(RAW/'manifest.json'),
        raw_events=len(e),catalog_items=len(catalog.ids),archive_events=int((e.day<'2015-06-01').sum()),
        metadata_cutoff_exclusive='2015-06-01',archive_sessions=archive[2].shape[0]))


def load(split):
    p=ROOT/split
    return dict(X=np.load(p/'features.npy',mmap_mode='r'),items=np.load(p/'items.npy',mmap_mode='r'),labels=np.load(p/'labels.npy',mmap_mode='r'),
        published=np.load(p/'published.npy',mmap_mode='r'),offsets=np.load(p/'offsets.npy'),meta=pd.read_parquet(p/'meta.parquet'))


def distances(data):
    matrix=sparse.load_npz(ROOT/'property_matrix.npz').tocsr();ids=np.load(ROOT/'catalog_ids.npy');sizes=np.load(ROOT/'catalog_sizes.npy');index={int(i):j for j,i in enumerate(ids)}
    for a,b in zip(data['offsets'][:-1],data['offsets'][1:]):
        rows=np.array([index[int(i)] for i in data['items'][a:b]])
        inter=(matrix[rows]@matrix[rows].T).toarray();union=sizes[rows,None]+sizes[rows][None,:]-inter
        D=1-np.divide(inter,union,out=np.zeros_like(inter),where=union>0);np.fill_diagonal(D,0)
        yield D


def assess(data,scores,with_sweep=False):
    metrics={name:np.empty((len(data['meta']),3)) for name in scores}
    sweep=np.empty((len(data['meta']),3,len(GRID),3),dtype=np.float32) if with_sweep else None
    for j,((a,b),D) in enumerate(zip(zip(data['offsets'][:-1],data['offsets'][1:]),distances(data))):
        for name,pred in scores.items():
            order=np.argsort(-pred[a:b],kind='stable')[:10];metrics[name][j]=calibrated_metrics(order,data['labels'][a:b],D)
        if with_sweep:
            for fi,floor in enumerate(['none','top30','score03']):
                for wi,weight in enumerate(GRID):sweep[j,fi,wi]=calibrated_metrics(floor_order(scores['tree'][a:b],D,weight,floor),data['labels'][a:b],D)
    return metrics,sweep


def train_and_freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'frozen.json').exists():return
    training=load('train');validation=load('validation');tm=training['meta']
    eligible=tm.index[tm.positives>0].to_numpy();offsets=training['offsets'];records=[]
    predictions={}
    for seed in [4285,4286,4287]:
        path=OUT/f'tree_{seed}.txt'
        if path.exists():
            model=lgb.Booster(model_file=str(path))
        else:
            if seed==4285:take=eligible
            else:
                rng=np.random.default_rng(seed);ep=tm.loc[eligible,'episode'].unique();draw=rng.choice(ep,len(ep),replace=True)
                by_episode={key:g.index.to_numpy() for key,g in tm[tm.positives>0].groupby('episode')}
                take=np.concatenate([by_episode[e] for e in draw])
            rows=np.concatenate([np.arange(offsets[j],offsets[j+1]) for j in take]);groups=np.diff(offsets)[take]
            started=time.perf_counter()
            model=lgb.train(dict(objective='lambdarank',metric='ndcg',num_leaves=31,learning_rate=.05,min_data_in_leaf=150,
                lambda_l2=2,num_threads=4,seed=4285,deterministic=True,force_col_wise=True,verbosity=-1),
                lgb.Dataset(feature_view(training['X'][rows],5,1.),label=training['labels'][rows],group=groups),num_boost_round=123)
            model.save_model(str(path));print('Fitted replication model',seed,'seconds',time.perf_counter()-started,flush=True)
        pred=model.predict(feature_view(validation['X'],5,1.),num_threads=4);predictions[f'tree_{seed}']=pred
        records.append(dict(seed=seed,path=str(path),sha256=digest(path)))
    scores={'popularity':np.array(validation['X'][:,2]),'session_knn':np.array(validation['X'][:,16])+1e-6*validation['X'][:,2],
        'published_candidate_diversity':np.array(validation['published']), 'tree':predictions['tree_4285'],**predictions}
    metrics,sweep=assess(validation,scores,True);mask=validation['meta'].positives.to_numpy()>0
    np.savez_compressed(OUT/'validation_metrics.npz',names=list(metrics),values=np.stack(list(metrics.values()),axis=1),sweep=sweep)
    means=sweep[mask].astype(np.float64).mean(axis=0);reference=means[0,-1];m=validation['meta'];count=m.history_count.to_numpy();div=m.history_diversity.to_numpy()
    threshold=float(m.loc[count>=2,'history_diversity'].median());policies={};rows=[]
    for kind in ['fixed','gate','threshold','continuous']:
        for fi,floor in enumerate(['none','top30','score03']):
            best=None;quality=(-np.inf,-np.inf)
            for weight in GRID:
                for tau in ([5] if kind=='fixed' else [1,2,3,5]):
                    if kind=='fixed':weights=np.full(len(m),weight)
                    elif kind=='gate':weights=np.where(count>=tau,weight,1.)
                    elif kind=='threshold':weights=np.where((count>=tau)&(div>=threshold),weight,1.)
                    else:weights=1-(1-weight)*np.minimum(1,count/tau)*div
                    index=np.abs(weights[:,None]-GRID[None,:]).argmin(axis=1)
                    mean=sweep[np.arange(len(m)),fi,index][mask].astype(np.float64).mean(axis=0)
                    record=dict(kind=kind,floor=floor,floor_index=fi,weight=float(weight),tau=tau,threshold=threshold,
                        validation_ndcg=float(mean[0]),validation_recall=float(mean[1]),validation_ild=float(mean[2]))
                    rows.append(record)
                    if np.all(mean[:2]>=.99*reference[:2]-1e-12) and tuple(mean[[2,0]])>quality:best=record;quality=tuple(mean[[2,0]])
            assert best is not None;policies[f'{kind}_{floor}']=best
    pd.DataFrame(rows).to_csv(OUT/'policy_validation_search.csv',index=False)
    write_json(OUT/'frozen.json',dict(protocol_sha256=digest(PROTOCOL),data_sha256=digest(ROOT/'complete.json'),models=records,policies=policies,
        frozen_at=pd.Timestamp.now(tz='Asia/Singapore').isoformat(),primary_seed=4285,
        interpretation='No confirmation outcomes used for fitting, selecting seed, or selecting policies'))
    print('Independent-dataset model/policy hashes frozen.',flush=True)


def confirm():
    if (OUT/'confirmation_receipt.json').exists():return
    frozen=json.loads((OUT/'frozen.json').read_text());assert digest(PROTOCOL)==frozen['protocol_sha256']
    data=load('confirmation');data['meta'].to_parquet(OUT/'confirmation_metadata.parquet',index=False)
    scores={'popularity':np.array(data['X'][:,2]),'session_knn':np.array(data['X'][:,16])+1e-6*data['X'][:,2],
        'published_candidate_diversity':np.array(data['published'])}
    for r in frozen['models']:
        assert digest(r['path'])==r['sha256']
        scores[f'tree_{r["seed"]}']=lgb.Booster(model_file=r['path']).predict(feature_view(data['X'],5,1.),num_threads=4)
    primary=scores['tree_4285'];policy_indices={};m=data['meta'];count=m.history_count.to_numpy();div=m.history_diversity.to_numpy()
    for name,p in frozen['policies'].items():
        w=p['weight'];tau=p['tau']
        if p['kind']=='fixed':weights=np.full(len(m),w)
        elif p['kind']=='gate':weights=np.where(count>=tau,w,1.)
        elif p['kind']=='threshold':weights=np.where((count>=tau)&(div>=p['threshold']),w,1.)
        else:weights=1-(1-w)*np.minimum(1,count/tau)*div
        policy_indices[name]=np.abs(weights[:,None]-GRID[None,:]).argmin(axis=1)
    names=list(scores)+list(frozen['policies']);values=np.empty((len(m),len(names),3),dtype=np.float32)
    for j,((a,b),D) in enumerate(zip(zip(data['offsets'][:-1],data['offsets'][1:]),distances(data))):
        for k,name in enumerate(names):
            if name in scores:order=np.argsort(-scores[name][a:b],kind='stable')[:10]
            else:order=floor_order(primary[a:b],D,GRID[policy_indices[name][j]],frozen['policies'][name]['floor'])
            values[j,k]=calibrated_metrics(order,data['labels'][a:b],D)
        if (j+1)%1000==0:print('Independent confirmation',j+1,'/',len(m),flush=True)
    np.savez_compressed(OUT/'confirmation_metrics.npz',values=values,names=names)
    write_json(OUT/'confirmation_receipt.json',dict(queries=len(m),retrieved_queries=int((m.positives>0).sum()),
        frozen_sha256=digest(OUT/'frozen.json'),metrics_sha256=digest(OUT/'confirmation_metrics.npz'),
        source='Independent Retailrocket dataset; changed next-view task; no target injection; no best-seed selection',
        completed_at=pd.Timestamp.now(tz='Asia/Singapore').isoformat()))
    report()


def report():
    cache=np.load(OUT/'confirmation_metrics.npz');values=cache['values'].astype(np.float64);names=cache['names'].tolist()
    m=pd.read_parquet(OUT/'confirmation_metadata.parquet');positive=m.positives.to_numpy()>0
    conditional=values[positive].mean(axis=0);full=values.mean(axis=0)
    draws=cluster_bootstrap(values[positive],m.loc[positive,'session_id'].to_numpy(),1000)
    rows=[];comparisons=[]
    for j,name in enumerate(names):
        row=dict(method=name,all_queries=len(m),retrieved_queries=int(positive.sum()),retrieval_coverage=float(positive.mean()),
            conditional_ndcg10=conditional[j,0],conditional_hit10=conditional[j,1],conditional_ild10=conditional[j,2],
            unconditional_ndcg10=full[j,0],unconditional_hit10=full[j,1],all_query_ild10=full[j,2])
        for k,metric in enumerate(['ndcg','hit','ild']):row[metric+'_low'],row[metric+'_high']=np.quantile(draws[:,j,k],[.025,.975])
        rows.append(row)
        for baseline in ['tree_4285','fixed_top30','fixed_score03']:
            ref=names.index(baseline)
            for k,metric in enumerate(['ndcg10','hit10','ild10']):
                delta=draws[:,j,k]-draws[:,ref,k];relative=100*(draws[:,j,k]/draws[:,ref,k]-1)
                comparisons.append(dict(method=name,baseline=baseline,metric=metric,absolute_delta=conditional[j,k]-conditional[ref,k],
                    relative_percent=100*(conditional[j,k]/conditional[ref,k]-1),absolute_low=np.quantile(delta,.025),absolute_high=np.quantile(delta,.975),
                    relative_low=np.quantile(relative,.025),relative_high=np.quantile(relative,.975)))
    pd.DataFrame(rows).to_csv(OUT/'confirmation_summary.csv',index=False);pd.DataFrame(comparisons).to_csv(OUT/'paired_differences.csv',index=False)
    print(pd.DataFrame(rows)[['method','conditional_ndcg10','conditional_hit10','conditional_ild10','unconditional_hit10']].to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','confirm','report','all']);a=p.parse_args().stage
    if a in ['prepare','all']:prepare()
    if a in ['train','all']:train_and_freeze()
    if a in ['confirm','all']:confirm()
    if a=='report':report()


if __name__=='__main__':main()
