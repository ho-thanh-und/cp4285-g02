"""Reconstruct causal features and benchmark warm end-to-end query inference."""
import json
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from .benchmark import Catalog
from .core import tokens
from .protocol_core import DELTAS, LAMBDAS, causal_indices, floor_order
from .benchmark_data import OUT, event_table, historical_index, knn_scores, load, write_json
from .benchmark_experiment import feature_view, policy_indices, subset


def candidate_features(items, history, kinds, query_tokens, query_category, user_id, catalog, archive):
    """Only supplied pre-query context enters this reconstruction."""
    count,cat_total,incidence,sizes,users=archive
    n=len(history);unique=list(dict.fromkeys(history));lookup={v:i for i,v in enumerate(unique)}
    hist=catalog.similarity(items,unique) if unique else np.zeros((len(items),1),dtype=np.float32)
    seq_cols=np.array([lookup[i] for i in history],dtype=int)
    weighted=[]
    for delta in DELTAS:
        w=delta**np.arange(n-1,-1,-1)*kinds
        weighted.append(hist[:,seq_cols]@(w/w.sum()) if n else np.zeros(len(items)))
    reverse_unique=list(dict.fromkeys(reversed(history)))
    if len(reverse_unique)>1:
        hd=1-catalog.similarity(unique)
        diversity=float(hd.sum()/(len(unique)*(len(unique)-1)))
        contradiction=float(1-catalog.similarity(reverse_unique[:1],reverse_unique[1:2])[0,0])
    else:diversity=contradiction=0.
    known=pd.notna(user_id);profile=users.get(int(user_id),[]) if known else []
    pop=np.array([count.get(i,0) for i in items])
    X=np.column_stack([
        np.arange(len(items))/max(1,len(items)-1),1/(np.arange(len(items))+1),np.log1p(pop),
        [count.get(i,0)/max(1,cat_total[catalog.category.get(i,-1)]) for i in items],
        catalog.price[[catalog.row[i] for i in items]],
        [len(catalog.item_tokens[i]&query_tokens)/max(1,len(query_tokens)) for i in items],
        [query_category!=0 and catalog.category.get(i)==query_category for i in items],
        [i in lookup for i in items],hist.max(axis=1),
        hist[:,lookup[history[-1]]] if n else np.zeros(len(items)),
        np.full(len(items),n),np.full(len(items),diversity),np.full(len(items),contradiction),
        np.full(len(items),known),[i in profile for i in items],
        catalog.similarity(items,profile).max(axis=1) if profile else np.zeros(len(items)),
        knn_scores(items,unique,catalog,incidence,sizes),*weighted]).astype(np.float32)
    return X,dict(history_count=n,history_diversity=diversity,contradiction=contradiction,known=known)


def benchmark():
    path=OUT/'end_to_end_latency.json'
    if path.exists():return
    print('Benchmarking causal feature reconstruction and end-to-end local inference...',flush=True)
    frozen=json.loads((OUT/'frozen.json').read_text());record=frozen['selected']
    catalog=Catalog();q=pd.read_parquet('data/processed/queries_metadata.parquet')
    events,_=event_table(q);archive=historical_index(events,catalog)
    data=subset(load('validation'));rng=np.random.default_rng(4285)
    sample=rng.choice(len(data['group']),min(1000,len(data['group'])),replace=False)
    wanted=set(data['meta'].query_id.iloc[sample])
    query_context={}
    for chunk in pd.read_csv('data/raw/train-queries.csv',sep=';',usecols=['queryId','searchstring.tokens','categoryId'],chunksize=50000):
        for row in chunk[chunk.queryId.isin(wanted)].to_dict('records'):
            query_context[row['queryId']]=(tokens(row['searchstring.tokens']),row['categoryId'])
    lookup=q.set_index('queryId');groups=events.groupby('episode',sort=False).indices
    etime=events.timeframe.to_numpy();eitem=events.itemId.to_numpy();eqid=events.queryId.to_numpy();ekind=events.kind.to_numpy()
    model=lgb.Booster(model_file=str(OUT/'models'/f'{record["name"]}.txt'))
    durations={n:[] for n in ['popularity','session_knn','relevance','fixed_top30','continuous_top30']}
    max_error=0.
    for j in sample:
        meta=data['meta'].iloc[j];row=lookup.loc[meta.query_id]
        a,b=data['offsets'][j:j+2];items=data['items'][a:b].tolist()
        t=time.perf_counter()
        count=archive[0];pop=np.array([count.get(i,0) for i in items]);np.argsort(-pop,kind='stable')[:10]
        durations['popularity'].append(time.perf_counter()-t)
        t=time.perf_counter()
        indices=groups.get(row.episode,np.array([],dtype=int));idx=indices[causal_indices(etime[indices],row.timeframe)]
        idx=idx[(eqid[idx]!=meta.query_id)&np.array([i in catalog.row for i in eitem[idx]],dtype=bool)]
        history=eitem[idx].tolist();kinds=ekind[idx]
        history_seconds=time.perf_counter()-t
        t=time.perf_counter();knn=knn_scores(items,list(dict.fromkeys(history)),catalog,archive[2],archive[3]);np.argsort(-(knn+1e-6*np.log1p(pop)),kind='stable')[:10]
        durations['session_knn'].append(history_seconds+time.perf_counter()-t)
        t=time.perf_counter();qt,category=query_context[meta.query_id]
        X,context=candidate_features(items,history,kinds,qt,category,row.userId,catalog,archive)
        scores=model.predict(feature_view(X,record['tau'],record['delta']),num_threads=1)
        np.argsort(-scores,kind='stable')[:10]
        relevance_seconds=history_seconds+time.perf_counter()-t
        durations['relevance'].append(relevance_seconds)
        # Validation cache parity checks the actual reconstruction, not a mock.
        max_error=max(max_error,float(np.max(np.abs(X-data['X'][a:b]))))
        for name in ['fixed_top30','continuous_top30']:
            t=time.perf_counter();D=1-catalog.similarity(items);np.fill_diagonal(D,0)
            p=frozen['policies'][name];weight=LAMBDAS[policy_indices(p,pd.DataFrame([context]))[0]]
            floor_order(scores,D,weight,p['floor'])
            durations[name].append(relevance_seconds+time.perf_counter()-t)
    assert max_error<1e-5, f'Serving feature reconstruction differs from the prepared features: {max_error}'
    write_json(path,dict(queries=len(sample),components={name:dict(mean_ms=1000*np.mean(v),p50_ms=1000*np.median(v),p95_ms=1000*np.quantile(v,.95)) for name,v in durations.items()},
        max_feature_reconstruction_error=max_error,
        population='Same seeded validation-query sample for every method, including cold and no-click queries',
        scope='Warm local end-to-end inference: in-memory event lookup, required feature construction, single-thread model prediction, token distances and greedy reranking. Popularity needs no history lookup. Archive construction, disk reads and network transport excluded. Latencies measured while the confirmation process may be active.'))
    print(f'End-to-end inference checked on {len(sample)} queries; feature parity max error {max_error:g}.',flush=True)


if __name__=='__main__':benchmark()
