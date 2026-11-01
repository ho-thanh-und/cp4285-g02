"""Bounded latency harness with real popularity and session-kNN retrieval."""
import time
import numpy as np
import pandas as pd
from .benchmark import Catalog
from .protocol_core import causal_indices
from .benchmark_data import OUT, event_table, historical_index, knn_scores, load, write_json
from .benchmark_experiment import subset


def benchmark():
    path=OUT/'baseline_latency.json'
    if path.exists():return
    print('Benchmarking actual popularity and session-kNN retrieval...',flush=True)
    catalog=Catalog()
    queries=pd.read_parquet('data/processed/queries_metadata.parquet')
    events,_=event_table(queries)
    count,cat_total,incidence,sizes,_=historical_index(events,catalog)
    data=subset(load('validation'))
    sample=np.random.default_rng(4285).choice(len(data['group']),min(2000,len(data['group'])),replace=False)
    lookup=queries.set_index('queryId')
    groups=events.groupby('episode',sort=False).indices
    etime=events.timeframe.to_numpy();eitem=events.itemId.to_numpy();eqid=events.queryId.to_numpy()
    durations={n:[] for n in ['history_lookup','global_popularity','category_popularity','session_knn','token_distances']}
    for j in sample:
        row=lookup.loc[data['meta'].query_id.iloc[j]]
        a,b=data['offsets'][j:j+2];items=data['items'][a:b].tolist()
        t=time.perf_counter()
        indices=groups.get(row.episode,np.array([],dtype=int))
        idx=indices[causal_indices(etime[indices],row.timeframe)]
        history=list(dict.fromkeys(eitem[i] for i in idx if eqid[i]!=row.name and eitem[i] in catalog.row))
        durations['history_lookup'].append(time.perf_counter()-t)
        t=time.perf_counter();pop=np.array([count.get(i,0) for i in items]);np.argsort(-pop,kind='stable')[:10]
        durations['global_popularity'].append(time.perf_counter()-t)
        t=time.perf_counter()
        cp=np.array([count.get(i,0)/max(1,cat_total[catalog.category.get(i,-1)]) for i in items]);np.argsort(-cp,kind='stable')[:10]
        durations['category_popularity'].append(time.perf_counter()-t)
        t=time.perf_counter();knn=knn_scores(items,history,catalog,incidence,sizes);np.argsort(-(knn+1e-6*np.log1p(pop)),kind='stable')[:10]
        durations['session_knn'].append(time.perf_counter()-t)
        t=time.perf_counter();catalog.similarity(items)
        durations['token_distances'].append(time.perf_counter()-t)
    write_json(path,dict(queries=len(sample),population='Seeded common 10% validation subset, all queries including cold and zero-click',
        components={name:dict(mean_ms=1000*np.mean(v),p50_ms=1000*np.median(v),p95_ms=1000*np.quantile(v,.95)) for name,v in durations.items()},
        note='Warm in-memory single-query feature retrieval/ranking components. Archive construction, I/O and full tree-feature construction excluded. Session-kNN includes actual neighbor retrieval; token distances measured separately.'))


if __name__=='__main__':benchmark()
