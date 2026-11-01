"""Common-query diagnostics of frozen legacy tree/RecBole checkpoints.

No checkpoint fitting or selection occurs here. These are post-confirmation
diagnostics with explicitly different original training protocols.
"""
from collections import defaultdict
import json
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from .benchmark import Catalog
from .core import tokens
from .benchmark_data import load,digest,write_json
from .benchmark_experiment import subset,score_queries
from .policies import cluster_bootstrap

OUT=Path('results/submission/legacy_comparison')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=['results/model.txt','results/iteration3/gru_ce.pth','results/iteration3/sas_ce.pth']
    write_json(OUT/'diagnostic_protocol.json',dict(status='Post-confirmation diagnostic, not a new selection or independent confirmation',
        checkpoint_sha256={p:digest(p) for p in paths},
        task='Exactly the session benchmark confirmation queries, candidates and observed click labels',
        context='Legacy tree scored with original daily click history and separately with strict 20m click history; neural checkpoints use strictly causal 20m click prefixes',
        neural_fallback='The same fixed pre-April graded popularity feature as the session benchmark baseline, not the user-ID flag at feature column 13',
        caveat='Training feature sets and original training procedures differ; this is a common-population practical comparator, not an isolated architectural ablation'))
    data=subset(load('test'),1.,fresh=True);meta=data['meta'];catalog=Catalog()
    clicks=pd.read_parquet('data/processed/clicks.parquet')
    events=defaultdict(list)
    for row in clicks.itertuples(index=False):events[row.sessionId,row.eventdate].append((int(row.timeframe),int(row.queryId),int(row.itemId)))
    for e in events.values():e.sort()
    wanted=set(meta.query_id);queries={}
    for chunk in pd.read_csv('data/raw/train-queries.csv',sep=';',chunksize=50000,dtype={'searchstring.tokens':str}):
        for row in chunk[chunk.queryId.isin(wanted)].to_dict('records'):queries[row['queryId']]=row
    scores={}
    for mode in ['daily','strict_20m']:
        prediction_path=OUT/f'legacy_tree_{mode}.npy'
        if prediction_path.exists():scores[f'legacy_tree_{mode}']=np.load(prediction_path);continue
        X=np.empty((len(data['items']),12),dtype=np.float32);prefixes=np.zeros((len(meta),10),dtype=np.int32)
        counts=[]
        for j,row in enumerate(meta.itertuples(index=False)):
            q=queries[row.query_id];history=[];seen=set()
            for timestamp,source,item in reversed(events[row.session_id,row.date]):
                if timestamp>=q['timeframe'] or source==row.query_id or item not in catalog.row or item in seen:continue
                if mode=='strict_20m' and q['timeframe']-timestamp>1_200_000:continue
                history.append(item);seen.add(item)
                if len(history)==10:break
            a,b=data['offsets'][j:j+2]
            X[a:b]=catalog.features(data['items'][a:b].tolist(),history,tokens(q['searchstring.tokens']),q['categoryId'])
            prefixes[j,:len(history)]=list(reversed(history));counts.append(len(history))
        prediction=lgb.Booster(model_file='results/model.txt').predict(X,num_threads=1)
        np.save(prediction_path,prediction);scores[f'legacy_tree_{mode}']=prediction
        if mode=='strict_20m':np.save(OUT/'strict_click_prefixes.npy',prefixes);np.save(OUT/'strict_click_counts.npy',counts)
        del X
    from .neural_search import configuration
    from .recbole_model import candidate_scores
    from recbole.utils import get_model
    import torch
    torch.set_num_threads(1)
    class PopFallback:
        def __getitem__(self,key):
            rows,column=key
            if column!=13:raise ValueError('Unexpected legacy fallback feature access')
            return np.asarray(data['X'][rows,2])
    nn_data=dict(prefixes=np.load(OUT/'strict_click_prefixes.npy'),items=data['items'],offsets=data['offsets'],X=PopFallback())
    for variant in ['gru_ce','sas_ce']:
        pred_path=OUT/f'{variant}.npy'
        if pred_path.exists():scores[variant]=np.load(pred_path);continue
        checkpoint=torch.load(f'results/iteration3/{variant}.pth',map_location='cpu',weights_only=False)
        config=configuration(variant,'cpu')
        class Shape:
            def num(self,field):return checkpoint['n_items']
        model=get_model(checkpoint['model_name'])(config,Shape());model.load_state_dict(checkpoint['state_dict']);model.eval()
        start=time.perf_counter();prediction=candidate_scores(model,checkpoint['mapping'],nn_data,batch_size=256)
        np.save(pred_path,prediction);scores[variant]=prediction
        print('Scored frozen legacy neural model',variant,'seconds',time.perf_counter()-start,flush=True)
    cache=np.load('results/session_benchmark/confirmation_metrics.npz');names=cache['names'].tolist()
    values={'benchmark_tree':cache['values'][:,names.index('relevance'),:2].astype(np.float64),
            'graded_popularity':cache['values'][:,names.index('global_popularity'),:2].astype(np.float64)}
    values.update({name:score_queries(pred,data)[:,:2] for name,pred in scores.items()})
    counts=np.load(OUT/'strict_click_counts.npy');positive=meta.positives.to_numpy()>0;rows=[]
    for slice_name,mask in [('all_positive',positive),('with_20m_click_history',positive&(counts>0)),('without_20m_click_history',positive&(counts==0))]:
        array=np.stack(list(values.values()),axis=1)[mask];draws=cluster_bootstrap(array,meta.loc[mask,'session_id'].to_numpy(),1000)
        mean=array.mean(axis=0)
        for j,name in enumerate(values):
            difference=100*(draws[:,j]/draws[:,0]-1)
            rows.append(dict(segment=slice_name,model=name,queries=int(mask.sum()),ndcg10=mean[j,0],recall10=mean[j,1],
                ndcg_relative_to_benchmark=100*(mean[j,0]/mean[0,0]-1),
                ndcg_relative_low=np.quantile(difference[:,0],.025),ndcg_relative_high=np.quantile(difference[:,0],.975)))
    pd.DataFrame(rows).to_csv(OUT/'common_query_comparison.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':run()
