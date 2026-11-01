"""Matched treatment diagnostics after confirmation; no new model selection.

Cross label construction with negative weighting to separate the two changes
that were confounded in the first exposure experiment. Fixed 123-tree budget.
"""
import json
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from .benchmark_data import load,write_json,digest
from .benchmark_experiment import subset,feature_view,score_queries
from .protocol_core import exposure_weights
from .policies import cluster_bootstrap

OUT=Path('results/submission/factorial')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    specs=[dict(name=f'{label}_unseen_{weight:g}',label=label,unseen=weight,ablation='none')
        for label in ['binary','click_purchase','graded_views'] for weight in [1.,.2,0.]]
    specs += [dict(name=name,label='binary',unseen=1.,ablation=name) for name in ['no_session','no_identity','no_knn','no_content_history']]
    protocol=dict(status='Exploratory post-confirmation treatment diagnosis; no confirmation-based model or hyperparameter selection',
        specs=specs,training_fraction=.1,tau=5,delta=1.,trees=123,leaves=31,seed=4285,
        rationale='Common data, tree budget and feature schema. Original graded-versus-binary comparison changed labels and weights simultaneously.',
        ablation_definitions={'no_session':'Remove all session-derived fields while retaining old persistent profile features',
            'no_identity':'Mask all three persistent identity features', 'no_knn':'Remove the collaborative session-neighbor score',
            'no_content_history':'Remove direct token/exact-item session features while retaining collaborative kNN context'})
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:write_json(path,protocol)
    training=subset(load('train'));val=subset(load('validation'));confirmation=subset(load('test'),1.,fresh=True)
    X=feature_view(training['X'],5,1.);vx=feature_view(val['X'],5,1.);cx=feature_view(confirmation['X'],5,1.)
    weights={}
    for value in [1.,.2,0.]:
        w=np.ones(len(training['labels']),dtype=np.float32)
        if value!=1.:
            for a,b in zip(training['offsets'][:-1],training['offsets'][1:]):w[a:b]=exposure_weights(training['labels'][a:b],training['X'][a:b,7]>0,value)
        weights[value]=w
    records=[];metrics=[]
    for spec in specs:
        name=spec['name'];receipt=OUT/f'{name}.json';cached=OUT/f'{name}_metrics.npy'
        if receipt.exists() and cached.exists():
            records.append(json.loads(receipt.read_text()));metrics.append(np.load(cached));continue
        y=training['labels'] if spec['label']=='binary' else np.where(training['grades']>=2,training['grades'],0) if spec['label']=='click_purchase' else training['grades']
        w=weights[spec['unseen']].copy();w[y>0]=1.
        cols=list(range(18))
        if spec['ablation']=='no_session':cols=[i for i in cols if i not in [7,8,9,10,11,12,16,17]]
        elif spec['ablation']=='no_identity':cols=[i for i in cols if i not in [13,14,15]]
        elif spec['ablation']=='no_knn':cols.remove(16)
        elif spec['ablation']=='no_content_history':cols=[i for i in cols if i not in [7,8,9,10,11,12,17]]
        started=time.perf_counter()
        model=lgb.train(dict(objective='lambdarank',metric='ndcg',label_gain=[0,1,2,4],learning_rate=.05,num_leaves=31,
            min_data_in_leaf=150,lambda_l2=2,num_threads=4,seed=4285,deterministic=True,force_col_wise=True,verbosity=-1),
            lgb.Dataset(X[:,cols],label=y,weight=w,group=training['group']),num_boost_round=123)
        model_path=OUT/f'{name}.txt';model.save_model(str(model_path))
        fit_seconds=time.perf_counter()-started
        prediction=model.predict(vx[:,cols],num_threads=4);validation=score_queries(prediction,val)[:,:2]
        prediction=model.predict(cx[:,cols],num_threads=4);result=score_queries(prediction,confirmation)[:,:2]
        np.save(cached,result)
        record=dict(**spec,fit_seconds=fit_seconds,features=cols,model_sha256=digest(model_path),
            validation_ndcg=float(validation[val['meta'].positives>0,0].mean()),
            confirmation_ndcg=float(result[confirmation['meta'].positives>0,0].mean()))
        write_json(receipt,record);records.append(record);metrics.append(result)
        print(name,'validation',record['validation_ndcg'],'confirmation diagnostic',record['confirmation_ndcg'],flush=True)
    pd.DataFrame(records).to_csv(OUT/'treatment_runs.csv',index=False)
    array=np.stack(metrics,axis=1);positive=confirmation['meta'].positives.to_numpy()>0
    boot=cluster_bootstrap(array[positive],confirmation['meta'].loc[positive,'session_id'].to_numpy(),1000)
    mean=array[positive].mean(axis=0);names=[s['name'] for s in specs];rows=[]
    for j,name in enumerate(names):
        # Compare every treatment with binary naive and the same label's naive
        # condition, making the isolated weighting effect directly reviewable.
        for baseline in set(['binary_unseen_1',specs[j]['label']+'_unseen_1']):
            b=names.index(baseline)
            for k,metric in enumerate(['ndcg10','recall10_calibrated']):
                draws=100*(boot[:,j,k]/boot[:,b,k]-1)
                rows.append(dict(method=name,baseline=baseline,metric=metric,score=mean[j,k],
                    relative_percent=100*(mean[j,k]/mean[b,k]-1),low=np.quantile(draws,.025),high=np.quantile(draws,.975)))
    pd.DataFrame(rows).to_csv(OUT/'paired_effects.csv',index=False)
    np.savez_compressed(OUT/'diagnostic_metrics.npz',values=array,names=names)
    write_json(OUT/'receipt.json',dict(protocol_sha256=digest(path),queries=len(confirmation['meta']),positive_queries=int(positive.sum()),
        interpretation='Post-confirmation exploratory diagnostics, not an untouched test and not a selected replacement model'))


if __name__=='__main__':run()
