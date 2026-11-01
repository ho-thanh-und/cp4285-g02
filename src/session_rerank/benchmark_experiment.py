"""Execute validation search, four training scales, frozen policies and confirmation.

Reproduce: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -u
    -m session_rerank.benchmark_experiment all
"""
import argparse
import json
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from .benchmark import Catalog
from .protocol_core import DELTAS, TAUS, LAMBDAS, calibrated_metrics, exposure_weights, floor_order
from .benchmark_data import ROOT, OUT, NAMES, load, write_json, digest


def feature_view(X, tau=3, delta=.85, anonymous=False):
    # A single decay summary enters each model. Other deltas are not visible.
    x = np.array(X[:,list(range(17))+[NAMES.index(f'decay_{delta}')]],copy=True)
    confidence = np.minimum(1., x[:,10]/tau)
    x[:,7] *= confidence
    for column in [8,9,17]:
        x[:,column] = confidence*x[:,column] + (1-confidence)*x[:,5]
    x[:,11] *= confidence
    x[:,16] *= confidence
    if anonymous:
        x[:,13:16] = 0
    return x


def subset(data, fraction=.1, fresh=False):
    meta = data['meta']
    mask = meta.fraction_rank.to_numpy() < fraction
    if fresh:
        forbidden=set()
        protocol=json.loads(Path('config/session_benchmark_protocol.json').read_text())
        for path in protocol['previously_inspected_confirmation_exclusion']:
            forbidden.update(pd.read_parquet(path).sessionId.astype(int))
        mask &= ~meta.session_id.isin(forbidden).to_numpy()
    queries = np.flatnonzero(mask)
    group = np.diff(data['offsets'])[queries]
    rows = np.concatenate([np.arange(data['offsets'][i],data['offsets'][i+1]) for i in queries])
    return dict(X=np.asarray(data['X'][rows]),items=np.asarray(data['items'][rows]),
        labels=np.asarray(data['labels'][rows]),grades=np.asarray(data['grades'][rows]),
        offsets=np.r_[0,np.cumsum(group)],meta=meta.iloc[queries].reset_index(drop=True),group=group)


def score_queries(scores, data, distances=None):
    result = np.zeros((len(data['group']),3))
    for j,(a,b) in enumerate(zip(data['offsets'][:-1],data['offsets'][1:])):
        order=np.argsort(-scores[a:b],kind='stable')[:10]
        result[j]=calibrated_metrics(order,data['labels'][a:b],None if distances is None else distances[j])
    return result


def positive_mean(values, data):
    return values[data['meta'].positives.to_numpy()>0].mean(axis=0)


def fit(train, val, name, tau=3, delta=.85, leaves=31, unseen=.2, binary=False,
        anonymous=False, filtered=False, augment=False):
    model_path=OUT/'models'/f'{name}.txt'
    receipt_path=model_path.with_suffix('.json')
    prediction_path=OUT/'predictions'/f'{name}.npy'
    if model_path.exists() and receipt_path.exists() and prediction_path.exists():
        return json.loads(receipt_path.read_text()),np.load(prediction_path)
    started=time.perf_counter()
    X=feature_view(train['X'],tau,delta,anonymous)
    y=train['labels'] if binary else train['grades']
    weights=np.ones(len(y),dtype=np.float32)
    if not binary:
        for a,b in zip(train['offsets'][:-1],train['offsets'][1:]):
            weights[a:b]=exposure_weights(train['labels'][a:b],train['X'][a:b,7]>0,unseen)
            weights[a:b][y[a:b]>0]=1.
    groups=train['group']
    if filtered:
        m=train['meta']
        keep=~(m.anomalous.to_numpy() | (m.candidate_count.to_numpy()==1) |
               ((m.category_count.to_numpy()>1)&(m.contradiction.to_numpy()>.8)))
        rows=np.repeat(keep,groups)
        X,y,weights,groups=X[rows],y[rows],weights[rows],groups[keep]
    if augment:
        # Label-preserving alternate historical context. Replace the most recent
        # history item with the candidate of greatest token similarity to it.
        # Retain original examples, append perturbed contexts at half weight.
        catalog=Catalog()
        alternate=X.copy()
        for j,(a,b) in enumerate(zip(train['offsets'][:-1],train['offsets'][1:])):
            if train['meta'].history_count.iloc[j]<1:
                continue
            source=int(np.argmax(train['X'][a:b,9]))
            sim=catalog.similarity(train['items'][a:b].tolist(),[int(train['items'][a+source])])[:,0]
            alternate[a:b,9]=sim
            alternate[a:b,17]=.5*alternate[a:b,17]+.5*sim
            alternate[a:b,8]=np.maximum(alternate[a:b,8],sim)
            alternate[a:b,7]=0;alternate[a+source,7]=1
        X=np.concatenate([X,alternate]); y=np.tile(y,2)
        weights=np.r_[weights,.5*weights];groups=np.tile(groups,2)
    vx=feature_view(val['X'],tau,delta,anonymous)
    model=lgb.LGBMRanker(objective='lambdarank',n_estimators=160,learning_rate=.05,
        num_leaves=leaves,max_depth=-1,min_child_samples=150,reg_lambda=2.,
        n_jobs=4,random_state=4285,deterministic=True,force_col_wise=True,
        label_gain=[0,1,2,4],verbosity=-1)
    model.fit(X,y,group=groups,sample_weight=weights,
        eval_set=[(vx,val['labels'])],eval_group=[val['group']],eval_at=[10],
        callbacks=[lgb.early_stopping(20,verbose=False)])
    training_seconds=time.perf_counter()-started
    t=time.perf_counter();prediction=model.predict(vx,num_threads=4)
    predict_seconds=time.perf_counter()-t
    scores=positive_mean(score_queries(prediction,val),val)
    record=dict(name=name,tau=tau,delta=delta,leaves=leaves,unseen=unseen,binary=binary,
        anonymous=anonymous,filtered=filtered,augment=augment,train_queries=int(len(groups)),
        train_candidates=int(len(y)),trees=model.best_iteration_,training_seconds=training_seconds,
        prediction_ms_per_query=1000*predict_seconds/len(val['group']),ndcg10=scores[0],recall10=scores[1],
        model_sha256=None)
    model.booster_.save_model(str(model_path))
    record['model_sha256']=digest(model_path)
    np.save(prediction_path,prediction)
    write_json(receipt_path,record)
    pd.DataFrame({'feature':[NAMES[i] for i in list(range(17))]+[f'decay_{delta}'],
        'gain':model.booster_.feature_importance('gain')}).to_csv(model_path.with_suffix('.importance.csv'),index=False)
    print(f'{name}: NDCG {scores[0]:.6f}, Recall {scores[1]:.6f}; {training_seconds:.1f}s, {model.best_iteration_} trees',flush=True)
    return record,prediction


def development():
    OUT.mkdir(parents=True,exist_ok=True)
    for path in ['models','predictions','figures']:
        (OUT/path).mkdir(exist_ok=True)
    if (OUT/'frozen.json').exists():
        print('Session benchmark frozen; development is complete.',flush=True)
        return
    train_data=load('train');val_data=load('validation')
    train=subset(train_data);val=subset(val_data)
    write_json(OUT/'development_population.json',dict(train_queries=len(train['group']),validation_queries=len(val['group']),
        validation_positives=int((val['meta'].positives>0).sum()),validation_meta_sha256=digest(ROOT/'validation'/'meta.parquet')))
    records=[]
    # All 20 prescribed threshold/decay combinations are independently fitted.
    for tau in TAUS:
        for delta in DELTAS:
            record,_=fit(train,val,f'dynamics_t{tau}_d{delta}',tau=tau,delta=delta)
            records.append(record)
    best=max(records,key=lambda r:(r['ndcg10'],r['recall10']))
    tau,delta=best['tau'],best['delta']
    variants=[('naive',dict(binary=True)),('naive15',dict(binary=True,leaves=15)),
        ('naive63',dict(binary=True,leaves=63)),('unseen0',dict(unseen=0.)),
        ('leaves15',dict(leaves=15)),('leaves63',dict(leaves=63)),
        ('anonymous',dict(anonymous=True)),('filtered',dict(filtered=True)),
        ('augmented',dict(augment=True))]
    for name,kwargs in variants:
        record,_=fit(train,val,name,tau=tau,delta=delta,**kwargs);records.append(record)
    pd.DataFrame(records).to_csv(OUT/'validation_search.csv',index=False)
    # Hold identity ablation out of the deployment choice so its matched control
    # remains available. Other variants may win; no family is assumed superior.
    best=max([r for r in records if not r['anonymous']],key=lambda r:(r['ndcg10'],r['recall10']))
    write_json(OUT/'development_choice.json',best)
    params={k:best[k] for k in ['tau','delta','leaves','unseen','binary','filtered','augment']}
    scale_records=[]
    for fraction in [.1,.25,.5,1.]:
        training=subset(train_data,fraction)
        record,_=fit(training,val,f'scale_{fraction:g}',**params)
        record['fraction']=fraction;scale_records.append(record)
        del training
    pd.DataFrame(scale_records).to_csv(OUT/'scalability.csv',index=False)
    selected=max(scale_records,key=lambda r:(r['ndcg10'],r['recall10']))
    # Fully retrained identity control at exactly the selected scale/config.
    training=subset(train_data,selected['fraction'])
    control,_=fit(training,val,'matched_anonymous',anonymous=True,**params)
    del training,train,train_data
    write_json(OUT/'selected_relevance.json',selected)
    write_json(OUT/'matched_anonymous.json',control)
    select_policies(val,selected,control)


def policy_values(kind, count, diversity, weight, tau=3, threshold=.5):
    confidence=np.minimum(1.,count/max(1,tau))
    if kind=='fixed':
        return np.full(len(count),weight)
    if kind=='gate':
        return np.where(count>=tau,weight,1.)
    if kind=='threshold':
        return np.where((count>=tau)&(diversity>=threshold),weight,1.)
    if kind=='continuous':
        return 1-(1-weight)*confidence*diversity
    raise ValueError(kind)


def build_policy_cache(data,scores,split):
    path=OUT/f'{split}_policy_cache.npz'
    floors=['none','top30','score03']
    if path.exists():
        return np.load(path)['values']
    catalog=Catalog()
    values=np.empty((len(data['group']),len(floors),len(LAMBDAS),3),dtype=np.float32)
    started=time.perf_counter()
    for j,(a,b) in enumerate(zip(data['offsets'][:-1],data['offsets'][1:])):
        D=1-catalog.similarity(data['items'][a:b].tolist());np.fill_diagonal(D,0)
        for f,floor in enumerate(floors):
            for w,weight in enumerate(LAMBDAS):
                order=floor_order(scores[a:b],D,weight,floor)
                values[j,f,w]=calibrated_metrics(order,data['labels'][a:b],D)
        if (j+1)%2000==0:
            print(f'{split} MMR sweep: {j+1:,}/{len(data["group"]):,}; {time.perf_counter()-started:.0f}s',flush=True)
    np.savez_compressed(path,values=values)
    return values


def controller_features(meta):
    return np.column_stack([np.log1p(meta.history_count),meta.history_diversity,meta.contradiction,meta.known.astype(int)])


def policy_indices(policy, meta, learned=None):
    if policy['kind']=='learned':
        weights=learned.predict(controller_features(meta))
    else:
        weights=policy_values(policy['kind'],meta.history_count.to_numpy(),meta.history_diversity.to_numpy(),
            policy['weight'],policy['tau'],policy.get('threshold',.5))
    # The continuous controller uses the same declared lambda decision grid.
    return np.abs(weights[:,None]-np.array(LAMBDAS)[None,:]).argmin(axis=1)


def choose_feasible(candidates, meta, cache, mask, reference, learned=None):
    best=None;best_score=(-np.inf,-np.inf)
    floor=reference[:2]*.99
    for p in candidates:
        idx=policy_indices(p,meta,learned)
        values=cache[np.arange(len(meta)),p['floor_index'],idx]
        mean=values[mask].mean(axis=0)
        if np.all(mean[:2]>=floor-1e-9) and tuple(mean[[2,0]])>best_score:
            best=dict(p,validation_ndcg=float(mean[0]),validation_recall=float(mean[1]),validation_ild=float(mean[2]))
            best_score=tuple(mean[[2,0]])
    return best


def select_policies(val, selected, control):
    scores=np.load(OUT/'predictions'/f'{selected["name"]}.npy')
    cache=build_policy_cache(val,scores,'validation')
    meta=val['meta'];positive=meta.positives.to_numpy()>0
    ref=cache[positive,0,-1].mean(axis=0)
    threshold=float(meta.loc[meta.history_count>=2,'history_diversity'].median())
    if not np.isfinite(threshold):threshold=.5
    policies={}
    search=[]
    for kind in ['fixed','gate','threshold','continuous']:
        for floor_index,floor in enumerate(['none','top30','score03']):
            candidates=[dict(kind=kind,floor=floor,floor_index=floor_index,weight=w,tau=tau,threshold=threshold)
                for w in LAMBDAS for tau in (TAUS if kind!='fixed' else [3])]
            selected_policy=choose_feasible(candidates,meta,cache,positive,ref)
            assert selected_policy is not None
            policies[f'{kind}_{floor}']=selected_policy
            for p in candidates:
                idx=policy_indices(p,meta)
                mean=cache[np.arange(len(meta)),floor_index,idx][positive].mean(axis=0)
                search.append(dict(**p,ndcg10=mean[0],recall10=mean[1],ild10=mean[2]))
    pd.DataFrame(search).to_csv(OUT/'policy_validation_search.csv',index=False)
    # Fit a small continuous controller on one half of validation session IDs,
    # then make the feasibility/selection decision on the other half only.
    unique=meta.session_id.unique().copy();rng=np.random.default_rng(4285);rng.shuffle(unique)
    fitmask=meta.session_id.isin(unique[:len(unique)//2]).to_numpy()&positive
    checkmask=(~meta.session_id.isin(unique[:len(unique)//2]).to_numpy())&positive
    row_values=cache[:,1]
    reward=row_values[:,:,2] - 8*np.maximum(0,row_values[:,-1:,0]-row_values[:,:,0]) - 4*np.maximum(0,row_values[:,-1:,1]-row_values[:,:,1])
    target=np.array(LAMBDAS)[reward.argmax(axis=1)]
    learned=DecisionTreeRegressor(max_depth=2,min_samples_leaf=100,random_state=4285)
    learned.fit(controller_features(meta)[fitmask],target[fitmask])
    import pickle
    with open(OUT/'learned_controller.pkl','wb') as f:
        pickle.dump(learned,f)
    candidate=dict(kind='learned',floor='top30',floor_index=1)
    learned_choice=choose_feasible([candidate],meta,cache,checkmask,cache[checkmask,0,-1].mean(axis=0),learned)
    policies['learned_top30']=learned_choice or dict(kind='fixed',floor='top30',floor_index=1,weight=1.,tau=3,
        fallback_reason='Learned controller failed the 1% NDCG/Recall constraint on its internal validation holdout')
    payload=dict(protocol_sha256=digest('config/session_benchmark_protocol.json'),selected=selected,
        matched_anonymous=control,policies=policies,learned_controller_sha256=digest(OUT/'learned_controller.pkl'),
        diagnostic_models=[json.loads((OUT/'models'/f'{name}.json').read_text()) for name in
            ['naive','unseen0','filtered','augmented',f'dynamics_t{selected["tau"]}_d{selected["delta"]}']],
        scale_models=pd.read_csv(OUT/'scalability.csv').to_dict('records'),
        validation_policy_cache_sha256=digest(OUT/'validation_policy_cache.npz'),
        frozen_at=pd.Timestamp.now(tz='Asia/Singapore').isoformat(),
        confirmation_opened=False,selection_scope='All choices use development/validation only; this replaces no prior result')
    write_json(OUT/'frozen.json',payload)
    print('Model/policies frozen before confirmation outcome scoring.',flush=True)


def verify_confirmation_receipt(frozen):
    receipt=json.loads((OUT/'confirmation_receipt.json').read_text())
    checks=[(OUT/'frozen.json',receipt['frozen_sha256']),
            (OUT/'confirmation_metrics.npz',receipt['metrics_sha256'])]
    records=[frozen['selected'],frozen['matched_anonymous']]+frozen['diagnostic_models']+frozen['scale_models']
    checks.extend((OUT/'models'/f'{m["name"]}.txt',m['model_sha256']) for m in records)
    for path,expected in checks:
        if digest(path)!=expected:raise RuntimeError(f'Frozen confirmation input/output changed: {path}')
    meta=pd.read_parquet(OUT/'confirmation_metadata.parquet')
    ids=__import__('hashlib').sha256(meta.query_id.to_numpy().tobytes()).hexdigest()
    if ids!=receipt['query_ids_sha256']:raise RuntimeError('Confirmation query IDs or order changed')
    if (len(meta)!=receipt['queries'] or int(meta.candidate_count.sum())!=receipt['candidates']
            or int((meta.positives>0).sum())!=receipt['positive_queries']):
        raise RuntimeError('Confirmation population differs from its receipt')


def confirmation():
    frozen=json.loads((OUT/'frozen.json').read_text())
    if digest('config/session_benchmark_protocol.json')!=frozen['protocol_sha256']:
        raise RuntimeError('Protocol changed after freezing')
    if (OUT/'confirmation_receipt.json').exists():
        verify_confirmation_receipt(frozen)
        print('Confirmation already recorded; reusing existing results.',flush=True)
        return
    for model in [frozen['selected'],frozen['matched_anonymous']]+frozen['diagnostic_models']+frozen['scale_models']:
        if digest(OUT/'models'/f'{model["name"]}.txt')!=model['model_sha256']:
            raise RuntimeError('Frozen model hash mismatch')
    if digest(OUT/'learned_controller.pkl')!=frozen['learned_controller_sha256']:
        raise RuntimeError('Frozen controller hash mismatch')
    data=subset(load('test'),1.,fresh=True)
    data['meta'].to_parquet(OUT/'confirmation_metadata.parquet',index=False)
    selected=frozen['selected'];control=frozen['matched_anonymous']
    predictions={};latency={}
    # All scale models were fixed by validation before this function is called.
    scales=pd.DataFrame(frozen['scale_models'])
    for record in scales.to_dict('records')+[control]+frozen['diagnostic_models']:
        name=record['name']
        model=lgb.Booster(model_file=str(OUT/'models'/f'{name}.txt'))
        started=time.perf_counter()
        features=feature_view(data['X'],record['tau'],record['delta'],bool(record['anonymous']))
        prediction=model.predict(features,num_threads=4)
        latency[name]=1000*(time.perf_counter()-started)/len(data['group'])
        predictions[name]=prediction
        del features
    scores=predictions[selected['name']]
    # Compute only the frozen policies, not a fresh confirmation policy sweep.
    import pickle
    with open(OUT/'learned_controller.pkl','rb') as f:learned=pickle.load(f)
    weights={name:policy_indices(p,data['meta'],learned) for name,p in frozen['policies'].items()}
    model_scores={'original_serp':-data['X'][:,0],
        'global_popularity':data['X'][:,2], 'category_popularity':data['X'][:,3],
        'session_knn':data['X'][:,16]+1e-6*data['X'][:,2],
        'relevance':scores,'anonymous':predictions[control['name']]}
    model_scores.update({r['name']:predictions[r['name']] for r in scales.to_dict('records')})
    model_scores.update({r['name']:predictions[r['name']] for r in frozen['diagnostic_models']})
    names=list(model_scores)+list(frozen['policies'])
    metrics=np.empty((len(data['group']),len(names),3),dtype=np.float32)
    policy_timing={n:0. for n in frozen['policies']}
    started=time.perf_counter();examples=[]
    for j,(a,b) in enumerate(zip(data['offsets'][:-1],data['offsets'][1:])):
        items=data['items'][a:b].tolist()
        # Construct once for all policies, and report shared similarity cost.
        if j==0:catalog=Catalog()
        D=1-catalog.similarity(items);np.fill_diagonal(D,0)
        orders={name:np.argsort(-s[a:b],kind='stable')[:10] for name,s in model_scores.items()}
        for name,p in frozen['policies'].items():
            t=time.perf_counter()
            orders[name]=floor_order(scores[a:b],D,LAMBDAS[weights[name][j]],p['floor'])
            policy_timing[name]+=time.perf_counter()-t
        for m,name in enumerate(names):
            metrics[j,m]=calibrated_metrics(orders[name],data['labels'][a:b],D)
        row=data['meta'].iloc[j]
        if len(examples)<16 and row.history_count>=2:
            examples.append(dict(query_id=int(row.query_id),session_id=int(row.session_id),timeframe=int(row.timeframe),
                history_count=int(row.history_count),diversity=float(row.history_diversity),contradiction=float(row.contradiction),
                relevant=[i for i,positive in zip(items,data['labels'][a:b]) if positive],
                top10={n:[items[k] for k in orders[n]] for n in ['original_serp','relevance','fixed_top30','continuous_top30']}))
        if (j+1)%5000==0:
            print(f'Confirmation: {j+1:,}/{len(data["group"]):,}; {time.perf_counter()-started:.0f}s',flush=True)
    np.savez_compressed(OUT/'confirmation_metrics.npz',values=metrics,names=names)
    write_json(OUT/'confirmation_examples.json',examples)
    write_json(OUT/'confirmation_timing.json',dict(prediction_ms_per_query=latency,
        mmr_ms_per_query={n:1000*t/len(data['group']) for n,t in policy_timing.items()},
        combined_similarity_all_methods_and_metrics_seconds=time.perf_counter()-started,
        note='Prediction timings include feature transformation and inference. MMR timings exclude shared token distance construction; the combined total includes evaluation overhead and is not production latency.'))
    write_json(OUT/'confirmation_receipt.json',dict(queries=len(data['group']),candidates=len(data['labels']),
        positive_queries=int((data['meta'].positives>0).sum()),frozen_sha256=digest(OUT/'frozen.json'),
        metrics_sha256=digest(OUT/'confirmation_metrics.npz'),
        query_ids_sha256=__import__('hashlib').sha256(data['meta'].query_id.to_numpy().tobytes()).hexdigest(),
        completed_at=pd.Timestamp.now(tz='Asia/Singapore').isoformat()))
    print('Reserved confirmation evaluation complete.',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['development','confirmation','checks','report','all'],
        help='all runs experiments and numerical checks; report generation requires the explicit report stage')
    args=p.parse_args(argv)
    if args.stage in ['development','all']:development()
    if args.stage in ['confirmation','all']:confirmation()
    if args.stage in ['checks','report','all']:
        from .protocol_audit import audit
        audit()
        from .latency_benchmark import benchmark
        benchmark()
        from .proxy_analysis import proxy_report
        proxy_report()
        from .case_studies import case_studies
        case_studies()
        from .serving_benchmark import benchmark as serving_benchmark
        serving_benchmark()
        from .report_generation import summarize
        summarize(write_documents=args.stage=='report')


if __name__=='__main__':
    main()
