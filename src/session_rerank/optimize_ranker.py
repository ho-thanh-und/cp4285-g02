"""Validation-guided relevance model experiments, independent of confirmation."""
import argparse
import json
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from .optimization_data import load_features, FEATURE_NAMES
from .recbole_model import ndcg_by_query

OUT=Path('results/optimization')


def train(name,columns,leaves=31,query_cap=None,variant='base'):
    if (OUT/'frozen_selection.json').exists():raise RuntimeError('Model selection already frozen')
    if (OUT/f'{name}.txt').exists():raise RuntimeError('Model already recorded')
    training=load_features('train',variant)
    validation=load_features('validation',variant)
    names=list(FEATURE_NAMES)
    if variant=='views':
        from .view_features import VIEW_NAMES
        names+=VIEW_NAMES
    started=time.perf_counter()
    if query_cap and query_cap<len(training['meta']):
        chosen=np.sort(np.random.default_rng(4285).choice(len(training['meta']),query_cap,replace=False))
        rows=np.concatenate([np.arange(training['offsets'][i],training['offsets'][i+1]) for i in chosen])
        X=training['X'][rows,:columns]
        y=training['y'][rows]
        groups=np.diff(training['offsets'])[chosen]
    else:
        X=training['X'][:,:columns]
        y=training['y']
        groups=np.diff(training['offsets'])
    model=lgb.LGBMRanker(objective='lambdarank',n_estimators=800,learning_rate=.05,num_leaves=leaves,
        max_depth=-1,min_child_samples=150,reg_lambda=5.,n_jobs=4,random_state=4285,
        deterministic=True,force_col_wise=True,verbosity=-1,max_bin=127,
        subsample=.85,subsample_freq=1,colsample_bytree=.9,lambdarank_truncation_level=20)
    categorical=['item_category'] if columns>names.index('item_category') else []
    print(f'Training {name}: {len(groups):,} queries, {len(y):,} candidates, {columns} features, {leaves} leaves',flush=True)
    model.fit(X,y,group=groups,feature_name=names[:columns],categorical_feature=categorical,
        eval_set=[(validation['X'][:,:columns],validation['y'])],eval_group=[np.diff(validation['offsets'])],
        eval_at=[10],callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(50)])
    prediction=model.predict(validation['X'][:,:columns])
    value=float(ndcg_by_query(prediction,validation['y'],validation['offsets']).mean())
    model.booster_.save_model(str(OUT/f'{name}.txt'))
    np.save(OUT/f'{name}_validation.npy',prediction)
    summary=dict(name=name,validation_ndcg10=value,training_queries=len(groups),training_candidates=len(y),
                 features=columns,variant=variant,num_leaves=leaves,best_iteration=model.best_iteration_,seconds=time.perf_counter()-started)
    (OUT/f'{name}.json').write_text(json.dumps(summary,indent=2)+'\n')
    pd.DataFrame({'feature':names[:columns],'gain':model.booster_.feature_importance('gain')}).to_csv(OUT/f'{name}_feature_importance.csv',index=False)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--features',choices=['baseline','enhanced','views'],default='enhanced')
    parser.add_argument('--leaves',type=int,default=31)
    parser.add_argument('--queries',type=int)
    args=parser.parse_args()
    columns=12 if args.features=='baseline' else len(FEATURE_NAMES)
    if args.features=='views':
        from .view_features import VIEW_NAMES
        columns+=len(VIEW_NAMES)
    train(args.name,columns,args.leaves,args.queries,'views' if args.features=='views' else 'base')


if __name__=='__main__':main()
