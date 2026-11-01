"""Secondary held-out token-overlap proxy for queries without click targets."""
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from .benchmark import Catalog
from .benchmark_data import OUT, load, write_json
from .protocol_core import LAMBDAS, floor_order
from .benchmark_experiment import feature_view, policy_indices


def proxy_report():
    frozen=json.loads((OUT/'frozen.json').read_text());record=frozen['selected']
    test=load('test');meta=pd.read_parquet(OUT/'confirmation_metadata.parquet')
    zero=meta[meta.positives==0].reset_index(drop=True)
    index=pd.Series(np.arange(len(test['meta'])),index=test['meta'].query_id)
    model=lgb.Booster(model_file=str(OUT/'models'/f'{record["name"]}.txt'))
    catalog=Catalog();rows=[];missing=0
    for row in zero.itertuples(index=False):
        j=index.loc[row.query_id];a,b=test['offsets'][j:j+2]
        target=np.flatnonzero(test['grades'][a:b]>0)
        if not len(target):
            missing+=1
            continue
        items=test['items'][a:b].tolist()
        D=1-catalog.similarity(items);np.fill_diagonal(D,0)
        scores=model.predict(feature_view(test['X'][a:b],record['tau'],record['delta']),num_threads=1)
        orders={'original_serp':np.arange(min(10,b-a)),'relevance':np.argsort(-scores,kind='stable')[:10]}
        one=pd.DataFrame([row._asdict()])
        for name in ['fixed_top30','continuous_top30']:
            p=frozen['policies'][name];w=LAMBDAS[policy_indices(p,one)[0]]
            orders[name]=floor_order(scores,D,w,p['floor'])
        for name,order in orders.items():
            rows.append(dict(query_id=row.query_id,method=name,held_out_candidate_view_targets=len(target),
                token_overlap_proxy10=float((1-D[np.ix_(order,target)]).max(axis=1).mean())))
    frame=pd.DataFrame(rows,columns=['query_id','method','held_out_candidate_view_targets','token_overlap_proxy10'])
    frame.to_csv(OUT/'zero_click_token_proxy.csv',index=False)
    summary=frame.groupby('method').agg(queries=('query_id','size'),token_overlap_proxy10=('token_overlap_proxy10','mean')).reset_index()
    summary.to_csv(OUT/'zero_click_token_proxy_summary.csv',index=False)
    write_json(OUT/'zero_click_proxy_scope.json',dict(zero_click_queries=len(zero),without_observed_future_candidate_view=missing,
        targets='Future same-episode candidate item views strictly before next query and within 20m, as weak labels prepared before scoring',
        metric='Mean over recommended top10 items of maximum token Jaccard similarity to any observed future candidate-view target',
        limitation='Secondary observed-view proxy. It does not infer relevance for unobserved items, and shares lexical tokens with the diversity objective. No parameters selected from these outcomes.'))


if __name__=='__main__':proxy_report()
