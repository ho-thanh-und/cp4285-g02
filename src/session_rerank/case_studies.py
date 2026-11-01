"""Grounded early/later query pairs from the same reserved confirmation episode."""
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from .benchmark import Catalog
from .protocol_core import LAMBDAS, floor_order
from .benchmark_data import OUT, load, write_json
from .benchmark_experiment import feature_view, policy_indices


def case_studies():
    frozen=json.loads((OUT/'frozen.json').read_text());record=frozen['selected']
    meta=pd.read_parquet(OUT/'confirmation_metadata.parquet')
    data=load('test');index=pd.Series(np.arange(len(data['meta'])),index=data['meta'].query_id)
    model=lgb.Booster(model_file=str(OUT/'models'/f'{record["name"]}.txt'));catalog=Catalog()
    cases=[]
    groups=meta.groupby('episode').history_count.agg(['size','min','max'])
    eligible=groups[(groups['size']>=2)&(groups['min']<=2)&(groups['max']>=5)].index[:4]
    for episode in eligible:
        group=meta[meta.episode==episode].sort_values(['timeframe','query_id'])
        pair=group.iloc[[0,-1]]
        turns=[]
        for _,row in pair.iterrows():
            j=index.loc[row.query_id];a,b=data['offsets'][j:j+2];items=data['items'][a:b].tolist()
            scores=model.predict(feature_view(data['X'][a:b],record['tau'],record['delta']),num_threads=1)
            D=1-catalog.similarity(items);np.fill_diagonal(D,0)
            orders={'original_serp':np.arange(min(10,b-a)),'relevance':np.argsort(-scores,kind='stable')[:10]}
            for name in ['fixed_top30','continuous_top30']:
                p=frozen['policies'][name];weight=LAMBDAS[policy_indices(p,pd.DataFrame([row]))[0]]
                orders[name]=floor_order(scores,D,weight,p['floor'])
            clicked=[i for i,y in zip(items,data['labels'][a:b]) if y]
            chosen={name:[items[i] for i in order] for name,order in orders.items()}
            visible=set(clicked)|set(sum(chosen.values(),[]))
            turns.append(dict(query_id=int(row.query_id),timeframe=int(row.timeframe),
                history_count=int(row.history_count),history_diversity=float(row.history_diversity),
                clicked_items=clicked,top10=chosen,
                item_tokens={str(i):sorted(catalog.item_tokens[i]) for i in visible}))
        cases.append(dict(episode=int(episode),session_id=int(pair.session_id.iloc[0]),turns=turns,
            interpretation='Earlier and later observed context in one episode. Anonymized tokens do not establish exploration, focus or subjective intent.'))
    write_json(OUT/'paired_session_cases.json',cases)


if __name__=='__main__':case_studies()
