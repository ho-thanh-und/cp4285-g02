"""Optional strict same-date relative-clock view prefix features.

The extra feature family assumes timeframe fields share the supplied session's
relative clock. Keep the click-only model as an ablation of this assumption.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .benchmark import Catalog
from .core import prior_click_items
from .optimization_data import ROOT,load_features

VIEW_NAMES=['prior_view_exact','prior_view_recency','prior_view_max_similarity',
            'prior_view_mean_similarity','prior_view_last_similarity','prior_view_count',
            'price_from_last_view','last_view_age_log']


def build(split,root=None,output=None,freeze_path='results/optimization/frozen_selection.json'):
    root=ROOT if root is None else Path(root)
    output=Path('results/optimization') if output is None else Path(output)
    if split=='confirmation' and not Path(freeze_path).exists():
        raise RuntimeError('Freeze model selection before confirmation feature access')
    data=load_features(split,root=root)
    path=root/split/'features_views.npy'
    if path.exists():raise RuntimeError('View feature cache already exists')
    catalog=Catalog()
    q=pd.read_parquet(root/f'{split}_queries.parquet').set_index('queryId')
    keys=set(zip(data['meta'].session_id,data['meta'].date))
    views=pd.read_csv('data/raw/train-item-views.csv',sep=';')
    events=defaultdict(list)
    for row in views.itertuples(index=False):
        if (row.sessionId,row.eventdate) in keys:
            events[row.sessionId,row.eventdate].append((row.timeframe,-1,row.itemId))
    for rows in events.values():rows.sort()
    X=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,
        shape=(len(data['X']),data['X'].shape[1]+len(VIEW_NAMES)))
    X[:,:data['X'].shape[1]]=data['X']
    available=0
    for j,row in enumerate(data['meta'].itertuples(index=False)):
        start,end=data['offsets'][j:j+2]
        items=data['items'][start:end].tolist()
        timestamp=int(q.loc[row.query_id,'timeframe'])
        source=events[row.session_id,row.date]
        prefix=prior_click_items(source,row.query_id,timestamp)
        if prefix:
            available+=1
            similarity=catalog.similarity(items,prefix)
            recent=max(time for time,_,_ in source if time<timestamp)
            last_price=catalog.price[catalog.row[prefix[0]]]
            extra=np.column_stack([
                [item in prefix for item in items],
                [1/(prefix.index(item)+1) if item in prefix else 0 for item in items],
                similarity.max(axis=1),similarity.mean(axis=1),similarity[:,0],
                np.full(len(items),len(prefix)),
                [abs(catalog.price[catalog.row[item]]-last_price) for item in items],
                np.full(len(items),np.log1p(timestamp-recent))])
        else:
            extra=np.zeros((len(items),len(VIEW_NAMES)),dtype=np.float32)
            extra[:,-2:]=-1
        X[start:end,data['X'].shape[1]:]=extra
        if (j+1)%10000==0:print(f'{split}: added strict prior views for {j+1:,} queries',flush=True)
    X.flush()
    report=dict(queries=len(data['meta']),queries_with_prior_views=available,
                fraction_with_prior_views=available/len(data['meta']),features=VIEW_NAMES,
                assumption='Within a supplied sessionId and date, view and query timeframe fields share a relative clock. Only view timeframe < query timeframe enters features.',
                control='40-feature model omits this entire family; primary click prefix and target definitions remain unchanged')
    output.mkdir(parents=True,exist_ok=True)
    (output/f'{split}_view_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['train','validation','confirmation'],required=True)
    build(parser.parse_args().split)


if __name__=='__main__':main()
