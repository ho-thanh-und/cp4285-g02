"""Temporal feature tables for the second, RecBole-backed experiment.

All per-item feedback aggregates use a one-day embargo. Validation and
confirmation caps prevent any target in those splits entering the aggregates.
The original experiment's inputs and outputs are never overwritten.
"""
import argparse
from collections import defaultdict, Counter
from datetime import date
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import Catalog, FEATURES
from .core import tokens, prior_click_items

ROOT = Path('data/processed_v2')
ORIGIN = date(2016, 1, 1).toordinal()
STAT_NAMES = ['click_all', 'click_7d', 'click_30d', 'impression_all', 'impression_7d',
              'impression_30d', 'ctr_all', 'ctr_7d', 'ctr_30d', 'view_all', 'view_7d',
              'view_30d', 'purchase_all', 'purchase_30d', 'click_trend']
EXTRA_NAMES = ['recent_click_rank', 'historical_click_rank', 'token_count', 'item_category',
               'query_token_count', 'candidate_count', 'history_count', 'price_from_last_click',
               'price_from_history_mean', 'price_from_candidate_mean', 'session_elapsed_log',
               'query_has_tokens', 'recent_ctr_rank']
FEATURE_NAMES = FEATURES + STAT_NAMES + EXTRA_NAMES


def day_index(value):
    return date.fromisoformat(str(value)).toordinal() - ORIGIN


def available_day(query_date, cap_date=None):
    end = day_index(query_date) - 2
    return min(end, day_index(cap_date)) if cap_date else end


def known_item_rows(item_ids,id_to_row):
    """Return valid event positions and catalog rows; unknown IDs never index -1."""
    item_ids=np.asarray(item_ids,dtype=np.int64)
    mapped=np.full(len(item_ids),-1,dtype=np.int32)
    valid=(item_ids>=0)&(item_ids<len(id_to_row))
    mapped[valid]=id_to_row[item_ids[valid]]
    positions=np.flatnonzero(mapped>=0)
    return positions,mapped[positions]


def window_counts(cumulative, end, days=None):
    if end < 0:
        return np.zeros(cumulative.shape[1], dtype=np.float32)
    current = cumulative[min(end, len(cumulative)-1)]
    if days is None or end-days < 0:
        return current
    return current - cumulative[end-days]


class TemporalStats:
    def __init__(self, path=ROOT/'daily_counts.npz'):
        with np.load(path) as archive:
            self.counts = {name: np.cumsum(archive[name], axis=0, dtype=np.float32)
                           for name in ['click', 'impression', 'view', 'purchase']}
        self.current_end = None
        self.current = None

    def snapshot(self, end):
        if self.current_end == end:
            return self.current
        c = [window_counts(self.counts['click'], end, w) for w in [None, 7, 30]]
        e = [window_counts(self.counts['impression'], end, w) for w in [None, 7, 30]]
        v = [window_counts(self.counts['view'], end, w) for w in [None, 7, 30]]
        p = [window_counts(self.counts['purchase'], end, w) for w in [None, 30]]
        ctr = [(click+1)/(exposure+20) for click, exposure in zip(c,e)]
        trend = (c[1]+1)/(c[2]*7/30+3)
        self.current = np.column_stack([*[np.log1p(x) for x in c],
            *[np.log1p(x) for x in e], *ctr, *[np.log1p(x) for x in v],
            *[np.log1p(x) for x in p], trend]).astype(np.float32)
        self.current_end = end
        return self.current


def reserve_confirmation():
    """Deterministic outcome-blind reservation reproducible from a clean checkout."""
    ROOT.mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet('data/processed/queries_metadata.parquet')
    previous=pd.read_parquet('data/processed/test.parquet')
    pool=meta[(meta.eventdate>='2016-05-16') & ~meta['is.test'] & ~meta.sessionId.isin(previous.sessionId)]
    chosen=pool.sample(n=10000,random_state=20261026).sort_values(['eventdate','sessionId','timeframe','queryId'])
    digest=hashlib.sha256(','.join(map(str,sorted(chosen.queryId))).encode()).hexdigest()
    expected=json.loads(Path('config/optimization_protocol.json').read_text())['confirmation_query_ids_sha256']
    if digest!=expected:raise RuntimeError('Raw data does not reproduce the declared confirmation sample')
    path=ROOT/'confirmation_ids.parquet'
    if path.exists():
        if chosen.queryId.tolist()!=pd.read_parquet(path).queryId.tolist():
            raise RuntimeError('Existing confirmation IDs differ from reproducible reservation')
    else:chosen.to_parquet(path,index=False)
    return chosen


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'daily_counts.npz').exists():
        raise RuntimeError('Prepared v2 data already exists; reuse it rather than overwrite the sample')
    meta = pd.read_parquet('data/processed/queries_metadata.parquet')
    confirmation = reserve_confirmation()
    train = meta[meta.eventdate.between('2016-04-01','2016-04-29') & ~meta['is.test']]
    train = train.sample(n=min(60000,len(train)),random_state=42852026)
    validation = pd.read_parquet('data/processed/validation.parquet')
    ids = {name: set(frame.queryId) for name,frame in [('train',train),('validation',validation),('confirmation',confirmation)]}
    selected = {name: [] for name in ids}
    products = pd.read_csv('data/raw/products.csv', sep=';')
    n_items = len(products)
    id_to_row = np.full(int(products.itemId.max())+1,-1,dtype=np.int32)
    id_to_row[products.itemId] = np.arange(n_items)
    n_days = day_index('2016-06-02')+1
    impression = np.zeros((n_days,n_items),dtype=np.float32)
    total=0
    for chunk in pd.read_csv('data/raw/train-queries.csv',sep=';',chunksize=25000,
                              dtype={'items':str,'searchstring.tokens':str}):
        for name, wanted in ids.items():
            selected[name].append(chunk[chunk.queryId.isin(wanted)].copy())
        for row in chunk.loc[~chunk['is.test'],['eventdate','items']].itertuples(index=False):
            items=np.fromstring(row.items,sep=',',dtype=np.int64)
            np.add.at(impression[day_index(row.eventdate)],id_to_row[items],1)
        total+=len(chunk)
        if total%100000==0: print(f'Temporal impression audit: {total:,} queries',flush=True)
    for name,frames in selected.items():
        q=pd.concat(frames,ignore_index=True).sort_values(['eventdate','sessionId','timeframe','queryId'])
        assert len(q)==len(ids[name])
        q.to_parquet(ROOT/f'{name}_queries.parquet',index=False)
    arrays={'impression':impression}
    clicks=pd.read_parquet('data/processed/clicks.parquet')
    for kind,events in [
        ('click',clicks.drop_duplicates(['queryId','itemId'])),
        ('view',pd.read_csv('data/raw/train-item-views.csv',sep=';')),
        ('purchase',pd.read_csv('data/raw/train-purchases.csv',sep=';'))]:
        values=np.zeros_like(impression)
        positions,rows=known_item_rows(events.itemId.to_numpy(),id_to_row)
        days=events.eventdate.map(day_index).to_numpy()[positions]
        # Some purchase dates lie outside the query observation interval.
        # They cannot inform any prediction in this benchmark.
        in_range=(days>=0)&(days<n_days)
        np.add.at(values,(days[in_range],rows[in_range]),1)
        arrays[kind]=values
    np.savez_compressed(ROOT/'daily_counts.npz',**arrays)
    # RecBole sessions use ordinal positions after sorting the original clocks.
    # This avoids float32 precision loss in large relative-millisecond clocks.
    sequence=clicks[clicks.eventdate<'2016-04-30'].drop_duplicates(['sessionId','eventdate','timeframe','itemId'])
    sequence=sequence.sort_values(['sessionId','eventdate','timeframe','queryId','itemId'])
    sequence['user_id']=sequence.sessionId.astype(str)+'_'+sequence.eventdate
    sequence['timestamp']=sequence.groupby('user_id',sort=False).cumcount()+1
    atomic=sequence[['user_id','itemId','timestamp']].rename(columns={
        'user_id':'user_id:token','itemId':'item_id:token','timestamp':'timestamp:float'})
    directory=Path('data/recbole/diginetica_session')
    directory.mkdir(parents=True,exist_ok=True)
    atomic.to_csv(directory/'diginetica_session.inter',sep='\t',index=False)
    info=dict(training_query_sample=len(train),validation_sample=len(validation),confirmation_sample=len(confirmation),
              recbole_training_interactions=len(atomic),recbole_training_daily_sessions=sequence.user_id.nunique(),
              stat_features=STAT_NAMES,feature_names=FEATURE_NAMES,
              impression_denominator='Released-label queries only; hidden-label impressions excluded from CTR denominators',
              recbole_sequence_note='Training uses ordered click sequences including repeated clicks; query inference uses the same distinct-click prefixes as experiment 1')
    Path('results/optimization').mkdir(parents=True,exist_ok=True)
    Path('results/optimization/preparation.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2),flush=True)


def rank_fraction(values):
    return np.argsort(np.argsort(values,kind='stable'),kind='stable')/max(1,len(values)-1)


def build_features(split,root=None,protocol_path='config/optimization_protocol.json',freeze_path='results/optimization/frozen_selection.json'):
    root=ROOT if root is None else Path(root)
    if split=='confirmation' and not Path(freeze_path).exists():
        raise RuntimeError('Confirmation outcomes remain locked until model selection is frozen')
    target=root/split
    if target.exists():
        raise RuntimeError(f'{target} already exists; cached features are immutable')
    target.mkdir()
    catalog=Catalog()
    stats=TemporalStats()
    queries=pd.read_parquet(root/f'{split}_queries.parquet')
    if split=='confirmation':
        protocol=json.loads(Path(protocol_path).read_text())
        digest=hashlib.sha256(','.join(map(str,sorted(queries.queryId))).encode()).hexdigest()
        if digest!=protocol['confirmation_query_ids_sha256']:
            raise RuntimeError('Confirmation IDs differ from the reserved protocol')
        old=pd.read_parquet('data/processed/test.parquet')
        if set(queries.sessionId)&set(old.sessionId):
            raise RuntimeError('Confirmation session IDs overlap the inspected first test')
    keys=set(zip(queries.sessionId,queries.eventdate))
    selected=set(queries.queryId)
    events=defaultdict(list)
    targets=defaultdict(set)
    clicks=pd.read_parquet('data/processed/clicks.parquet')
    for row in clicks.itertuples(index=False):
        if row.queryId in selected: targets[row.queryId].add(row.itemId)
        if (row.sessionId,row.eventdate) in keys and row.itemId in catalog.row:
            events[row.sessionId,row.eventdate].append((row.timeframe,row.queryId,row.itemId))
    for values in events.values(): values.sort()
    capacity=int(queries['items'].str.count(',').add(1).sum())
    X=np.lib.format.open_memmap(target/'features.npy',mode='w+',dtype=np.float32,shape=(capacity,len(FEATURE_NAMES)))
    y=np.lib.format.open_memmap(target/'labels.npy',mode='w+',dtype=np.int8,shape=(capacity,))
    item_array=np.lib.format.open_memmap(target/'items.npy',mode='w+',dtype=np.int32,shape=(capacity,))
    cursor=0
    offsets=[0]
    metadata=[]
    prefixes=[]
    excluded=Counter()
    cap={'train':None,'validation':'2016-04-29','selection':'2016-04-29','confirmation':'2016-05-14'}[split]
    for i,row in enumerate(queries.to_dict('records')):
        items=[int(x) for x in row['items'].split(',')]
        relevance=np.array([x in targets[row['queryId']] for x in items],dtype=np.int8)
        if not relevance.any():
            excluded['no_observed_positive_in_candidates']+=1
            continue
        history=prior_click_items(events[row['sessionId'],row['eventdate']],row['queryId'],row['timeframe'])
        base=catalog.features(items,history,tokens(row['searchstring.tokens']),row['categoryId'])
        rows=np.array([catalog.row[x] for x in items])
        aggregate=stats.snapshot(available_day(row['eventdate'],cap))[rows]
        price=catalog.price[rows]
        hist_price=np.array([catalog.price[catalog.row[x]] for x in history])
        extras=np.column_stack([
            rank_fraction(aggregate[:,1]),rank_fraction(aggregate[:,0]),
            [len(catalog.item_tokens[x]) for x in items],
            [catalog.category.get(x,-1) for x in items],
            np.full(len(items),len(tokens(row['searchstring.tokens']))),
            np.full(len(items),len(items)),np.full(len(items),len(history)),
            np.abs(price-hist_price[0]) if history else np.full(len(items),-1),
            np.abs(price-hist_price.mean()) if history else np.full(len(items),-1),
            price-price.mean(),np.full(len(items),np.log1p(row['timeframe'])),
            np.full(len(items),bool(tokens(row['searchstring.tokens']))),rank_fraction(aggregate[:,7])])
        features=np.column_stack([base,aggregate,extras]).astype(np.float32)
        n=len(items)
        X[cursor:cursor+n]=features
        y[cursor:cursor+n]=relevance
        item_array[cursor:cursor+n]=items
        cursor+=n
        offsets.append(cursor)
        if len(history)>=2:
            sim=catalog.similarity(history)
            diversity=float((1-sim)[~np.eye(len(history),dtype=bool)].mean())
        else: diversity=float('nan')
        metadata.append(dict(query_id=row['queryId'],session_id=row['sessionId'],date=row['eventdate'],
            history_count=len(history),history_diversity=diversity,
            category_count=row['candidate_category_count'] if 'candidate_category_count' in row else len({catalog.category[x] for x in items})))
        prefix=np.zeros(10,dtype=np.int32)
        prefix[:len(history)]=list(reversed(history))
        prefixes.append(prefix)
        if (i+1)%10000==0:print(f'{split}: {i+1:,} queries / {cursor:,} candidates',flush=True)
    X.flush(); y.flush(); item_array.flush()
    np.save(target/'offsets.npy',np.asarray(offsets,dtype=np.int64))
    np.save(target/'prefixes.npy',np.asarray(prefixes,dtype=np.int32))
    pd.DataFrame(metadata).to_parquet(target/'metadata.parquet',index=False)
    info=dict(sampled_queries=len(queries),evaluated_queries=len(metadata),candidates=cursor,
              capacity=capacity,excluded=dict(excluded),feature_names=FEATURE_NAMES,statistics_cap=cap)
    (target/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2),flush=True)


def load_features(split,variant='base',root=None):
    target=(ROOT if root is None else Path(root))/split
    manifest=json.loads((target/'manifest.json').read_text())
    n=manifest['candidates']
    if variant not in ['base','views']:raise ValueError(variant)
    filename='features.npy' if variant=='base' else 'features_views.npy'
    return dict(X=np.load(target/filename,mmap_mode='r')[:n],
                y=np.load(target/'labels.npy',mmap_mode='r')[:n],
                items=np.load(target/'items.npy',mmap_mode='r')[:n],
                offsets=np.load(target/'offsets.npy'),prefixes=np.load(target/'prefixes.npy'),
                meta=pd.read_parquet(target/'metadata.parquet'),manifest=manifest)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','features'])
    parser.add_argument('--split',choices=['train','validation','confirmation'],default='train')
    args=parser.parse_args()
    prepare() if args.action=='prepare' else build_features(args.split)


if __name__=='__main__':main()
