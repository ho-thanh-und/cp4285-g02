"""Build full-candidate, causal, memory-mapped session benchmark from raw CSVs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
from .benchmark import Catalog
from .core import tokens
from .protocol_core import WINDOW, DELTAS, causal_indices, nested_fractions

ROOT = Path('data/session_benchmark')
OUT = Path('results/session_benchmark')
NAMES = ['position', 'reciprocal_position', 'popularity', 'category_popularity',
         'price', 'query_overlap', 'category_match', 'prior_exact', 'prior_max',
         'prior_last', 'prior_count', 'prior_diversity', 'contradiction',
         'known_user', 'user_exact', 'user_similarity', 'session_knn'] + [f'decay_{d}' for d in DELTAS]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x,np.generic) else str(x))+'\n')
    temporary.replace(path)


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def event_table(queries):
    raw = Path('data/raw')
    clicks = pd.read_csv(raw/'train-clicks.csv', sep=';').merge(
        queries[['queryId','sessionId','userId','eventdate']], on='queryId', validate='many_to_one')
    clicks['kind'] = 2.
    frames = [clicks]
    for name, kind in [('views',1.),('purchases',4.)]:
        f = pd.read_csv(raw/f'train-item-{name}.csv' if name=='views' else raw/'train-purchases.csv', sep=';')
        f['kind'] = kind
        f['queryId'] = -1
        frames.append(f)
    events = pd.concat([f[['sessionId','userId','eventdate','timeframe','itemId','kind','queryId']] for f in frames],ignore_index=True).drop_duplicates()
    assign_episodes(queries,events)
    return events.sort_values(['episode','timeframe','kind','itemId']).reset_index(drop=True), clicks


def assign_episodes(queries, events):
    """Mutate episode IDs on both tables using a joint, stable event timeline."""
    # Include queries as boundary events: an intervening query prevents an
    # artificial inactivity split even when there are no intervening item events.
    timeline = pd.concat([
        queries[['sessionId','eventdate','timeframe']].assign(source=0, original=np.arange(len(queries))),
        events[['sessionId','eventdate','timeframe']].assign(source=1, original=np.arange(len(events)))
    ]).sort_values(['sessionId','eventdate','timeframe','source'])
    new = (timeline.sessionId.ne(timeline.sessionId.shift()) |
           timeline.eventdate.ne(timeline.eventdate.shift()) |
           timeline.timeframe.diff().gt(WINDOW))
    timeline['episode'] = new.cumsum().astype('int32')
    qmap = timeline[timeline.source==0].set_index('original').episode
    emap = timeline[timeline.source==1].set_index('original').episode
    queries['episode'] = qmap.reindex(np.arange(len(queries))).to_numpy()
    events['episode'] = emap.reindex(np.arange(len(events))).to_numpy()


def historical_index(events, catalog):
    """Fixed pre-April archive, strictly before every training/evaluation query."""
    past = events[(events.eventdate<'2016-03-31') & events.itemId.isin(catalog.row)]
    count = past.groupby('itemId').kind.sum().to_dict()
    cat_total = Counter()
    for item, value in count.items():
        cat_total[catalog.category.get(item,-1)] += value
    # Actual session nearest-neighbour retrieval over a historical episode-item
    # incidence matrix (not an item-repeat heuristic).
    distinct = past[['episode','itemId']].drop_duplicates()
    rows, uniques = pd.factorize(distinct.episode)
    cols = np.array([catalog.row[x] for x in distinct.itemId])
    incidence = sparse.csr_matrix((np.ones(len(rows),dtype=np.float32),(rows,cols)),shape=(len(uniques),len(catalog.ids)))
    sizes = np.sqrt(np.asarray(incidence.sum(axis=1)).ravel())
    users = {}
    known = past.dropna(subset=['userId']).sort_values(['eventdate','timeframe'])
    for uid, group in known.groupby('userId',sort=False):
        users[int(uid)] = list(dict.fromkeys(reversed(group.itemId.tolist())))[:50]
    return count, cat_total, incidence, sizes, users


def knn_scores(items, history, catalog, incidence, sizes, k=100):
    if not history:
        return np.zeros(len(items),dtype=np.float32)
    cols = [catalog.row[x] for x in set(history)]
    overlap = np.asarray(incidence[:,cols].sum(axis=1)).ravel()
    neighbors = np.flatnonzero(overlap)
    if not len(neighbors):
        return np.zeros(len(items),dtype=np.float32)
    similarity = overlap[neighbors]/(sizes[neighbors]*np.sqrt(len(cols)))
    take = np.argsort(-similarity, kind='stable')[:k]
    return np.asarray(incidence[neighbors[take]][:,[catalog.row[x] for x in items]].T @ similarity[take]).ravel()


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'complete.json').exists():
        recorded=json.loads((ROOT/'complete.json').read_text())
        if any(digest(Path('data/raw')/name)!=value for name,value in recorded['raw_hashes'].items()):
            raise RuntimeError('Raw inputs changed after preparation; preserve this run and use a separate experiment directory')
        print('Session benchmark data already complete; reusing recorded data.',flush=True)
        return
    if not Path('data/raw/train-queries.csv').exists():
        subprocess.run([sys.executable,'scripts/download_data.py'],check=True)
    if not all(Path(p).exists() for p in ['data/processed/clicks.parquet','data/processed/exposures.parquet']):
        subprocess.run([sys.executable,'-m','session_rerank.prepare'],check=True)
    started = time.perf_counter()
    q = pd.read_csv('data/raw/train-queries.csv',sep=';',dtype={'items':str,'searchstring.tokens':str})
    events, clicks = event_table(q)
    catalog = Catalog()
    count, cat_total, incidence, sizes, users = historical_index(events,catalog)
    targets = clicks.groupby('queryId').itemId.agg(set).to_dict()
    # Purchase attribution requires an associated clicked item, same episode,
    # after this query and before the next query / 20m, whichever comes first.
    next_time = q.sort_values(['episode','timeframe','queryId']).groupby('episode').timeframe.shift(-1)
    q['next_time'] = next_time.reindex(q.index).fillna(np.inf)
    q['split'] = np.select([
        q.eventdate.between('2016-04-01','2016-04-28'),
        q.eventdate.between('2016-05-01','2016-05-14'),
        q.eventdate.between('2016-05-16','2016-06-01')],['train','validation','test'],default='archive')
    eligible = q[(q.split!='archive') & ~q['is.test']].copy()
    eligible['has_click'] = eligible.queryId.isin(targets)
    events_count = events.groupby('episode').size()
    # Length bucket is total event length, available only for sampling, never a feature.
    summaries = eligible.groupby(['episode','split']).agg(
        date=('eventdate','min'), known=('userId',lambda x: bool(x.notna().any())),
        positive=('has_click','max'), queries=('queryId','size')).reset_index()
    summaries['length_bucket'] = pd.cut(summaries.episode.map(events_count).fillna(0),[-1,0,2,4,np.inf],labels=['0','1-2','3-4','5+']).astype(str)
    summaries['week'] = pd.to_datetime(summaries.date).dt.strftime('%G-%V')
    summaries = nested_fractions(summaries)
    assert summaries.episode.is_unique, 'Episode straddles chronological split'
    eligible['fraction_rank'] = eligible.episode.map(summaries.set_index('episode').fraction_rank)
    summaries.to_parquet(ROOT/'sessions.parquet',index=False)
    all_audit = dict(total_queries=len(q),total_episodes=int(q.episode.nunique()),
        anonymous_fraction=float(q.userId.isna().mean()), hidden_label_queries=int(q['is.test'].sum()),
        prior_archive_events=len(events[events.eventdate<'2016-03-31']),
        session_knn_archive_sessions=incidence.shape[0],
        note='100% means all released-label queries in the predeclared chronological intervals, with all candidates; hidden-label queries are never scored as zero-clicks.')
    # Arrays keep variable-length event histories without a Python object per row.
    etime = events.timeframe.to_numpy()
    eitem = events.itemId.to_numpy()
    ekind = events.kind.to_numpy()
    eqid = events.queryId.to_numpy()
    groups = events.groupby('episode',sort=False).indices
    session_clicks = events[events.kind==2].groupby('episode').size().to_dict()
    for split in ['train','validation','test']:
        folder = ROOT/split; folder.mkdir(exist_ok=True)
        if (folder/'manifest.json').exists():
            continue
        selected = eligible[eligible.split==split].sort_values(['eventdate','episode','timeframe','queryId'])
        lists = [list(dict.fromkeys(map(int,s.split(',')))) for s in selected['items']]
        offsets = np.r_[0,np.cumsum([len(x) for x in lists])]
        np.save(folder/'offsets.npy',offsets)
        shape = (int(offsets[-1]),len(NAMES))
        X = np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype='float32',shape=shape)
        item_out = np.lib.format.open_memmap(folder/'items.npy',mode='w+',dtype='int32',shape=(shape[0],))
        label_out = np.lib.format.open_memmap(folder/'labels.npy',mode='w+',dtype='int8',shape=(shape[0],))
        grade_out = np.lib.format.open_memmap(folder/'grades.npy',mode='w+',dtype='int8',shape=(shape[0],))
        metadata=[]
        for j,(row, original_items) in enumerate(zip(selected.to_dict('records'),lists)):
            a,b = offsets[j:j+2]
            valid = np.array([i in catalog.row and bool(catalog.item_tokens[i]) for i in original_items])
            # Preserve supplied candidates. Missing metadata gets zero content
            # features and is audited; the current raw catalog is expected complete.
            if not valid.all():
                raise ValueError(f'Missing catalog/tokens in query {row["queryId"]}; audit before continuing')
            items = original_items
            event_rows = groups.get(row['episode'],np.array([],dtype=int))
            times = etime[event_rows]
            idx = event_rows[causal_indices(times,row['timeframe'])]
            idx = idx[(eqid[idx]!=row['queryId']) & np.array([i in catalog.row for i in eitem[idx]],dtype=bool)]
            history = eitem[idx].tolist()
            kinds = ekind[idx]
            n = len(history)
            unique = list(dict.fromkeys(history))
            hist = catalog.similarity(items,unique) if unique else np.zeros((len(items),1),dtype=np.float32)
            lookup = {v:i for i,v in enumerate(unique)}
            seq_cols = np.array([lookup[v] for v in history],dtype=int)
            weighted = []
            for d in DELTAS:
                if n:
                    w = d**np.arange(n-1,-1,-1)*kinds
                    weighted.append(hist[:,seq_cols] @ (w/w.sum()))
                else:
                    weighted.append(np.zeros(len(items)))
            # Sequential contradiction: distance between the two latest distinct
            # observed items, not future targets or an inferred mental state.
            reverse_unique = list(dict.fromkeys(reversed(history)))
            if len(reverse_unique)>1:
                hd = 1-catalog.similarity(unique)
                diversity = float(hd.sum()/(len(unique)*(len(unique)-1)))
                contradiction = float(1-catalog.similarity(reverse_unique[:1],reverse_unique[1:2])[0,0])
            else:
                diversity=contradiction=0.
            profile = users.get(int(row['userId']),[]) if pd.notna(row['userId']) else []
            pop = np.array([count.get(i,0) for i in items])
            qt = tokens(row['searchstring.tokens'])
            features = np.column_stack([
                np.arange(len(items))/max(1,len(items)-1),1/(np.arange(len(items))+1),np.log1p(pop),
                [count.get(i,0)/max(1,cat_total[catalog.category.get(i,-1)]) for i in items],
                catalog.price[[catalog.row[i] for i in items]],
                [len(catalog.item_tokens[i]&qt)/max(1,len(qt)) for i in items],
                [row['categoryId']!=0 and catalog.category.get(i)==row['categoryId'] for i in items],
                [i in lookup for i in items],hist.max(axis=1),
                hist[:,lookup[history[-1]]] if n else np.zeros(len(items)),
                np.full(len(items),n),np.full(len(items),diversity),np.full(len(items),contradiction),
                np.full(len(items),pd.notna(row['userId'])),[i in profile for i in items],
                catalog.similarity(items,profile).max(axis=1) if profile else np.zeros(len(items)),
                knn_scores(items,unique,catalog,incidence,sizes), *weighted]).astype(np.float32)
            clicked = targets.get(row['queryId'],set())
            labels = np.array([i in clicked for i in items],dtype=np.int8)
            future = event_rows[(times>row['timeframe']) & (times<min(row['next_time'],row['timeframe']+WINDOW))]
            bought = set(eitem[future[ekind[future]==4.]])
            viewed = set(eitem[future[ekind[future]==1.]])
            grades = np.array([3 if i in clicked and i in bought else 2 if i in clicked else 1 if i in viewed else 0 for i in items],dtype=np.int8)
            X[a:b]=features; item_out[a:b]=items; label_out[a:b]=labels; grade_out[a:b]=grades
            metadata.append(dict(query_id=row['queryId'],session_id=row['sessionId'],episode=row['episode'],date=row['eventdate'],
                timeframe=row['timeframe'],history_count=n,history_diversity=diversity,contradiction=contradiction,
                known=pd.notna(row['userId']),has_user_profile=bool(profile),fraction_rank=row['fraction_rank'],
                candidate_count=len(items),category_count=len({catalog.category.get(i,-1) for i in items}),
                positives=int(labels.sum()),anomalous=session_clicks.get(row['episode'],0)>100))
            if (j+1)%5000==0:
                X.flush()
                print(f'{split}: {j+1:,}/{len(selected):,} queries; elapsed {time.perf_counter()-started:.0f}s',flush=True)
        for arr in [X,item_out,label_out,grade_out]:
            arr.flush()
        pd.DataFrame(metadata).to_parquet(folder/'meta.parquet',index=False)
        write_json(folder/'manifest.json',dict(queries=len(selected),candidates=shape[0],features=NAMES,query_ids_sha256=hashlib.sha256(selected.queryId.to_numpy().tobytes()).hexdigest()))
        del X,item_out,label_out,grade_out
    write_json(ROOT/'complete.json',dict(**all_audit,preparation_seconds=time.perf_counter()-started,
        raw_hashes={p.name:digest(p) for p in Path('data/raw').glob('*.csv')}))
    print('All session benchmark data prepared.',flush=True)


def load(split):
    p = ROOT/split
    return dict(X=np.load(p/'features.npy',mmap_mode='r'),items=np.load(p/'items.npy',mmap_mode='r'),
        labels=np.load(p/'labels.npy',mmap_mode='r'),grades=np.load(p/'grades.npy',mmap_mode='r'),
        offsets=np.load(p/'offsets.npy'),meta=pd.read_parquet(p/'meta.parquet'))


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
