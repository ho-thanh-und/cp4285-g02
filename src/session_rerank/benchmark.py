"""Train a small LambdaMART ranker; cache validation/test rankings and metrics.

Run development first, freeze policies from validation, then explicitly evaluate
the held-out test once. Labels are binary distinct query-associated item clicks.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import lightgbm as lgb
from .core import tokens, prior_click_items, greedy_rerank, ranking_metrics

WEIGHTS = np.array([1., .995, .99, .98, .95, .9, .8, .6, 0.])
FEATURES = ["position_fraction", "reciprocal_position", "log_clicks", "smoothed_ctr", "price_log2", "query_token_overlap", "query_category_match", "prior_exact_click", "prior_item_recency", "history_max_similarity", "history_mean_similarity", "last_item_similarity"]


class Catalog:
    def __init__(self):
        p = pd.read_csv('data/raw/products.csv', sep=';')
        c = pd.read_csv('data/raw/product-categories.csv', sep=';')
        self.ids = p.itemId.to_numpy()
        self.row = {int(item):i for i,item in enumerate(self.ids)}
        self.item_tokens = {int(item):tokens(text) for item,text in zip(p.itemId,p['product.name.tokens'])}
        self.price = p.pricelog2.fillna(0).to_numpy(dtype=np.float32)
        self.category = c.set_index('itemId').categoryId.to_dict()
        vocab = {}
        rows, cols = [], []
        for row, item in enumerate(self.ids):
            for token in self.item_tokens[item]:
                if token not in vocab:
                    vocab[token] = len(vocab)
                rows.append(row)
                cols.append(vocab[token])
        self.matrix = csr_matrix((np.ones(len(rows), dtype=np.float32),(rows,cols)),shape=(len(p),len(vocab)))
        self.sizes = np.asarray(self.matrix.sum(axis=1)).ravel()
        a = pd.read_parquet('data/processed/clicks.parquet')
        early = a[a.eventdate < '2016-03-31'].drop_duplicates(['queryId','itemId'])
        self.pop = Counter(early.itemId)
        exposure = pd.read_parquet('data/processed/exposures.parquet')
        self.exposure = dict(zip(exposure.itemId, exposure.exposures))

    def similarity(self, ids, other=None):
        other = ids if other is None else other
        a = np.array([self.row[x] for x in ids])
        b = np.array([self.row[x] for x in other])
        intersection = (self.matrix[a] @ self.matrix[b].T).toarray()
        union = self.sizes[a,None] + self.sizes[b][None,:] - intersection
        return np.divide(intersection, union, out=np.zeros_like(intersection), where=union>0)

    def features(self, items, history, query_tokens, query_category):
        n = len(items)
        hist = self.similarity(items,history) if history else np.zeros((n,1),dtype=np.float32)
        pop = np.array([self.pop[x] for x in items])
        exposures = np.array([self.exposure.get(x,0) for x in items])
        overlap = np.array([len(self.item_tokens[x] & query_tokens) / max(1,len(query_tokens)) for x in items])
        return np.column_stack([
            np.arange(n)/max(1,n-1), 1/(np.arange(n)+1), np.log1p(pop),
            (pop+1)/(exposures+20), self.price[[self.row[x] for x in items]], overlap,
            [self.category.get(x,-1)==query_category and query_category!=0 for x in items],
            [x in history for x in items], [1/(history.index(x)+1) if x in history else 0 for x in items],
            hist.max(axis=1), hist.mean(axis=1), hist[:,0],
        ]).astype(np.float32)


def load_examples(name, catalog):
    q = pd.read_parquet(f'data/processed/{name}.parquet')
    keys = set(zip(q.sessionId,q.eventdate))
    clicks = pd.read_parquet('data/processed/clicks.parquet')
    history_events = defaultdict(list)
    targets = defaultdict(set)
    selected = set(q.queryId)
    for row in clicks.itertuples(index=False):
        if row.queryId in selected:
            targets[row.queryId].add(row.itemId)
        if (row.sessionId,row.eventdate) in keys and row.itemId in catalog.row:
            history_events[row.sessionId,row.eventdate].append((row.timeframe,row.queryId,row.itemId))
    for key in history_events:
        history_events[key].sort()
    examples = []
    excluded = Counter()
    for row in q.to_dict('records'):
        # Deduplicate candidates, retaining the original display order.
        items = list(dict.fromkeys(int(x) for x in row['items'].split(',')))
        if any(x not in catalog.row or not catalog.item_tokens[x] for x in items):
            excluded['missing_product_or_tokens'] += 1
            continue
        relevance = np.array([x in targets[row['queryId']] for x in items],dtype=np.int8)
        if not relevance.any():
            excluded['no_observed_positive_in_candidates'] += 1
            continue
        events = history_events[row['sessionId'],row['eventdate']]
        history = prior_click_items(events,row['queryId'],row['timeframe'])
        if len(history)>=2:
            sim = catalog.similarity(history)
            diversity = float((1-sim)[~np.eye(len(history),dtype=bool)].mean())
        else:
            diversity = float('nan')
        features = catalog.features(items,history,tokens(row['searchstring.tokens']),row['categoryId'])
        examples.append(dict(query_id=row['queryId'],session_id=row['sessionId'],date=row['eventdate'],
                             items=items,history=history,history_count=len(history),history_diversity=diversity,
                             category_count=row['candidate_category_count'],features=features,relevance=relevance))
    print(f'{name}: {len(examples):,} evaluated queries; exclusions: {dict(excluded)}',flush=True)
    Path(f'results/{name}_eligibility.json').write_text(json.dumps(dict(sampled_queries=len(q),evaluated_queries=len(examples),exclusions=dict(excluded)),indent=2)+'\n')
    return examples


def train(examples):
    X = np.concatenate([x['features'] for x in examples])
    y = np.concatenate([x['relevance'] for x in examples])
    group = [len(x['items']) for x in examples]
    model = lgb.LGBMRanker(objective='lambdarank',n_estimators=120,learning_rate=.05,num_leaves=15,
                          max_depth=6,min_child_samples=100,reg_lambda=1.,n_jobs=4,random_state=4285,
                          deterministic=True,force_col_wise=True,verbosity=-1)
    print(f'Training LambdaMART on {len(group):,} queries / {len(y):,} candidates...',flush=True)
    model.fit(X,y,group=group,feature_name=FEATURES)
    Path('results/model.txt').write_text(model.booster_.model_to_string())
    pd.DataFrame({'feature':FEATURES,'gain':model.booster_.feature_importance('gain')}).to_csv('results/feature_importance.csv',index=False)
    return model.booster_


def evaluate(examples, catalog, model, name):
    output = Path('results/cache')
    output.mkdir(parents=True,exist_ok=True)
    all_features = np.concatenate([x['features'] for x in examples])
    prediction = model.predict(all_features,num_threads=4)
    metrics = np.empty((len(examples),len(WEIGHTS),3))
    baseline = np.empty((len(examples),3,3))
    metadata = []
    ranking_examples = []
    cursor = 0
    started = time.perf_counter()
    for i,x in enumerate(examples):
        n = len(x['items'])
        scores = prediction[cursor:cursor+n]
        cursor += n
        D = 1-catalog.similarity(x['items'])
        np.fill_diagonal(D,0)
        for j,w in enumerate(WEIGHTS):
            order = greedy_rerank(scores,D,float(w))
            metrics[i,j] = ranking_metrics(order,x['relevance'],D)
        positions = np.arange(n)
        popularity = x['features'][:,2]
        session = x['features'][:,7]*100 + x['features'][:,8]*10 + popularity
        for j,order in enumerate([positions,np.argsort(-popularity,kind='stable'),np.argsort(-session,kind='stable')]):
            baseline[i,j] = ranking_metrics(order,x['relevance'],D)
        if len(ranking_examples)<12 and x['history_count']>=2:
            ranking_examples.append(dict(query_id=int(x['query_id']),date=x['date'],history=x['history'],
                candidate_ids=x['items'],relevant_ids=[a for a,b in zip(x['items'],x['relevance']) if b],
                top10_by_weight={str(w):[x['items'][a] for a in greedy_rerank(scores,D,float(w))] for w in WEIGHTS}))
        metadata.append({k:x[k] for k in ['query_id','session_id','date','history_count','history_diversity','category_count']})
        if (i+1)%1000==0:
            print(f'{name}: scored {i+1:,}/{len(examples):,} ({time.perf_counter()-started:.1f}s)',flush=True)
    np.savez_compressed(output/f'{name}.npz',metrics=metrics,baselines=baseline,weights=WEIGHTS)
    pd.DataFrame(metadata).to_parquet(output/f'{name}_metadata.parquet',index=False)
    Path(f'results/{name}_ranking_examples.json').write_text(json.dumps(ranking_examples,indent=2,default=int)+'\n')
    seconds=time.perf_counter()-started
    Path(f'results/{name}_timing.json').write_text(json.dumps(dict(queries=len(examples),seconds=seconds,
        ms_per_query_all_nine_weights_plus_metrics=1000*seconds/len(examples),note='Sequential local CPU benchmark; includes distance construction, nine re-rankings and metrics; excludes model prediction and data loading.'),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['development','test'],default='development')
    args=parser.parse_args()
    if args.stage=='development' and Path('results/frozen_policies.json').exists():
        raise RuntimeError('This experiment is frozen. Reproduce or explore in a separate checkout/output directory.')
    if args.stage=='test' and not Path('results/frozen_policies.json').exists():
        raise RuntimeError('Freeze policies from validation before unlocking test evaluation')
    if args.stage=='test':
        from .policies import file_hash
        frozen=json.loads(Path('results/frozen_policies.json').read_text())
        if frozen['model_sha256']!=file_hash('results/model.txt') or frozen['protocol_sha256']!=file_hash('config/protocol.json'):
            raise RuntimeError('Model or protocol changed after policy freeze')
    catalog=Catalog()
    if args.stage=='development':
        training=load_examples('train',catalog)
        model=train(training)
        del training
        examples=load_examples('validation',catalog)
        evaluate(examples,catalog,model,'validation')
    else:
        if Path('results/cache/test.npz').exists():
            raise RuntimeError('Test cache already exists. Reuse it for reporting; do not silently rerun.')
        model=lgb.Booster(model_file='results/model.txt')
        evaluate(load_examples('test',catalog),catalog,model,'test')


if __name__=='__main__':
    main()
