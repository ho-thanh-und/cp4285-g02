"""Independent raw-log evaluation audit; no project ranking/evaluator imports.

Recomputes every later-period released-label query directly from raw CSVs and
compares the original-order scores, candidate order and labels with cached data.
Also checks raw event histories against a direct sid/date/time filter.
"""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

OUT=Path('results/submission/audit')


def raw_metrics(items, clicked):
    relevant=[item in clicked for item in items]
    count=sum(relevant)
    dcg=sum(1/math.log2(rank+2) for rank,value in enumerate(relevant[:10]) if value)
    ideal=sum(1/math.log2(rank+2) for rank in range(min(10,count)))
    return (dcg/ideal if ideal else 0.,sum(relevant[:10])/min(10,count) if count else 0.),relevant


def audit():
    OUT.mkdir(parents=True,exist_ok=True)
    targets=pd.read_csv('data/raw/train-clicks.csv',sep=';').groupby('queryId').itemId.agg(set).to_dict()
    meta=pd.read_parquet('data/session_benchmark/test/meta.parquet')
    offsets=np.load('data/session_benchmark/test/offsets.npy')
    items_cache=np.load('data/session_benchmark/test/items.npy',mmap_mode='r')
    labels_cache=np.load('data/session_benchmark/test/labels.npy',mmap_mode='r')
    features=np.load('data/session_benchmark/test/features.npy',mmap_mode='r')
    lookup=dict(zip(meta.query_id,range(len(meta))))
    fresh=pd.read_parquet('results/session_benchmark/confirmation_metadata.parquet')
    fresh_ids=set(fresh.query_id)
    old=pd.read_parquet('data/processed/test.parquet')
    old_ids=set(old.queryId)
    iteration=pd.read_parquet('data/iteration3/confirmation/metadata.parquet')
    iter_ids=set(iteration.query_id)
    records=[];sklearn_checks=0
    for chunk in pd.read_csv('data/raw/train-queries.csv',sep=';',chunksize=50000,dtype={'items':str}):
        selected=chunk[(chunk.eventdate>='2016-05-16')&~chunk['is.test']]
        for row in selected.to_dict('records'):
            query=row['queryId']
            items=list(dict.fromkeys(map(int,row['items'].split(','))))
            clicked=targets.get(query,set())
            metric,relevant=raw_metrics(items,clicked)
            j=lookup[query];a,b=offsets[j:j+2]
            assert items==items_cache[a:b].tolist(), f'Candidate/order mismatch at {query}'
            assert relevant==labels_cache[a:b].astype(bool).tolist(),f'Raw target mismatch at {query}'
            np.testing.assert_allclose(features[a:b,0],np.arange(len(items))/max(1,len(items)-1),atol=1e-7)
            if sum(relevant) and sklearn_checks<1000 and len(items)>1:
                reference=ndcg_score(np.array([relevant],dtype=float),-np.arange(len(items),dtype=float)[None,:],k=10,ignore_ties=True)
                assert abs(reference-metric[0])<1e-12
                sklearn_checks+=1
            records.append(dict(query_id=query,session_id=row['sessionId'],date=row['eventdate'],
                candidate_count=len(items),clicked_candidates=sum(relevant),clicked_outside_candidates=len(clicked-set(items)),
                first_clicked_rank=min((k+1 for k,v in enumerate(relevant) if v),default=np.nan),
                ndcg10=metric[0],recall10_calibrated=metric[1],
                fresh_confirmation=query in fresh_ids,original_test=query in old_ids,iteration3=query in iter_ids))
        print(f'Raw audit: {len(records):,} evaluated queries',flush=True)
    raw=pd.DataFrame(records).set_index('query_id')
    raw.to_parquet(OUT/'raw_query_scores.parquet')
    cached=np.load('results/session_benchmark/confirmation_metrics.npz')
    i=cached['names'].tolist().index('original_serp')
    expected=raw.loc[fresh.query_id,['ndcg10','recall10_calibrated']].to_numpy()
    error=float(np.max(np.abs(expected-cached['values'][:,i,:2])))
    assert error<1e-7
    rows=[]
    for name,mask in [('all_later_period',np.ones(len(raw),dtype=bool)),
        ('original_test',raw.original_test),('iteration3_sample',raw.iteration3),('fresh_confirmation',raw.fresh_confirmation)]:
        frame=raw[mask];positive=frame[frame.clicked_candidates>0]
        rows.append(dict(cohort=name,queries=len(frame),positive_queries=len(positive),
            ndcg10=float(positive.ndcg10.mean()),recall10_calibrated=float(positive.recall10_calibrated.mean()),
            mean_candidates=float(frame.candidate_count.mean()),fraction_top10_click=float((positive.first_clicked_rank<=10).mean())))
    pd.DataFrame(rows).to_csv(OUT/'original_order_cohorts.csv',index=False)
    # Same-query v1 original-order cache agreement, independently of new code.
    old_meta=pd.read_parquet('results/cache/test_metadata.parquet')
    old_cache=np.load('results/cache/test.npz')['baselines'][:,0,0]
    old_error=float(np.max(np.abs(raw.loc[old_meta.query_id,'ndcg10'].to_numpy()-old_cache)))
    assert old_error<1e-10
    history_result=history_audit(meta,features,offsets)
    result=dict(status='passed',raw_queries=len(raw),raw_candidate_and_label_comparisons=len(raw),
        sklearn_ndcg_checks=sklearn_checks,new_original_order_max_error=error,old_original_order_max_error=old_error,
        original_ranker_label_correction='The iteration3 key original_ranker is results/model.txt, a trained 12-feature LambdaMART. It is not the untouched SERP order.',
        correct_comparison='Earlier untouched original_order NDCG is about .073, new untouched original_serp about .071; the alleged .183-to-.071 raw-order collapse mixed two different models.',
        historical_context=history_result,
        source='Independent raw CSV join and hand-formula evaluator cross-checked with sklearn; no project evaluator imports')
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'AUDIT.md').write_text('''# Independent evaluation audit

The alleged original-order collapse was a naming error in the assessment. The
earlier `original_ranker` row is a trained LambdaMART reference loaded from
`results/model.txt`. It is not the supplied candidate order. The untouched
order was approximately 0.073 NDCG on the original test sample and 0.071 on
the session benchmark confirmation sample. These scores are consistent in scale.

Every released-label query in the later evaluation period was reconstructed
directly from raw candidate lists and query-associated clicks. Candidate order,
binary targets and position features match the new cached arrays. All new
confirmation original-order metrics match the independent calculation, and
the earlier original-order cache also agrees on exactly the same queries.
The first 1,000 eligible NDCG calculations were independently checked against
scikit-learn. Detailed counts, maximum errors and direct history checks are
recorded in `independent_audit.json` and `original_order_cohorts.csv`.

This verifies the stated offline task. It does not establish shopper benefit,
causal exposure correction, or generalization to another dataset. Earlier and
new learned rankers still use different features and populations; their scores
must be compared on common inputs before making a superiority claim.
''')
    print(json.dumps(result,indent=2),flush=True)


def history_audit(meta,features,offsets):
    queries=pd.read_parquet('data/processed/queries_metadata.parquet')
    clicks=pd.read_csv('data/raw/train-clicks.csv',sep=';').merge(queries[['queryId','sessionId','eventdate']],on='queryId',validate='many_to_one')
    clicks['kind']=2.
    frames=[clicks]
    for file,kind in [('train-item-views.csv',1.),('train-purchases.csv',4.)]:
        f=pd.read_csv('data/raw/'+file,sep=';');f['queryId']=-1;f['kind']=kind;frames.append(f)
    events=pd.concat([f[['sessionId','eventdate','timeframe','itemId','queryId','kind']] for f in frames],ignore_index=True).drop_duplicates()
    catalog=pd.read_csv('data/raw/products.csv',sep=';')
    token_map={
        int(item):set(map(int,value.split(','))) for item,value in zip(catalog.itemId,catalog['product.name.tokens'])}
    valid=set(token_map)
    groups={key:g.sort_values(['timeframe','kind','itemId']).to_dict('records') for key,g in
        events[events.eventdate>='2016-05-16'].groupby(['sessionId','eventdate'],sort=False)}
    # Oversample contexts with history while retaining seeded cold examples.
    selected=pd.concat([meta[meta.history_count>0].sample(n=min(500,int((meta.history_count>0).sum())),random_state=4285),
        meta[meta.history_count==0].sample(n=min(500,int((meta.history_count==0).sum())),random_state=4285)])
    qlookup=queries.set_index('queryId');items=np.load('data/session_benchmark/test/items.npy',mmap_mode='r')
    index=pd.Series(np.arange(len(meta)),index=meta.query_id)
    maximum=0.
    for row in selected.itertuples(index=False):
        q=qlookup.loc[row.query_id]
        history=[e for e in groups.get((row.session_id,row.date),[]) if
            0<int(q.timeframe)-int(e['timeframe'])<=1_200_000 and e['queryId']!=row.query_id and e['itemId'] in valid]
        j=index.loc[row.query_id];a,b=offsets[j:j+2]
        assert len(history)==row.history_count
        for candidate in range(a,min(b,a+20)):
            ts=token_map[int(items[candidate])]
            for k,delta in enumerate([.5,.7,.85,.95,1.]):
                weights=[delta**(len(history)-1-i)*e['kind'] for i,e in enumerate(history)]
                similarity=[len(ts&token_map[e['itemId']])/len(ts|token_map[e['itemId']]) for e in history]
                value=sum(w*s for w,s in zip(weights,similarity))/sum(weights) if weights else 0.
                maximum=max(maximum,abs(value-float(features[candidate,17+k])))
    assert maximum<1e-6
    return dict(queries=len(selected),history_count_match=True,manual_weighted_jaccard_max_error=maximum,
        membership='Direct same-sessionId/date filter with strict timestamp inequality and 1,200,000 lookback; no episode extractor import')


if __name__=='__main__':audit()
