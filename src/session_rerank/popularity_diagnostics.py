"""Literature-motivated descriptive diagnostics, never a policy selector.

References: Wang (2023), arXiv:2312.07855; Ludewig et al. (2021),
doi:10.1007/s11257-020-09277-1. Popularity is an archive count, not an
identified exposure propensity. All fixed 5/10/20% head definitions are shown.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .optimization_data import load_features
from .optimization_report import hybrid_scores
from .benchmark_experiment import score_queries
from .benchmark_data import write_json,digest
from .policies import cluster_bootstrap

OUT=Path('results/submission/popularity_diagnostics')


def head_items(items,popularity,fraction):
    items=np.asarray(items);popularity=np.asarray(popularity)
    if not 0<fraction<1 or items.shape!=popularity.shape or not len(items):
        raise ValueError('Nonempty aligned items/counts and a fraction in (0,1) required')
    if len(np.unique(items))!=len(items):raise ValueError('Head definition needs unique items')
    return items[np.lexsort((items,-popularity))[:max(1,int(np.ceil(fraction*len(items))))]]


def uniform_head_share(head_counts,candidate_counts,k=10):
    head_counts=np.asarray(head_counts);candidate_counts=np.asarray(candidate_counts)
    if (candidate_counts<=0).any() or (head_counts<0).any() or (head_counts>candidate_counts).any():
        raise ValueError('Invalid candidate/head counts')
    slots=np.minimum(k,candidate_counts)
    return float(np.sum(slots*head_counts/candidate_counts)/slots.sum())


def target_strata(head_positive_counts,positive_counts):
    h=np.asarray(head_positive_counts);p=np.asarray(positive_counts)
    return dict(head_only=(p>0)&(h==p),tail_only=(p>0)&(h==0),mixed=(h>0)&(h<p),no_click=p==0)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    root=Path('results/submission/collaborative_comparison')
    d=load_features('confirmation','views',root='data/collaborative_comparison')
    stored=np.load(root/'per_query_metrics.npz')
    assert np.array_equal(stored['query_ids'],d['meta'].query_id)
    tree=np.load(root/'collab31.npy');neural=np.load(root/'gru_ce.npy')
    predictions=dict(supplied_order=-d['X'][:,0],past_week_popularity=d['X'][:,13],
        collaborative_tree=tree,original_hybrid=hybrid_scores(tree,neural,d,.9))
    adapted=dict(labels=d['y'],offsets=d['offsets'],group=np.diff(d['offsets']))
    names=list(predictions);values=np.stack([score_queries(s,adapted)[:,:2] for s in predictions.values()],axis=1)
    for name in ['collaborative_tree','original_hybrid']:
        np.testing.assert_allclose(values[:,names.index(name)],stored['values'][:,stored['names'].tolist().index(name)],rtol=0,atol=1e-14)
    items=np.asarray(d['items']);unique,first,inverse=np.unique(items,return_index=True,return_inverse=True)
    popularity=np.asarray(d['X'][:,12]);archive=popularity[first]
    assert np.array_equal(popularity,archive[inverse])
    offsets=d['offsets'];lengths=np.diff(offsets);positives=np.add.reduceat(d['y'],offsets[:-1])
    orders={name:np.concatenate([np.arange(a,b)[np.argsort(-scores[a:b],kind='stable')[:10]]
        for a,b in zip(offsets[:-1],offsets[1:])]) for name,scores in predictions.items()}
    exposure=[];strata_rows=[];paired=[]
    for fraction in [.05,.1,.2]:
        head=head_items(unique,archive,fraction);mask=np.isin(items,head)
        candidate_head=np.add.reduceat(mask,offsets[:-1]);clicked_head=np.add.reduceat(mask*d['y'],offsets[:-1])
        expected=uniform_head_share(candidate_head,lengths)
        for name,index in orders.items():
            share=float(mask[index].mean())
            exposure.append(dict(head_fraction=fraction,method=name,queries=len(d['meta']),
                candidate_occurrence_head_share=float(mask.mean()),uniform_top10_expected_head_share=expected,
                clicked_item_head_share=float(clicked_head.sum()/positives.sum()),head_exposure_share=share,
                exposure_minus_uniform_percentage_points=100*(share-expected)))
        groups=target_strata(clicked_head,positives)
        assert sum(int(m.sum()) for m in groups.values())==len(d['meta'])
        for stratum,take in groups.items():
            if not take.any():continue
            mean=values[take].mean(axis=0)
            for j,name in enumerate(names):
                strata_rows.append(dict(head_fraction=fraction,stratum=stratum,method=name,queries=int(take.sum()),
                    ndcg10=mean[j,0],calibrated_recall10=mean[j,1]))
            pair=values[take][:,[names.index('collaborative_tree'),names.index('original_hybrid')]]
            draws=cluster_bootstrap(pair,d['meta'].loc[take,'session_id'].to_numpy(),1000)
            for k,metric in enumerate(['ndcg10','calibrated_recall10']):
                delta=draws[:,1,k]-draws[:,0,k]
                paired.append(dict(head_fraction=fraction,stratum=stratum,queries=int(take.sum()),metric=metric,
                    hybrid_minus_tree=float(pair[:,1,k].mean()-pair[:,0,k].mean()),
                    low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975))))
    pd.DataFrame(exposure).to_csv(OUT/'opportunity_and_exposure.csv',index=False)
    pd.DataFrame(strata_rows).to_csv(OUT/'target_popularity_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'hybrid_increment_by_target_popularity.csv',index=False)
    write_json(OUT/'scope.json',dict(status='Post-confirmation descriptive evaluation; no fitting, selection or test-set retuning',
        archive='Original comparator cumulative click counts capped at May14, before all evaluated queries',
        head_definitions=[.05,.1,.2],ties='Ascending item ID',
        uniform_reference='Uniform sampling without replacement from each supplied candidate list; short-list sizes respected',
        strata='All clicked candidates in head, all in tail, or mixed; mutually exclusive. Outcome-defined diagnostics cannot be serving gates.',
        limitations='Popularity counts are not exposure propensities. Uniform opportunity is a descriptive reference, not a fair or useful target distribution. No causal, protected-group fairness, shopper-benefit or independent-confirmation claim.',
        input_receipt_sha256=digest(root/'receipt.json'),
        research=['https://arxiv.org/abs/2312.07855','https://doi.org/10.1007/s11257-020-09277-1']))
    print(pd.DataFrame(exposure).query('head_fraction==0.1').to_string(index=False),flush=True)
    print(pd.DataFrame(strata_rows).query('head_fraction==0.1').to_string(index=False),flush=True)


if __name__=='__main__':run()
