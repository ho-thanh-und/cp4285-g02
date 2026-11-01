"""Descriptive item-exposure audit of existing frozen pipeline scores.

No protected attributes or seller ownership are available. These statistics
measure catalog visibility, not demographic fairness or causal welfare.
"""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .optimization_data import load_features
from .optimization_report import hybrid_scores
from .benchmark_data import write_json, digest

OUT=Path('results/submission/exposure')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    root=Path('results/submission/collaborative_comparison')
    receipt=json.loads((root/'receipt.json').read_text())
    assert digest(root/'per_query_metrics.npz')==receipt['metrics_sha256']
    d=load_features('confirmation','views',root='data/collaborative_comparison')
    tree=np.load(root/'collab31.npy');neural=np.load(root/'gru_ce.npy')
    predictions=dict(supplied_order=-d['X'][:,0],past_week_popularity=d['X'][:,13],
        collaborative_tree=tree,original_hybrid=hybrid_scores(tree,neural,d,.9))
    # All dates share the same May14 archive cap. Check each repeated item's
    # archive frequency is constant, rather than silently pooling moving clocks.
    item_ids=np.asarray(d['items']);unique,first,inverse=np.unique(item_ids,return_index=True,return_inverse=True)
    population=np.asarray(d['X'][:,12]);archive=population[first]
    assert np.array_equal(population,archive[inverse])
    head_count=max(1,int(np.ceil(.1*len(unique))))
    head=set(unique[np.lexsort((unique,-archive))[:head_count]])
    cold=set(unique[archive==0]);rows=[]
    for name,scores in predictions.items():
        exposures=Counter();repeat=0;hits=0;discounted_head=0.;discounted_total=0.
        for j,(a,b) in enumerate(zip(d['offsets'][:-1],d['offsets'][1:])):
            order=np.argsort(-scores[a:b],kind='stable')[:10];chosen=d['items'][a:b][order]
            exposures.update(map(int,chosen));history=set(d['prefixes'][j]);repeat+=sum(i in history for i in chosen)
            hits+=int(d['y'][a:b][order].sum())
            weights=1/np.log2(np.arange(len(chosen))+2)
            discounted_head+=sum(w for item,w in zip(chosen,weights) if item in head);discounted_total+=weights.sum()
        total=sum(exposures.values());counts=np.array(list(exposures.values()),dtype=float)
        rows.append(dict(method=name,queries=len(d['meta']),eligible_unique_items=len(unique),exposure_slots=total,
            exposed_unique_items=len(exposures),eligible_item_coverage=len(exposures)/len(unique),
            head_exposure_share=sum(n for i,n in exposures.items() if i in head)/total,
            discounted_head_exposure_share=discounted_head/discounted_total,
            archive_zero_click_exposure_share=sum(n for i,n in exposures.items() if i in cold)/total,
            exposure_hhi=float(np.sum((counts/total)**2)),prior_clicked_item_exposure_share=repeat/total,
            observed_click_hits=hits))
    pd.DataFrame(rows).to_csv(OUT/'item_exposure.csv',index=False)
    write_json(OUT/'scope.json',dict(status='Descriptive post-confirmation audit; no model selection',
        population='Same 41792 positive benchmark queries, supplied candidate pools, original frozen pipeline archive/context rules',
        head_definition='Top ceil(10% * unique eligible items) by cumulative archive clicks through May14; ties by item ID',
        coverage_denominator='Unique items occurring anywhere in these supplied lists, not the full product catalog',
        interpretation='Exposure concentration and retrieval confinement can inform stakeholder analysis; neither these metrics nor clicked-item repeats establish shopper harm or seller/demographic fairness.',
        input_receipt_sha256=digest(root/'receipt.json'),scores_sha256={p.name:digest(p) for p in [root/'collab31.npy',root/'gru_ce.npy']}))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':run()
