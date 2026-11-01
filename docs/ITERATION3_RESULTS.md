# Third experiment results

All main confirmation comparisons use the same **9,865 queries and 1,492,509 supplied candidates**. The initial reservation contained 10,000 queries; 135 without an observed clicked candidate were excluded under the unchanged eligibility rule. Both earlier inspected test/confirmation session-ID sets were excluded before new fitting. This is another outcome sample in the same date period, not an independent temporal replication.

## Main comparison

| Method | NDCG@10 | Recall@10 | Token Jaccard ILD@10 |
|---|---:|---:|---:|
| original_ranker | 0.18261 | 0.30450 | 0.90329 |
| previous_hybrid | 0.22412 | 0.36918 | 0.90878 |
| best_tree | 0.23146 | 0.38503 | 0.90828 |
| best_neural | 0.20369 | 0.34038 | 0.91374 |
| best_fusion | 0.23116 | 0.38470 | 0.90887 |
| anonymous_control | 0.23075 | 0.38559 | 0.90944 |
| fixed | 0.22970 | 0.38258 | 0.93312 |
| adaptive_diversity | 0.22998 | 0.38231 | 0.91581 |

## What improved?

The selected model is the 31-leaf collaborative-feature tree fused with RecBole GRU4Rec cross-entropy scores at tree weight **0.9**, only where click history exists. The original validation sample nominated it; the fresh 7,921-query model-selection sample chose it over the other four nominees. Neither stage accessed the new confirmation outcomes.

The selected NDCG is **0.23116**, versus **0.22412** for the previous frozen hybrid on these exact queries. The absolute improvement is **+0.00704**, paired 95% session-bootstrap interval **+0.00439 to +0.00948**. The relative improvement is **+3.14%**, interval **+1.94% to +4.25%**. Recall improves **+4.20% relative**. Against the original basic ranker on this same sample, the NDCG gain is **+26.59% relative**.

The component tree scores 0.23146, slightly above its selected fusion. The neural fusion contribution again does not establish an independent confirmation gain over that tree. The frozen primary selection remains unchanged. The anonymous-user control scores 0.23075, showing little point-estimate difference from the richer identity-aware tree; these data have few usable persistent profiles.

## RecBole comparison on identical queries

The following is a diagnostic comparison of already frozen neural predictions, computed after selection. It did not change nominees or model selection.

| RecBole model | NDCG@10 | Absolute difference from original GRU/BPR | 95% interval for difference |
|---|---:|---:|---|
| GRU4Rec, BPR | 0.18556 | +0.00000 | [+0.00000, +0.00000] |
| GRU4Rec, cross-entropy | 0.20025 | +0.01469 | [+0.01211, +0.01734] |
| SASRec, cross-entropy | 0.20369 | +0.01813 | [+0.01539, +0.02081] |

The new RecBole configurations improve materially over the earlier GRU/BPR model, but remain below the stronger feature-based rankings. This comparison changes architecture/hyperparameters as well as loss, so it does not isolate a causal effect of cross-entropy alone.

## Diversity findings

With relevance weight 0.7, fixed diversification increases Jaccard ILD **+2.67% relative** for **-0.63% relative NDCG** against selected relevance-only scoring. Relative paired intervals are **+2.59% to +2.74%** for ILD and **-1.17% to -0.05%** for NDCG. Cosine ILD increases **+4.03%**, a numerical distance robustness check rather than independent semantic validation.

The session-diversity search again chooses weight 0.6 in both branches. The threshold is inactive and the method equals its history-length gate. It offers a different relevance/diversity point but no demonstrated information gain from the diversity threshold.

## Data availability findings

- On original development queries, only 114/5,932 have a usable earlier persistent-user click profile. On this confirmation sample, that count is 197/9,865, approximately **2%**.
- Among released-label queries, missing user IDs rise from 0% in January/February to about 47% in March, 86% in April and 88% in May. This is identifier availability in the released data, not a claim about shopper behaviour.
- Repeat-click targets are much easier than novel targets in the baseline error audit. Adding more model complexity cannot supply intent that the input never records.
- Richer pre-query browsing/search/filter/cart signals, stable anonymous cross-visit IDs, readable product/query attributes, and reliable exposure/timestamp data are the most useful additions. More rows can also help, but the current experiment does not establish a sample-size requirement or promise near-perfect ranking.

## Scope of the result

This is a reproducible incremental improvement on the unchanged full-candidate task. It does not establish a general NDCG level across datasets, shopper satisfaction, sales effects or causal user-intent changes. Repeated same-period development and multiple reserved samples limit broader generalization; all earlier positive and negative results remain available.

## Recorded provenance

- Frozen selection SHA256: `22c8afe427cadaf4739474a13f7a63712827188458c64bda013c8f74de83f423`
- Confirmation query-ID SHA256: `b997c8051a4598558525a5d3325983f06851324ba63b056fad0ae29c75714758`
- Full model/config hashes: `results/iteration3/frozen_selection.json`.
- Raw per-query metrics: `results/iteration3/confirmation_per_query.npz`.
- Paired improvement table: `results/iteration3/improvement_vs_previous.csv`.
- Methods and reproduction details: `docs/ITERATION3_METHODS.md`.
