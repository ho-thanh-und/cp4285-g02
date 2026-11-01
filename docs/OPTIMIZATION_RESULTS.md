# Optimization results, 26 October 2026

All confirmation rows use the same 9,883 queries and 1,481,803 supplied candidates. Of the 10,000 reserved queries, 117 had no observed positive in the candidate list and were excluded using the same rule as the first experiment. The confirmation sample excludes all original session IDs in the earlier inspected test. It remains a sample from the same broader date period, not an independent dataset.

## Full experiment ledger

Validation guided model, epoch, score-fusion and diversification selection. Confirmation results did not change the frozen selection. All model variants are retained below, including those that failed to improve.

| Method | Validation NDCG@10 | Confirmation NDCG@10 | Confirmation Recall@10 | Jaccard ILD@10 | Cosine ILD@10 |
|---|---:|---:|---:|---:|---:|
| random_order | 0.07195 | 0.07541 | 0.14128 | 0.94213 | 0.91113 |
| supplied_order | 0.07124 | 0.07212 | 0.13240 | 0.88184 | 0.82591 |
| recent_popularity | 0.19958 | 0.18805 | 0.32362 | 0.91548 | 0.87222 |
| original_ranker | 0.20933 | 0.19066 | 0.31465 | 0.89644 | 0.84599 |
| baseline_20k | 0.21332 | 0.18905 | 0.31345 | 0.89854 | 0.84918 |
| enhanced_20k | 0.23641 | 0.22307 | 0.36833 | 0.90029 | 0.85163 |
| enhanced_60k | 0.24089 | 0.22696 | 0.37427 | 0.89941 | 0.85034 |
| enhanced_60k_63 | 0.24000 | 0.22287 | 0.36875 | 0.90385 | 0.85685 |
| views_20k | 0.23716 | 0.22646 | 0.37400 | 0.89912 | 0.85036 |
| views_60k | 0.24386 | 0.22872 | 0.37621 | 0.89730 | 0.84753 |
| views_60k_63 | 0.24394 | 0.22676 | 0.37564 | 0.89997 | 0.85085 |
| recbole_gru4rec | 0.19417 | 0.18066 | 0.31286 | 0.91684 | 0.87433 |
| selected_relevance | 0.24500 | 0.22476 | 0.37174 | 0.90278 | 0.85461 |
| fixed | 0.24141 | 0.22261 | 0.36937 | 0.92974 | 0.89357 |
| history_gate | 0.24225 | 0.22443 | 0.37240 | 0.91378 | 0.87024 |
| adaptive_diversity | 0.24225 | 0.22443 | 0.37240 | 0.91378 | 0.87024 |
| progression | 0.24330 | 0.22450 | 0.37208 | 0.91037 | 0.86538 |

## Frozen primary result

The selected model combines `views_60k_63` and RecBole GRU4Rec with tree weight 0.75 on nonempty click histories. Each score is min-max normalized within its query. Queries without a click prefix retain the tree ranking. Selection reached 0.24500 validation NDCG@10. Its confirmation NDCG is 0.22476, compared with 0.19066 for the original ranker evaluated on these exact confirmation queries.

The absolute NDCG improvement is **+0.03410**, with paired 95% session-cluster bootstrap interval **+0.02977 to +0.03913**. The relative improvement is **+17.89%**, interval **+15.56% to +20.87%**. Recall improves by **+18.14% relative**. Intervals are conditional on this split and these trained models; they do not cover training randomness or dataset variation.

## What actually helped?

- At the same 20k-query training cap, richer features increase validation NDCG from 0.21332 to 0.23641. Their confirmation scores are 0.18905 and 0.22307. This supports the value of features and recent information beyond merely adding training examples.
- Expanding the enhanced training pool to 59,594 evaluable queries raises validation NDCG to 0.24089 and confirmation NDCG to 0.22696. This no-view model gives a substantial gain without relying on cross-table view-clock interpretation.
- Browsing-view features give a small validation improvement. Their same-date shared-clock assumption remains explicit, with the 40-feature no-view model as a control.
- Increasing tree complexity from 31 to 63 leaves did not consistently help. The highest validation score among the tree models did not remain the highest on confirmation.
- RecBole GRU4Rec alone scores 0.19417 on validation and 0.18066 on confirmation. It does not beat the original ranker. It is an actual trained RecBole baseline, not evidence that a neural model must win.
- Adding GRU4Rec improves the selected tree on validation (0.24394 to 0.24500) but reduces it on confirmation (0.22676 to 0.22476). The fusion gain did not generalize. Do not claim that the RecBole component caused the overall confirmation improvement.
- The unselected 31-leaf view model scores 0.22872 on confirmation. That is a diagnostic result, not a replacement selected after inspecting confirmation.

## Diversification with the stronger scorer

The validation-selected fixed relevance weight is 0.7. Against selected relevance-only ranking on confirmation, it changes NDCG by **-0.00215 absolute / -0.95% relative**, paired relative interval **-1.52% to -0.42%**. Jaccard ILD increases **+0.02695 absolute / +2.99% relative**, interval **+2.89% to +3.09%**. Cosine ILD also increases **+4.56% relative**, supporting numerical robustness to the distance measure, not independently proving semantic usefulness.

The adaptive diversity search again chooses equal weights (0.6) in both diversity branches with minimum history two. It is exactly a history gate. Against the fixed policy it has higher NDCG (+0.82% relative) and lower Jaccard diversity (-1.72% relative), so it should be described as a different trade-off rather than a universally worse ranking. There is no evidence here that the diversity threshold itself adds information.

## Metric calibration and subgroup analysis

Random ranking gives 0.07541 confirmation NDCG@10. The selected ranking gives approximately three times that value. NDCG is not a percentage of correctly predicted products. It varies strongly with candidate-list size and the number of clicked targets.

For lists with more than 50 candidates (8,273 queries), original and selected NDCG are 0.15175 and 0.18109. For 11–50 candidates they are 0.33174 and 0.39280. Keeping the full candidate lists is important: changing to shorter/sampled lists could inflate the number while changing the task.

The no-click-history subgroup contains 6,646 queries. NDCG increases from 0.16652 to 0.20907 there, showing that improvement is not confined to repeated-click sessions. The earlier-date subgroup improves from 0.18899 to 0.23104 and the later subgroup from 0.19177 to 0.22056. These subgroup analyses are descriptive, not adjusted multiple hypothesis tests.

## Runtime issue and recovery

The first confirmation execution hung while copying PyTorch checkpoint tensors after LightGBM had activated a different macOS OpenMP runtime. A native stack trace identified a barrier wait. No confirmation metrics were produced in that attempt. The process was stopped, RecBole inference was moved to a fresh subprocess, and the same frozen checkpoints/configuration were evaluated. This was an execution fix, not refitting or outcome-guided selection.

## Recorded provenance

- Frozen selection SHA256: `d3f6069c4af78cda1917fe4d527ec86b900769df7fd41b08c3c795136fe9f5b1`
- Confirmation query-ID SHA256: `d43bae3ec0c4940a0061d25b49defe9a5f149a6806269a2969d82becec0802ae`
- Model/config hashes are in `results/optimization/frozen_selection.json`.
- Per-query metrics are in `results/optimization/confirmation_per_query.npz`.
- Full paired differences are in `results/optimization/confirmation_paired_differences.csv`.
- Figures are in `results/optimization/figures/`.

## Submission interpretation

The research code, reproducible setup, actual RecBole training, model ablations, frozen confirmation and uncertainty estimates now form a substantially stronger core experiment. The final presentation should emphasize the measured benefit of recent information, the limits of a neural/hybrid model, and the lack of added value from the tested session-diversity threshold. It must not promise satisfaction, sales gains, causal intent changes or semantic variety from these offline proxies. Team review/reproduction and the final presentation deck are still required.
