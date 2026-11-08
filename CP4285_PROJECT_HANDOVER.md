# CP4285: project and experiment handover

This document describes the work completed so far for the anonymous-session recommendation project. It is written for teammates who have not followed the implementation. Results are validation results, not final confirmation results. The latest saved results cover models trained with 2,000 and 10,000 queries.

## Research question and hypothesis

Can a recommendation list balance relevance and variety using the behaviour observed in the current anonymous session? The hypothesis is that similar interactions suggest focused interest and should keep more weight on relevance; interactions across different types of products may indicate broader interest and support more variety. If the interactions change during the session, the desired balance may also change.

The current fixed 6 + 4 policy is a useful baseline for that question. It uses the same six/four allocation throughout a session. It does not, by itself, test whether the allocation should change as browsing changes. The exponential adaptive policy uses measured session variety to change the relevance weight; it is a separate experiment.

## Data and session boundaries

The experiments use Diginetica shopping logs, including queries, candidate products and their original order, clicks, item views, purchases, product attributes and categories. A query is one search request with a list of candidate products. The current code reranks the logged candidate list. It does not implement an independent full-catalog retrieval stage that first reduces millions of items to 500.

A daily session combines a source session identifier and date, then replaces it with an anonymous key. A user may have multiple sessions, but the model only uses actions strictly earlier in the **current** session. It does not use a persistent user ID or previous sessions. A session with hidden-label queries is excluded by the sampling protocol. Whole sessions are kept together; target query counts can be exceeded slightly.

The 2,000- and 10,000-query samples are nested, and both use the same fixed validation queries. The earlier metadata audit identified 145,039 eligible queries in the April 1–29 training window; this is not the total number of raw queries. `all` selects the eligible training set.

| Requested training queries | Actual queries | Daily sessions | Train queries with matched purchase | Matched train purchase events | Unmatched train purchases | Validation queries with attributed purchase |
|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 2,000 | 1,299 | 46 | 46 | 26 | 10 |
| 10,000 | 10,000 | 6,481 | 193 | 196 | 119 | 10 |

Both sizes use 503 validation queries from 334 sessions. Click NDCG has 501 eligible validation queries; purchase recall has only 10 purchase-bearing queries. The 2,000-query run has 295,890 candidate rows and 1,993 click-positive training queries; the 10,000-query run has 1,460,212 candidate rows and 9,962 click-positive training queries. Queries without a positive label for the given objective are excluded from fitting that ranker's groups.

## Preprocessing and purchase attribution

1. Select complete, eligible sessions using query metadata and chronological split windows.
2. Read query candidates, product data, clicks, views and purchase events for the selected sessions. Preserve candidate order and original rank for audit.
3. For each query, build a history from events strictly before its timestamp. Exclude its own outcome and future events. Cap the earlier event history at 100 in the recorded runs.
4. Create one feature row per candidate. Aggregate product popularity from earlier training days; use the completed training-period counts for later validation queries.
5. Mark an observed click as a positive click label when it occurs strictly after its query and matches a candidate. Record clicks that do not match candidate lists separately.
6. Attribute each purchase to the unique most recent strictly earlier query in the same daily session that contained the purchased item. Tied latest query times are ambiguous. Purchases without a matching earlier candidate stay in the audit rather than being guessed.

This is a **last-touch heuristic**, not proof that the query caused the purchase. Some products have several eligible earlier queries. The audit records the count of such cases and unmatched purchases.

## Features and models

Product/query features include log price, product category, query-category match, query-token overlap and earlier training click/view popularity. Session features include earlier click/view counts, whether the candidate was previously clicked/viewed, time since its last interaction, and category/token similarity to the session's earlier products. Purchase-aware variants also use earlier purchase popularity and, for the session model, earlier session purchase counts. These features are available before the query; labels are outcomes used after the query for fitting/evaluation.

All six named models are **LightGBM LambdaRank variants**. They are not six different neural architectures. The full-list candidate scores are used to rank products.

| Model | Current-session features | Training gains |
|---|---|---|
| `click_product` | No | Unobserved 0; clicked 1 |
| `click_session` | Yes | Unobserved 0; clicked 1 |
| `purchase_product` | No | Unobserved 0; clicked 1; attributed purchase 3; includes earlier purchase popularity |
| `purchase_session` | Yes | Unobserved 0; clicked 1; attributed purchase 3; includes earlier purchase features |
| `purchase_session_gain_2` | Yes | Purchase gain 2 sensitivity fit |
| `purchase_session_gain_5` | Yes | Purchase gain 5 sensitivity fit |

For example, if A is unobserved, B is clicked, and C is clicked and purchased, the default purchase-aware gains are 0, 1 and 3. C gets its highest grade once; click and purchase gains are not added. An unobserved item is not necessarily disliked: logging and the original ranking affect what could be clicked.

LambdaRank trains the model to improve the ordering of differently graded candidates within each query, emphasizing high ranks. Its scores are ordering scores rather than calibrated click or purchase probabilities.

## Training, tuning, and early stopping

The 2,000- and 10,000-query models were each trained from scratch. The configuration for those saved runs used:

| Parameter | Value |
|---|---|
| Learning rate | 0.05 |
| Number of leaves | 15 or 31 |
| Minimum rows per leaf | 100 or 300 |
| Maximum boosting rounds | 500 |
| Early stopping patience | 30 rounds |
| Evaluation cutoff | top 10 |
| LambdaRank truncation level | 13 |
| Random seed / threads | 4285 / 4 |
| `max_bin`, `max_depth` | 255, -1 (unlimited depth) |
| L1 / L2 regularization | 0 / 0 |
| Feature / bagging fraction | 1 / 1; bagging frequency 0 |

The two leaf settings form four combinations per nominal model. Four nominal models therefore have 16 trials; the gain-2 and gain-5 sensitivity models each add one fit using the nominal purchase/session model's selected leaf settings. Each trial saves its best iteration and validation results. Click models use binary click NDCG for early stopping; purchase-aware models use their training gain mapping for their monitored NDCG. The common primary validation endpoint is click NDCG@10 when choosing a trial, including for purchase-aware models. Sensitivity models are diagnostic and are not nominated as the nominal scorer automatically.

Early stopping means a validation metric failed to improve for 30 rounds; the model keeps its best iteration. The progress log prints every 25 rounds, so it does not show every evaluated iteration. A low best iteration alone does not prove a bug or establish why performance peaked early.

The comparison uses models fitted at their respective sizes and their own validation-selected settings. This is a learning curve under a shared tuning procedure, rather than a fixed-hyperparameter-only comparison. Multiple configurations have been inspected on validation, which makes those scores exploratory.

## Recommendation policies tested

### Exponential adaptive relevance/diversity weighting

For each query, the system computes diversity H of distinct prior clicked/viewed products in the current session. Known pair distances use an equal blend of product-name token Jaccard distance and category distance. The model scores candidate relevance. Greedy reranking selects items by combining normalized relevance with average distance to already selected items.

The normalized exponential curve uses the configured relevance bounds `alpha_min=0.2`, `alpha_max=1.0`, with diversity weight `1-alpha`:

```text
alpha(H) = alpha_min + (alpha_max-alpha_min)
           * (exp(-beta*H)-exp(-beta)) / (1-exp(-beta))
```

Rates beta 0.1, 0.5, 1, 2 and 5 were evaluated. A higher beta generally lowers alpha faster as H increases. When session diversity cannot be measured, the configured fallback alpha is 0.8. History **time decay was off** in these tests; beta refers to the response to session variety, not the ageing rate of past actions. Models were reused across rates, and this reranking did not modify the training loss.

### Fixed 6 + 4 policy

1. Keep the six products with the highest relevance scores.
2. From the remaining logged candidates, choose four products for high pairwise diversity **among those four**.
3. Sort the selected four by relevance and append them after the first six.

The implementation seeds with the most distant remaining pair, then greedily adds the candidate with the greatest total distance to the selected tail. It is an approximation and does not guarantee the globally optimal four-item set. Unknown pair distances receive zero reward during selection; reported ILD excludes unknown comparisons. A candidate list shorter than K produces a shorter result. No extra minimum-relevance threshold was imposed on the four diversity selections.

## Metrics and their populations

| Metric | Meaning | Population / caveat |
|---|---|---|
| Click NDCG@10 | Click gain discounted by `log2(rank+1)`, divided by the ideal candidate ordering | Equal-query mean among queries with a clicked candidate; not a CTR percentage |
| Purchase-aware NDCG@10 | Same normalized ranking measure with common gains 0, 1, 3 | Equal-query mean where an ideal gain exists; a purchase gets its highest grade once |
| Click recall@10 | Clicked candidate products retrieved in the top 10 / all clicked candidates | Queries with at least one clicked candidate |
| Purchase recall@10 | Attributed purchased candidates retrieved in the top 10 / all attributed purchased candidates | Only 10 purchase-bearing validation queries; unmatched purchases are outside this denominator |
| Head click / gain NDCG@6 | Relevance of the first six, normalized to the ideal top-six ordering | Six-plus-four analysis; first six are identical between compared lists |
| Tail ILD@4 | Mean known pair distance among the last four recommendations | Requires known pairs; measures mutual diversity within the tail |
| Overall ILD@10 | Mean known pair distance among all top-10 products | Reported separately for token, category and 50/50 combined distances |
| Pair coverage | Known pairs / all possible pairs | Prevents unknown metadata from looking like measured diversity |
| Training and reranking time | Observed elapsed time for model fits or per-query selection | Depends on hardware and number of trials; not an online latency benchmark |

Purchase recall is averaged over eligible queries, not necessarily the same as the fraction of purchase events retrieved. A no-click or no-gain query is excluded from the respective NDCG denominator. Logged-click NDCG does not directly measure diversity, and unclicked candidates may be unseen or relevant. Evaluate relevance and diversity together rather than treating either as a complete measure.

The professor's earlier suggestions included semantic similarity/accuracy and overlap/relevance distance. The current implementation uses anonymized name-token Jaccard and category distances, **not a learned semantic embedding**. It does not currently report a validated semantic-accuracy measure, a separate overlap relevance-distance metric, online CTR, online conversion or user satisfaction. These should not be claimed as measured outcomes.

**Later implementation update:** A configurable top-K1 relevance filter has
since been added to the 6 + 4 policy, with K1=500 by default. It operates on the
logged, scored candidate list. The results below predate that change and do not
measure the K1-filtered policy.

## Results: fixed 6 + 4 policy

The tables compare relevance-only to 6 + 4 on identical validation queries. Each arrow is before → after. Head click NDCG@6 is the same for both policies because the first six items are retained. Gain NDCG uses the common 0/1/3 evaluation gains for all models, including sensitivity fits. Tail ILD compares the last four products only; it does not measure their distance from the first six.

### Models trained with 2,000 queries

| Model | Head click NDCG@6 | Tail ILD@4 | Click NDCG@10 | Gain NDCG@10 | Purchase recall@10 |
|---|---:|---:|---:|---:|---:|
| click_product | 0.1129 | 0.4562 → 0.5192 | 0.1295 → 0.1280 | 0.1295 → 0.1280 | 0.0000 → 0.0000 |
| click_session | 0.1477 | 0.4480 → 0.5174 | 0.1629 → 0.1579 | 0.1632 → 0.1578 | 0.1000 → 0.0000 |
| purchase_product | 0.1135 | 0.4605 → 0.5184 | 0.1274 → 0.1218 | 0.1274 → 0.1218 | 0.1000 → 0.1000 |
| purchase_session | 0.1539 | 0.4536 → 0.5162 | 0.1713 → 0.1706 | 0.1710 → 0.1703 | 0.1000 → 0.1000 |
| purchase_session_gain_2 | 0.1554 | 0.4573 → 0.5167 | 0.1721 → 0.1700 | 0.1729 → 0.1704 | 0.2000 → 0.1000 |
| purchase_session_gain_5 | 0.1381 | 0.4522 → 0.5160 | 0.1614 → 0.1568 | 0.1620 → 0.1570 | 0.2000 → 0.1000 |

### Models trained with 10,000 queries

| Model | Head click NDCG@6 | Tail ILD@4 | Click NDCG@10 | Gain NDCG@10 | Purchase recall@10 |
|---|---:|---:|---:|---:|---:|
| click_product | 0.1401 | 0.4633 → 0.5186 | 0.1595 → 0.1576 | 0.1601 → 0.1581 | 0.3000 → 0.3000 |
| click_session | 0.1636 | 0.4531 → 0.5185 | 0.1941 → 0.1921 | 0.1958 → 0.1939 | 0.4000 → 0.4000 |
| purchase_product | 0.1306 | 0.4664 → 0.5180 | 0.1511 → 0.1530 | 0.1511 → 0.1530 | 0.1000 → 0.1000 |
| purchase_session | 0.1705 | 0.4615 → 0.5190 | 0.1962 → 0.1920 | 0.1980 → 0.1936 | 0.4000 → 0.3000 |
| purchase_session_gain_2 | 0.1708 | 0.4594 → 0.5192 | 0.1934 → 0.1909 | 0.1953 → 0.1928 | 0.4000 → 0.3000 |
| purchase_session_gain_5 | 0.1616 | 0.4578 → 0.5191 | 0.1866 → 0.1858 | 0.1884 → 0.1876 | 0.4000 → 0.4000 |

For the nominal `purchase_session` model at 10,000 queries, the tail ILD increased from 0.4615 to 0.5190, while whole-list click NDCG fell from 0.1962 to 0.1920 and purchase recall fell from 0.40 to 0.30. Head relevance stayed the same. The purchase recall denominator is only 10 queries, so the apparent difference is highly sensitive to individual queries.

![Six-plus-four comparison across models](results/six_four_evaluation/20261107T092814Z/six_four_comparison.png)

![Purchase metrics across models](results/six_four_evaluation/20261107T092814Z/purchase_metrics_comparison.png)

## Results: exponential relevance-weight rates

Each cell is **click NDCG@10 / combined ILD@10** after exponential reranking. All rates reuse the same fitted models, validation set, K=10, alpha bounds and fallback. History time decay is off. The six-plus-four policy is a separate fixed-quota experiment and does not use these beta values.

### Models trained with 2,000 queries

| Model | Beta 0.1 | Beta 0.5 | Beta 1 | Beta 2 | Beta 5 |
|---|---:|---:|---:|---:|---:|
| click_product | 0.1290 / 0.4925 | 0.1291 / 0.4929 | 0.1293 / 0.4931 | 0.1292 / 0.4933 | 0.1289 / 0.4939 |
| click_session | 0.1569 / 0.4872 | 0.1568 / 0.4875 | 0.1567 / 0.4877 | 0.1568 / 0.4880 | 0.1549 / 0.4899 |
| purchase_product | 0.1237 / 0.4935 | 0.1237 / 0.4937 | 0.1234 / 0.4940 | 0.1231 / 0.4943 | 0.1228 / 0.4948 |
| purchase_session | 0.1681 / 0.4846 | 0.1682 / 0.4848 | 0.1682 / 0.4850 | 0.1688 / 0.4856 | 0.1681 / 0.4873 |
| purchase_session_gain_2 | 0.1703 / 0.4851 | 0.1705 / 0.4853 | 0.1705 / 0.4854 | 0.1697 / 0.4861 | 0.1696 / 0.4876 |
| purchase_session_gain_5 | 0.1558 / 0.4880 | 0.1558 / 0.4884 | 0.1560 / 0.4886 | 0.1560 / 0.4892 | 0.1555 / 0.4905 |

### Models trained with 10,000 queries

| Model | Beta 0.1 | Beta 0.5 | Beta 1 | Beta 2 | Beta 5 |
|---|---:|---:|---:|---:|---:|
| click_product | 0.1604 / 0.4824 | 0.1603 / 0.4828 | 0.1604 / 0.4830 | 0.1609 / 0.4838 | 0.1603 / 0.4852 |
| click_session | 0.1911 / 0.4807 | 0.1910 / 0.4810 | 0.1911 / 0.4815 | 0.1912 / 0.4823 | 0.1901 / 0.4842 |
| purchase_product | 0.1493 / 0.4851 | 0.1491 / 0.4853 | 0.1488 / 0.4855 | 0.1497 / 0.4864 | 0.1494 / 0.4876 |
| purchase_session | 0.1977 / 0.4792 | 0.1977 / 0.4794 | 0.1972 / 0.4799 | 0.1973 / 0.4809 | 0.1959 / 0.4831 |
| purchase_session_gain_2 | 0.1889 / 0.4793 | 0.1888 / 0.4796 | 0.1888 / 0.4800 | 0.1896 / 0.4811 | 0.1887 / 0.4829 |
| purchase_session_gain_5 | 0.1869 / 0.4803 | 0.1868 / 0.4808 | 0.1866 / 0.4814 | 0.1868 / 0.4821 | 0.1850 / 0.4843 |

For the nominal 10,000-query `purchase_session` model, relevance-only click NDCG/ILD were 0.1962/0.4669. Beta 0.1 produced 0.1977/0.4792; beta 5 produced 0.1959/0.4831. The small validation differences are exploratory. Do not interpret a higher offline NDCG as proof of higher live CTR.

![Exponential rates, 10,000-query models](results/exponential_decay_10000/20261107T091326Z/all_models_exponential_rates.png)

## Update, 8 November 2026

Later work is recorded in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md): dataset facts
(95% of queries are category pages, session lengths, clicks versus views), a
fixed-alpha versus adaptive-alpha comparison, a rank-weighted two-tower filter,
the filter → LightGBM → rerank pipeline, and the Phase 1 decisions (ranker
features use clicks only; train on all sessions). The sections below describe
the state on 7 November.

## What is complete and what comes next

**Complete:** anonymous session preprocessing; candidate, interaction and purchase audits; LightGBM ranking variants and gain sensitivity; manual learning curves for 2,000 and 10,000 queries; fixed 6 + 4 and exponential reranking evaluations on validation; CSVs and graphs.

**Not yet complete:** manual 50,000-query and full eligible-data runs; a systematic relevance guard for the four diverse products; adaptive 6 + 4 quota or a within-session trajectory study for the revised policy; a frozen selection and fresh confirmation evaluation for these new experiments; confidence intervals for the reported new policy comparisons; and online assessment of clicks or purchases.

Suggested continuation: inspect actual recommended lists and tail relevance; repeat the same protocol at larger sizes; choose an explicit relevance/diversity decision rule and assess purchase support; freeze model/policy choices before opening fresh confirmation sessions; report paired uncertainty and metadata coverage. The final report should state that the fixed quota and the exponential adaptive curve answer different questions.

## Files and reproduction

- [Manual training and configuration guide](TRAINING_CONFIG_GUIDE.md): parameter definitions, commands, formulas and log interpretation.
- [Research and training plan](TRAINING_PLAN.md): protocol, hypotheses, splits and planned evaluation.
- [Main entry point](main.py): readable high-level calls. See `manual_training_main()`.
- [Training implementation](src/session_rerank/expanded_training.py): LightGBM settings, trials and early stopping.
- [Preprocessing implementation](src/session_rerank/expanded_data.py): anonymous samples, features and purchase attribution.
- [Reranking and metric functions](src/session_rerank/adaptive_policies.py) and [validation evaluation](src/session_rerank/configured_reranking.py).
- [Six-plus-four results CSV](results/six_four_evaluation/20261107T092814Z/comparison.csv).
- [Exponential results, 2,000](results/exponential_decay_2000/20261107T090922Z/comparison.csv) and [10,000](results/exponential_decay_10000/20261107T091326Z/comparison.csv).

The `results/` and `data/` directories are local and ignored by Git. Sharing this Markdown file alone does not transfer the CSVs, figures or data; export the accompanying artifacts separately if teammates need them. The raw dataset should be treated according to its applicable access terms.
