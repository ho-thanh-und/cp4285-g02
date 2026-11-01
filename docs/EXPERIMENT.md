# Experiment specification and interpretation

## Research question

Does observed session diversity help select the relevance/diversity balance beyond a tuned fixed weight or a rule that only checks history length?

H1: A policy conditioned on session diversity yields more lexical variety than fixed re-ranking while meeting the same validation relevance-loss budget.

H2: Session diversity adds value beyond a history-only gate. The progression heuristic is a secondary comparison, not an assumed law of shopping behavior.

## Data and operational session

The original archive has 923,127 queries, 573,935 supplied session IDs, 184,047 products and 1,217 categories. Of all candidate lists, 832,939 (90.23%) contain one category. There are 133,762,939 query/candidate pairs. No candidate rows lack product metadata or name tokens, and no candidate lists contain duplicate IDs. Missing user IDs affect 62.28% of queries.

There are 11,011 supplied session IDs spanning multiple dates. We define an operational session as `(sessionId, eventdate)` and use the most recent ten **distinct clicked products**, ordered by observed click timeframe, before the current query. Each click must come from another query and have timeframe strictly less than the current query. Equal timestamps and current-query clicks are excluded. No persistent user ID or future session summary enters the features.

This is a conservative, daily session definition, not a claim that the original sessions ended at midnight. We use the supplied IDs without inventing a 30-minute inactivity threshold. The summarization point is immediately before each query, not the end of the session. The minimum history needed to use pairwise diversity is a separate parameter, selected from 2, 3 and 5 distinct products. These choices should not be conflated with inactivity boundaries.

Only public query-associated clicks enter history. The 286,967 `is.test=True` queries lack released click labels. They are excluded from training targets and evaluation, not treated as zero-click negatives. Their released candidate impressions can contribute to historical exposure counts. Missing click labels also limit the history available to this benchmark.

Browsing views remain excluded because their timeframe range (12 to 1,199,992) differs substantially from the queries' range (0 to 12,780,662,909). This range difference flags an unresolved integration issue; it does not prove the fields use different units. Purchases do not define the primary target.

## Chronology and sampling

| Purpose | Source-query dates | Query sample cap |
|---|---|---:|
| Frozen item statistics | Before 31 March 2016 | All released eligible events/impressions |
| Ranker training | 1–29 April 2016 | 12,000 |
| Policy selection | 1–14 May 2016 | 6,000 |
| Held-out test | 16 May–1 June 2016 | 10,000 |

The dates 31 March, 30 April and 15 May form one-day gaps. The maximum observed source-query-to-click delay is 78,104,255 milliseconds, less than one day. The gap reduces the chance of label-time overlap at split boundaries when clicks inherit their source query's date. We do not add session-relative milliseconds to midnight to fabricate absolute timestamps.

Queries are sampled uniformly without replacement inside each date interval with seeds 4285, 4286 and 4287. They retain their whole candidate lists, not sampled candidate negatives. Historical statistics use all eligible earlier raw records, not the smaller sampled training set. The code records daily sample counts and hashes of selected query IDs. Sampling is not exactly stratified by day; the daily counts are available to inspect representativeness.

After sampling, queries without at least one observed positive in their supplied candidate list are excluded. This leaves 11,916 training, 5,932 validation and 9,869 test queries. Results estimate ranking quality conditional on this criterion, not across all traffic. Macro means weight queries equally. The same retained queries enter every method.

Daily sessions can share an original session ID across dates/splits. The ranker never uses session IDs as predictive features or earlier/later held-out labels in global statistics. Confidence intervals cluster by original supplied session ID to account for correlated queries. This protocol evaluates future-date queries; it is not an unseen-user or unseen-original-session benchmark.

## Relevance methods

1. Supplied candidate order.
2. Historical distinct-query click popularity, frozen before 31 March.
3. Session-repeat-first heuristic with historical popularity as fallback.
4. LambdaMART, trained on complete candidate lists of sampled training queries with binary query-associated click targets.

The learned ranker uses candidate position, reciprocal position, log popularity, a smoothed historical click/impression ratio, log price, query-token overlap, category match, prior exact clicks/recency, and Jaccard similarity to clicked history. The click/impression ratio counts each query/item positive once. The ranker has 120 trees, learning rate 0.05, 15 leaves and fixed seed 4285. Its hyperparameters were fixed before validation evaluation. See `benchmark.py` for the remaining fixed settings.

Candidate position is the position in the released list. It is useful predictive context, not an estimated exposure propensity. We do not interpret model performance as debiased relevance. The session-repeat baseline is our explicit heuristic, not a faithful reproduction of a published S-POP algorithm. The original competition baseline script is retained for provenance but not run or relabeled as our baseline.

## Diversity and policies

For product token sets A and B, distance is `1 - |A intersection B| / |A union B|`. Repeated name tokens are collapsed into sets. ILD is the mean distance of distinct pairs in the top `min(10, candidate_count)` products. Session diversity uses the same distance over the observed prefix. A single product provides no pairwise diversity estimate.

The greedy objective is `weight * normalized_relevance + (1 - weight) * mean_distance_to_selected_items`. Relevance is min-max scaled once per candidate list. Distances already lie in [0,1] and are not rescaled per greedy step. The highest-relevance item seeds the list. Stable ties keep supplied candidate order. Setting weight to one recovers the relevance ranking exactly.

Every policy uses the same ranker and candidate set. The candidate weights are `[1, .995, .99, .98, .95, .9, .8, .6, 0]`.

- Fixed: one global weight.
- History gate: one weight when the minimum history is available, otherwise relevance-only.
- Diversity threshold: separate weights for lower- and higher-diversity histories. The higher-diversity branch cannot put more weight on relevance than the narrow branch. Thresholds are the validation 25th/50th/75th percentiles for the eligible prefixes. Insufficient history falls back to relevance-only.
- Progression: start diversifying at two prior distinct clicks, then exponentially increase the relevance weight as history grows. This is a heuristic to test the meeting suggestion. Click count is an imperfect proxy for session progress and does not establish a change in intent.
- Shuffled control: apply the frozen diversity policy after permuting its diversity feature within exact history-count strata, seed 4285. No policy is fitted to the shuffled test data.

Validation selects the highest mean ILD policy in each family subject to mean NDCG at least 98% of the relevance ranker's value. This is a **relative 2% loss**, not an absolute 0.02 change. The threshold is an experiment design choice, not a business-derived utility guarantee. All policies include relevance-only as a feasible fallback. The search examines 9 fixed policies, 27 gates, 405 diversity-threshold policies and 27 progression policies.

The adaptive search includes equal branch weights deliberately. On validation it selected 0.6 in both branches with minimum history two. Therefore its threshold does not affect recommendations. Reporting the adaptive method as a successful personalized strategy would misstate the experiment.

## Metrics and uncertainty

NDCG@10 uses binary distinct clicked items, discounts `1/log2(rank+1)`, and ideal ranking over the same candidate set. Recall@10 divides recovered positives by all observed positives in that candidate set. There is no sampled-negative evaluation. ILD@10 is lexical diversity, and we report both absolute changes and percent changes from named baselines.

The bootstrap resamples original session IDs 1,000 times with replacement. Every method shares the same cluster draw, giving paired differences. Within each draw the estimator remains the total metric sum divided by total query count. Reported 95% percentile intervals quantify sampling variability conditional on this trained model/split, not uncertainty across retraining runs or alternate datasets. Multiple subgroup comparisons are descriptive. No formal population-wide causal claim or universal noninferiority claim follows.

The timing JSON explicitly measures distance construction, all nine re-rankings and metric computation. It excludes model prediction and data loading. It is not production end-to-end latency.

## Relation to the papers

Gharahighehi and Vens (2021, sections on balancing diversity/accuracy and greedy re-ranking) adapt SR/SKNN boosting strategies on four music datasets. Their greedy re-ranking comparison uses a fixed weight. Our adaptive weight is an adaptation of that idea for query-candidate ranking with anonymized product-name tokens, not an exact reproduction.

Yin et al. question a universal diversity/accuracy trade-off and show that accuracy-oriented methods can already yield substantial diversity. Our strong popularity and session-repeat baselines make that an empirically relevant concern here. Wu, Yan and Si's competition paper motivates statistic, query/item and session features. References and URLs are in the repository README.

## What would strengthen a subsequent iteration?

Keep the current test results fixed. Verify conclusions with a prespecified different time window or a newly reserved, session-disjoint confirmation sample. Compare daily and original-session history definitions on development data, audit view timing before inclusion, test lexical distance robustness, and inspect whether diversity gains correspond to interpretable product variation. A transformer is useful only if it answers a question that the current experiment leaves unresolved.
