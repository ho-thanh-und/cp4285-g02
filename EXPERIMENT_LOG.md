# Experiment log

This log records the analyses, experiments, decisions and code changes made
since the 10,000-query learning-curve run (7–8 November 2026). Every number is
on **validation** data (503 queries, 501 with a click) unless stated otherwise.
Validation scores are exploratory: several settings were inspected on the same
queries, and nothing here has been checked on selection or confirmation data.

Result folders under `results/` are local (ignored by Git). Each section names
the folder it comes from, so the numbers can be checked or regenerated.

Contents

1. [Code and environment changes](#1-code-and-environment-changes)
2. [Dataset facts](#2-dataset-facts)
3. [Experiment A: fixed alpha versus session-adaptive alpha](#3-experiment-a-fixed-alpha-versus-session-adaptive-alpha)
4. [Experiment B: two-tower first-stage filter](#4-experiment-b-two-tower-first-stage-filter)
5. [Experiment C: filter → LightGBM → rerank pipeline](#5-experiment-c-filter--lightgbm--rerank-pipeline)
6. [Experiment D (Phase 1): product views and training sessions](#6-experiment-d-phase-1-product-views-and-training-sessions)
   - [6a. Experiment E (Phase 2, step 1): category-page ranker](#6a-experiment-e-phase-2-step-1-category-page-ranker)
   - [6b. Experiment F (Phase 2, step 2): scaling to 50,000 training queries](#6b-experiment-f-phase-2-step-2-scaling-to-50000-training-queries)
   - [6c. Experiment G: classical baseline (item-kNN) and MRR](#6c-experiment-g-classical-baseline-item-knn-and-mrr)
7. [Decisions](#7-decisions)
8. [Plan and status](#8-plan-and-status)
9. [Known issues and caveats](#9-known-issues-and-caveats)
10. [Commands and files](#10-commands-and-files)

---

## 1. Code and environment changes

**LightGBM training on native Windows (fixed).** Training crashed with
`OSError: access violation` in `LGBM_DatasetSetField`. Bisecting showed the
cause is import order: importing pandas *before* LightGBM breaks LightGBM's
native library on this machine; importing LightGBM first works. Data, label
types and group sizes were ruled out. Earlier results were produced in WSL,
which is why this did not show up before.

- `src/session_rerank/__init__.py` now imports LightGBM first, so any script
  that imports `session_rerank` before pandas is safe.
- `tests/conftest.py` does the same for the test suite.
- Test suite on Windows: **156 of 158 pass** (as of §6c; 141 of 143 when
  first fixed). The 2 failures
  (`test_anonymous_study.py`, `test_expanded_study.py`, older study pipelines)
  fail while writing a temporary file at the end of a ~250-character path;
  most likely the Windows 260-character path limit, not a code fault. Not yet
  fixed.

**New modules** (each has a `main.py` action, a JSON config and tests):

| Module | Action | Config | Tests |
|---|---|---|---|
| `alpha_sweep.py` | `alpha-sweep` | `config/alpha_sweep.json` | `tests/test_alpha_sweep.py` |
| `two_tower.py` | `two-tower` | `config/two_tower.json` | `tests/test_two_tower.py` |
| `filter_pipeline.py` | `filter-pipeline` | `config/filter_pipeline.json` | `tests/test_filter_pipeline.py` |
| `event_ablation.py` | `event-ablation` | `config/event_ablation.json` | `tests/test_event_ablation.py` |
| `ranker_phase2.py` | `ranker-phase2` | `config/ranker_phase2.json`, `config/ranker_phase2_50k.json` | `tests/test_ranker_phase2.py` |
| `main.py` (`prepare_data_main`) | `prepare-data` | `config/learning_curve.json` | `tests/test_ranker_phase2.py` |
| `item_knn.py` | `item-knn` | `config/item_knn.json` (its `compare_with` lists the Phase 2 runs to join) | `tests/test_item_knn.py` |

**Analysis scripts** moved into the repository: `scripts/analysis/session_lengths.py`
and `scripts/analysis/click_view.py`. Other dataset facts below were measured
with short one-off queries; the method is described next to each number.

---

## 2. Dataset facts

### Size of the training pool

- 923,127 queries in total. The April 1–29 training window has 279,058 queries;
  **145,039** are eligible (whole sessions without any hidden-label query) in
  **93,231** daily sessions. (An earlier answer quoted 190,738; that counted
  visible queries individually and ignored that a session with any hidden query
  is excluded as a whole.)
- The pilot (`config/anonymous_samples.json`) used 2,000 training queries. It
  was described as a "small sample" pilot; an older protocol used 12,000. The
  learning curve now trains 2,000 and 10,000 queries; 50,000 and all remain.

### What a "query" is

- **95% of training queries have no search text.** They are category-page
  browses (`categoryId > 0`); the remaining 5% are text searches with
  `categoryId = 0`.
- Category pages show products from their category: 99.95% of logged
  candidates and of clicked products belong to the query's category. A category
  has a median of 513 products (76% of browse queries' categories have more
  than 200; 27% more than 1,000).
- Each logged list has **at most 200 products** (median 197). The models only
  ever rerank these lists. 90% of lists contain a single category, so category
  diversity is near zero by construction.
- 99% of post-query clicks are on a logged product (5,878 matched, 51 missing
  in the pilot sample). Products the site never showed have no click labels.

### Session lengths (`scripts/analysis/session_lengths.py`)

Session = source session ID + date. Project training pool (April, no hidden
query), 93,231 sessions:

| Per session | Mean | Median | 90th pct | Max | Sessions with ≥ 1 |
|---|---:|---:|---:|---:|---:|
| Queries | 1.56 | 1 | 3 | 54 | 100% |
| Clicks | 2.66 | 1 | 6 | 220 | 100% |
| Views | 2.86 | 2 | 7 | 87 | 74% |
| Orders | 0.035 | 0 | 0 | 3 | 3.4% |

- 72.7% of sessions have one query; 6% have four or more.
- Whole dataset (592,643 sessions): median 1 query, 1 click, 1 view; 2.2% of
  sessions buy.
- Every visible session has at least one click. The 35% of all sessions without
  clicks are almost all hidden-label (competition test) sessions: 73% of those
  show no click because their clicks were withheld. Training data is therefore
  "sessions where someone clicked".

### Clicks and views (`scripts/analysis/click_view.py`)

- 69.9% of clicks have a view of the same product in the same session; 67.4%
  have it at or after the click. Median delay from click to view: **1.5 s**
  (90th percentile 3.1 s, 99th 10.4 s).
- 66.2% of views have a click on the same product in the same session; about a
  third of views have no click behind them.
- 80.8% of sessions with a click also have a view.
- Organizer statement (competition forum, relayed from a search summary): clicks
  cover clicks on site recommendations and site search; views come from other
  navigation and **view data is sampled** (some views are missing).
  <https://competitions.codalab.org/forums/7901/1501>
- Reading: a view is mostly the product page opened by a click; the rest records
  other browsing. The dataset does not document how a shopper reached a page.

### Click rate by displayed rank (10,000-query training sample)

| Rank | 1 | 2–5 | 6–10 | 11–20 | 21–50 | 51–100 | 101–150 | 151–200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Click rate | 3.1% | 2.3% | 2.1% | 1.7% | 1.1% | 1.0% | 0.8% | 1.0% |
| Share of all clicks | 2% | 6% | 6% | 10% | 18% | 23% | 16% | 18% |

- Fitted examination curve: click rate ∝ rank^−0.304 (ranks 1–100). Much
  flatter than web search: rank 10 keeps two-thirds of rank 1's rate.
- 57% of clicks are below rank 50; median clicked rank 65–67. The site's
  original order is weak.
- These rates mix position bias with the site's own relevance ordering.

### First versus later queries in a session (10,000-query training sample)

| | 1st query | 2nd | 3rd+ |
|---|---:|---:|---:|
| Queries | 6,481 | 1,729 | 1,790 |
| Clicks per query | 1.51 | 1.67 | 1.73 |
| Median click rank | 65 | 62 | 60 |
| Token ILD of clicked products (2+ clicks) | 0.856 | 0.861 | 0.860 |
| Token ILD of the same number of random products | 0.941 | 0.947 | 0.943 |
| Same category as the previous query | – | 64% | 57% |

- First-query shoppers are not browsing at random: they click products more
  similar to each other than random ones, as later queries do. A random first
  list cost two-thirds of NDCG (0.078 vs 0.222 for popularity order).
- Clicked-product variety does not shrink over a session.
- 36% of second queries switch category: a candidate signal for adaptation.

### Validation history coverage

- 66.8% of validation queries have no earlier click/view in their session;
  7.2% have too little (fewer than two usable products); **26.0% (131 queries)**
  have measurable session variety H (median 0.50, interquartile 0.45–0.61).

### Diversity measures

- Token ILD@10 is close to its ceiling everywhere (≈ 0.89–0.96): most product
  names share no tokens. Category ILD@10 is ≈ 0.02 on logged lists. The 50/50
  combined ILD therefore moves very little under any policy.

---

## 3. Experiment A: fixed alpha versus session-adaptive alpha

`results/alpha_sweep/20261108T044340Z_queries_10000/` — `uv run main.py alpha-sweep --trained-run results/learning_curve/queries_10000`

**Definitions.** Greedy reranking picks each next product by
`alpha × relevance + (1 − alpha) × mean distance to products already chosen`.
alpha = 1 is relevance only. The exponential policy sets alpha per query from
session variety H, falling from 1.0 toward 0.2 as H rises; beta is the speed of
that fall; queries without measurable H use alpha = 0.8.

**Design.** Saved 10,000-query models, no retraining. Policies: relevance;
fixed alpha 0.95 … 0.2 (draws the trade-off curve); exponential beta 0.1, 0.5,
1, 2, 5; and a shuffle control that permutes each beta's alphas across the
measured-history queries. Each adaptive point is compared with the fixed curve
at the same ILD (NDCG gap), with 2,000 session-bootstrap replicates; reported
for all queries and for the 131 measured-history queries.

**Results (`purchase_session`, all queries).**

| Policy | Mean alpha | Click NDCG@10 | Combined ILD@10 |
|---|---:|---:|---:|
| Relevance | 1.00 | 0.1962 | 0.4669 |
| Fixed 0.8 | 0.80 | **0.1985** | 0.4774 |
| Exponential beta 0.1 | 0.74 | 0.1977 | 0.4792 |
| Exponential beta 5 | 0.66 | 0.1959 | 0.4831 |

- Across 6 models × 5 betas × 2 populations, every NDCG gap to the fixed curve
  lies between about −0.005 and +0.003; no positive gap has an interval above
  zero. The shuffle control scores the same as the matched policy.
- **Conclusion:** on validation, session-variety adaptation does not beat a
  well-chosen fixed alpha. Mild fixed diversification (alpha ≈ 0.8) adds ILD
  without losing NDCG.

---

## 4. Experiment B: two-tower first-stage filter

`uv run main.py two-tower --trained-run results/learning_curve/queries_10000`

**Rank-aware training.** Pairs of (clicked product, other product) with weights:

| Product | Weight | Examples (eta = 0.304) |
|---|---|---|
| Clicked at rank r | r^eta, capped at 10 | rank 1 → 1.0, rank 10 → 2.0, rank 100 → 4.0 |
| Shown at rank r, not clicked | r^−eta | rank 1 → 1.0, rank 10 → 0.5, rank 150 → 0.22 |
| Never shown (same category, or catalog for text search) | 0.5 | |

Pair loss: `w_click × w_other × softplus(score_other − score_click)`. eta is
fitted to the click-rate curve each run (`propensity_eta: "estimate"`).

**Towers.** Product: name tokens, category, price band, and (from v2)
popularity. Query: page category or search tokens, earlier clicked/viewed
products in the session (through the product tower), query position. Score is
a scaled cosine. Pool to filter: the query's category (category pages) or the
whole catalog (text search).

**Popularity without leakage (v2).** Daily clicks and views per product from the
full raw files over April 1–29 (334,858 clicks). A training query on day d sees
only days before d, as daily rates. Validation sees the whole window. The
popularity baseline uses the same whole-window counts.

### Runs

| Folder | Variant |
|---|---|
| `results/two_tower/20261108T050707Z_queries_10000` | v1: no popularity, eta estimated (0.304) |
| `results/two_tower/20261108T050906Z_queries_10000` | v1 with eta = 0 (no rank weighting); settings file records it |
| `results/two_tower/20261108T052545Z_queries_10000` | **v2: with popularity**, eta estimated |

Recall of clicked products when keeping the top K1 (v2 and baselines):

| K1 | Site order | Popularity | Tower v1 | Tower v2 | v2 − popularity (95% CI) |
|---:|---:|---:|---:|---:|---|
| Whole category/catalog | | | | | |
| 20 | 0.260 | 0.379 | 0.248 | 0.348 | −0.031 (−0.058, −0.006) |
| 50 | 0.459 | 0.581 | 0.448 | 0.572 | −0.009 (−0.035, +0.017) |
| 100 | 0.687 | 0.756 | 0.584 | 0.729 | −0.027 (−0.053, −0.001) |
| 200 | 1.000* | 0.882 | 0.712 | 0.853 | −0.030 (−0.053, −0.007) |
| Logged list only | | | | | |
| 20 | 0.260 | 0.484 | 0.401 | 0.451 | −0.033 (−0.061, −0.008) |
| 50 | 0.459 | 0.707 | 0.610 | 0.688 | −0.019 (−0.045, +0.008) |

\* The site's list contains every clicked product by construction.

- **Rank weighting** (v1 vs eta = 0): clearly better only at full-pool K1 = 50
  (+0.042, CI +0.016 to +0.069); other K1 within noise. One run each.
- **Popularity input** raised tower recall by about 0.10–0.15, but the tower
  remains slightly below plain popularity. Text searches (25 validation
  queries) fail for both (recall ≈ 0). v2 overfits after epoch 3 of 8.
- In v1, about half of the tower's top 100 were never shown by the site; those
  products can never count as hits.

**Top-10 lists, before LightGBM** (v2, re-scored after training with the new
top-10 metrics; future runs write `top10_summary.csv` themselves):

| Top 10 from | Click NDCG@10 | Combined ILD@10 | Category ILD | Never shown |
|---|---:|---:|---:|---:|
| Site order | 0.087 | 0.457 | 0.025 | 0% |
| Popularity (logged list) | **0.201** | 0.469 | 0.021 | 0% |
| Two-tower (logged list) | 0.172 | 0.464 | 0.023 | 0% |
| Popularity (whole category) | 0.164 | 0.470 | 0.019 | 23% |
| Two-tower (whole category) | 0.135 | 0.473 | 0.041 | 21% |
| Random (logged list) | 0.072 | 0.484 | 0.025 | 0% |
| *LightGBM `purchase_session`, relevance only* | *0.196* | *0.467* | | *0%* |

- **Sorting the logged list by April popularity (0.201) matches the best
  LightGBM model (0.196).** LightGBM's popularity features count only the
  sampled sessions; giving it full-April counts is a Phase 2 item.

---

## 5. Experiment C: filter → LightGBM → rerank pipeline

`results/filter_pipeline/20261108T054207Z_queries_10000/` — `uv run main.py filter-pipeline --trained-run results/learning_curve/queries_10000 --two-tower-run results/two_tower/20261108T052545Z_queries_10000`

For each validation query: choose a pool (logged list; two-tower or popularity
top K1 from the whole category; or those filters restricted to shown
products), compute the usual LightGBM features for every pool product, score
with a saved model, rerank with a policy, and measure the top 10. NDCG and
recall divide by **all** clicks of the query, so a pool that misses clicked
products is penalised. Check: the logged-list relevance row reproduces the
existing ILD exactly (0.4669); NDCG is 0.1949 rather than 0.1962 because the
denominator now includes the 0.6% of clicks on products outside the list.

`purchase_session`, K1 = 100:

| Pool | Policy | NDCG@10 | ILD@10 | Category ILD | Never shown |
|---|---|---:|---:|---:|---:|
| Logged list | relevance | 0.195 | 0.467 | 0.022 | 0% |
| Logged list | fixed 0.8 | **0.197** | 0.477 | 0.024 | 0% |
| Logged list | 6 + 4 | 0.191 | 0.485 | 0.031 | 0% |
| Two-tower, whole category | relevance | 0.154 | 0.480 | 0.044 | 26% |
| Two-tower, whole category | 6 + 4 | 0.150 | **0.494** | 0.046 | 26% |
| Popularity, whole category | relevance | 0.154 | 0.482 | 0.046 | 26% |
| Two-tower, shown only | relevance | 0.190 | 0.456 | 0.002 | 0% |
| Popularity, shown only | relevance | 0.191 | 0.458 | 0.001 | 0% |

Differences from the logged list (relevance policy, session bootstrap):

- Two-tower K1 = 100, whole category: NDCG −0.041 (−0.057, −0.027);
  ILD +0.013 (+0.006, +0.020).
- Two-tower K1 = 100, shown only: NDCG +0.000 (−0.010, +0.010); ILD −0.000.

**Conclusions.**

1. Cutting the logged list to the filter's top 100 does not lose NDCG.
2. The NDCG drop with never-shown products is a worst case: about 2.6 of 10
   slots go to products nobody could click. Offline data cannot say whether
   they are relevant.
3. Never-shown same-category products double category ILD (0.022 → 0.044), a
   larger diversity increase than any reranking policy achieves.
4. The two-tower and popularity filters are tied end to end.
5. Fixed alpha 0.8 is the best policy in the logged and two-tower pools
   (slightly above relevance-only); in the popularity pools it is slightly
   below relevance-only. No adaptive policy beats it in any pool.

---

## 6. Experiment D (Phase 1): product views and training sessions

`results/event_ablation/20261108T064707Z_queries_10000/` — `uv run main.py event-ablation --trained-run results/learning_curve/queries_10000`

**Design.** Rebuild the LightGBM features under three event rules and retrain
`click_session` and `purchase_session` on two training sets, each with the
run's full 4-setting grid (48 fits). Validation always uses all 503 queries.
Check passed: the rebuilt "all events" features equal the cached features.

- Event rules: `all_events` (current); `clicks_only` (no views anywhere in
  features or popularity); `unexplained_views` (drop a view when the same
  product was clicked in the same session up to 60 s earlier; removed 58% of
  views, keeping 8,049 of 19,255).
- Training sets: `all_sessions` (current, 9,962 click queries);
  `multi_click_sessions` (sessions with ≥ 2 clicks, 6,397 queries, −36%).

Click NDCG@10, validation-selected trial / mean over the 4 grid settings:

| Events | Training set | `click_session` | `purchase_session` |
|---|---|---|---|
| All events | All sessions | 0.194 / 0.189 | 0.196 / 0.189 |
| Clicks only | All sessions | 0.195 / 0.192 | **0.200** / 0.190 |
| Unexplained views | All sessions | 0.196 / 0.194 | 0.196 / 0.192 |
| All events | ≥ 2 clicks | 0.185 / 0.182 | 0.186 / 0.182 |
| Clicks only | ≥ 2 clicks | 0.180 / 0.179 | 0.189 / 0.181 |
| Unexplained views | ≥ 2 clicks | 0.183 / 0.180 | 0.185 / 0.181 |

- **Views:** every difference from the current setup lies between −0.002 and
  +0.005, and every 95% interval includes zero, overall and for queries with
  and without history.
- **≥ 2 click sessions:** −0.007 to −0.011 (grid mean) in every combination.
  Queries with history are unchanged (±0.002); the loss sits in queries without
  history, e.g. −0.015 (−0.029, −0.001) for clicks-only. A single-click query
  still teaches LambdaRank one winner against ~140 skipped products.
- **Hyperparameter sensitivity:** early stopping keeps 4–115 trees, and the
  four grid settings alone move NDCG by up to 0.017, more than any data choice.
  Comparisons should use the grid mean, not only the selected trial.

---

## 6a. Experiment E (Phase 2, step 1): category-page ranker

`results/ranker_phase2/20261108T073511Z_queries_10000/` — `uv run main.py ranker-phase2 --trained-run results/learning_curve/queries_10000`

**Design.** All variants follow the agreed data choices: clicks-only features,
all sessions, text-search queries left out of training and early stopping
(9,419 click-model training queries). Each variant retrains `click_session` and
`purchase_session` with the full 4-setting grid. Primary segment: the 476
category-page validation queries with a click (158 with session history, 318
without). References on the same queries: the site's original order, and the
logged list sorted by whole-window click popularity.

| Variant | Change |
|---|---|
| `base` | Current ranker under the agreed data choices |
| `popularity` | + daily click and purchase rates per product over Apr 1–29 from the full raw files (334,858 clicks); a training query on day d sees only days before d |
| `popularity_position` | + LightGBM position-bias correction (rank buckets 1, 2, 3, 4, 5, 6–10, 11–20, 21–50, 51–100, 101–200) |
| `popularity_xendcg` | popularity features with the `rank_xendcg` loss instead of LambdaRank |
| `popularity_lr002` | popularity features with learning rate 0.02 instead of 0.05 |

Click NDCG@10 on category pages, validation-selected trial / grid mean:

| Ranker | `click_session` | `purchase_session` |
|---|---|---|
| Site's original order | 0.083 | |
| **Popularity sort** | 0.198 | |
| `base` | 0.179 / 0.178 | 0.187 / 0.183 |
| `popularity` | 0.216 / 0.208 | 0.215 / 0.212 |
| `popularity_position` | **0.229** / 0.217 | 0.217 / 0.213 |
| `popularity_xendcg` | 0.221 / **0.219** | **0.228** / **0.223** |
| `popularity_lr002` | 0.215 / 0.209 | 0.211 / 0.205 |

Differences on category pages (session bootstrap, 95% intervals):

- **Popularity features vs `base`:** +0.028 to +0.037 in every model and
  statistic; every interval excludes zero (e.g. `click_session` grid mean
  +0.030, +0.013 to +0.048).
- **On top of popularity features:** `rank_xendcg` +0.011 (grid mean, both
  models; intervals −0.001 to +0.024 and −0.000 to +0.022), position correction
  +0.009 for `click_session` (−0.000 to +0.018) and +0.001 for
  `purchase_session`, learning rate 0.02 −0.007 to +0.001. None is clearly
  above zero alone.
- **Against the popularity sort:** `popularity_xendcg` is the only variant ahead
  with intervals above zero in all four model/statistic combinations:
  `click_session` +0.023 (+0.001, +0.047) selected and +0.021 (+0.002, +0.042)
  grid mean; `purchase_session` +0.030 (+0.008, +0.053) and +0.025 (+0.005,
  +0.047).
- **Where the gain comes from:** on category queries **with** session history
  every popularity variant beats the popularity sort by +0.07 to +0.10
  (intervals above zero). On queries **without** history they tie with it or
  fall slightly behind (−0.001 to −0.026). This matches the product decision
  that cold start uses popularity (§7).
- **Feature use** (`popularity`, gain share): `category_id` 50–56%, window click
  rate 21–23%, session token similarity 8–10%, time since the product was last
  seen in the session 5–6%.
- **Data choices:** `base` scores 0.188 on all queries against 0.195 for the
  Phase 1 clicks-only model; the difference comes from leaving text searches
  out of training and choosing trials on category pages, and is within the
  grid's own spread (§6).

**Conclusion.** Window popularity is the clear gain. The lead candidate is
`popularity_xendcg`, the first ranker that beats the popularity sort on category
pages with intervals above zero. It is chosen on validation after inspecting
five variants, so it must be confirmed on the selection set (Phase 5).

---

## 6b. Experiment F (Phase 2, step 2): scaling to 50,000 training queries

`results/ranker_phase2/20261108T075158Z_queries_50000/` — data prepared with
`uv run main.py prepare-data --queries 50000 --curve-root results/learning_curve_v2`, then
`uv run main.py ranker-phase2 --trained-run results/learning_curve_v2/queries_50000 --phase2-config config/ranker_phase2_50k.json`

**Data.** 50,000 training queries, 7,360,538 candidate rows (645 MB folder);
47,347 category-page click queries used for training. The 50k sample nests the
10k sample and uses the same 503 validation queries (both checked by the
preparation code). A new curve folder `results/learning_curve_v2` was needed
because the original `results/learning_curve` fingerprints code and config
that the uncommitted K1-filter edits changed; the data-preparation code itself
is unchanged.

**New `prepare-data` action** (`main.py`, `prepare_data_main`): runs only the
sampling and feature-cache steps of manual training, without fitting the six
learning-curve models. A later `train` with the same `--queries` and
`--curve-root` resumes from these stages. To fit 50k in memory (1.6 GB free of
15 GB), feature rebuilding now fills one array of the final size instead of
stacking copies, and Phase 2 reads only the candidate columns it needs.

**Variants** (`config/ranker_phase2_50k.json`): `popularity` (LambdaRank),
`popularity_xendcg`, `popularity_position`; `base` and learning rate 0.02 were
settled at 10k. Here "difference vs base" in `differences.csv` means vs the
`popularity` LambdaRank model.

Click NDCG@10 on category pages (476 queries), selected trial / grid mean:

| Ranker | 10k `click_session` | **50k `click_session`** | 10k `purchase_session` | **50k `purchase_session`** |
|---|---|---|---|---|
| Popularity sort | 0.198 | 0.198 | | |
| `popularity` | 0.216 / 0.208 | 0.231 / 0.229 | 0.215 / 0.212 | 0.230 / 0.223 |
| `popularity_xendcg` | 0.221 / 0.219 | 0.231 / 0.229 | 0.228 / 0.223 | **0.232 / 0.231** |
| `popularity_position` | 0.229 / 0.217 | **0.235** / 0.229 | 0.217 / 0.213 | 0.228 / 0.225 |

- **More data helps:** 50k vs 10k (grid mean, same variant) +0.008 to +0.021,
  every interval above zero (e.g. `popularity` `click_session` +0.021, +0.009
  to +0.034). Selected-trial gains are similar but their intervals touch zero.
- **The variants converge at 50k:** xendcg and position correction vs
  LambdaRank are within ±0.008 with intervals around zero (best: xendcg
  `purchase_session` grid mean +0.008, −0.002 to +0.017).
- **All three beat the popularity sort** on category pages: +0.025 to +0.037,
  intervals above zero in all 12 model/variant/statistic combinations (e.g.
  `popularity_xendcg` `purchase_session` grid mean +0.033, +0.012 to +0.056).
- **Segments:** with session history +0.10 to +0.12 over the popularity sort
  (intervals above zero); without history a tie (−0.012 to +0.002). xendcg is
  the closest to the popularity sort without history (−0.002 to +0.000).
- Early stopping still keeps few trees (6–90). Feature use (xendcg): category ID
  ~40%, window click rate ~32%, time since last seen in session 7–10%, session
  token similarity ~6%.

**Conclusion (6b).** The ranker keeps improving with data, and the loss choice and
position correction stop mattering at 50k. `popularity_xendcg` stays the lead
candidate: never behind the others, best `purchase_session` grid mean, and
level with the popularity sort on cold queries. The all-data size (~145k
queries, ~21M rows) would need roughly 3–4 GB at peak with the current code,
more than is free on this machine.

---

## 6c. Experiment G: classical baseline (item-kNN) and MRR

`results/item_knn/20261108T081521Z_queries_50000/` — `uv run main.py item-knn --trained-run results/learning_curve_v2/queries_50000`
(lead ranker rerun with MRR: `results/ranker_phase2/20261108T081226Z_queries_50000/`, `config/ranker_lead_50k.json`)

**MRR@10** (`reciprocal_rank` in `event_ablation.py`): 1 / position of the first
clicked product in the top 10, 0 if none, undefined for queries without a
clicked candidate (same population as NDCG). Added to every per-query result of
Phases 1–2 and the baseline. It lives outside `adaptive_policies.py` on
purpose: the learning-curve folders fingerprint that file.

**Item-kNN** (`item_knn.py`), the classical collaborative filter:
- Baskets: distinct clicked products per visible session over Apr 1–29 from
  the raw files (115,936 sessions, 56,479 with ≥ 2 products; 334,858 clicks;
  at most 50 most recent products per session).
- Similarity: co-click cosine, `co_clicks(i, j) / sqrt(sessions(i) × sessions(j))`,
  no shrinkage; 1,221,630 product pairs over 45,480 clicked products.
- Ranking: sum of similarities to the products clicked earlier in the session;
  ties and cold shoppers fall back to window click popularity, then the site's
  order. Without history it equals the popularity sort (checked by a test).
- Coverage: 164 validation queries have earlier clicks; in 130 of them at least
  one candidate was co-clicked with a history product.

The lead ranker was retrained to add MRR; it reproduced §6b exactly
(`click_session` 0.2290, `purchase_session` 0.2310 grid mean), confirming that
training is deterministic.

**Comparison on identical category-page queries** (476; LightGBM = grid mean of
`popularity_xendcg`, 50k):

| Method | NDCG@10 | Recall@10 | MRR@10 | NDCG@10 with history (158) | NDCG@10 no history (318) |
|---|---:|---:|---:|---:|---:|
| Site's original order | 0.083 | 0.146 | 0.073 | 0.079 | 0.085 |
| Popularity sort | 0.198 | 0.336 | 0.175 | 0.150 | 0.222 |
| Item-kNN (co-click CF) | 0.206 | 0.346 | 0.183 | 0.173 | 0.222 |
| LightGBM xendcg, `click_session` | 0.229 | 0.372 | 0.203 | 0.247 | 0.220 |
| **LightGBM xendcg, `purchase_session`** | **0.231** | **0.374** | **0.204** | **0.252** | 0.221 |

Differences (session bootstrap, 95% intervals), category pages:

- **Item-kNN vs popularity sort:** NDCG +0.008 (−0.005, +0.021); with history
  +0.023 (−0.014, +0.062). Helpful in direction, not clearly above zero.
- **LightGBM vs item-kNN:** `purchase_session` NDCG +0.025 (+0.003, +0.047),
  recall +0.028 (+0.001, +0.055), MRR +0.021 (−0.001, +0.045); with history
  NDCG +0.078 (+0.020, +0.143). `click_session` is similar (+0.023, +0.003 to
  +0.047).
- **LightGBM vs popularity sort:** NDCG +0.033 (+0.014, +0.056), recall +0.038
  (+0.010, +0.069), MRR +0.029 (+0.006, +0.053).
- Without history all methods except the site's order tie (differences within
  ±0.005).

**Conclusion.** The learned ranker beats both the non-personalised baseline and
the classical CF baseline on all three metrics, and the advantage sits in
sessions with history. Co-click CF alone is a weak personaliser here (most
sessions are short and one-category), but it is the natural model for the item
page.

---

## 7. Decisions

| Date | Decision | Evidence |
|---|---|---|
| 2026-11-08 | **Category-page ranker adds window popularity features; lead candidate is `rank_xendcg` with those features.** Position correction and learning rate 0.02 are not adopted for now. | §6a: popularity +0.03 (intervals above zero); xendcg is the only variant ahead of the popularity sort in every model/statistic. To be confirmed on selection data (Phase 5). |
| 2026-11-08 | **Item-kNN on session co-clicks is the classical CF baseline** (and the planned "You may also like" model); MRR@10 is reported with NDCG@10 and Recall@10. | §6c: LightGBM beats item-kNN by +0.025 NDCG (interval above zero); item-kNN beats popularity only in direction. |
| 2026-11-08 | **Train the category-page ranker on 50,000 queries** (data folder `results/learning_curve_v2/queries_50000`); keep `popularity_xendcg` as the lead. Position correction is dropped. | §6b: 50k adds +0.008 to +0.021 over 10k; at 50k all variants beat the popularity sort by about +0.03 and differ from each other by less than 0.01. All-data training does not fit in the free memory yet. |
| 2026-11-08 | **Final product: a recommendation website** with a home page, category pages and an item page with "You may also like" (user decision; §8.1). | Product goal set by the team. |
| 2026-11-08 | **Cold start uses popularity**, not collaborative filtering (user decision): popular overall or trending on the home page, popular within the category on category pages. Item-to-item CF is used on the item page, where the viewed item is the signal. | An anonymous first visit has no user ID and no interaction for CF to use; two-thirds of validation queries have no session history (§2). |
| 2026-11-08 | **The home page is evaluated online**: the team's website logs interactions (user decision). An offline next-item proxy remains optional. | Diginetica has no home-page impressions (§8.1). |
| 2026-11-08 | **Demo product names are generated by AI** (user decision). They are display-only: models, features and metrics keep using the anonymised token IDs. | The catalog has no real names or images (§8.1). |
| 2026-11-08 | **"You may also like" uses clicks only** (user decision). As understood: a clicked product is a positive; products not clicked (or never shown) are not treated as negatives, neither in training nor in evaluation. To be confirmed by the user. | Overrides the earlier proposal to use views for this page (§8.1, point 4). |
| 2026-11-08 | **Text-search queries are excluded from training for now** (user decision). | 5% of queries; both the two-tower and popularity fail on them (§4). |
| 2026-11-08 | **Ranker features use clicks only (no product views).** | §6: no measurable effect; simpler. Applies to the category-page ranker from Phase 2; Experiments A–C used all events. Views stay available for "You may also like", whose signal is product-page visits (§8.1). |
| 2026-11-08 | **Train the ranker on all sessions**, not only multi-click sessions. | §6: multi-click training loses ~0.01 NDCG, mostly on queries without history. |
| 2026-11-08 | Keep LightGBM LambdaMART as the main relevance model. | Popularity order ≈ LightGBM (§4); legacy benchmark: SASRec 0.204 vs tree 0.231, GRU4Rec fusion added nothing (README "Earlier reported results"). |
| 2026-11-08 | Do not use NeuMF. | Needs a learned embedding per user/session ID; 62–70% of rows have no user ID, the study is anonymous by design, and validation sessions are always new. The two-tower plays that role for unseen sessions. |
| 2026-11-08 | GRU4Rec only as an optional, time-boxed extra feature, trained on ≥ 2-click sessions. | Only ~⅓ of queries have earlier history; legacy fusion showed no gain. |
| 2026-11-08 | RankNet/BPR are treated as loss comparisons, not new models. | LambdaMART is RankNet's pairwise loss with NDCG weighting; the two-tower already uses a BPR-style loss. |
| 2026-11-08 | Do not randomise first-query lists. | §2: first-query clicks are as focused as later ones; random costs ⅔ of NDCG. |
| 2026-11-07 | K1 = 500 on the logged list is a no-op. | Logged lists hold at most 200 products; use K1 ≤ 50–100 as a real relevance guard, or a full-category filter (§5). |

---

## 8. Plan and status

| Phase | Content | Status |
|---|---|---|
| 0 | Windows training fix; commit current work; data-only "prepare all" step | **Done.** Windows fix; `prepare-data` (§6b). Committed 2026-11-08 on branch `refactor/readable-main-pipeline` in two commits: `a9b9a5d` (the earlier K1-filter edits and the team handover) and the commit after it (all work in this log). Not pushed |
| 1 | Views ablation; multi-click training filter | **Done** (§6, §7) |
| 2 | Relevance model: full-April popularity in LightGBM; position debias (LightGBM `position`); `lambdarank` vs `rank_xendcg`; RankNet MLP check; optional GRU4Rec feature; lower learning rate; scale to 50k and all queries | **Steps 1–2 done** (§6a: popularity, position, xendcg, learning rate; §6b: 50k queries). Left: all-data training (needs less memory per row or a larger machine); RankNet MLP check; optional GRU4Rec feature |
| 3 | Filter: retrain two-tower on all data; compare with popularity filter using the Phase 2 ranker; choose K1 and whether never-shown products are allowed | Pending |
| 4 | Diversity: fixed-alpha curve vs adaptive policies (session variety, query position, category switch); declared rule: most ILD within 5% NDCG loss | Pending |
| 5 | Choose on selection, freeze, evaluate confirmation; session-bootstrap intervals by segment | Pending |

### 8.0 Course requirements checklist (assessed 2026-11-08)

| Requirement | Already in place | Gap | Size |
|---|---|---|---|
| Well-defined user problem and objective | Research question (§3 of the handover), product brief (§8.1), per-page metrics | One consolidated statement for the report. The original "adaptive diversity" hypothesis was not supported (§3, §2), so the objective should read as relevance with controlled variety, with adaptation as a tested-and-rejected idea | Writing |
| At least one classical baseline (CF, MF) | **Done 2026-11-08 (§6c): item-kNN on session co-clicks**, plus popularity sort and the site's order | Originally: no collaborative-filtering model under the current protocol. The older pipeline's session co-occurrence graph and GRU4Rec/SASRec used persistent-user features and another protocol. Needed: item-kNN on session co-clicks (also the "You may also like" model). MF needs a learned vector per user or session, which new anonymous sessions do not have; its item factors can still be used item-to-item | Build |
| At least one neural or advanced model | Two-tower (neural, rank-weighted loss); LightGBM LambdaMART/xendcg (advanced) | Re-evaluate the two-tower under the final data choices (clicks only, 50k) in the same table as the other models | Small |
| Rigorous offline evaluation (NDCG, Recall@K, MRR) | NDCG@10, Recall@10, **MRR@10 (added 2026-11-08, §6c)**, recall at K1, ILD, session-bootstrap intervals, segments; one table of site order, popularity, item-kNN and LightGBM on identical queries (§6c) | **Everything is validation-only:** selection and confirmation (Phase 5) are needed for a held-out result. No single table compares every model under one protocol | Build |
| Error analysis of failure modes | Scattered findings: cold queries tie popularity, text search fails, never-shown products unjudgeable, token ILD ceiling, hyperparameter sensitivity, category ID dominates splits | A systematic analysis of the final model: where it loses to popularity (categories, deep-ranked clicks, unpopular or new products, session length), with example lists | Build + writing |
| Ethical analysis | Anonymous design (no user IDs); notes on position bias and click-only selection effect | Not written. Topics with measurable parts: popularity bias and exposure concentration (coverage, Gini of recommended products), position bias in logged clicks, privacy of the website's interaction logging, stakeholder impact (long-tail sellers, shoppers), AI-generated demo names | Measurement + writing |

Proposed order: (1) item-kNN baseline and MRR (**done**, §6c), (2) one comparison table on held-out
data (Phase 5), (3) error analysis, (4) ethics with exposure measurements,
(5) the written problem statement.

### 8.1 Product surfaces (user brief, 2026-11-08; open points 1–4 resolved)

The user's brief for the website, and the proposed data/model mapping:

| Page | User brief | Candidate pool | Cold start (no interaction yet) | After interaction | Diversity | Offline evaluation in Diginetica |
|---|---|---|---|---|---|---|
| Home page | No hard filter; cold start by collaborative filtering; then 6 relevance + 4; needs two-tower | Whole catalog | Popularity / trending (proposed: CF needs at least one interaction or user ID) | Two-tower retrieval → LightGBM-style scoring → 6 + 4 | 6 + 4 | No home-page impressions exist; proxy: predict the session's next clicked/viewed products from its history |
| Category page | Hard filter; cold start by CF; then rank by relevance; no two-tower | Products of the category | Popularity within the category | LightGBM ranker (Phase 2) | None (relevance only) | Directly the dataset's category-page queries (95% of queries): click NDCG@10 |
| Item page, "You may also like" | Diversity based on relevance from other pages; 6 + 4 | Products similar to / co-clicked with the viewed item | Item-to-item co-clicks (the item itself is the signal) | Same, personalised by session history | 6 + 4 | Clicks only: did the list contain products the shopper clicked later in the session; unclicked products are not penalised |

Points raised for the user (open):

1. *Resolved 2026-11-08: cold start uses popularity.* "Collaborative filtering"
   cannot work with zero interactions for an anonymous visitor; at cold start
   only aggregate signals exist (popularity, trending, category popularity).
   Item-to-item CF ("shoppers who viewed X also viewed Y") needs one
   interaction, which the item page always has.
2. *Resolved 2026-11-08: the website logs interactions and the home page is
   evaluated online.* The home page has no logged impressions in Diginetica, so
   offline it could only be judged by a next-item proxy.
3. *Resolved 2026-11-08: AI-generated product names for the demo, display
   only.* The catalog is anonymised (IDs, category IDs, price bands, name token
   IDs; no names or images).
4. *Resolved 2026-11-08: "You may also like" uses clicks only, with no penalty
   for products that were not clicked* (replaces the earlier proposal to use
   views). The item-to-item signal is therefore co-clicks within a session;
   48.5% of training-pool sessions have two or more clicks (§2).

---

## 9. Known issues and caveats

- **Never-shown products cannot be judged offline.** NDCG treats them as
  misses; report the shown-only result next to any full-category result.
- **Purchases are too rare for decisions:** 10 purchase-bearing validation
  queries; one query moves purchase recall by 0.1.
- **Text searches** are 5% of queries (25 in validation); the two-tower and
  popularity both fail on them.
- **Training data contains only sessions with a click** (§2), a selection
  effect to state in the report.
- **Two-tower popularity and history still use views** (raw clicks + views;
  session history of clicks and views). Align them with the clicks-only
  decision when the tower is retrained in Phase 3. Session variety H in the
  adaptive policies also uses clicks and views.
- Two older pipeline tests fail on Windows at a long temporary path (§1).
- A small reranking ILD (combined 0.46–0.49) reflects the token ceiling and the
  near-zero category spread of logged lists (§2), not a broken metric.

---

## 10. Commands and files

```powershell
uv run main.py alpha-sweep --trained-run results/learning_curve/queries_10000
uv run main.py two-tower --trained-run results/learning_curve/queries_10000
uv run main.py filter-pipeline --trained-run results/learning_curve/queries_10000 --two-tower-run results/two_tower/RUN
uv run main.py event-ablation --trained-run results/learning_curve/queries_10000
uv run main.py ranker-phase2 --trained-run results/learning_curve/queries_10000
uv run main.py prepare-data --queries 50000 --curve-root results/learning_curve_v2
uv run main.py ranker-phase2 --trained-run results/learning_curve_v2/queries_50000 --phase2-config config/ranker_phase2_50k.json
uv run main.py ranker-phase2 --trained-run results/learning_curve_v2/queries_50000 --phase2-config config/ranker_lead_50k.json
uv run main.py item-knn --trained-run results/learning_curve_v2/queries_50000
uv run python scripts/analysis/session_lengths.py data/raw OUTPUT_FOLDER
uv run python scripts/analysis/click_view.py data/raw
```

| Output file | Contents |
|---|---|
| `alpha_sweep/…/summary.csv` | Mean NDCG/ILD/alpha per model, policy and history group |
| `alpha_sweep/…/gap_vs_fixed_curve.csv` | Adaptive minus fixed-curve NDCG at equal ILD, with CI |
| `alpha_sweep/…/history_diagnostics.csv`, `adaptive_alpha_spread.csv` | Fallback shares, H and alpha distributions |
| `two_tower/…/retrieval_summary.csv` | Recall at each K1 by pool, method and segment |
| `two_tower/…/top10_summary.csv` | NDCG@10, ILD@10 and never-shown share of each method's top 10 |
| `two_tower/…/item_embeddings.npy`, `item_ids.npy` | Product vectors for use as a filter |
| `filter_pipeline/…/summary.csv`, `difference_vs_logged.csv` | End-to-end NDCG/ILD and differences with CI |
| `event_ablation/…/summary.csv`, `difference_vs_current.csv` | Phase 1 results by segment, with CI |
| `event_ablation/…/feature_gain.csv`, `audit.json` | Feature importance and per-fit settings |
| `ranker_phase2/…/summary.csv`, `differences.csv` | Phase 2 NDCG by segment, variant and model; differences vs `base` and vs the popularity sort |
| `ranker_phase2/…/validation_per_query.parquet` | Per-query NDCG (and MRR from §6c on) for every variant and reference, with query type and history flag |
| `item_knn/…/summary.csv`, `differences.csv` | NDCG/Recall/MRR of every method by segment; differences vs popularity sort and vs item-kNN |
| `item_knn/…/knn_settings.json` | Basket, similarity and coverage statistics |
