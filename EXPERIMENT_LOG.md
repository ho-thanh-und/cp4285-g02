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
   - [6d. Experiment H (Phase 5, category pages): held-out selection set](#6d-experiment-h-phase-5-category-pages-held-out-selection-set)
   - [6e. Experiment I (Phase 4, item page): "You may also like" with 6 + 4](#6e-experiment-i-phase-4-item-page-you-may-also-like-with-6--4)
   - [6f. Experiment J (Phase 5, item page): held-out selection week](#6f-experiment-j-phase-5-item-page-held-out-selection-week)
   - [6g. Experiment K (Phase 4, home page): session two-tower vs item-kNN](#6g-experiment-k-phase-4-home-page-session-two-tower-vs-item-knn)
   - [6h. Experiment L (Phase 5, home page): held-out selection week](#6h-experiment-l-phase-5-home-page-held-out-selection-week)
   - [6i. Error analysis of the frozen models (selection data)](#6i-error-analysis-of-the-frozen-models-selection-data)
   - [6j. Experiment M (home page): SASRec and GRU4Rec](#6j-experiment-m-home-page-sasrec-and-gru4rec)
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
- Test suite on Windows: **192 of 194 pass** (as of §6j, Transformers4Rec bridge; 141 of 143 when
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
| `heldout.py` | `heldout` | `config/heldout.json` | `tests/test_heldout.py` |
| `ymal.py` | `ymal` | `config/ymal.json` (validation week), `config/ymal_selection.json` (frozen policy, selection week) | `tests/test_ymal.py` |
| `error_analysis.py` | `error-analysis` | `config/error_analysis.json` | `tests/test_error_analysis.py` |
| `sequence_models.py` | `sequence-models` | `config/sequence_models.json` (April), `config/sequence_models_feb_apr.json` (February–April) | `tests/test_sequence_models.py` |
| `external_models.py` | `export-sequences`, `score-external` | `config/t4rec.json` | `tests/test_external_models.py` |
| `scripts/t4rec/` (`train_t4rec.py`, `requirements-t4rec.txt`, `t4rec.slurm`, `T4REC_GUIDE.md`) | Transformers4Rec in its own environment | `config/t4rec.json` | helpers in `tests/test_external_models.py` |
| `scripts/cluster/` (`make_sequence_grid.py`, `check_inputs.py`, `collect_sequence_grid.py`, `sequence_grid.slurm`, `sequence_grid.pbs`, `CLUSTER_GUIDE.md`) | cluster job array | generated `config/sequence_grid/` | `tests/test_cluster_scripts.py` |
| `home_page.py` | `home-page` | `config/home_page.json` (validation week), `config/home_page_selection.json` (frozen policy and model, selection week) | `tests/test_home_page.py` |

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

**Conclusion (6c).** The learned ranker beats both the non-personalised baseline and
the classical CF baseline on all three metrics, and the advantage sits in
sessions with history. Co-click CF alone is a weak personaliser here (most
sessions are short and one-category), but it is the natural model for the item
page.

---

## 6d. Experiment H (Phase 5, category pages): held-out selection set

`results/heldout/20261108T082748Z_selection/` — `uv run main.py heldout` (`config/heldout.json`)

**First held-out result.** These 1,000 selection queries (May 9–14, 614
sessions, 996 with a click, 962 category pages, 152,880 candidates) were never
used for any choice before this run.

**Procedure** (`heldout.py`), each step completed before the next:

1. **Freeze** (`frozen_choices.json`), written before any selection data was
   read: clicks-only features, all sessions, no text searches in training;
   lead ranker `popularity_xendcg` trained on 50k queries with each model's
   validation-selected setting (`click_session`: 31 leaves, min rows 100, 47
   trees; `purchase_session`: 15 leaves, min rows 100, 74 trees); item-kNN as
   in §6c; metrics NDCG@10, Recall@10, MRR@10; declared rule: choose the
   LightGBM model with the higher selection category-page click NDCG@10, ties to
   `click_session`. Recorded code state: commit `75367ca` plus uncommitted
   changes (the held-out module itself, committed afterwards).
2. **Retrain** the frozen models with that setting and tree count. Check
   passed: on validation they reproduce the Phase 2 per-query scores exactly.
3. **Rebuild the selection split** with the pipeline's sampling and database
   code. Check passed: query IDs equal the pilot's reserved selection IDs.
4. **Features and labels** as for validation; similarity and popularity from
   the April training window only.
5. **Confirmation stays sealed**: the code refuses to open it until the home
   page and "You may also like" choices are frozen.

**Results on selection, category pages (962 queries):**

| Method | NDCG@10 | Recall@10 | MRR@10 | NDCG with history (356) | NDCG no history (606) |
|---|---:|---:|---:|---:|---:|
| Site's original order | 0.065 | 0.118 | 0.056 | 0.063 | 0.067 |
| Popularity sort | 0.174 | 0.283 | 0.161 | 0.160 | 0.183 |
| Item-kNN (co-click CF) | 0.176 | 0.291 | 0.159 | 0.163 | 0.183 |
| LightGBM `click_session` | 0.203 | 0.322 | **0.186** | 0.223 | **0.192** |
| **LightGBM `purchase_session`** (chosen) | **0.204** | **0.326** | 0.185 | **0.229** | 0.189 |

Differences (session bootstrap, 95% intervals):

- **LightGBM vs popularity sort:** `purchase_session` NDCG +0.029 (+0.017,
  +0.043), recall +0.043 (+0.025, +0.062), MRR +0.024 (+0.011, +0.037). With
  history NDCG +0.069 (+0.040, +0.101). `click_session` is the same within
  0.005.
- **LightGBM vs item-kNN:** NDCG +0.028 (+0.014, +0.041), recall +0.035
  (+0.015, +0.055), MRR +0.027 (+0.013, +0.041).
- **Item-kNN vs popularity sort:** NDCG +0.001 (−0.008, +0.011); no measurable
  gain.
- **Without history** LightGBM is now slightly ahead of the popularity sort
  (`click_session` NDCG +0.009, +0.001 to +0.016), where validation showed a
  tie.

**Choice by the declared rule:** `purchase_session` 0.20346 vs `click_session`
0.20336 category-page NDCG, a difference of 0.0001 — effectively a tie that the
rule resolves to `purchase_session` (`selection_choice.json`).

**Reading.** Every validation conclusion held on untouched data: the learned
ranker beats both baselines on all three metrics with intervals above zero,
item-kNN adds nothing measurable over popularity, and the gain sits in sessions
with history. Absolute scores are lower than on validation (0.20 vs 0.23),
which reflects a different week of traffic, so only same-query comparisons
should be read.

---

## 6e. Experiment I (Phase 4, item page): "You may also like" with 6 + 4

`results/ymal/20261108T090850Z_validation/` — `uv run main.py ymal` (`config/ymal.json`)

**Design** (`ymal.py`), following the product rules (clicks only, no penalty
for unclicked products):

- **Item pages:** every product clicked in an eligible session (no hidden-label
  query) of the validation week, May 1–7, from the raw files: 18,645 sessions,
  49,845 clicks. A page is evaluable when the shopper clicked a different
  product later in the session (26,419 pages); 20,000 were sampled with a fixed
  seed. Later clicks are the hits; pages without one are skipped. The code
  accepts only the validation and selection weeks.
- **Pool:** the item's top co-click neighbours from April (item-kNN cosine,
  §6c), K1 = 20 or 50, filled with the item category's most popular products
  when there are too few. 17,997 pages (90%) have at least one neighbour.
- **Methods:** category popularity (cold baseline), relevance only, fixed
  alpha 0.8, and 6 + 4 (six most relevant, then four mutually diverse from the
  rest of the pool), each with K1 = 20 and 50.
- **Metrics:** Recall@10, hit rate, MRR@10, NDCG@10 against later clicks;
  combined/token/category ILD@10, distinct categories, share outside the item's
  category, and tail-to-head distance (positions 7–10 vs 1–6).

**Results** (20,000 pages; intervals from 2,000 session-bootstrap replicates):

| Method | Recall@10 | Hit rate | MRR@10 | NDCG@10 | ILD@10 | Category ILD | Categories in list | Tail-to-head |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Category popularity | 0.142 | 0.298 | 0.118 | 0.098 | 0.460 | 0.000 | 1.0 | 0.459 |
| **Relevance only** (item-kNN) | **0.148** | **0.316** | **0.136** | **0.107** | 0.617 | 0.299 | 2.4 | 0.628 |
| Fixed alpha 0.8, K1 = 20 | 0.140 | 0.306 | 0.133 | 0.103 | 0.650 | 0.355 | 2.8 | 0.656 |
| Fixed alpha 0.8, K1 = 50 | 0.139 | 0.304 | 0.132 | 0.102 | 0.659 | 0.373 | 2.9 | 0.666 |
| **6 + 4, K1 = 20** | 0.134 | 0.294 | 0.134 | 0.101 | 0.685 | 0.416 | 3.2 | 0.714 |
| **6 + 4, K1 = 50** | 0.128 | 0.284 | 0.133 | 0.098 | **0.727** | **0.497** | **3.7** | **0.775** |

(Relevance only is identical for K1 = 20 and 50: the pool size only matters to
the diversity policies.)

- **Item-kNN beats category popularity** on every relevance metric: recall
  +0.006 (+0.001, +0.012), hit +0.017, MRR +0.018, NDCG +0.010; all intervals
  above zero. Its lists already span 2.4 categories on average, against 1.0.
- **Cost and gain of diversity against relevance only** (all intervals exclude
  zero):

  | Policy | Recall | Hit rate | MRR | NDCG (relative) | ILD | Category ILD | Tail-to-head |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Fixed 0.8, K1 = 20 | −0.008 | −0.010 | −0.003 | −0.004 (−4.0%) | +0.033 | +0.056 | +0.028 |
  | Fixed 0.8, K1 = 50 | −0.009 | −0.012 | −0.004 | −0.005 (−5.0%) | +0.042 | +0.073 | +0.038 |
  | 6 + 4, K1 = 20 | −0.014 | −0.022 | −0.002 | −0.006 (−5.9%) | +0.068 | +0.116 | +0.086 |
  | 6 + 4, K1 = 50 | −0.020 | −0.032 | −0.004 | −0.009 (−8.3%) | +0.110 | +0.197 | +0.147 |

- **6 + 4 keeps the top of the list:** MRR drops by only 0.002–0.004 because
  positions 1–6 are unchanged; its cost sits in hits at positions 7–10. Fixed
  alpha changes the whole list and gives about half the diversity for about
  half the recall cost.
- **The diverse tail really differs from the head:** tail-to-head distance
  rises from 0.63 to 0.71 (K1 = 20) or 0.77 (K1 = 50).
- **Unlike category pages, ILD moves here**, because recommendations can cross
  categories (category ILD 0.30–0.50 against 0.02 on category pages).
- **Caveat:** later clicks can only land on products the shopper was shown on
  the site, and cross-category products were shown less often. The relevance
  cost of diversity is therefore likely overstated offline.

**Decision rule check.** The plan's rule (§8, Phase 4: most ILD within 5% NDCG
loss) admits fixed alpha 0.8 with K1 = 20 (−4.0%) and rejects 6 + 4 (−5.9% and
−8.3%); the product brief asked for 6 + 4 on this page.

**Decision (user, 2026-11-08): fixed alpha 0.8 with K1 = 20**, following the
declared rule. K1 = 50 was not chosen because it lands at −5.0%. Speed was
checked and does not separate the policies: on a 20-product pool fixed alpha
takes 0.08 ms and 6 + 4 0.06 ms, both dwarfed by the 4 ms needed for product
distances, which both policies use (and which can be precomputed per product
for the website).

---

## 6f. Experiment J (Phase 5, item page): held-out selection week

`results/ymal/20261108T092942Z_selection/` — `uv run main.py ymal --ymal-config config/ymal_selection.json`

**Procedure.** `frozen_choices.json` (policy `fixed_0.8_20`, the user's reason,
similarity settings, metrics, code state: commit `acd20f7` plus the uncommitted
freeze code) was written before the week was read. The code refuses the
selection week without a frozen policy. Week May 9–14: 19,270 eligible sessions,
55,157 clicks, 29,951 evaluable item pages, 20,000 sampled with the fixed seed;
17,625 (88%) have co-click neighbours. All policies are still scored on the
same pages for comparison.

| Method | Recall@10 | Hit rate | MRR@10 | NDCG@10 | ILD@10 | Category ILD | Categories in list | Tail-to-head |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Category popularity | 0.139 | 0.291 | 0.112 | 0.091 | 0.458 | 0.000 | 1.0 | 0.458 |
| Relevance only (item-kNN) | 0.145 | 0.304 | 0.128 | 0.101 | 0.620 | 0.304 | 2.5 | 0.631 |
| **Fixed alpha 0.8, K1 = 20 (frozen)** | 0.139 | 0.295 | 0.125 | 0.097 | 0.652 | 0.360 | 2.8 | 0.659 |
| 6 + 4, K1 = 20 | 0.132 | 0.283 | 0.126 | 0.095 | 0.687 | 0.421 | 3.2 | 0.716 |
| 6 + 4, K1 = 50 | 0.126 | 0.272 | 0.125 | 0.093 | 0.727 | 0.497 | 3.7 | 0.774 |

- **Frozen policy vs relevance only:** NDCG −0.003 (−3.3%, interval −0.0037
  to −0.0029), recall −0.006, MRR −0.003; ILD +0.033 (+0.032, +0.033), category
  ILD +0.056, tail-to-head +0.029. Inside the 5% rule, as on validation (−4.0%).
- **Frozen policy vs category popularity (the cold baseline):** NDCG +0.007
  (+0.003, +0.011), MRR +0.013 (+0.008, +0.019); recall and hit rate level
  (intervals around zero); ILD +0.194 and 2.8 categories instead of 1.0.
- **Item-kNN relevance vs category popularity** again beats it on every
  relevance metric (NDCG +0.010, +0.006 to +0.014).
- 6 + 4 with K1 = 20 costs −5.5% NDCG for +0.068 ILD, repeating the validation
  trade-off.

**Reading.** The item-page results held on the untouched week: item-kNN is the
right base for "You may also like", and the frozen fixed-alpha policy stays
within the declared relevance budget while adding measurable cross-category
variety.

---

## 6g. Experiment K (Phase 4, home page): session two-tower vs item-kNN

`results/home_page/20261108T095146Z_validation/` — `uv run main.py home-page` (`config/home_page.json`)

**Design** (`home_page.py`):

- **Session two-tower** trained on every eligible April session in the raw
  files: each prefix of a session's distinct clicks (last 20) predicts the next
  clicked product (128,825 examples). Session side: mean and last of the
  clicked products' vectors plus session length. Product side: name tokens,
  category, price band, leak-safe daily click and purchase rates. Sampled
  softmax against 10 uniform and 10 popularity-weighted random products; epoch
  chosen on a fixed 3,000-case subset of the validation week (best: epoch 5 of
  5; check recall@10 0.136, 0.123, 0.134, 0.135, 0.138). Model saved for reuse.
- **Cases:** in the validation week's 18,645 eligible sessions, every point
  after at least one click with a later click on a different product (26,419;
  20,000 sampled). 10 products from the whole catalog, excluding products
  already clicked; hits are the later clicks; no penalty otherwise. Cold start
  (no clicks) is popularity and can only be measured online.
- **Methods:** popularity, trending (last 7 days of April), item-kNN on the
  session's clicks (sum of co-click similarities), the two-tower; then relevance
  only, fixed alpha 0.8 and 6 + 4 over a 20-product pool of the two-tower and of
  item-kNN.

**Results** (20,000 cases; intervals from 2,000 session-bootstrap replicates):

| Method | Recall@10 | Hit rate | MRR@10 | NDCG@10 | ILD@10 | Category ILD | Categories in list | Outside last click's category |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.004 | 0.011 | 0.006 | 0.004 | 0.663 | 0.381 | 3.0 | 94% |
| Trending | 0.006 | 0.015 | 0.003 | 0.003 | 0.656 | 0.377 | 3.0 | 94% |
| **Item-kNN** | **0.157** | **0.329** | **0.149** | **0.121** | 0.628 | 0.318 | 2.5 | 31% |
| Two-tower | 0.137 | 0.302 | 0.123 | 0.100 | 0.441 | 0.012 | 1.0 | 1% |
| Two-tower + fixed 0.8 | 0.137 | 0.302 | 0.123 | 0.099 | 0.444 | 0.014 | 1.1 | 1% |
| Two-tower + 6 + 4 | 0.133 | 0.300 | 0.123 | 0.098 | 0.463 | 0.024 | 1.1 | 2% |
| **Item-kNN + fixed 0.8** | 0.151 | 0.321 | 0.147 | 0.117 | 0.667 | 0.388 | 2.9 | 35% |
| Item-kNN + 6 + 4 | 0.141 | 0.309 | 0.147 | 0.114 | 0.712 | 0.467 | 3.4 | 39% |

- **Catalog-wide popularity fails** (recall 0.004): without a category, the
  most popular products overall are rarely what a shopper clicks next.
  Personalisation after the first click is essential on this page.
- **Item-kNN beats the two-tower:** recall +0.020 (+0.013, +0.025), MRR +0.026
  (+0.018, +0.035), NDCG +0.021 (+0.015, +0.028). After one click the two are
  level on recall (0.174 each) with item-kNN ahead on NDCG (0.118 vs 0.109);
  after two or more clicks item-kNN is clearly ahead (NDCG 0.123 vs 0.095).
- **The two-tower is almost single-category:** 99% of its products share the
  last clicked product's category (category ILD 0.012). Its 20-product pool
  therefore leaves fixed alpha and 6 + 4 almost nothing to diversify (ILD +0.003
  and +0.022).
- **Diversity on item-kNN** (against item-kNN relevance only): fixed 0.8 costs
  NDCG −0.004 (−3.0%) for ILD +0.039 and 2.9 categories; 6 + 4 costs −5.9% for
  ILD +0.084 and 3.4 categories, the same trade-off as on the item page.
- **Rule check** (most ILD within 5% NDCG loss, applied to the best base): item-
  kNN + fixed alpha 0.8 passes (−3.0%); 6 + 4 does not (−5.9%).

**Reading.** On this proxy the classical co-click filter is both more accurate
and more diverse than the neural session model, which learned to stay inside
the last category. Possible two-tower improvements (more epochs, larger
embeddings, a category-mixing negative sampler) were not tried.

**Decision (user, 2026-11-08): item-kNN with fixed alpha 0.8 over a 20-product
pool** for the home page after the first click (popularity before it). This
replaces the brief's two-tower, following the data and the declared 5% rule.
The session two-tower stays in the study as the evaluated neural model.
Frozen in `config/home_page_selection.json` (frozen run = this folder).

---

## 6h. Experiment L (Phase 5, home page): held-out selection week

`results/home_page/20261108T102455Z_selection/` — `uv run main.py home-page --home-config config/home_page_selection.json`

**Procedure.** `frozen_choices.json` (policy `fixed_0.8_item_knn`, the user's
reason, saved model path; code: commit `56f562a` plus the uncommitted home-page
code) was written before the week was read. The saved session two-tower was
reused, not retrained. Week May 9–14: 19,270 eligible sessions, 29,951
evaluable cases, 20,000 sampled.

| Method | Recall@10 | Hit rate | MRR@10 | NDCG@10 | ILD@10 | Category ILD | Categories in list |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.001 | 0.005 | 0.001 | 0.001 | 0.662 | 0.379 | 3.0 |
| Trending | 0.002 | 0.009 | 0.002 | 0.001 | 0.656 | 0.377 | 3.0 |
| Item-kNN | 0.151 | 0.309 | 0.133 | 0.105 | 0.640 | 0.343 | 2.6 |
| Two-tower | 0.138 | 0.293 | 0.116 | 0.092 | 0.442 | 0.016 | 1.1 |
| **Item-kNN + fixed 0.8 (frozen)** | 0.146 | 0.302 | 0.130 | 0.102 | **0.680** | **0.414** | **3.1** |
| Item-kNN + 6 + 4 | 0.136 | 0.288 | 0.130 | 0.099 | 0.725 | 0.493 | 3.6 |

- **Frozen policy vs item-kNN relevance only:** NDCG −0.003 (−3.1%), ILD
  +0.040; inside the 5% rule, as on validation (−3.0%, +0.039).
- **Frozen policy vs the two-tower:** NDCG +0.010 (+0.005, +0.014), MRR
  +0.014 (+0.007, +0.020), recall +0.008 (+0.002, +0.014), ILD +0.238.
- **Item-kNN vs the two-tower:** NDCG +0.013 (+0.008, +0.017); the validation
  ranking held.
- Catalog-wide popularity again fails once a shopper has clicked (NDCG 0.001).

**Reading.** All three pages are now frozen and confirmed on the untouched
selection week. The confirmation week can be opened once, for all three.

---

## 6i. Error analysis of the frozen models (selection data)

`results/error_analysis/20261108T105517Z/` — `uv run main.py error-analysis` (`config/error_analysis.json`)

No new models. Category pages: the frozen LightGBM `purchase_session` against
the popularity sort, per query. Item page and home page: the selection cases are
rebuilt exactly (checked against the saved results) and each case gets context:
whether any later click was in the frozen 20-product pool, the category of the
later clicks, co-click neighbours, and April popularity. No bug or data problem
was found, so no model is changed before the confirmation week.

### Category pages (962 queries)

| Breakdown | Cases | LightGBM NDCG | Popularity NDCG | Difference | Win / loss |
|---|---:|---:|---:|---:|---|
| No session history | 606 | 0.189 | 0.183 | +0.006 | 11% / 11% |
| With session history | 356 | 0.229 | 0.160 | **+0.069** | 24% / 13% |
| 1st query in session | 593 | 0.186 | 0.180 | +0.006 | 11% / 11% |
| 3rd+ query in session | 190 | 0.272 | 0.176 | **+0.096** | 32% / 14% |
| First click at site rank 1–10 | 152 | 0.356 | 0.359 | −0.002 | 15% / 15% |
| First click at site rank 101+ | 295 | 0.115 | 0.081 | +0.034 | 13% / 8% |
| A clicked product had no April clicks | 150 | 0.067 | 0.053 | +0.014 | 10% / 6% |
| All clicked products had April clicks | 812 | 0.229 | 0.197 | +0.032 | 17% / 13% |

Failure modes:

1. **Most queries are missed by both methods:** in 58% neither LightGBM nor
   the popularity sort puts a clicked product in the top 10. Shoppers click
   deep (31% of first clicks are below the site's rank 100) among ~150
   candidates, so a top 10 often cannot reach them.
2. **No gain on the first query of a session** (+0.006): without history the
   ranker has little beyond popularity. The gain grows with history (3rd+
   query: +0.096).
3. **Products with no April clicks** (16% of queries click one): both methods
   rank them low (NDCG 0.07 vs 0.23); popularity-driven features cannot help.
4. **Over-personalisation:** the worst losses are queries where the clicked
   product was the list's most popular one and session features pushed it down
   (e.g. NDCG 1.00 → 0.00 in two queries with history).
5. **Category spread:** among categories with ≥ 15 queries, differences range
   from −0.001 to +0.124; some categories gain nothing.

### Item page, "You may also like" (20,000 pages)

| Breakdown | Pages | Frozen NDCG | Category-popularity NDCG | Frozen hit | Popularity hit |
|---|---:|---:|---:|---:|---:|
| A later click is in the 20-product pool | 7,325 (37%) | 0.241 | 0.150 | 0.715 | 0.473 |
| No later click in the pool | 12,675 (63%) | 0.014 | 0.057 | 0.052 | 0.185 |
| No co-click neighbours | 2,375 (12%) | 0.064 | 0.064 | 0.212 | 0.213 |
| Later clicks all in other categories | 2,975 (15%) | 0.001 | 0.000 | | |
| Item with 100+ April clicks | 1,342 | 0.158 | 0.099 | | |
| Item with 1–9 April clicks | 6,450 | 0.079 | 0.081 | | |

Failure modes:

1. **Candidate recall, not reranking, is the bottleneck:** only 37% of pages
   have any later click among the item's 20 co-click neighbours (53% among all
   its neighbours). When one is there, the frozen list hits 72% of the time;
   when none is, it hits 5% while category popularity still hits 19%.
2. **Items without co-click neighbours** (12%) fall back to category popularity.
3. **Category switches are unpredictable:** on 15% of pages every later click
   is in another category, and every method scores ~0.
4. **Rarely clicked items** (fewer than 10 April clicks, 43% of pages) gain
   nothing over category popularity; the gain sits in popular items (+0.059).

### Home page (20,000 cases)

| Breakdown | Cases | Frozen NDCG | Two-tower NDCG | Difference |
|---|---:|---:|---:|---:|
| A later click is in the 20-product pool | 8,398 (42%) | 0.242 | 0.147 | +0.095 |
| No later click in the pool | 11,602 (58%) | 0.000 | 0.052 | −0.052 |
| 1 click so far | 6,232 | 0.114 | 0.113 | +0.002 |
| 3–5 clicks so far | 5,374 | 0.104 | 0.082 | +0.022 |
| Last click on a product with no April clicks | 2,150 | 0.045 | 0.071 | **−0.025** |
| Last click on a product with 100+ April clicks | 1,342 | 0.158 | 0.116 | +0.042 |
| Later clicks all in other categories | 2,975 | 0.004 | 0.000 | +0.004 |

Failure modes:

1. **Candidate recall again:** 58% of cases have no later click in the
   item-kNN pool, and the frozen list scores 0 there; the two-tower still
   reaches 0.052 on those cases.
2. **Cold products favour the neural model:** when the last click is on a
   product with no April clicks (11%), co-click data is empty and the content-
   based two-tower (name, category, price) wins (−0.025 for item-kNN).
3. **One click is not enough to beat the two-tower** (tie); item-kNN pulls
   ahead with 2–5 clicks.
4. **Category switches** (15%) are missed by every method.

### Implications (not applied before confirmation)

- The main limit on both item-kNN pages is **which products enter the pool**,
  not how the pool is ordered. A larger or blended pool would raise the
  ceiling.
- A **hybrid** that falls back to the two-tower when the clicked products have
  no co-click neighbours or no April clicks would address the cold-product
  failure on the home page. This is future work: changing a frozen model now
  would require a new selection run.
- On category pages, the ranker helps exactly where session information exists
  and cannot fix products without history; this supports popularity for cold
  start and motivates the cold-product limitation in the report.

---

## 6j. Experiment M (home page): SASRec and GRU4Rec

`uv run main.py sequence-models` (`config/sequence_models.json`, `sequence_models.py`)

**Design.** Both models read a session's clicks in order and predict the next
clicked product. Training: every eligible April session in the raw files,
clicks only, products in order of first click, the last 20; next-product
prediction at every position of the 42,299 sessions with two or more known
clicks (126,406 positions). Each model learns one 64-dimensional vector per
product clicked in April (40,683 products; 3.3% of cases have no known click
and fall back to popularity; 94% of cases have a later click inside this
vocabulary). Sampled softmax against 2,000 shared products per batch (half
uniform, half popularity-weighted, accidental hits masked); Adam, learning
rate 0.001, batch 128 sessions, dropout 0.2. GRU4Rec: one GRU layer. SASRec: 2
causal self-attention blocks, 2 heads, position embeddings. The epoch with the
best recall@10 on the same 3,000-case validation subset as the session
two-tower is kept. Evaluation on exactly the 20,000 cases of the home-page run
(§6g), joined with its saved results for paired differences.

**Run 1, failed to train** (`results/sequence_models/20261108T110911Z_validation/`,
10 epochs). SASRec's loss stayed at 7.6 ≈ ln(2,001), chance level against 2,001
candidates (recall@10 0.001); GRU4Rec was still improving at epoch 10 (check
recall 0.020). Cause: PyTorch's default embedding initialisation (standard
deviation 1) gives huge dot-product scores and saturates the softmax (SASRec's
first-epoch loss was 21.8). Not a model result.

**Run 2** (`results/sequence_models/20261108T111930Z_validation/`): embeddings
initialised with standard deviation 0.02, as in the reference implementations,
and 20 epochs. SASRec's check recall peaked at epoch 6 (0.158) and stayed near
0.147–0.153 afterwards; GRU4Rec was still improving at epoch 20 (0.109).
Training took about 2–4 minutes per model on the laptop CPU.

| Method | Recall@10 | Hit rate | MRR@10 | NDCG@10 | ILD@10 |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.004 | 0.011 | 0.006 | 0.004 | 0.663 |
| Item-kNN | 0.157 | 0.329 | 0.149 | 0.121 | 0.628 |
| Item-kNN + fixed 0.8 (frozen home policy) | 0.151 | 0.321 | 0.147 | 0.117 | 0.667 |
| Session two-tower | 0.137 | 0.302 | 0.123 | 0.100 | 0.441 |
| **SASRec** | **0.169** | **0.352** | **0.160** | **0.129** | 0.555 |
| GRU4Rec (under-trained) | 0.111 | 0.246 | 0.105 | 0.084 | 0.677 |

NDCG@10 by clicks so far:

| Clicks so far | Item-kNN | Two-tower | SASRec | GRU4Rec |
|---|---:|---:|---:|---:|
| 1 | 0.118 | 0.109 | **0.131** | 0.089 |
| 2 | 0.123 | 0.095 | **0.130** | 0.085 |
| 3–5 | 0.110 | 0.083 | **0.110** | 0.066 |
| 6+ | 0.137 | 0.110 | **0.148** | 0.098 |

- **SASRec beats item-kNN:** NDCG +0.008 (+0.004, +0.012), recall +0.012
  (+0.007, +0.017), MRR +0.010 (+0.005, +0.016); and the frozen policy by NDCG
  +0.012 (+0.008, +0.016). It is ahead or level in every clicks-so-far group.
- **But it is less diverse:** ILD 0.555 against 0.628 (item-kNN) and 0.667
  (frozen policy).
- **GRU4Rec** is below item-kNN (NDCG −0.037) but was still improving when
  training stopped; a longer run is needed before judging it.
- **Caveats:** the epoch was chosen on 3,000 of the 20,000 evaluation cases
  (as for the two-tower), so validation numbers are slightly optimistic; the
  selection week is the clean check. Only one hyperparameter setting was run.

**Reading.** Correctly trained, the transformer model is the most accurate
home-page method on validation, while item-kNN with fixed alpha stays the most
diverse.

**Decision (user, 2026-11-08): keep item-kNN + fixed alpha 0.8 on the home
page for now.** SASRec is reported as the most accurate home-page model on
validation. Next step for it: train on more data than April alone and test
whether its advantage grows; any later switch must be frozen on validation
before SASRec is run on the selection week.

**More data and a tuning grid (prepared 2026-11-08, not yet run).** Training
history available before May (eligible sessions, distinct clicks):

| Training window | Sessions | Sessions with ≥ 2 clicks | Training positions | Products |
|---|---:|---:|---:|---:|
| April (runs above) | 93,231 | 42,299 | 128,825 | 40,683 |
| March–April | 159,813 | 72,812 | 223,520 | 52,286 |
| February–April | 207,181 | 94,406 | 291,928 | 58,806 |
| January–April | 209,144 | 94,676 | 292,304 | 58,908 |

January adds almost nothing, so February–April is the full history (2.3× the
April training positions). The sequence models now take a `training_window`
setting, checked to end before 2016-04-30, and
`config/sequence_models_feb_apr.json` uses February–April. One run of both
models would take about 25–30 minutes on the laptop (the April 20-epoch run
took 15.5). At the user's request, a 12-setting tuning grid is packaged for a
compute cluster in `scripts/cluster/` (job array, one setting per task, CPU
only): SASRec embedding 64/128 × 1/2 blocks × dropout 0.2/0.5 (30 epochs) and
GRU4Rec embedding 64/128 × learning rate 0.001/0.003 (60 epochs). Steps are in
`scripts/cluster/CLUSTER_GUIDE.md`; `collect_sequence_grid.py` ranks the
settings by validation NDCG with intervals against item-kNN and the frozen
policy. `.gitattributes` keeps the job scripts on Unix line endings.

**Cluster environment (user choice, 2026-11-08): Miniconda + pip.**
`requirements.txt` pins the versions the experiments ran with (Python 3.11,
numpy 1.26.4, pandas 2.3.3, pyarrow 14.0.2, scipy 1.12.0, LightGBM 4.7.0, CPU
torch 2.5.1). The older `requirements.lock.txt` pins different, untested
versions and should not be used. The job scripts activate the conda
environment `cp4285`.

**Transformers4Rec (requested 2026-11-08; built, not yet run).** Findings from
its PyPI packages: latest release 23.12.0 (January 2024, none since); it requires
Hugging Face `transformers` < 4.31 and `merlin-models`; `merlin-core` 23.8.0
requires pandas < 1.6, dask, dask-cuda, numba and tensorflow-metadata, and
`merlin-dataloader` lists Python 3.8–3.10. It therefore cannot share the main
environment (pandas 2.3, Python 3.11). **User decision: a separate conda
environment** (`cp4285-t4r`, Python 3.10, `scripts/t4rec/requirements-t4rec.txt`).

It connects through files, so no project code runs in the old environment:
`main.py export-sequences` (`external_models.py`) writes the training sessions,
the exact home-page cases (checked against the saved home-page run) and the
product vocabulary; `scripts/t4rec/train_t4rec.py` trains the library's
documented XLNet next-item setup (causal masking, tied embeddings; best epoch by
recall@10 on the same 3,000 check cases as the other neural models) and writes
ranked predictions per case; `main.py score-external` scores them with the same
metrics and compares them with every saved home-page method (and SASRec/GRU4Rec
runs listed in `config/t4rec.json`) with session-bootstrap intervals. The export
and scoring steps are tested; the training script has only its helper functions
tested, because the library cannot be installed on the laptop, so it has a
`--smoke` mode for a first check on the cluster. `scripts/t4rec/t4rec.slurm` runs
all three steps; `scripts/t4rec/T4REC_GUIDE.md` has the instructions.

**GPU jobs (2026-11-08).** At the user's request, two Slurm scripts use the NUS
`gpu` partition header (`--gpus=h100-96:1`): `scripts/cluster/sequence_grid_gpu.sh`
runs the 12 grid settings one after another on one GPU (skipping finished ones),
and `scripts/t4rec/t4rec_gpu.sh` runs the three Transformers4Rec steps. Both stop
early if PyTorch cannot see the GPU. Any GPU type suffices (the models need
well under 2 GB of GPU memory). The sequence models now train on the GPU when
one is visible (`device`: `auto`, `cpu`, `cuda`) and save CPU-loadable weights;
only the CPU path is tested here. The CUDA builds of torch are a one-time
replacement described in the guides. The guides also record the first install
problems on the cluster: Python 3.11 was only available from `conda-forge`, the
home-directory disk quota was exceeded, and a batch job that did not activate
the environment ran pip in the `base` environment (Python 3.13). The first GPU grid job (an
A100 80 GB, MIG slice of 40 GB; `torch 2.5.1+cu124` saw the GPU and the input
check passed) then failed on every setting with `No module named
'session_rerank'`: pip installs only the libraries, not the project in `src/`.
`main.py` and `tests/conftest.py` now add `src/` to the search path themselves
(checked by running `main.py` from a copy with the installed package hidden). The
second GPU job reached training on `cuda` and then failed in SASRec: its
position indices and causal mask were created on the CPU while the model was
on the GPU. Both are now created on the input's device; a test checks that every
tensor the model code creates names its device (it fails on the old code). The
CPU tests cannot run the GPU path itself, so the next cluster run is the check.
The third GPU job ran correctly (`device: cuda`; first setting's check recall
peaked at 0.171 against 0.158 for the April-only SASRec; SASRec with 2 blocks
and dropout 0.5 reached 0.180). Run one after another, each setting took about
14.5 minutes, mostly CPU work (per-epoch check of 3,000 cases, scoring 20,000
cases, re-reading the raw CSV), so the 12 settings needed about 3 hours. With
only one GPU available, `sequence_grid_gpu.sh` now runs four settings at a time
on that GPU (16 CPU cores, 4 threads each), with one log per setting; the grid
itself is unchanged.
The re-submitted job (1097527) ran the older one-at-a-time script: it skipped the
4 finished settings and trained the 4 dim-128 SASRec settings in about an hour.
Check recall@10 so far:
- dropout 0.5 clearly beats 0.2: with 0.2, recall peaks at epoch 3 (0.161-0.162
  at dim 128) and then falls while the loss keeps dropping, i.e. overfitting;
- dim 128 does not beat dim 64 (best: dim 64, 2 blocks, dropout 0.5, at least
  0.180; dim 128, 2 blocks, dropout 0.5: 0.176).
A second GPU became available, so the script gained `ONLY` (run only settings
whose name contains a word): the 4 GRU4Rec settings run there with
`ONLY=gru4rec`, and the old job is cancelled once its last SASRec setting is
done.

## 6k. Exposure and popularity bias (ethics, selection data, 2026-11-09)

**Question.** Which products get shown, and how concentrated is exposure on
popular products, for every evaluated method on all three pages?

**Method** (`uv run main.py exposure`, `config/exposure.json`,
`src/session_rerank/exposure.py`, `tests/test_exposure.py`).
- Rebuild the selection-week top-10 lists of every method:
  - home page: 20,000 cases
  - item page: 20,000 pages
  - category pages: 966 queries, using the saved LightGBM boosters and the
    selection features rebuilt
- Each rebuilt list's NDCG must equal the saved value (category pages: per-query
  click NDCG of every method; candidates equal the saved table). **All three
  pages passed.** To share code, `home_page.case_lists`/`load_frozen_model`
  and `ymal.YmalData`/`page_lists` were factored out of the evaluation loops;
  their tests still pass.
- Popularity is April clicks. The **head** is the most-clicked products holding
  50% of April clicks: 3,761 products (2% of 184,047). The tail is 41,754
  other clicked products. **138,532 products (75%) had no April click.**
- Every slot counts once. Coverage and Gini are measured over the whole catalog
  (home, item) or over the products the site listed (category pages, 36,036).
- Reference: the products shoppers clicked next (home and item pages: later
  clicks; category pages: clicked listed products).

**Results** (`results/exposure/20261108T160349Z/summary.csv`):

| Page | Method | Distinct shown | Coverage | Gini | Head share | Tail | No-click | Head ratio to shoppers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Home | popularity | 12 | 0.000 | 1.000 | 1.000 | 0 | 0 | 3.32 |
| Home | trending | 13 | 0.000 | 1.000 | 1.000 | 0 | 0 | 3.32 |
| Home | item_knn | 27,461 | 0.149 | 0.938 | 0.324 | 0.676 | 0 | 1.07 |
| Home | **fixed_0.8_item_knn (frozen)** | 27,431 | 0.149 | 0.937 | 0.313 | 0.687 | 0 | 1.04 |
| Home | six_four_item_knn | 27,653 | 0.150 | 0.937 | 0.305 | 0.695 | 0 | 1.01 |
| Home | two_tower | 27,670 | 0.150 | 0.949 | 0.391 | 0.484 | 0.125 | 1.30 |
| Home | shopper clicks | 14,095 | 0.077 | 0.964 | 0.301 | 0.557 | 0.141 | 1.00 |
| Item | category_popularity | 8,012 | 0.044 | 0.989 | 0.728 | 0.264 | 0.008 | 2.42 |
| Item | relevance_20 | 32,070 | 0.174 | 0.920 | 0.302 | 0.695 | 0.003 | 1.00 |
| Item | **fixed_0.8_20 (frozen)** | 32,296 | 0.175 | 0.918 | 0.289 | 0.708 | 0.004 | 0.96 |
| Item | six_four_20 | 31,997 | 0.174 | 0.919 | 0.282 | 0.715 | 0.003 | 0.94 |
| Category | site_order | 2,818 | 0.078 | 0.963 | 0.249 | 0.504 | 0.247 | 0.62 |
| Category | popularity_sort | 2,796 | 0.078 | 0.964 | 0.722 | 0.270 | 0.008 | 1.80 |
| Category | item_knn | 3,961 | 0.110 | 0.941 | 0.627 | 0.365 | 0.008 | 1.57 |
| Category | **lightgbm_purchase_session (frozen)** | 3,349 | 0.093 | 0.954 | 0.681 | 0.298 | 0.021 | 1.70 |
| Category | shopper clicks | 1,357 | 0.038 | 0.967 | 0.400 | 0.491 | 0.109 | 1.00 |

**Reading.**
- Cold-start popularity is the most concentrated: 12 products for every home
  visitor.
- The frozen item-kNN lists on the home and item pages match shoppers' own head
  share (ratio 0.96–1.04) and show about twice as many distinct products as
  shoppers clicked.
- Fixed 0.8 and 6 + 4 lower the head share by only 1–2 points. They diversify
  within a list, not across the catalog.
- Item-kNN cannot show products without April clicks, though 14% of shoppers'
  next clicks went to such products. The two-tower is the only method that
  shows them (12.5% of slots), but it leans more on the head (ratio 1.30).
- Category pages trade accuracy for exposure:
  - the site's order is the most long-tail (25% head, 25% no-click) and the
    least accurate (NDCG 0.065);
  - the frozen LightGBM ranker (NDCG 0.203) puts 68% of slots on the 2% head,
    1.7 times shoppers' share, and only 2.1% on no-click products.
- Gini is near 1 for every method because the catalog is large, so coverage and
  head share are the measures to report.

The written section is `ETHICS.md`. It also covers stakeholders, feedback loops,
training-data biases (position, click-only sessions, never-shown products),
privacy of the demo website's logging, transparency, AI-generated demo names
and limitations. Mitigations proposed there (exploration slots on category
pages, capping popularity features, filling item-kNN pools by content) are
not evaluated.

Also: pytest's default temporary folder is blocked in this environment, so
tests now run with `--basetemp` in the session scratchpad. Full suite: **203
passed**, 2 known long-path failures.

---

## 7. Decisions

| Date | Decision | Evidence |
|---|---|---|
| 2026-11-08 | **Category-page ranker adds window popularity features; lead candidate is `rank_xendcg` with those features.** Position correction and learning rate 0.02 are not adopted for now. | §6a: popularity +0.03 (intervals above zero); xendcg is the only variant ahead of the popularity sort in every model/statistic. To be confirmed on selection data (Phase 5). |
| 2026-11-08 | **Home page stays on item-kNN + fixed alpha 0.8 for now**, although SASRec is more accurate on validation (user decision). SASRec to be retested with more training data. | §6j: SASRec NDCG +0.008 over item-kNN but ILD 0.555 vs 0.667 for the frozen policy; one untuned setting. |
| 2026-11-08 | **Home page uses item-kNN on the session's clicks with fixed alpha 0.8 over a 20-product pool** after the first click, popularity before it (user decision, replaces the brief's two-tower); frozen before the selection week is opened. | §6g: item-kNN beats the session two-tower (NDCG +0.021, interval above zero) and is far more diverse; fixed 0.8 costs −3.0% NDCG for +0.039 ILD. |
| 2026-11-08 | **"You may also like" uses item-kNN with fixed alpha 0.8 over a 20-product pool** (user decision, replaces the brief's 6 + 4 for this page); frozen before the selection week is opened. | §6e: within the declared 5% NDCG rule (−4.0%), +0.033 ILD; 6 + 4 gave twice the diversity at −5.9%. Speed is equal. |
| 2026-11-08 | **Category-page choices frozen; `purchase_session` (popularity + xendcg, 50k) is the category-page ranker.** | §6d: frozen before the selection split was read; selection confirms LightGBM over both baselines; the declared rule picks `purchase_session` by 0.0001 NDCG (a practical tie). |
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
| 0 | Windows training fix; commit current work; data-only "prepare all" step | **Done.** Windows fix; `prepare-data` (§6b). Committed 2026-11-08 on branch `refactor/readable-main-pipeline` in two commits: `a9b9a5d` (the earlier K1-filter edits and the team handover) and `75367ca` (all work in this log); both pushed to `origin` the same day |
| 1 | Views ablation; multi-click training filter | **Done** (§6, §7) |
| 2 | Relevance model: full-April popularity in LightGBM; position debias (LightGBM `position`); `lambdarank` vs `rank_xendcg`; RankNet MLP check; optional GRU4Rec feature; lower learning rate; scale to 50k and all queries | **Steps 1–2 done** (§6a: popularity, position, xendcg, learning rate; §6b: 50k queries). Left: all-data training (needs less memory per row or a larger machine); RankNet MLP check; optional GRU4Rec feature |
| 3 | Filter: retrain two-tower on all data; compare with popularity filter using the Phase 2 ranker; choose K1 and whether never-shown products are allowed | Pending |
| 4 | Diversity: fixed-alpha curve vs adaptive policies (session variety, query position, category switch); declared rule: most ILD within 5% NDCG loss | **Item page evaluated on validation** (§6e); fixed alpha 0.8 (K1 = 20) chosen and frozen 2026-11-08. **Home page evaluated on validation** (§6g); item-kNN + fixed alpha 0.8 chosen and frozen 2026-11-08 |
| 5 | Choose on selection, freeze, evaluate confirmation; session-bootstrap intervals by segment | **All three pages frozen and evaluated on selection** (§6d, §6f, §6h). Confirmation week not opened yet |

### 8.0 Course requirements checklist (assessed 2026-11-08)

| Requirement | Already in place | Gap | Size |
|---|---|---|---|
| Well-defined user problem and objective | Research question (§3 of the handover), product brief (§8.1), per-page metrics | One consolidated statement for the report. The original "adaptive diversity" hypothesis was not supported (§3, §2), so the objective should read as relevance with controlled variety, with adaptation as a tested-and-rejected idea | Writing |
| At least one classical baseline (CF, MF) | **Done 2026-11-08 (§6c): item-kNN on session co-clicks**, plus popularity sort and the site's order | Originally: no collaborative-filtering model under the current protocol. The older pipeline's session co-occurrence graph and GRU4Rec/SASRec used persistent-user features and another protocol. Needed: item-kNN on session co-clicks (also the "You may also like" model). MF needs a learned vector per user or session, which new anonymous sessions do not have; its item factors can still be used item-to-item | Build |
| At least one neural or advanced model | Two-tower (neural, rank-weighted loss); LightGBM LambdaMART/xendcg (advanced) | Re-evaluate the two-tower under the final data choices (clicks only, 50k) in the same table as the other models | Small |
| Rigorous offline evaluation (NDCG, Recall@K, MRR) | NDCG@10, Recall@10, **MRR@10 (added 2026-11-08, §6c)**, recall at K1, ILD, session-bootstrap intervals, segments; one table of site order, popularity, item-kNN and LightGBM on identical queries (§6c); **held-out selection table for category pages (§6d, 2026-11-08)** | Confirmation (final held-out test) still sealed until the other pages are frozen. Originally: **everything was validation-only:** selection and confirmation (Phase 5) are needed for a held-out result. No single table compares every model under one protocol | Build |
| Error analysis of failure modes | **Done 2026-11-08 (§6i)** for all three pages on selection data. Earlier, scattered findings: cold queries tie popularity, text search fails, never-shown products unjudgeable, token ILD ceiling, hyperparameter sensitivity, category ID dominates splits | A systematic analysis of the final model: where it loses to popularity (categories, deep-ranked clicks, unpopular or new products, session length), with example lists | Build + writing |
| Ethical analysis | **Done 2026-11-09 (§6k, `ETHICS.md`)**: exposure measured on all three pages; written section drafted. Originally: anonymous design (no user IDs); notes on position bias and click-only selection effect | Originally not written. Topics with measurable parts: popularity bias and exposure concentration (coverage, Gini of recommended products), position bias in logged clicks, privacy of the website's interaction logging, stakeholder impact (long-tail sellers, shoppers), AI-generated demo names | Measurement + writing |

Proposed order: (1) item-kNN baseline and MRR (**done**, §6c), (2) one comparison table on held-out
data (Phase 5; **selection done for all three pages**, §6d/§6f/§6h), (3) error analysis (**done**, §6i), (4) ethics with exposure measurements (**done**, §6k),
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
uv run main.py heldout
uv run main.py ymal
uv run main.py ymal --ymal-config config/ymal_selection.json
uv run main.py home-page
uv run main.py home-page --home-config config/home_page_selection.json
uv run main.py error-analysis
uv run main.py exposure
uv run main.py sequence-models
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
| `heldout/…/frozen_choices.json`, `selection_choice.json` | Choices frozen before opening the split; the declared rule's outcome |
| `heldout/…/summary.csv`, `differences.csv`, `selection_per_query.parquet` | Held-out NDCG/Recall/MRR by method and segment, with intervals |
| `heldout/…/click_session.txt`, `purchase_session.txt`, `selection.sqlite` | Frozen models and the rebuilt selection split |
| `ymal/…/summary.csv`, `differences.csv`, `per_page.parquet` | Item-page relevance and diversity by method; differences vs category popularity and vs relevance only |
| `home_page/…/summary.csv`, `differences.csv`, `per_case.parquet` | Home-page relevance and diversity by method and clicks so far; differences vs popularity and vs the two-tower |
| `home_page/…/session_two_tower.pt`, `training_log.csv` | Saved session two-tower (reused by a selection run) and its epoch log |
| `sequence_models/…/summary.csv`, `differences.csv`, `per_case.parquet`, `sasrec.pt`, `gru4rec.pt`, `vocabulary_rows.npy` | SASRec/GRU4Rec results joined with the home-page methods; saved models and product vocabulary |
| `error_analysis/…/*_breakdown.csv`, `category_pages_by_category.csv`, `category_pages_worst_losses.csv`, `overview.json` | Win/tie/loss and mean NDCG per case property for each page; headline shares |
