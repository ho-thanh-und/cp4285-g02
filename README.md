# Session-Aware Product Re-ranking

The agreed expansion, purchase-aware training, decay experiments, and final-report
requirements are recorded in [TRAINING_PLAN.md](TRAINING_PLAN.md). Read that plan
before continuing implementation or drafting conclusions. The commands below
describe the implemented pilot workflow.


## Step 1 for the anonymous session study

Create the new small sample tables without training models:

```powershell
uv run main.py
# Equivalent explicit action:
uv run main.py prepare
```

Read `src/session_rerank/samples.py` for this preparation workflow. Its settings
are in `config/anonymous_samples.json`. Targets are 2,000 training queries,
500 validation queries, 1,000 selection queries, and 1,000 reserved confirmation
queries. Sampling keeps complete daily sessions, so a target can be exceeded.
Personal user IDs and post-query duration are not exported.

The output folder is `data/anonymous_samples/`. It contains:

| Table | Purpose |
|---|---|
| `products.parquet` | Static product/category/token metadata |
| `sessions.parquet` | Complete selected anonymous daily sessions and split assignment |
| `queries.parquet` | Train/validation/selection queries and history availability |
| `candidates.parquet` | Original ranked candidate lists and observed click labels |
| `clicks.parquet` | Click events, original rank, match status, and click order |
| `events.parquet` | Complete query/click/view/purchase logs for visible sessions |
| `histories.parquet` | Strictly earlier same-session events allowed for each query |
| `confirmation_ids.parquet` | Reserved confirmation metadata, without outcomes |

`audit.json` records data quality and sample counts. `manifest.json` records
settings, source checksums, table schemas, and prepared-file checksums. A second
prepare run verifies and reuses identical outputs; changed settings or source
files require a fresh output directory rather than silently replacing samples.

Generate a review containing all column distributions, a column dictionary,
actual sample rows, table relationships, and a complete SQLite copy:

```powershell
uv run python scripts/review_samples.py
```

Open `results/anonymous-samples-review.html` in a browser. All charts work offline.
The full data is in `results/anonymous-samples.sqlite`; the profiles are also saved
as JSON beside the HTML. Graphs use all prepared rows; displayed example rows
are only samples. Clicks with missing/ambiguous candidate matches retain NULL
ranks and a diagnostic status. Nonclicked candidates are not confirmed dislikes.

Histories currently retain at most 100 earlier events per query; the complete
session remains in `events`. Equal timestamps are grouped, not assigned a known
true order. Session boundaries use source session ID plus source date; this is
not a verified inactivity timeout. Confirmation outcomes are not materialized.

## Step 2: train and compare LightGBM rankers

```powershell
uv run main.py pilot-train
```

This runs Step 1 first and reuses the prepared tables when their checksums match.
The default command and the explicit `prepare` action run Step 1 by itself.

Start in `main.py` and Ctrl+click `train_rankers` to follow the two model fits.
The feature builder in `ranker_features.py` joins the query, candidate, and
product tables and adds earlier activity. Both models learn from the same query
groups and use the same validation grid. The second model also sees earlier
clicks and views in the current session. The original search order is a baseline
and breaks ties between equal model scores.

Each run saves its own folder under `results/anonymous_rankers/` containing both
models, feature gains, tuning results in `run.json`, NDCG@10 and Recall@10 in
`validation_metrics.csv`, and one row per validation candidate in
`validation_scores.parquet`. Queries with no recorded click on a shown product
are counted separately and still get scores. The ranking metrics cover queries
with at least one observed candidate click. Selection and confirmation are
reserved for later stages.

## Define product diversity

```powershell
uv run main.py diversity --ranker-run results/anonymous_rankers/20261104T192951Z-321c6432
```

Supply the folder from the training run you want to review. Start at
`define_product_diversity` in `src/session_rerank/diversity.py`.

Category distance is 0 for matching category IDs and 1 for different IDs.
Product-name token distance uses Jaccard distance, measuring anonymized token
overlap. The initial combined distance gives each component weight 0.5.
These weights are a declared starting definition.

ILD@10 averages the distances of all 45 unordered pairs among ten products.
Shorter lists use their actual pair count. Zero or one product has undefined
ILD. Missing metadata produces unknown pair distances. Each component reports
its known-pair coverage, and the blend requires both components to be known.
Token distance is a lexical proxy, and category IDs supply no hierarchy.

Each review writes `measure.json`, `validation_diversity_by_query.csv`, and
`validation_diversity_summary.csv` under a new `results/anonymous_diversity/`
folder. The review measures the original order and both saved model orderings
on all validation queries, including queries with zero clicks. It records
the dataset and score fingerprints. Category, token, and blended ILD are
reported separately for inspection.

## Complete the remaining study stages

```powershell
uv run main.py run --ranker-run results/anonymous_rankers/20261104T192951Z-321c6432
```

Read the calls in `main.py`. Detailed comments explain each stage.

1. `tune_diversification()` compares relevance-only, quota, fixed-weight and
   session-adaptive methods on validation. The gate uses training-prefix ILD
   quantiles. The sparse-history default comes from fixed-weight validation tuning.
2. `compare_on_selection()` evaluates one validation-nominated setting per family
   on the reserved selection queries.
3. `freeze_and_evaluate()` saves input/code fingerprints and choices before
   reading confirmation candidate lists and click outcomes.
4. `save_study_report()` writes a single offline `report.html`, figures, CSV
   statistics, confidence intervals and recommendation examples.

The declared choice rule maximizes combined ILD with at most 5% relative
NDCG loss and no ILD loss against relevance-only on the same split. Ties prefer
NDCG, then the simpler family, then policy ID. All approaches use the same trained
session-aware LightGBM model and original candidate lists. The 50/50 category/token
blend is retained. Missing distances earn zero diversity reward during reranking
and are reported with coverage during evaluation. History variety uses unique
earlier clicked/viewed products, with fewer than two measurable items treated
as sparse history.

The report includes all frozen approaches and an original-order diagnostic,
even if the selection winner loses quality on confirmation. Metrics use equal
weight per query. Relevance excludes zero-click candidate lists, while diversity
includes all measurable lists. Paired 95% confidence intervals use 2,000 bootstrap
samples of entire daily sessions. Runtime measures and descriptive gain/loss
examples are saved alongside the report.

Each execution has its own `results/anonymous_study/RUN` folder. A completion
manifest records each stage's output hashes. To continue an interrupted execution:

```powershell
uv run main.py resume --study-run results/anonymous_study/RUN
```

Use the original `--config` if a custom sample protocol was used. Resume verifies
the recorded scientific code and input files, reuses completed stages, and keeps
frozen choices intact. A lock prevents concurrent writers in one study folder.
After a process crash, remove its stale `running.lock` only after verifying
that its process has stopped.

## Analysis tools added in November 2026

Results, decisions and the phase plan are in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
Each tool reuses a learning-curve run folder and evaluates on validation only.

```powershell
# Fixed-alpha trade-off curve versus session-adaptive alpha, with a shuffle control.
uv run main.py alpha-sweep --trained-run results/learning_curve/queries_10000
# Rank-weighted two-tower filter (product name, category, price, popularity).
uv run main.py two-tower --trained-run results/learning_curve/queries_10000
# Filter K1 products -> LightGBM scores -> rerank -> NDCG@10 and ILD@10.
uv run main.py filter-pipeline --trained-run results/learning_curve/queries_10000 --two-tower-run results/two_tower/RUN
# Phase 1: retrain rankers with/without product views and on all vs multi-click sessions.
uv run main.py event-ablation --trained-run results/learning_curve/queries_10000
# Phase 2: category-page ranker variants (window popularity, position correction, loss, learning rate).
uv run main.py ranker-phase2 --trained-run results/learning_curve/queries_10000
# Prepare a larger size's data without fitting models, then run Phase 2 on it.
uv run main.py prepare-data --queries 50000 --curve-root results/learning_curve_v2
uv run main.py ranker-phase2 --trained-run results/learning_curve_v2/queries_50000 --phase2-config config/ranker_phase2_50k.json
# Classical CF baseline (item-kNN on session co-clicks), compared with the runs in config/item_knn.json.
uv run main.py item-knn --trained-run results/learning_curve_v2/queries_50000
```

Settings live in `config/alpha_sweep.json`, `config/two_tower.json`,
`config/filter_pipeline.json`, `config/event_ablation.json`,
`config/ranker_phase2.json`, `config/ranker_phase2_50k.json`,
`config/ranker_lead_50k.json` and `config/item_knn.json`. Rankers report
click NDCG@10, Recall@10 and MRR@10. Dataset analyses
used for the log are in `scripts/analysis/`.

On native Windows, LightGBM training crashes when pandas is imported first.
`session_rerank/__init__.py` and `tests/conftest.py` import LightGBM first; keep
that order in new scripts.

## Earlier benchmark for reference

The earlier complete experiment is kept in
`src/session_rerank/pipeline/legacy.py`, under `run_legacy_experiment()`.
Its features include personal and older-session context, so it is distinct from
the new anonymous-session study. Its existing full-run command remains:

```powershell
uv run cp4285 run
```

The following file map describes that earlier benchmark:

| Step | File in `src/session_rerank/pipeline` | What happens |
|---|---|---|
| 1. Prepare the dataset | `dataset.py` | Audit the CSVs; make time-based samples, histories, and tree features; reserve selection and confirmation queries. No models are trained here. |
| 2. Train multiple models | `training.py` | Fit the original reference, two view-feature trees, three contextual trees (including the anonymous-user control), GRU4Rec/BPR, GRU4Rec/CE, and SASRec/CE. |
| 3. Compare models | `comparison.py` | Nominate tree, neural, and combined-score candidates using validation; choose the highest NDCG on the fresh selection sample. |
| 4. Choose diversification | `diversification.py` | Compare fixed relevance/diversity weights, history gates, and adaptive policies on the original validation sample. |
| 5. Freeze and evaluate | `evaluation.py` | Save model fingerprints and choices before evaluating reserved confirmation queries; measure relevance and token variety. |
| 6. Save results | `reporting.py` | Write tables, recommendation examples, statistics, and charts. |

`state.py` holds the small result objects and saved-file paths shared by these
steps. The existing lower-level code still implements the scientific work:

- `prepare.py`, `optimization_data.py`, `view_features.py`, and `iteration3.py`:
  samples, causal history, and features.
- `benchmark.py`, `optimize_ranker.py`, `recbole_model.py`, and `neural_search.py`:
  reference, tree, and neural training.
- `iteration3_report.py`, `optimization_report.py`, `policies.py`, and `core.py`:
  score combinations, re-ranking, metrics, policy search, and reporting.

The earlier benchmark preserves its feature definitions, model parameters,
selection rules, diversification search, and statistical calculations. The
results documented below are the original reported experiment, not a new full
training run of this branch.

## Follow the code in VS Code

Open the `cp4285-g02` repository folder itself in VS Code. Start with `main.py`,
then Ctrl+click a stage function (or press F12) to open its implementation. Each
stage is imported directly from the file that defines it. Every application
function has a docstring explaining its purpose; larger functions also describe
their inputs, returned values, and saved files. Hover over a call to read it.

The `[tool.pyright]` setting in `pyproject.toml` adds `src` to the editor's import
search path, following the [Pyright configuration documentation](https://github.com/microsoft/pyright/blob/main/docs/configuration.md).
If navigation still does not work, use **Python: Select Interpreter** in the
command palette and choose the project's `.venv\Scripts\python.exe`, then run
**Developer: Reload Window**. Microsoft Python and Pylance are required.

When reading the code, these names recur:

- **Query:** one search with its own candidate products.
- **History/prefix:** items clicked before that query, with no target/future clicks.
- **Features (`X`):** numeric information used to score each candidate.
- **Labels (`y`):** whether the candidate received an observed click for that query.
- **Offsets:** boundaries separating each query's rows in the flattened arrays.
- **Validation:** data for model development and diversification tuning.
- **Selection:** a separate sample used to choose among nominated models.
- **Confirmation:** reserved queries evaluated after all choices are frozen.
- **NDCG@10:** relevance quality near the top of the recommendation list.
- **ILD@10:** average token dissimilarity among the top ten items.
- **Frozen choices:** recorded models/policies that cannot change after evaluation.

## Run on Windows with uv

Open PowerShell in `C:\Nam-projects\Cp4285-Diginetica\cp4285-g02`:

```powershell
uv sync --locked
uv run main.py --help
uv run pytest -q
```

Run the new anonymous sample preparation:

```powershell
uv run main.py
```

`uv run cp4285 run` and `uv run python -m session_rerank.iteration3 run` call
the earlier six-step benchmark in `pipeline/legacy.py`. The existing `cp4285 prepare`, `context`, and `train`
commands, including their options, remain available for individual legacy tasks;
`cp4285 prepare` only reserves iteration-3 queries and requires earlier processed
data. Use `main.py` for preparation from a fresh checkout.

The first `uv run` can perform the environment sync automatically. There is no
manual activation step and no `.env` or API key is required. The project uses
Python 3.11 (`.python-version`), a reproducible `uv.lock`, CPU-only PyTorch on
Windows/Linux, and the default `dev` group for pytest. The older
`requirements*.lock.txt` files record earlier environments and are not the
installation source for this uv setup.

The six original Diginetica CSVs live directly in this repository's `data/raw`
folder. The local dataset was consolidated here from the parent folder without
copying or changing the CSV contents. The local move record, including file sizes
and SHA-256 checksums, is `data/source/dataset_relocation.json`.

Treat raw CSVs as read-only. The `/data/` rule in `.gitignore` keeps these large
local files out of Git commits. Other machines must supply the six CSVs in
`data/raw` before running the anonymous sampler. The earlier benchmark also
has a public-data downloader.

## Earlier benchmark results and restarting

The final report files are saved under `results/iteration3/`:

- `confirmation_summary.csv`: relevance and token-variety results with intervals.
- `confirmation_paired_differences.csv`, `confirmation_slices.csv`, and
  `improvement_vs_previous.csv`: statistical comparisons and subgroup results.
- `confirmation_examples.json`: example recommendation lists.
- `figures/`: relevance, diversification, and model-comparison charts.
- `frozen_selection.json` and `confirmation_receipt.json`: recorded choices,
  input fingerprints, and evaluation receipt.

Earlier stages also save model files, validation/selection tables, and audits.
Processed data is under `data/processed`, `data/processed_v2`, and `data/iteration3`;
reference and earlier-model outputs remain under `results/` and
`results/optimization/`. Full preparation and training require more disk space
than the environment alone.

Completed preparation and training stages are reused. After candidates have
been recorded for fresh selection, training is disabled. After choices are
frozen, they cannot be replaced by rerunning the pipeline. Confirmation outcomes
are cached in `confirmation_per_query.npz`; rerunning `uv run cp4285 run`
reuses them and regenerates report files. A reporting failure can be retried without evaluating
confirmation again. Existing completed experiments using the earlier cache
format remain readable alongside their examples and receipt files.

## Refactor verification

All 59 tests pass in the Linux Python 3.11 verification environment, including
the real-data split check. A small deterministic experiment matches the original
pipeline's model choices, policies, per-query relevance/diversity metrics, and
recommendation examples. New sample tests cover complete sessions, anonymous
exports, strict history cutoffs, original click ranks, and confirmation
separation. Additional checks cover freeze protection, cached
confirmation reuse, and the training variants. Both run commands' help screens
were verified, and Windows x64 dependency planning passed. The checks ran in
Linux; the Windows runtime and full model training were not exercised as part
of this refactor.

## Earlier benchmark methodology

Re-rank Diginetica's complete candidate lists using information available before
each query. Daily sessions provide up to ten distinct prior clicked products.
Historical statistics and collaborative item associations use earlier data with
an embargo; persistent-user context has an anonymous-user control.

Train temporal/collaborative LambdaMART models and **RecBole GRU4Rec and SASRec**.
Compare BPR and cross-entropy configurations. Nominate models on development data,
choose on a fresh model-selection sample, then freeze before confirmation.
The selected hybrid uses 90% normalized tree score and 10% GRU4Rec/CE score
when click history exists, otherwise the tree score.

Greedy re-ranking balances relevance and token Jaccard distance. Select fixed
and session-adaptive weights within a 2% validation NDCG loss budget.

## Earlier reported results

All rows use the same **9,865 newly reserved confirmation queries and 1,492,509
candidates**. Session IDs exclude both earlier test/confirmation samples.

| Method | NDCG@10 | Token ILD@10 |
|---|---:|---:|
| Original ranker | 0.18261 | 0.90329 |
| Previous hybrid | 0.22412 | 0.90878 |
| RecBole SASRec/CE | 0.20369 | 0.91374 |
| Selected collaborative hybrid | 0.23116 | 0.90887 |
| Hybrid + fixed diversification | 0.22970 | 0.93312 |
| Hybrid + adaptive diversification | 0.22998 | 0.91581 |

The selected model improves NDCG **3.1%** over the previous hybrid
(paired 95% interval: **1.9–4.3%**). Its component tree scores 0.23146,
so fusion adds no demonstrated confirmation benefit. Fixed diversification adds
**2.7% token variety** for a **0.6% NDCG reduction**. The adaptive policy again
selects equal diversity-branch weights and becomes a history-length gate.

These offline results measure recorded clicks and lexical variety, not shopper
satisfaction or sales.


## Six relevant items and four diverse items

The current `config/learning_curve.json` uses `top_k: 10` and, under
`reranking`, `policy_type: "diverse_tail"`, `relevance_slots: 6`, and `k1: 500`.
First score the logged candidates and retain the top K1 by relevance. Keep the
first six as the relevance head. From the remainder of K1, choose four items
with high **mutual** diversity and sort those four by relevance. If fewer than
K1 candidates are logged, use the available candidates. K1 must be at least
`top_k`. This filter occurs after model scoring; it is not a full-catalog retrieval stage. It seeds the most distant remaining pair and
then greedily adds the item with the largest total distance to the selected
four-item group. This is an approximation; it does not guarantee the globally
most diverse set. Missing pair distances receive zero selection reward, and
unknown pairs are excluded from reported ILD with coverage recorded.

`reranking_metrics.csv` reports `head_click_ndcg` and `head_gain_ndcg` for the
first six, `tail_diversity_ild` for the final four, and overall top-K metrics.
NDCG measures logged relevance; ILD measures variety. A lower NDCG can accompany
more diversity, but unclicked items in historical logs are not necessarily
irrelevant. Purchase metrics remain useful, with small-sample caution.

The earlier saved 6 + 4 results were measured before the K1 filter was added;
they should not be labelled as K1-filtered results. Use a new experiment to
measure K1. The configured candidate-pool size is saved per query and its mean
is included in `reranking_metrics.csv`.

Use `policy_type: "adaptive"` to return to the existing weighted decay policy.
Decay settings apply to that policy, not the six-and-four policy. Since this
changes the experiment code and configuration, use a new `--curve-root` when
training another size, for example `results/six_four`.

## Manual training sizes and hyperparameter tuning

Read [TRAINING_CONFIG_GUIDE.md](TRAINING_CONFIG_GUIDE.md) for the full config,
decay examples, early-stopping explanation, and Windows comparison recovery.

Use **`config/learning_curve.json`** for manual training. Each command prepares
and trains exactly one size, reviews the configured reranker on validation, redraws the learning
curve, and stops. Start the next command only when you are ready:

```powershell
uv run main.py train --queries 2000
uv run main.py train --queries 10000
uv run main.py train --queries 50000
uv run main.py train --queries all
```

`all` means every eligible training query (currently 145,039). Sizes are targets:
whole sessions can put the actual count slightly above the requested count.
The samples are nested with the same seed, and validation queries/labels remain
identical. Every size trains fresh models. Selection and confirmation outcomes
are not loaded. Repeating the same command verifies and resumes its completed
stages. Source or settings changes need a new experiment directory.

Always supply the sample size as a CLI argument. The editable config contains
hyperparameters and experiment settings; it has no sample-size default:

```powershell
uv run main.py train --queries 2000 --training-config config/learning_curve.json
```

`--queries` is required for `train`, including with a custom config. Use a positive
integer or `all`. The selected size is still recorded in each run's generated
metadata for reproducibility. `output_directory` chooses the experiment
folder. `primary_metric` stays fixed across all sizes; the default is
`click_ndcg`, with graded purchase-aware NDCG and purchase recall also reported.

The `training` object contains the editable LightGBM settings:

| Setting | What to change |
|---|---|
| `learning_rate` | Update size for each tree; smaller values usually need more trees |
| `num_boost_round` | Maximum number of trees per trial |
| `early_stopping_rounds` | Stop after this many rounds without validation improvement |
| `log_evaluation_period` | Print validation progress every this many rounds; 0 disables iteration logs |
| `num_leaves` | List of tree sizes to test; more leaves allow more complex rules |
| `min_data_in_leaf` | List of minimum leaf sizes; larger values constrain complexity |
| `max_depth` | Maximum tree depth; -1 means unlimited |
| `max_bin` | Maximum feature histogram bins |
| `lambda_l1`, `lambda_l2` | Nonnegative regularization strengths |
| `feature_fraction` | Fraction of features sampled per tree |
| `bagging_fraction`, `bagging_freq` | Row sampling fraction and frequency; frequency 0 disables bagging |
| `lambdarank_truncation_level` | Rank range emphasized by the ranking loss |

The same config also controls the recommendation list, outside the `training`
object. `top_k` is the number of products returned, and the cutoff for validation
NDCG/recall and early stopping. Lists shorter than K retain all available items.
The LambdaRank loss has its own `lambdarank_truncation_level`, commonly K+3.

| Ranking control | Meaning |
|---|---|
| `top_k` | List size and evaluation cutoff, default 10 |
| `reranking.enabled` | Compare the configured policy with relevance-only after fitting |
| `decay_function_type` | `fixed`, `linear`, or `exponential` relevance-weight mapping |
| `decay_function_rate` | Positive dimensionless curve rate; larger means less relevance weight at the same history diversity |
| `alpha_min`, `alpha_max` | Lower/upper relevance-weight bounds; diversity weight is 1-alpha |
| `fixed_alpha` | Relevance weight when function type is `fixed` |
| `fallback_alpha` | Relevance weight with insufficient or unmeasurable history |
| `history_decay_type` | `none`, `linear`, or `exponential` ageing of earlier interactions |
| `history_decay_rate_per_minute` | Positive age-decay rate; larger means older interactions matter less |
| `history_max_events` | Shared cap on earlier same-session feature/reranking events |
| `minimum_history_items` | Minimum distinct usable history items for adaptive weighting |
| `diversity_metric` | `token`, `category`, or `combined` (equal token/category blend) |

Every row except `top_k` belongs inside the `reranking` object. History decay and
adaptive weighting are separate operations. Given observed history diversity H,
linear relevance weight is `min + (max-min)*max(0,1-rate*H)`. Rate 1 reproduces
the original linear curve. Exponential uses
`min + (max-min)*(exp(-rate*H)-exp(-rate))/(1-exp(-rate))` with shared endpoints.
`fixed_alpha` applies only to a fixed policy; the function rate applies to linear
or exponential policies.

History time weights instead use age in minutes: linear
`max(0,1-rate*age)` reaches zero after `1/rate` minutes; exponential
`exp(-rate*age)` has half-life `ln(2)/rate` minutes. A rate of approximately
0.138629 means a five-minute exponential half-life. History type `none` disables
age weighting. Undefined history diversity uses `fallback_alpha`.

Decay is applied during reranking, after relevance training. It is not an extra
term in the LambdaRank loss. Manual validation saves relevance-only and configured
policy metrics side by side in `reranking_metrics.csv`, individual lists in
`validation_reranking.parquet`, and prefixed `reranked_*` metrics in the comparison
CSV. In the curve chart, solid lines are relevance-only and dashed lines show the
configured policy. Selection and confirmation stay unopened.

`seed` and `threads` are top-level controls. The two lists form a Cartesian grid:
`num_leaves: [15,31]` and `min_data_in_leaf: [100,300]` test four combinations per
nominal model. Other settings are scalar and apply to every trial. Gain-sensitivity
models reuse the best nominal purchase/session leaf settings. Use single-value
lists if you want to isolate the data-size effect with fixed hyperparameters.
Using the same grid at every size measures performance after the same tuning
procedure. Early stopping uses the model's click or graded training objective;
the fixed primary validation endpoint chooses the best trial.

For a different learning rate or grid, copy the config and set a new output
folder so the previous results remain comparable:

```powershell
Copy-Item config/learning_curve.json config/learning_curve_lr003.json
# Edit learning_rate to 0.03 and output_directory to results/tuning/lr003.
uv run main.py train --queries 2000 --training-config config/learning_curve_lr003.json
```

Alternatively, `--curve-root results/tuning/lr003` overrides the output directory.
Keep one config unchanged while expanding its sizes. K and decay changes also
require a separate experiment folder. The explicit `pilot-train` command retains the earlier click-only pilot workflow.

Review completed sizes without training anything:

```powershell
uv run main.py compare
uv run main.py compare --training-config config/learning_curve_lr003.json
```

The experiment folder contains `comparison.csv` and `learning_curve.png`.
Each `queries_2000`, `queries_10000`, etc. folder contains models, audited sample
counts, `<model>_trials.csv` with every attempted setting and score, matching JSON
records with the exact shared config, and `learning_metrics.csv`. The console
shows feature batches, model/trial numbers, iteration scores, best iterations,
and trial durations. The chart uses validation data, so final scientific claims
still require a later frozen confirmation study.

## Expanded anonymous-session study

Implementation is ready. The manual learning-curve path above is the entry point
for gradual training; the expanded command below runs the complete final study. The full-data experiment has
**not been run**: the preliminary preparation attempt was stopped before model
training at the user's request. Fresh confirmation outcomes remain unopened.

The commented `expanded_main()` in `main.py` runs the expanded experiment when
you choose to start it:

```powershell
uv run main.py expand
```

It uses every eligible training query, keeps the pilot validation and selection
queries fixed, and reserves 2,000 fresh confirmation queries in whole daily
sessions. Exact settings are in `config/expanded_study.json`. Read
`TRAINING_PLAN.md` for the research protocol and feature/label boundaries.

The run prints its new folder under `results/expanded_study/`. To continue an
interrupted run, supply that exact folder:

```powershell
uv run main.py expand-resume --study-run results/expanded_study/<run-folder>
```

Completed stages are verified by their saved file hashes and reused. Resume
requires the same scientific source files. Choices are frozen before fresh
confirmation events are loaded. The original pilot commands and outputs remain
available.

The stages prepare a SQLite database and batched feature caches; train click and
purchase-aware LightGBM LambdaRank models; compare fixed, quota, threshold,
linear and exponential relevance weights; test history time decay separately;
select and freeze settings; evaluate confirmation; and write a standalone white
`report.html`. The report embeds its charts and links the discussion to saved
CSV/Parquet tables, model trials, purchase audits, and session trajectories.

Purchase labels use a unique latest preceding same-session query that contained
the item. Unmatched and ambiguous purchases stay in the audit. The declared gains
are 0 (unobserved), 1 (click), and 3 (purchase). Sensitivity fits use purchase
gains 2 and 5, with common evaluation gains. Purchase-aware NDCG becomes primary
only with at least 100 purchase-bearing training queries and 20 validation
queries; otherwise click NDCG remains primary. This rule is saved before scores.

Verification uses Linux/Python 3.11, including a real LightGBM
synthetic run through reservation, training, selection, freeze, confirmation,
reporting, and checkpoint reuse. Full-data run outcomes are recorded separately
inside each run folder. Windows uses the same uv entry point.
