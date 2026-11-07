# CP4285 manual training and configuration guide

Updated: 7 November 2026.

Use this guide to train one sample size at a time, change model hyperparameters,
configure the relevance/diversity balance, and review the measured impact.
The configuration file is [config/learning_curve.json](config/learning_curve.json).
The high-level implementation is [main.py](main.py), in `manual_training_main()`.

Your 2,000-query run completed training and validation. Its final comparison step
hit a Windows path-separator bug. The code now normalizes that lookup; the repair
does not require retraining. No additional real-data training is started by this
code/documentation update.

## 1. Working folder and manual workflow

Run commands in PowerShell from:

```powershell
cd C:\Nam-projects\Cp4285-Diginetica\cp4285-g02
```

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

## 2. Complete example configuration

This is the current configuration. JSON does not allow comments, so the setting
explanations are in the tables above. Edit the actual JSON file rather than this
example when running an experiment.

```json
{
  "pilot_directory": "data/anonymous_samples",
  "output_directory": "results/learning_curve",
  "primary_metric": "click_ndcg",
  "query_batch_size": 500,
  "seed": 4285,
  "threads": 4,
  "label_gains": [
    0,
    1,
    3
  ],
  "purchase_gain_sensitivity": [
    2,
    5
  ],
  "training": {
    "learning_rate": 0.05,
    "num_boost_round": 500,
    "early_stopping_rounds": 30,
    "log_evaluation_period": 25,
    "num_leaves": [
      15,
      31
    ],
    "min_data_in_leaf": [
      100,
      300
    ],
    "max_depth": -1,
    "max_bin": 255,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "lambdarank_truncation_level": 13
  },
  "top_k": 10,
  "reranking": {
    "enabled": true,
    "decay_function_type": "linear",
    "decay_function_rate": 1.0,
    "alpha_min": 0.2,
    "alpha_max": 1.0,
    "fallback_alpha": 0.8,
    "fixed_alpha": 0.8,
    "history_decay_type": "none",
    "history_decay_rate_per_minute": 0.13862943611198905,
    "history_max_events": 100,
    "minimum_history_items": 2,
    "diversity_metric": "combined"
  }
}
```

`label_gains` records the declared main gain scheme `[0,1,3]`: unobserved, clicked,
and purchased. A purchased candidate receives the highest grade once; click and
purchase gains are not added together. The main experiment currently requires
that declared scheme. `purchase_gain_sensitivity` controls additional diagnostic
purchase-gain fits, initially 2 and 5; all are measured with common evaluation
gains. It is not a list of additional classes.

`primary_metric` can be `click_ndcg` or `gain_ndcg` for manual curves. It stays
fixed across sizes, even when the purchase count grows. `seed` controls the model
randomness. The nested sampling order comes from the original pilot protocol,
whose manifest is recorded with the experiment. `query_batch_size` controls how
many query groups are expanded together when constructing features.

## 3. How the two decay controls differ

The hypothesis is that focused browsing supports more relevance weight, while
varied browsing can support more diversity. Only earlier actions in the same
anonymous daily session contribute. Previous sessions are not available.

### Relevance-weight decay

`decay_function_type` and `decay_function_rate` map observed session diversity H
to relevance weight alpha. H ranges from 0 to 1. These rates have no time unit.

For the default linear rate 1 and bounds 0.2 to 1.0:

| Observed history diversity H | Relevance weight alpha | Diversity weight 1-alpha |
|---|---|---|
| 0.1, focused | 0.92 | 0.08 |
| 0.8, varied | 0.36 | 0.64 |

A linear rate above 1 reaches the minimum relevance weight sooner. A rate below
1 decreases it more slowly and may not reach that minimum at H=1. The normalized
exponential curve shares its endpoints for every positive rate.

During greedy reranking, each next item balances normalized relevance and its
mean distance to the already selected items. The first item is seeded by highest
relevance. This is a greedy list builder, not an exact search over every possible
set of K products.

### History time decay

`history_decay_type` and `history_decay_rate_per_minute` determine how much an
older interaction contributes to measured session diversity. These rates use
minutes. They can help when broad browsing becomes focused later in a session.

| Type and rate | Example |
|---|---|
| `none` | Earlier retained interactions have equal weight |
| `linear`, rate 0.2 per minute | Weight is 0.5 after 2.5 minutes and 0 after 5 minutes |
| `exponential`, rate approximately 0.138629 per minute | Weight is approximately 0.5 after 5 minutes and 0.25 after 10 minutes |

History diversity uses distinct earlier clicked/viewed products. Each product
keeps its latest eligible interaction. Pair weights are the products of the two
item weights; H is the weighted average of known pair distances. Purchases can
contribute relevance features but are not included as a separate interaction type
in this diversity calculation.

The history cap counts retained earlier events, including event types beyond
clicks/views. A product with missing metadata cannot satisfy the minimum usable
history requirement. Insufficient usable products, missing pair distances, or
zero effective weight trigger `fallback_alpha`. More actions alone do not force
more diversity.

## 4. What each training command fits

Each size fits four main relevance models:

| Model | Training signals and features |
|---|---|
| `click_product` | Click labels with product/query features |
| `click_session` | Click labels plus earlier current-session features |
| `purchase_product` | Click/purchase grades with product/query features and prior purchase popularity |
| `purchase_session` | Click/purchase grades plus current-session features and earlier purchase counts |

With the default two-by-two grid, these models use 16 training trials. The two
purchase-gain sensitivity models add one trial each, using the nominal purchase
session model's chosen leaf settings. That is 18 fits per sample size with the
default configuration. The same early-stopping message can appear once per fit.

The fitting objective is LightGBM LambdaRank. Decay is applied in the validation
reranking stage after fitting. The history cap also changes the history available
to relevance feature construction. Selection and confirmation are later stages
and are not opened by these manual commands.

## 5. Understanding early stopping

An early-stopping message is normal training behavior. With
`early_stopping_rounds: 30`, training stops after 30 consecutive rounds without a
better monitored validation score. The model retains the best iteration.
This matches [LightGBM's early-stopping documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html).

For example, your click/session trial reported:

```text
[25] valid_0's rank_ndcg_at_10: 0.155368
Early stopping, best iteration is:
[3]  valid_0's rank_ndcg_at_10: 0.162854
```

Iteration 3 was best. Training continued through the patience window, and later
rounds did not beat that score. It did not stop after building only three trees
without checking later rounds. Iteration logs appear only every 25 rounds under
the default logging setting, so not every attempted round is printed.

A small best iteration does not by itself establish a defect. It means the
monitored validation metric peaked early for that trial. Assess the saved scores,
features and larger-sample results before changing the stopping rule. The log
alone does not establish why validation stopped improving.

Click models monitor binary click NDCG. Purchase-aware models monitor the fitted
graded-gain NDCG. The `Completed ... click_ndcg=...` line then reports the shared
primary endpoint. These can differ slightly because they measure different gains.
The best trial across grid settings is chosen by the common primary endpoint.

The categorical-bin warnings in your log did not stop training. The actual
failure was the final `KeyError`, after all model fits and validation reranking.

## 6. Windows comparison failure and recovery

The saved feature receipt used a key such as:

```text
train_features\queries.parquet
```

The comparison code looked up `train_features/queries.parquet`. That spelling
mismatch caused the error after `learning_metrics.csv` had already been saved.
The reporting code now normalizes receipt keys before lookup and still checks
the original file hashes. Models, feature values and validation scores are not
changed by that repair.

Regenerate the comparison without training any model:

```powershell
uv run main.py compare
```

The existing experiment's original source fingerprints are preserved in its
reporting-patch record when compatibility metadata is updated. Model files and
training receipts must still match their recorded hashes. You can then start
10,000 queries manually when ready; there is no need to refit 2,000 queries just
to recover the comparison chart.

Repeating an unchanged training command reuses completed stages and models. A
partially completed model fit is retrained; this is not a checkpoint after every
tree or every trial. Normal interruption and recovery do not automatically
advance to the next sample size.

## 7. Reading the metrics

| Metric | Meaning |
|---|---|
| Click NDCG@K | Rewards clicked candidates near the top; normalized against their ideal order |
| Purchase-aware NDCG@K | Uses the declared click/purchase gains in the same calculation |
| Click recall@K | Clicked candidate products retrieved divided by all clicked candidate products |
| Purchase recall@K | Attributed purchased candidates retrieved divided by all attributed purchased candidates |
| ILD@K | Mean known pair distance among recommended products |

Metrics average eligible queries equally. Queries with no relevant candidate do
not contribute to that relevance endpoint. Purchase recall is conditional on an
attributed purchase in the candidate list; it does not measure purchases outside
those lists. Missing metadata and undefined diversity are not positive evidence
of variety.

The combined distance is an equal blend of token Jaccard distance and category
distance. It measures lexical/category variety, not a learned semantic embedding.
Offline ranking results do not establish actual CTR or purchase-conversion gains.

Use validation to decide which configuration to investigate. Do not use repeated
validation tuning as final proof. A later study must freeze its choices before
opening reserved confirmation outcomes.

## 8. Implementation map

| File | Role |
|---|---|
| [main.py](main.py) | High-level manual stage calls and CLI |
| [config/learning_curve.json](config/learning_curve.json) | Editable settings |
| [expanded_data.py](src/session_rerank/expanded_data.py) | Nested samples, purchase attribution, causal features and batches |
| [expanded_training.py](src/session_rerank/expanded_training.py) | LambdaRank fits, grids, logging and checkpoints |
| [ranking_settings.py](src/session_rerank/ranking_settings.py) | Validation of K, decay controls, bounds and units |
| [adaptive_policies.py](src/session_rerank/adaptive_policies.py) | Weight functions, history diversity, list building and metrics |
| [configured_reranking.py](src/session_rerank/configured_reranking.py) | Relevance-only versus configured-policy validation |
| [learning_curve.py](src/session_rerank/learning_curve.py) | Experiment identity, saved comparisons and curve chart |
| [TRAINING_PLAN.md](TRAINING_PLAN.md) | Research protocol and later selection/confirmation study |
