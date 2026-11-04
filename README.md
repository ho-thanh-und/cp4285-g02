# Session-Aware Product Re-ranking

## Step 1 for the anonymous session study

Create the new small sample tables without training models:

```powershell
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

The `prepare` action is the new anonymous study's step 1. The default
`uv run main.py` and `uv run cp4285 run` still execute the earlier benchmark,
whose later features include personal/older-session context. Those later stages
have not yet been converted to the new anonymous sample schema.

## Start here: `main.py`

The complete experiment is expressed as six steps in the repository's root
`main.py`:

```python
prepare_dataset()
train_models()
comparison = compare_models()
settings = choose_diversification(comparison)
results = freeze_and_evaluate(comparison, settings)
save_results(results)
```

Read `main.py` first, then follow each function into `src/session_rerank/pipeline`.
The functions return named results where the next step needs them. Large datasets
and fitted models are saved on disk rather than passed around in memory.

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

The refactor preserves the existing feature definitions, model parameters,
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

Run the complete experiment when sufficient disk space is available:

```powershell
uv run main.py
```

`uv run cp4285 run` and `uv run python -m session_rerank.iteration3 run` call
the same six-step flow. The existing `cp4285 prepare`, `context`, and `train`
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
`data/raw` or allow the pipeline's public-data downloader to obtain them.

## Saved results and restarting

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
are cached in `confirmation_per_query.npz`; rerunning `main.py` reuses them and
regenerates report files. A reporting failure can be retried without evaluating
confirmation again. Existing completed experiments using the earlier cache
format remain readable alongside their examples and receipt files.

## Refactor verification

All 44 tests pass in the Linux Python 3.11 verification environment, including
the real-data split check. A small deterministic experiment matches the original
pipeline's model choices, policies, per-query relevance/diversity metrics, and
recommendation examples. New sample tests cover complete sessions, anonymous
exports, strict history cutoffs, original click ranks, and confirmation
separation. Additional checks cover freeze protection, cached
confirmation reuse, and the training variants. Both run commands' help screens
were verified, and Windows x64 dependency planning passed. The checks ran in
Linux; the Windows runtime and full model training were not exercised as part
of this refactor.

## Methodology

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

## Results

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
