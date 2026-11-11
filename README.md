# Session-Aware Recommendations for an Anonymous Online Shop

CP4285 project, group g02. Dataset: Diginetica (CIKM Cup 2016, Track 2).

This branch holds the final state of the project: the frozen three-page
recommender (category, item, home page), its confirmation-week results, the
demo website, and everything needed to reproduce them. The full iterative
development history, earlier pilot studies and an older pre-pivot benchmark
are kept on `refactor/readable-main-pipeline` for reference; they are not
needed to reproduce the reported results.

**Start here:**

- [REPORT.md](REPORT.md): the project report (problem, data, methods,
  evaluation, final-test results, error analysis, ethics).
- [ETHICS.md](ETHICS.md): the full ethical analysis with exposure measurements.
- [HANDOFF.md](HANDOFF.md): running the frozen models behind the demo website
  (`uv run main.py handoff`, then `uv run main.py serve`).
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md): every experiment, decision and code
  change behind the final results.
- [AI_USE_DECLARATION.md](AI_USE_DECLARATION.md): how an AI coding assistant
  was used during development.
- [TRAINING_PLAN.md](TRAINING_PLAN.md) and
  [TRAINING_CONFIG_GUIDE.md](TRAINING_CONFIG_GUIDE.md): the training protocol
  and the `config/learning_curve.json` field reference used for the category
  ranker.

## The final system

| Page | Before any click (cold start) | After the first click | Variety |
|---|---|---|---|
| Category page | LightGBM (works without history) | LightGBM with session features | none (relevance only) |
| Item page, "You may also like" | item-kNN on the viewed product | (same) | fixed alpha 0.8 over top 20 |
| Home page | April popularity | SASRec (dim 64, 2 blocks, dropout 0.5) | 6 + 4 over its top 20 |

Choices were made on a validation week, checked on a selection week, then
tested once on a final confirmation week. See `REPORT.md` §5 for the numbers.

## Running it

```bash
uv sync --locked
uv run pytest -q
```

The six raw Diginetica CSVs go in `data/raw/` (gitignored; `scripts/download_data.py`
fetches the public release). Key reproduction commands (`REPORT.md` §10):

```bash
uv run main.py train --queries 50000           # category-page LightGBM ranker
uv run main.py item-knn --trained-run <run>     # classical co-click baseline
uv run main.py ranker-phase2 --trained-run <run>
uv run main.py sequence-models                  # SASRec / GRU4Rec, home page
uv run main.py heldout                          # category pages, selection week
uv run main.py ymal --ymal-config config/ymal_selection.json
uv run main.py home-page
uv run main.py error-analysis
uv run main.py confirmation                     # the final test, once
uv run main.py exposure                         # ethics measurements
uv run main.py handoff                          # export models for the website
uv run main.py serve --handoff results/handoff/<timestamp>
uv run python scripts/report/make_figures.py
uv run python scripts/report/audit_report.py    # re-checks REPORT.md's numbers
```

`--help` lists every action and its config flag. Settings live in
`config/*.json`; each run folder under `results/` (gitignored) records its own
settings and, for held-out runs, its frozen choices and code version.

The SASRec/GRU4Rec grid ran on a GPU cluster (`scripts/cluster/`); an optional
Transformers4Rec cross-check lives in `scripts/t4rec/`
(`main.py export-sequences` / `score-external`, own conda env). Dataset facts
cited in the report come from `scripts/analysis/`.

## Demo website

```bash
uv run main.py handoff
uv run main.py serve --handoff results/handoff/<timestamp>
uv run python scripts/run_demo.py
```

Serves `web/` against the exported, frozen models. Product names and images in
`web/` are AI-generated placeholders for the dataset's anonymised products
(`ETHICS.md` §7); every reported metric is computed on the original anonymised
IDs.

## Code map

- `main.py`: one CLI action per stage; `--help` documents every command and
  config flag.
- `src/session_rerank/`: the implementation. Each stage is imported directly
  from the file that defines it — Ctrl+click a call in `main.py` to open it.
- `config/*.json`: settings for every stage (validation/selection/confirmation
  variants are separate files, e.g. `ymal.json` / `ymal_selection.json` /
  `ymal_confirmation.json`).
- `tests/`: one file per module, plus `conftest.py` (LightGBM must import
  before pandas on native Windows; see its docstring).

On native Windows, LightGBM training crashes if pandas is imported first;
`src/session_rerank/__init__.py` and `tests/conftest.py` import LightGBM first,
so keep that order in new scripts.
