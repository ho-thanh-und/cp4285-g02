# Session Aware Product Re-Ranking

CP4285 Group 2 research project. This repository studies whether session context
and item diversification improve the ranking of Diginetica's supplied product
candidates. The implementation includes LambdaMART rankers, session and
collaborative features, RecBole models, ranking and diversity metrics, audits,
and reproducible experiment protocols.

## Get started

### Requirements

- Git
- Python 3.11
- About 5 GB of free disk space for the source archive, prepared data, and
  generated outputs. Neural model training needs more memory and disk space.

From the repository root:

```sh
git clone <group-repository-url>
cd cp4285-g02
python3.11 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The install includes the project dependencies, including CPU/GPU PyTorch and
RecBole. If your team uses a specific accelerator build of PyTorch, install that
build first using the official PyTorch instructions, then install the project
dependencies. Keep credentials and machine-specific settings out of Git.

### Download and prepare data

The public Diginetica archive is downloaded on first preparation. The data is
not stored in Git.

```sh
python scripts/download_data.py
python -m session_rerank.prepare
```

The downloader verifies the archive format, extracts the raw CSVs, and records
checksums in `data/source/manifest.json`. Do not edit the raw files. Keep
`data/`, `results/`, model checkpoints, and logs local; they are ignored by
Git. Each contributor should run the preparation step on their own machine.

## Run experiments

Start with the lightweight baseline workflow:

```sh
python -m session_rerank.benchmark --stage development
python -m session_rerank.run_experiment all
```

The optimization workflow is driven by `src/session_rerank/optimization_data.py`,
`optimize_ranker.py`, `recbole_model.py`, and `run_experiment.py`; protocols live
in `config/`. The later session benchmark is run in stages so that its selection
and reserved confirmation are explicit:

```sh
python -m session_rerank.benchmark_data
python -m session_rerank.benchmark_experiment development
python -m session_rerank.benchmark_experiment confirmation
python -m session_rerank.benchmark_experiment checks
python -m session_rerank.benchmark_experiment report
```

Confirmation is intended to be run only after model and policy choices have
been frozen. Do not use its outcomes to tune a model and then present those same
outcomes as an untouched evaluation. Outputs are written below `results/`; data
for the session benchmark is under `data/session_benchmark/`.

Run the automated checks locally before opening a pull request:

```sh
python -m pytest
```

For reproducibility, record the command, protocol file, input manifest/checksum,
Python version, and relevant environment details with any new result. Use fixed
seeds and keep development/validation decisions separate from confirmation.

## Project layout

| Path | Purpose |
|---|---|
| `src/session_rerank/` | Data preparation, features, models, evaluation, audits, and reports |
| `config/` | Versioned experiment protocols and model settings |
| `tests/` | Regression and protocol checks |
| `scripts/` | Data download and repository utilities |
| `docs/` | Methods, experiment notes, results interpretation, and planning |
| `data/` | Downloaded and prepared local data; ignored by Git |
| `results/` | Generated tables, plots, logs, and checkpoints; ignored except `results/shared/` |
| `results/shared/` | Reviewed, compact artifacts intended for peer review in Git |
| `EXPERIMENT_STATUS.md` | Current experiment status and links to local outputs |
| `LITERATURE_NOTES.md` | References and caveats for behavioral claims |

`docs/README_FULL.md` provides the longer research narrative. The result
artifacts themselves are generated locally and are not available to a new clone
unless explicitly shared or regenerated.

## Contributing as a group

1. Pull the latest `main` and create a focused branch, for example
   `docs/methods-summary` or `experiment/retailrocket-check`.
2. Before starting a new run, agree on its question, data split, metrics, and
   protocol. Add or update a JSON protocol in `config/` before inspecting
   confirmation outcomes.
3. Commit source code, tests, configuration, and concise documentation. Do not
   commit raw data, generated result folders, checkpoints, private data, or
   environment files.
4. Put new methods and interpretations in `docs/` or update the relevant
   methods/results page. A useful result note states the question, command,
   dataset/version, split, metric, uncertainty, limitations, and output paths.
5. Keep intermediate outputs in ignored `results/`. Put final, compact tables
   and figures that peers should review in `results/shared/`; include the
   generating command and protocol in a nearby Markdown note or relevant page
   under `docs/`. See [`results/shared/README.md`](results/shared/README.md).
6. Run `python -m pytest`, inspect `git status`, and open a pull request with a
   short summary, reproduction command, and any remaining questions.

If a result cannot be regenerated by every teammate because of size or compute
limits, document who produced it, the exact environment and command, input
checksums, and where the team can retrieve the artifact. Never silently replace
another person's experiment output or frozen confirmation record.

## Current findings

The longer confirmed results, comparisons, and interpretation limits are in
[`docs/README_FULL.md`](docs/README_FULL.md) and
[`docs/OPTIMIZATION_RESULTS.md`](docs/OPTIMIZATION_RESULTS.md). They distinguish
the supplied-order baseline, trained rankers, hybrid models, and diversity
policies; scores from different protocols and query populations should not be
compared as if they were one experiment.

## Remaining work

- Agree on the final research question and which experiment provides the
  headline result; clearly label exploratory analyses.
- Independently reproduce the selected pipeline from a clean environment and
  record any differences in versions, data hashes, or outputs.
- Finish the group report/poster or presentation, check every figure and
  citation, and ensure every member can explain the data split and limitations.
- Keep the interpretation bounded: logged-click ranking and lexical diversity
  do not establish shopper satisfaction, sales lift, fairness, or causal
  off-policy reward. A validated online or identified exposure study would be
  needed for those claims.
- Update this list as tasks are assigned and completed; keep owners and review
  dates in the team's project tracker rather than inventing assignments here.

## License and data

This repository contains research code and experiment configurations. Check the
course/team policy before redistributing source datasets, generated artifacts,
or third-party materials. The original data downloader records source details
and checksums locally.
