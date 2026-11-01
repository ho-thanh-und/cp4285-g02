# When does session-aware diversification help?

CP4285 Group 2: Nguyen Dinh Nam, R Arulkumaran, Ho Chi Thanh.

This repository implements the team's STePS project: re-rank Diginetica's
supplied search candidates and test whether session diversity adds value beyond
a tuned fixed weight and a history-length rule. It includes a RecBole GRU4Rec
model, stronger temporal-feature LambdaMART models, validation-only model and
policy selection, and a reserved confirmation sample with paired uncertainty
estimates. The first experiment is preserved as a reference.

## Current confirmation experiment

All rows below use the same 9,883 evaluable confirmation queries and 1,481,803
supplied candidates. Model and policy selection used validation only. The new
sample excludes every supplied session ID in the first experiment's test.

| Method | NDCG@10 | Recall@10 | Token Jaccard ILD@10 |
|---|---:|---:|---:|
| Original relevance ranker | 0.19066 | 0.31465 | 0.89644 |
| RecBole GRU4Rec with cold-history fallback | 0.18066 | 0.31286 | 0.91684 |
| Enhanced 40-feature ranker, 60k query cap | 0.22696 | 0.37427 | 0.89941 |
| Validation-selected tree + RecBole hybrid | 0.22476 | 0.37174 | 0.90278 |
| Selected hybrid + fixed diversification | 0.22261 | 0.36937 | 0.92974 |
| Selected hybrid + session-diversity policy | 0.22443 | 0.37240 | 0.91378 |

The frozen selected hybrid improves NDCG by **+0.03410 absolute / +17.89%
relative** against the original model on the same queries. The paired 95%
session-cluster bootstrap interval for the relative gain is **+15.56% to
+20.87%**. Random ranking gives 0.07541 NDCG on this population. NDCG is not a
percentage of correctly predicted products.

The selected hybrid uses 75% normalized tree score and 25% normalized RecBole
score when click history exists. The RecBole contribution improves validation
but reduces confirmation performance compared with that tree alone (0.22676).
The unselected 31-leaf view model scores 0.22872 on confirmation. We retain the
validation selection rather than choosing a different model after observing
confirmation. The gains primarily come from stronger temporal/context features;
RecBole alone does not outperform the original ranker.

Fixed diversification increases Jaccard ILD by **+2.99% relative** for **-0.95%
relative NDCG**, compared with the selected hybrid. Cosine ILD also increases,
providing a numerical distance check. The session-diversity search again picks
equal weights in both diversity branches, so it becomes a history-length gate.
It preserves more relevance and produces less diversity than the fixed rule,
without establishing added value from the session-diversity threshold.

The full experiment ledger, confidence intervals, subgroup analysis and figures
are generated under `results/optimization/`. This folder and the local methods
notes in `docs/` remain Git-ignored. The reproducible source/configuration and
the measured summaries in this README form the submission repository.

## Preserved first experiment

Policies were selected on validation and frozen before test evaluation. All rows
below use the same 9,869 evaluable test queries. This is a sample of the
original query-ranking task, not the commonly used processed next-item benchmark
and not the competition leaderboard protocol.

| Method                      | NDCG@10 | Recall@10 | Token ILD@10 |
| --------------------------- | ------: | --------: | -----------: |
| Supplied candidate order    | 0.07315 |   0.13879 |      0.88424 |
| Historical popularity       | 0.15942 |   0.27105 |      0.90979 |
| Session repeat + popularity | 0.18677 |   0.30250 |      0.91101 |
| Learned relevance ranker    | 0.19463 |   0.31725 |      0.89664 |
| Fixed re-ranking            | 0.19209 |   0.31441 |      0.91377 |
| History-length gate         | 0.19219 |   0.31430 |      0.90978 |
| Session-diversity policy    | 0.19219 |   0.31430 |      0.90978 |
| Session-progression rule    | 0.19281 |   0.31468 |      0.90639 |

The fixed policy increased ILD by **0.01714 absolute / 1.91% relative**, while
NDCG decreased by **0.00255 absolute / 1.31% relative**, against the same
relevance ranker. Paired session-cluster bootstrap 95% intervals were +1.84% to
+1.98% for relative ILD change and -1.78% to -0.81% for relative NDCG change.

The selected diversity policy assigned weight 0.6 to both narrow and broad
histories. It therefore collapsed to the history-length gate. Against fixed
re-ranking, its NDCG difference was +0.000105 (95% interval -0.000903 to
+0.001048) and its ILD difference was -0.003997 (interval -0.004746 to
-0.003251). These results provide no evidence that the tested session-diversity
threshold improves on the simpler fixed policy. They do not rule out better
adaptive methods.

The operational session is the supplied session ID restricted to one event date,
with up to ten distinct clicked products strictly before the current query. Item
statistics use source queries before 31 March 2016. Ranker training uses 1–29
April, policy selection uses 1–14 May, and the test uses 16 May–1 June. One-day
gaps separate these periods. See `config/protocol.json` for the experiment
specification.

Local planning documents, slides, downloaded data and generated results are
excluded from Git. The result table above records the first experiment. Run the
commands below to generate the underlying tables, model and figures.

## Reproduce the RecBole experiment

Use Python 3.11 and `uv` from the repository root. RecBole 1.2.1 requires the
pinned compatible environment below. The original archive is about 356 MiB
and expands to about 901 MiB. Allow additional disk space for memory-mapped
features and models. A CPU laptop with approximately 16 GiB RAM can run the
experiment. Training time depends on the hardware.

```bash
uv venv --python 3.11 .venv-recbole
uv pip install --python .venv-recbole/bin/python -r requirements-recbole.lock.txt
uv pip install --python .venv-recbole/bin/python --no-deps .
.venv-recbole/bin/python -m session_rerank.run_experiment prepare
.venv-recbole/bin/python -m session_rerank.run_experiment train
.venv-recbole/bin/python -m session_rerank.run_experiment freeze
.venv-recbole/bin/python -m session_rerank.run_experiment confirm
.venv-recbole/bin/python -m pytest -q
```

A fresh checkout contains source and configuration only. The commands generate
`data/` and `results/` locally. If you already have a frozen local experiment,
reproduce in a separate copy/checkout to preserve it. The policy-freeze and test
stages refuse to overwrite their frozen/cache outputs. Reporting may be rerun
from the same cache without fitting or selecting anything. The model and
protocol hashes must match their frozen values before confirmation evaluation.
`run_experiment all` performs these stages in order from a fresh checkout.
`requirements.lock.txt` retains the original experiment's environment, while
`requirements-recbole.lock.txt` is the current reproducible environment.

RecBole builds the sequential dataset, samples training negatives, trains its
GRU4Rec implementation with BPR loss and selects the checkpoint using our full
query-candidate validation metric. The validation adapter retains the original
task rather than substituting leave-one-out or sampled-candidate evaluation.
The model's query prefixes use the same strict prior-click definition as the
first experiment. Scores fall back to recent popularity for empty/unseen
histories, and cold candidates remain in the evaluation.

The richer rankers use 40 features, or 48 with a separately evaluated view-prefix
family. Training aggregates use dates at least two days before each query.
Validation aggregates stop on 29 April, and confirmation aggregates stop on
14 May. Same-day view features assume a shared session-relative clock and use
strictly earlier view timeframes. The 40-feature control omits that assumption.
The new confirmation sample excludes every supplied session ID in the first
test sample. Its ID hash is fixed in `config/optimization_protocol.json`.

`data/source/manifest.json`, generated by the downloader, records original file
IDs, byte counts and SHA256 digests. The downloader checks the response format
and safe ZIP member paths. It downloads but never runs the competition baseline.
Public Drive availability may change. The original public folder is linked in
the manifest. No images or additional private datasets are needed.

## Source files and generated outputs

- `src/session_rerank/core.py`: timestamp-safe history selection, greedy
  re-ranking, binary NDCG/Recall and Jaccard distance.
- `src/session_rerank/prepare.py`: full-data audit, chronological samples,
  exposure statistics.
- `src/session_rerank/benchmark.py`: relevance features, LambdaMART training and
  per-query evaluation.
- `src/session_rerank/policies.py`: validation-only selection, frozen policies,
  paired bootstrap and charts.
- `config/protocol.json`: declared split dates, selection objective and policy
  families for the first experiment.
- `config/optimization_protocol.json`: second experiment and reserved sample.
- `config/recbole_gru4rec.yaml`: actual RecBole model/data/training configuration.
- `src/session_rerank/optimization_data.py`: embargoed statistics and feature caches.
- `src/session_rerank/recbole_model.py`: RecBole training and same-task validation.
- `src/session_rerank/optimize_ranker.py`: feature/model ablations.
- `src/session_rerank/optimization_report.py`: validation selection and confirmation.
- `src/session_rerank/run_experiment.py`: reproducible staged command-line entry point.

The following files are generated locally and excluded from Git:

- `results/data_audit.json`: full raw-data counts, candidate/category audit and
  sample identifiers.
- `results/test_summary.csv`: method means with 95% intervals.
- `results/test_paired_differences.csv`: paired absolute and relative
  differences.
- `results/test_slices.csv`: descriptive history/category/date breakdowns.
- `results/test_ranking_examples.json`: a small deterministic set of anonymized
  ranking examples for inspection.
- `results/optimization/validation_models.csv`: all second-experiment model scores.
- `results/optimization/frozen_selection.json`: selected models/policies and hashes.
- `results/optimization/confirmation_summary.csv`: confirmation means and intervals.
- `results/optimization/confirmation_paired_differences.csv`: paired comparisons.
- `results/optimization/figures/`: result figures generated from the measured data.

## Limits that belong in the presentation

The evaluation measures ranking of clicks recorded under the old system. It
cannot establish how clicks would change under a newly displayed list,
satisfaction, purchases, or removal of filter bubbles. Unclicked candidates are
unobserved relevance, not confirmed dislikes. Product names are anonymized
tokens, so ILD measures lexical variety, not independently validated semantic or
useful variety. The re-ranker directly optimizes this distance, making a
diversity increase alone insufficient evidence of utility. Only 27.7% of
evaluable test queries in the first experiment have two prior distinct observed
clicked items under the conservative definition. All conclusions are conditional on the sampled dates,
supplied candidates, visible labels, and the operational session definition.

The first test has been inspected. The second experiment reserves a new sample
from the same broader dates with disjoint supplied session IDs, and freezes
selection before evaluating it. This is fresh confirmation of outcomes in the
same period, not an independent dataset or a separate temporal replication.

## Sources

- [RecBole sequential recommendation](https://recbole.io/docs/get_started/started/sequential.html)
  and [GRU4Rec configuration](https://recbole.io/docs/user_guide/model/sequential/gru4rec.html).

- [Original CIKM Cup 2016 Track 2](https://competitions.codalab.org/competitions/11161)
  and
  [public data folder](https://drive.google.com/drive/folders/0B7XZSACQf0KdXzZFS21DblRxQ3c?resourcekey=0-3k4O5YlwnZf0cNeTZ5Y_Uw).
- Gharahighehi and Vens (2021),
  [Personalizing Diversity Versus Accuracy in Session-Based Recommender Systems](https://doi.org/10.1007/s42979-020-00399-2).
  The original adaptive SR/SKNN strategy differs from this project's
  adaptive-weight re-ranker.
- Wu, Yan and Si (2017),
  [Ensemble Methods for Personalized E-Commerce Search Challenge at CIKM Cup 2016](https://arxiv.org/abs/1708.04479).
  The initial slides attributed this title to different authors; use these
  authors in the final references.
- Yin, Fang, Sun and Ong,
  [Understanding Diversity in Session-Based Recommendation](https://arxiv.org/abs/2208.13453),
  with [published article DOI](https://doi.org/10.1145/3600226). This motivates
  testing simple accuracy-oriented methods and avoiding an assumed universal
  relevance/diversity trade-off.
- [4DBInfer, NeurIPS 2024, dataset appendix D.3](https://proceedings.neurips.cc/paper_files/paper/2024/file/2fd67447702c8eff5683dda507a1b0a2-Paper-Datasets_and_Benchmarks_Track.pdf)
  documents relative session timeframes and missing absolute session start
  times. We do not synthesize absolute session start times.

Implementation and analysis were prepared with AI assistance.
