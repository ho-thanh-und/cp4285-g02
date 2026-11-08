# Running Transformers4Rec on the home-page cases

Transformers4Rec (NVIDIA Merlin) trains an XLNet session model here and is
compared with item-kNN, the frozen home-page policy, the session two-tower and
any SASRec/GRU4Rec run on exactly the same cases and metrics
(EXPERIMENT_LOG.md §6g, §6j).

It cannot share the main environment: its last release (23.12.0, January 2024)
needs Hugging Face `transformers` < 4.31, and Merlin needs pandas < 1.6 and
Python ≤ 3.10. So it runs in its own conda environment, connected through files:

| Step | Environment | Command | Writes |
|---|---|---|---|
| 1. Export | `cp4285` | `python main.py export-sequences` | `results/t4rec/RUN/` with `train_sessions.parquet`, `cases.parquet`, `vocabulary.parquet`, `export.json` |
| 2. Train and predict | `cp4285-t4r` | `python scripts/t4rec/train_t4rec.py --export RUN` | `predictions.parquet`, `t4rec_training_log.csv`, `t4rec_xlnet.pt` |
| 3. Score | `cp4285` | `python main.py score-external --export RUN --label t4rec_xlnet` | `RUN/scored_t4rec_xlnet/` with summary and differences |

Settings (training window, model size, epochs, top-N) are in
`config/t4rec.json`; by default training uses February–April, like
`config/sequence_models_feb_apr.json`.

**Status:** steps 1 and 3 are tested in this repository. Step 2 follows the
library's documentation but has not been run (the library cannot be installed
on the development laptop). Start with the smoke test below and expect to adjust
small API details in `build_model`, `scores_from_output` or the training call.

## 1. Prerequisites

Steps 1–3 of `scripts/cluster/CLUSTER_GUIDE.md`: the repository, `data/raw`,
the two result files, Miniconda and the `cp4285` environment.

## 2. Create the Transformers4Rec environment

```bash
conda create -n cp4285-t4r python=3.10 -y
```

```bash
conda activate cp4285-t4r
```

```bash
python -m pip install -r scripts/t4rec/requirements-t4rec.txt
```

Check that it imports:

```bash
python -c "import transformers4rec.torch as tr; from merlin.schema import Schema; print('ok')"
```

If pip reports a version conflict, keep the pins for `transformers4rec`,
`merlin-*`, `pandas` and `transformers` and relax the others. For a GPU, delete
the `--extra-index-url` line in the requirements file before installing.

## 3. Smoke test

Trains for one epoch on 2,000 sessions and predicts 500 cases; nothing is
scored. On a login node or an interactive session:

```bash
conda activate cp4285
```

```bash
python main.py export-sequences
```

```bash
conda activate cp4285-t4r
```

```bash
python scripts/t4rec/train_t4rec.py --export "$(cat results/t4rec/latest_export.txt)" --smoke
```

Or as a job:

```bash
sbatch --export=ALL,SMOKE=1 scripts/t4rec/t4rec.slurm
```

It should end with `wrote predictions_smoke.parquet`. If it fails inside the
library, the traceback points to the call to adjust.

## 4. Full run

Set the partition/account lines in `scripts/t4rec/t4rec.slurm` if needed (and
`--gres=gpu:1` for a GPU), then:

```bash
sbatch scripts/t4rec/t4rec.slurm
```

The job runs all three steps, switching environments (`CONDA_ENV`, default
`cp4285`; `T4REC_ENV`, default `cp4285-t4r`). The log in `logs/` shows one line
per epoch and, at the end, the scored NDCG@10 and recall@10.

## 5. Results

`RUN/scored_t4rec_xlnet/summary.csv` lists every method by clicks so far, and
`differences.csv` gives Transformers4Rec minus each other method with 95%
session-bootstrap intervals. To include a SASRec/GRU4Rec run in the same table,
add its folder to `compare_sequence_runs` in `config/t4rec.json` and rerun
step 3. Copy the run folder back and record the outcome in EXPERIMENT_LOG.md.
As for every model, these are validation results; any model the home page
switches to must be frozen before the selection week.
