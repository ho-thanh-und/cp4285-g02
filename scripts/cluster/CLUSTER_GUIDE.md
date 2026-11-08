# Running the SASRec / GRU4Rec grid on a compute cluster

This runs 12 sequence-model settings on February–April training data (about
207,000 sessions, 292,000 training positions) as one job array: each task
trains one model and evaluates it on the same 20,000 validation-week home-page
cases as every other home-page method (EXPERIMENT_LOG.md §6g, §6j). Each task
needs about 8 CPU cores, 16 GB memory and an estimated 30–90 minutes; a GPU is
not used.

| Settings | Varied | Fixed |
|---|---|---|
| SASRec × 8 | embedding 64 / 128, attention blocks 1 / 2, dropout 0.2 / 0.5 | 30 epochs, learning rate 0.001 |
| GRU4Rec × 4 | embedding 64 / 128, learning rate 0.001 / 0.003 | 60 epochs, dropout 0.2 |

Every setting keeps the epoch with the best recall@10 on the validation check
subset. The grid is chosen on validation only; whichever setting wins must be
frozen before it is run on the selection week.

## 1. Get the code onto the cluster

Push the branch from your laptop first (the cluster clones from GitHub):

```bash
git push origin refactor/readable-main-pipeline
```

On the cluster login node:

```bash
git clone https://github.com/ho-thanh-und/cp4285-g02.git
cd cp4285-g02
git checkout refactor/readable-main-pipeline
```

## 2. Copy the inputs that are not in Git

Results and data are ignored by Git. From the laptop's `cp4285-g02` folder,
copy the raw CSVs (about 900 MB) and three small result files. Replace
`USER@CLUSTER` and the target path. Check that the dataset's terms allow
copying it to the cluster.

```bash
rsync -av data/raw/ USER@CLUSTER:cp4285-g02/data/raw/
```

```bash
rsync -avR results/learning_curve_v2/queries_50000/products.parquet results/home_page/20261108T095146Z_validation/per_case.parquet USER@CLUSTER:cp4285-g02/
```

(`-R` keeps the `results/...` folder structure.)

## 3. Create the Python environment

Python 3.11 is required. With `uv` (recommended, uses the project's lock file):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

If `uv` is not allowed, use a virtual environment with the packages the grid needs:

```bash
python3.11 -m venv .venv
.venv/bin/pip install "numpy<2" "pandas>=2.2,<3" "scipy>=1.12,<1.13" "pyarrow>=14,<15" lightgbm psutil "matplotlib>=3.9,<3.10" "torch==2.5.1" --extra-index-url https://download.pytorch.org/whl/cpu
```

## 4. Generate the grid and check the inputs

```bash
.venv/bin/python scripts/cluster/make_sequence_grid.py --threads 8
```

```bash
.venv/bin/python scripts/cluster/check_inputs.py
```

The check must print `Ready.` before you submit.

## 5. Submit

Open the job script, set the partition/queue and account lines if your cluster
requires them, and keep `--cpus-per-task` equal to `--threads` above.

Slurm:

```bash
sbatch scripts/cluster/sequence_grid.slurm
```

PBS:

```bash
qsub scripts/cluster/sequence_grid.pbs
```

Monitor with `squeue -u $USER` (Slurm) or `qstat -u $USER` (PBS). Each task
writes its log to `logs/`; training prints one line per epoch.

## 6. Collect the results

When all tasks have finished:

```bash
.venv/bin/python scripts/cluster/collect_sequence_grid.py
```

It ranks the settings by validation NDCG@10, shows the difference from
item-kNN and from the frozen home-page policy with 95% intervals, flags settings
that were still improving at their last epoch, and lists any setting without a
finished run. Copy the results back to the laptop:

```bash
rsync -av USER@CLUSTER:cp4285-g02/results/sequence_grid/ results/sequence_grid/
```

Then record the outcome in EXPERIMENT_LOG.md.
