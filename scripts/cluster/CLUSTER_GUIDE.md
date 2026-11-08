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

## 3. Create the Python environment (Miniconda + pip)

Python 3.11 is required. `requirements.txt` pins the exact versions the
experiments ran with (CPU build of torch). If Miniconda is not installed yet:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

Create the environment and install the packages. Use the `conda-forge`
channel: on the NUS cluster the `defaults` channel did not offer Python 3.11.
The `&&` stops the chain if a step fails, so pip never installs into the wrong
Python:

```bash
conda create -n cp4285 -c conda-forge --override-channels python=3.11 -y && conda activate cp4285 && python --version && python -m pip install --no-cache-dir -r requirements.txt
```

`python --version` must print `Python 3.11.x`. Install on the login node, not in
a batch job: a batch job starts a fresh shell that may run the `base`
environment (Python 3.13), where pyarrow 14.0.2 has no ready-made package and
fails to build. Any batch script you write must itself run
`source ~/miniconda3/etc/profile.d/conda.sh` and `conda activate cp4285`, as the
project's job scripts do.

The project code itself (`src/session_rerank`) is not installed by pip;
`main.py` and the tests add `src/` to Python's search path themselves. (Before
commit after `1ee7b9e`, every job failed with `No module named 'session_rerank'`;
with an older checkout, submit with `--export=ALL,PYTHONPATH=$HOME/cp4285-g02/src`.)

**Disk quota.** The environment needs about 1 GB with CPU torch (about 3 GB with
the CUDA build) and the copied data about 0.9 GB. Check with `quota -s`; clear
caches with `conda clean --all -y` and `rm -rf ~/.cache/pip`. If the home quota
is too small, create the environment on scratch storage with
`conda create -p /path/to/scratch/envs/cp4285 ...` and submit with
`--export=ALL,CONDA_ENV=/path/to/scratch/envs/cp4285`.

The job scripts activate `cp4285` from `~/miniconda3`. If Miniconda lives
elsewhere or the environment has another name, set them when submitting, e.g.
`sbatch --export=ALL,CONDA_SH=/path/to/etc/profile.d/conda.sh,CONDA_ENV=myenv ...`.

## 4. Generate the grid and check the inputs

With the environment active:

```bash
python scripts/cluster/make_sequence_grid.py --threads 8
```

```bash
python scripts/cluster/check_inputs.py
```

The check must print `Ready.` before you submit.

## 5. Submit

Open the job script, set the partition/queue and account lines if your cluster
requires them, and keep `--cpus-per-task` equal to `--threads` above. The CPU
scripts write their logs to `logs/`, which must exist before submitting
(Slurm opens the log file before the script starts):

```bash
mkdir -p logs
```

Slurm (12 CPU tasks in parallel):

```bash
sbatch scripts/cluster/sequence_grid.slurm
```

Slurm on one GPU (all 12 settings one after another; see "Using a GPU" below):

```bash
sbatch scripts/cluster/sequence_grid_gpu.sh
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
python scripts/cluster/collect_sequence_grid.py
```

It ranks the settings by validation NDCG@10, shows the difference from
item-kNN and from the frozen home-page policy with 95% intervals, flags settings
that were still improving at their last epoch, and lists any setting without a
finished run. Copy the results back to the laptop:

```bash
rsync -av USER@CLUSTER:cp4285-g02/results/sequence_grid/ results/sequence_grid/
```

Then record the outcome in EXPERIMENT_LOG.md.

## Using a GPU

`scripts/cluster/sequence_grid_gpu.sh` runs all 12 settings on one GPU (the
header uses the NUS `gpu` partition and `--gpus=TYPE:1`). Any GPU type is
enough: the models need well under 2 GB of GPU memory, so a smaller type than
`h100-96` usually starts sooner (`sinfo -o "%P %G"` lists the types). The models
train on the GPU automatically when PyTorch can see one (`device` in the config:
`auto`, `cpu` or `cuda`); scoring the 20,000 cases still runs on the CPU.

`requirements.txt` installs the CPU build of torch. For the GPU, replace it once
with the CUDA build of the same version (about 2.5 GB more; mind the quota):

```bash
conda activate cp4285
```

```bash
python -m pip uninstall -y torch
```

```bash
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
```

The job checks this first and stops with a message if PyTorch cannot see the
GPU, so no GPU time is wasted. Its log is `seq_grid_gpu_JOBID.out` in the
repository root, and it ends by running the collector.
