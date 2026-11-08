#!/bin/bash
# Slurm array job: one sequence-model setting per task (see CLUSTER_GUIDE.md).
# Submit from the repository root:  sbatch scripts/cluster/sequence_grid.slurm
#
# ---- Adjust these for your cluster ------------------------------------------
#SBATCH --job-name=cp4285-seq
#SBATCH --array=1-12                 # one task per line of config/sequence_grid/grid_index.txt
#SBATCH --cpus-per-task=8            # keep equal to --threads given to make_sequence_grid.py
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/seq_%A_%a.out
##SBATCH --partition=CHANGE_ME       # remove one '#' and set your partition if required
##SBATCH --account=CHANGE_ME         # remove one '#' and set your project/account if required
# -----------------------------------------------------------------------------

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Python of the project environment (created with `uv sync` or pip, see the guide).
PYTHON="${PYTHON:-.venv/bin/python}"
CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" config/sequence_grid/grid_index.txt)
if [ -z "${CONFIG}" ]; then
    echo "No config for array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname): ${CONFIG}"
"${PYTHON}" main.py sequence-models --sequence-config "${CONFIG}"
