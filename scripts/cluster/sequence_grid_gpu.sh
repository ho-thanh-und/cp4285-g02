#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=cp4285_seq_grid
#SBATCH --gpus=h100-96:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=seq_grid_gpu_%j.out
#
# All 12 SASRec / GRU4Rec settings, one after another on one GPU.
# Submit from the repository root:  sbatch scripts/cluster/sequence_grid_gpu.sh
#
# GPU: any type works. These models need well under 2 GB of GPU memory, so a
# smaller GPU type (list them with: sinfo -o "%P %G") usually starts sooner.
# Memory and time: 32 GB and 8 hours are ample; most of each setting's time is
# CPU scoring of the 20,000 cases, not training.
#
# Needs the CUDA build of torch in the environment (requirements.txt installs
# the CPU build). One-time setup, see CLUSTER_GUIDE.md "Using a GPU".
# Re-submitting skips settings that already have a finished run.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
set +u
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-cp4285}"
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

nvidia-smi || true
python -c "import sys, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 'torch cannot see the GPU: install the CUDA build (CLUSTER_GUIDE.md, Using a GPU)')"

if [ ! -f config/sequence_grid/grid_index.txt ]; then
    python scripts/cluster/make_sequence_grid.py --threads "${SLURM_CPUS_PER_TASK}"
fi
python scripts/cluster/check_inputs.py

while read -r CONFIG; do
    [ -z "${CONFIG}" ] && continue
    OUT=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['output_directory'])" "${CONFIG}")
    if ls "${OUT}"/*_validation/summary.csv >/dev/null 2>&1; then
        echo "=== skip (already finished): ${CONFIG}"
        continue
    fi
    echo "=== $(date '+%H:%M:%S') ${CONFIG}"
    python main.py sequence-models --sequence-config "${CONFIG}" || echo "=== FAILED: ${CONFIG}"
done < config/sequence_grid/grid_index.txt

python scripts/cluster/collect_sequence_grid.py
