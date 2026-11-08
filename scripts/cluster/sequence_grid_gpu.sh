#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --job-name=cp4285_seq_grid
#SBATCH --gpus=a100-40:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=seq_grid_gpu_%j.out
#
# The 12 SASRec / GRU4Rec settings on ONE GPU, several at the same time.
# Submit from the repository root:  sbatch scripts/cluster/sequence_grid_gpu.sh
# Run 3 at a time instead of 4:     sbatch --export=ALL,PARALLEL=3 scripts/cluster/sequence_grid_gpu.sh
# Only settings whose name contains a word (e.g. on a second GPU):
#                                   sbatch --export=ALL,ONLY=gru4rec scripts/cluster/sequence_grid_gpu.sh
#
# Why parallel on one GPU: each model needs well under 2 GB of GPU memory, and
# most of each setting's time is CPU work (the per-epoch check and the final
# scoring of 20,000 cases). So PARALLEL settings share the GPU and each gets
# cpus-per-task / PARALLEL CPU threads. Each setting writes its own log to
# grid_logs/NAME.log. Re-submitting skips settings that already finished.
#
# Needs the CUDA build of torch in the environment (CLUSTER_GUIDE.md, "Using a GPU").

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
set +u
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-cp4285}"
set -u

PARALLEL="${PARALLEL:-4}"
ONLY="${ONLY:-}"
THREADS=$(( SLURM_CPUS_PER_TASK / PARALLEL ))
[ "${THREADS}" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
mkdir -p grid_logs

nvidia-smi || true
python -c "import sys, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 'torch cannot see the GPU: install the CUDA build (CLUSTER_GUIDE.md, Using a GPU)')"

# Same settings as before; only the thread count per setting changes.
python scripts/cluster/make_sequence_grid.py --threads "${THREADS}"
python scripts/cluster/check_inputs.py
echo "Running ${PARALLEL} settings at a time, ${THREADS} CPU threads each${ONLY:+, only names containing '${ONLY}'}."

run_setting() {
    local config="$1" name
    name=$(basename "${config}" .json)
    echo "=== $(date '+%H:%M:%S') start ${name}"
    if python main.py sequence-models --sequence-config "${config}" > "grid_logs/${name}.log" 2>&1; then
        echo "=== $(date '+%H:%M:%S') done  ${name}"
    else
        echo "=== $(date '+%H:%M:%S') FAILED ${name} (see grid_logs/${name}.log)"
    fi
}

while read -r CONFIG; do
    [ -z "${CONFIG}" ] && continue
    if [ -n "${ONLY}" ] && [[ "${CONFIG}" != *"${ONLY}"* ]]; then
        continue
    fi
    OUT=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['output_directory'])" "${CONFIG}")
    if ls "${OUT}"/*_validation/summary.csv >/dev/null 2>&1; then
        echo "=== skip (already finished): ${CONFIG}"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]; do
        wait -n || true
    done
    run_setting "${CONFIG}" &
done < config/sequence_grid/grid_index.txt
wait

python scripts/cluster/collect_sequence_grid.py
