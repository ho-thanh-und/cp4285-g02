#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --job-name=cp4285_t4rec
#SBATCH --gpus=h100-96:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=t4rec_gpu_%j.out
#
# Transformers4Rec on the home-page cases, on one GPU (see T4REC_GUIDE.md):
#   1. cp4285      export training sessions and cases     (main.py export-sequences)
#   2. cp4285-t4r  train XLNet and write predictions      (scripts/t4rec/train_t4rec.py)
#   3. cp4285      score and compare with other methods  (main.py score-external)
# Submit from the repository root:  sbatch scripts/t4rec/t4rec_gpu.sh
# Smoke test first:                 sbatch --export=ALL,SMOKE=1 scripts/t4rec/t4rec_gpu.sh
#
# GPU: any type works; this XLNet (64-dimensional, 2 layers) needs a few GB at
# most. The cp4285-t4r environment needs the CUDA build of torch 2.1.2 (see
# T4REC_GUIDE.md, "Using a GPU"); otherwise training falls back to the CPU.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
set +u
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
set -u
activate() { set +u; conda activate "$1"; set -u; }

nvidia-smi || true

activate "${CONDA_ENV:-cp4285}"
python main.py export-sequences --t4rec-config config/t4rec.json
EXPORT=$(cat results/t4rec/latest_export.txt)
echo "Export folder: ${EXPORT}"

activate "${T4REC_ENV:-cp4285-t4r}"
python -c "import sys, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 'torch in the Transformers4Rec environment cannot see the GPU: install the CUDA build (T4REC_GUIDE.md, Using a GPU)')"
if [ "${SMOKE:-0}" = "1" ]; then
    python scripts/t4rec/train_t4rec.py --export "${EXPORT}" --smoke
    echo "Smoke run finished: ${EXPORT}/predictions_smoke.parquet (not scored)."
    exit 0
fi
python scripts/t4rec/train_t4rec.py --export "${EXPORT}"

activate "${CONDA_ENV:-cp4285}"
python main.py score-external --export "${EXPORT}" --label t4rec_xlnet --t4rec-config config/t4rec.json
