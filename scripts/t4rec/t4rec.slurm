#!/bin/bash
# Transformers4Rec on the home-page cases (see scripts/t4rec/T4REC_GUIDE.md).
# Three steps, two conda environments:
#   1. cp4285      export training sessions and cases     (main.py export-sequences)
#   2. cp4285-t4r  train XLNet and write predictions      (scripts/t4rec/train_t4rec.py)
#   3. cp4285      score and compare with other methods  (main.py score-external)
# Submit from the repository root:  sbatch scripts/t4rec/t4rec.slurm
# Smoke test first:                 sbatch --export=ALL,SMOKE=1 scripts/t4rec/t4rec.slurm
#
# ---- Adjust these for your cluster ------------------------------------------
#SBATCH --job-name=cp4285-t4rec
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/t4rec_%j.out
##SBATCH --gres=gpu:1                # remove one '#' for a GPU (and use the CUDA torch build)
##SBATCH --partition=CHANGE_ME       # remove one '#' and set your partition if required
##SBATCH --account=CHANGE_ME         # remove one '#' and set your project/account if required
# -----------------------------------------------------------------------------

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
set +u
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
set -u

activate() { set +u; conda activate "$1"; set -u; }

activate "${CONDA_ENV:-cp4285}"
python main.py export-sequences --t4rec-config config/t4rec.json
EXPORT=$(cat results/t4rec/latest_export.txt)
echo "Export folder: ${EXPORT}"

activate "${T4REC_ENV:-cp4285-t4r}"
if [ "${SMOKE:-0}" = "1" ]; then
    python scripts/t4rec/train_t4rec.py --export "${EXPORT}" --smoke
    echo "Smoke run finished: ${EXPORT}/predictions_smoke.parquet (not scored)."
    exit 0
fi
python scripts/t4rec/train_t4rec.py --export "${EXPORT}"

activate "${CONDA_ENV:-cp4285}"
python main.py score-external --export "${EXPORT}" --label t4rec_xlnet --t4rec-config config/t4rec.json
