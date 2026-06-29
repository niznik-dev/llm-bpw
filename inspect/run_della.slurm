#!/bin/bash
#SBATCH --job-name=llm-bpw-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=msalganik
#SBATCH --constraint=gpu80
#SBATCH --output=logs/slurm-%j.out
#
# llm-bpw Inspect probe on della. Usage: sbatch inspect/run_della.slurm

set -euo pipefail

CONDA_ENV="cruijff"
MODEL_NAME="Qwen3-14B"
MODEL_PATH="/scratch/gpfs/MSALGANIK/pretrained-llms/${MODEL_NAME}"
GRID="${PWD}/data/grid.csv"
PROMPT="baseline"
LOG_DIR="${PWD}/logs"

module purge
module load anaconda3/2025.6
conda activate "${CONDA_ENV}"

export HF_HUB_OFFLINE=1                    # no network lookups on compute nodes

mkdir -p "${LOG_DIR}"

/usr/bin/time -p inspect eval src/inspect_task.py@bpw \
    --model "hf/${MODEL_NAME}" \
    -M model_path="${MODEL_PATH}" \
    -T grid_path="${GRID}" \
    -T prompt="${PROMPT}" \
    --log-dir "${LOG_DIR}"

# Convert the newest .eval log to data/results.csv for plotting.
python src/inspect_to_csv.py --latest --log-dir "${LOG_DIR}" --out data/results.csv

echo "Done. Plot with: python src/plot_results.py --results data/results.csv"
