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
# llm-bpw — Inspect AI fertility probe on della (large HF model).
#
# Mirrors the cruijff_kit Inspect pattern: the hf/ provider loads the model
# weights in-process from a /scratch path; Inspect handles batching + logging.
# The .eval log it writes is turned into a plot-ready results.csv afterwards
# (see inspect_to_csv.py) so the SAME plotters work as on the laptop.
#
# Usage:
#   cd /scratch/gpfs/MSALGANIK/<you>/GitHub/llm-bpw
#   sbatch inspect/run_della.slurm
#
# Sizing notes:
#   - A ~32B model in bf16 (~64 GB) fits on one 80 GB A100 (gpu80). Keep gpu:1.
#   - A ~72B model (~144 GB) needs two: set --gres=gpu:2 above and add
#     `-M device=auto` to the inspect call below to shard across them.

set -euo pipefail

# --- knobs ------------------------------------------------------------------
CONDA_ENV="cruijff"                       # has inspect_ai + torch/transformers
MODEL_NAME="Qwen3-14B"                     # HF model dir name (provider = hf/)
MODEL_ROOT="/scratch/gpfs/MSALGANIK/pretrained-llms"
MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
GRID="${PWD}/data/grid.csv"               # absolute: Inspect runs from elsewhere
PROMPT="era_prior"                        # prompt variant from probe.py
LOG_DIR="${PWD}/logs"
# Qwen3 is a reasoning model; inspect_task.py disables thinking by default
# (disable_thinking=True -> appends /no_think). Pass -T disable_thinking=false
# in the inspect call below to let it think.
# ----------------------------------------------------------------------------

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
