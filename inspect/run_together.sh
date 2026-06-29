#!/usr/bin/env bash
# llm-bpw Inspect probe via Together.ai — hosted models, no local GPU.
#
# Reuses the same task (src/inspect_task.py) as the della run; only the provider
# changes. Reads TOGETHER_API_KEY from a gitignored .env that Inspect loads
# automatically (so the key never lands in your shell env). Copy .env.example
# to .env and fill it in first.
#
# Usage:
#   bash inspect/run_together.sh
#   MODEL=together/Qwen/Qwen2.5-72B-Instruct-Turbo PROMPT=era_prior bash inspect/run_together.sh
set -euo pipefail

MODEL="${MODEL:-together/Qwen/Qwen2.5-72B-Instruct-Turbo}"
GRID="${GRID:-$PWD/data/grid.csv}"
PROMPT="${PROMPT:-baseline}"
LOG_DIR="${LOG_DIR:-$PWD/logs}"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy .env.example to .env and add TOGETHER_API_KEY." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

inspect eval src/inspect_task.py@bpw \
  --model "$MODEL" \
  -T grid_path="$GRID" \
  -T prompt="$PROMPT" \
  --log-dir "$LOG_DIR"

python src/inspect_to_csv.py --latest --log-dir "$LOG_DIR" --out data/results.csv
echo "Done. Plot: python src/plot_results.py --results data/results.csv"
