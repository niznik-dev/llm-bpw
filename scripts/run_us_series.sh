#!/usr/bin/env bash
# Run the 9-model cross-model probe on the US grid into today's runs folder.
# Idempotent + resumable: skips any model that already has a folder under
# runs/<today>/us_period/, so a failed/partial series can be re-run to fill gaps.
# Layout: runs/<date>/us_period/<model>_<noun>/ (--group keeps same-day experiments
# from colliding).
#
#   bash scripts/run_us_series.sh
#
# Flags per model mirror the Denmark series: streaming-only
# tiers get --stream; heavy reasoners get a big token budget + --allow-thinking.
set -u
GRID="data/grid_usa.csv"
GROUP="us_period"
RUNS="data/runs/$(date +%Y%m%d)/$GROUP"

MODELS=(
  "together/openai/gpt-oss-20b|--stream"
  "together/openai/gpt-oss-120b|--stream"
  "together/Qwen/Qwen3.6-Plus|--stream"
  "together/Qwen/Qwen3.7-Plus|--stream"
  "together/Qwen/Qwen3.7-Max|--stream"
  "together/MiniMaxAI/MiniMax-M3|--stream"
  "together/google/gemma-4-31B-it|--allow-thinking --max-tokens 1024"
  "together/nvidia/nemotron-3-ultra-550b-a55b|--stream --allow-thinking --max-tokens 4096"
  "together/zai-org/GLM-5.2|--stream --allow-thinking --max-tokens 4096"
)

for entry in "${MODELS[@]}"; do
  model="${entry%%|*}"; flags="${entry##*|}"; leaf="${model##*/}"
  if compgen -G "$RUNS/${leaf}_*" > /dev/null 2>&1; then
    echo "SKIP $leaf (already has a folder under $RUNS)"; continue
  fi
  echo "=== RUN $leaf : $flags ==="
  python src/run_probe.py --model "$model" --grid "$GRID" --group "$GROUP" $flags \
    || echo "!! FAILED: $leaf (continuing to next model)"
done

echo "=== US series done. Folders under $RUNS/ ==="
ls -1 "$RUNS" 2>/dev/null
