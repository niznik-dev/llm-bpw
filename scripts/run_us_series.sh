#!/usr/bin/env bash
# Run the US PERIOD probe across the 7-model subset into
# runs/<date>/us_period_think{ON,OFF}/ (via --group). Idempotent + resumable: skips
# any model that already has a folder, so a failed/partial series can be re-run to
# fill gaps.
#
#   bash scripts/run_us_series.sh              # thinking ON (default)
#   THINKING=off bash scripts/run_us_series.sh # thinking OFF ablation
#
# THINKING is the same standard as run_dk_cohort.sh: defaults ON (lower RMSE), state
# ALWAYS in the folder name, driven by the model_meta.yaml registry (thinking_on/off)
# via --thinking. The legacy --allow-thinking soft-switch and the per-model token
# budgets are gone — everything now flows through $COMMON below.
#
# Roster matches DK: the 7-model subset (gpt-oss 20b/120b retired — no per-model
# thinking control in the registry, so they can't honor --thinking on|off).
#
# Streaming comes from model_meta.yaml `stream:` (run_probe applies it per model).
set -u
# THINKING=on (default) | off. Budget block is kept byte-identical to
# run_dk_cohort.sh (one standard). ON keeps a generous 50k ceiling — most models
# finish ~1k; the tail past 50k is GLM's genuinely-unbounded ~2-3%, which retry_nulls
# owns at a higher ceiling. OFF clamps to 32 tokens: 6/7 models emit a bare decimal
# in <=6 tok (verified 2026-07-02 logs), so 32 is ~5x headroom with zero truncation
# on real answers — it just caps MiniMax's OFF token-salad early instead of letting
# it burn to the 512 default.
THINKING="${THINKING:-on}"
GRID="data/grids/grid_usa.csv"
if [ "$THINKING" = "on" ]; then
  GROUP="us_period_thinkON"
  BUDGET="--max-tokens 50000"   # GLM/nemotron tail; most finish ~1k
else
  GROUP="us_period_thinkOFF"
  BUDGET="--max-tokens 32"      # bare decimal is <=6 tok; caps MiniMax salad early
fi
RUNS="data/runs/$(date +%Y%m%d)/$GROUP"
COMMON="--grid $GRID --thinking $THINKING --group $GROUP $BUDGET"
echo "### thinking=$THINKING -> $RUNS ###"

# "<together id>|<extra flags>" — the per-model flag seam (none needed now: thinking
# + budget are set globally above via $COMMON). Roster identical to run_dk_cohort.sh.
MODELS=(
  "together/Qwen/Qwen3.6-Plus|"
  "together/Qwen/Qwen3.7-Plus|"
  "together/Qwen/Qwen3.7-Max|"
  "together/google/gemma-4-31B-it|"
  "together/nvidia/nemotron-3-ultra-550b-a55b|"
  "together/zai-org/GLM-5.2|"
  "together/MiniMaxAI/MiniMax-M3|"
)

for entry in "${MODELS[@]}"; do
  model="${entry%%|*}"; flags="${entry##*|}"; leaf="${model##*/}"
  if compgen -G "$RUNS/${leaf}_*" > /dev/null 2>&1; then
    echo "SKIP $leaf (already has a folder under $RUNS)"; continue
  fi
  echo "=== RUN $leaf : ${flags:-(common only)} ==="
  python src/run_probe.py --model "$model" $COMMON $flags \
    || echo "!! FAILED: $leaf (continuing to next model)"
done

echo "=== US series done. Folders under $RUNS/ ==="
ls -1 "$RUNS" 2>/dev/null
