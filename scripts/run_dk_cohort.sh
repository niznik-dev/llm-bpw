#!/usr/bin/env bash
# Run the Denmark COHORT probe (thread B) across the 7-model subset into a dated
# cohort runs folder. Idempotent + resumable: skips any model that already has a
# folder, so a failed/partial series can be re-run to fill the gaps.
#
#   bash scripts/run_dk_cohort.sh          # from the repo root
#
# THINKING IS FORCED OFF for every model, verified empirically (2026-07-02) to
# suppress reasoning (4-5 output tokens/call, 0 nulls). --thinking off makes
# run_probe look up the per-model control in model_meta.yaml (thinking_off) and
# pass it as GenerateConfig(extra_body=...). Vendor surfaces differ:
#   Qwen / Gemma / Nemotron -> chat_template_kwargs.enable_thinking=false
#   GLM-5.2                 -> reasoning.enabled=false
#   MiniMax-M3              -> thinking.type=disabled
#
# STREAMING QUIRK (also verified): Qwen *requires* --stream (else 400
# streaming_required); MiniMax-M3 *hangs* with --stream, so it runs non-streamed.
# gpt-oss dropped (weak boom-readers, not worth the cost).
#
# Cohort grid = data/grids/grid_denmark_cohort.csv (cohorts 1933/1945/1955/1974,
# Shared-A). Baseline for scoring = data/baselines/hfd_denmark_cohort_asfr.csv.
set -u
# THINKING=off (default) | on. The ON run uses a generous token ceiling so we can
# LEARN each model's real budget from the logs (thinking OFF needs almost none).
THINKING="${THINKING:-off}"
GRID="data/grids/grid_denmark_cohort.csv"
if [ "$THINKING" = "on" ]; then
  RUNS="data/runs/$(date +%Y%m%d)_dk_cohort_thinkON"
  BUDGET="--max-tokens 50000"   # catches GLM's runaway cells (most finish ~1k)
else
  RUNS="data/runs/$(date +%Y%m%d)_dk_cohort"
  BUDGET=""
fi
COMMON="--prompt cohort_baseline --grid $GRID --thinking $THINKING --runs-dir $RUNS $BUDGET"
echo "### thinking=$THINKING -> $RUNS ###"

# "<together id>|<extra flags>" — streaming only where verified.
MODELS=(
  "together/Qwen/Qwen3.6-Plus|--stream"
  "together/Qwen/Qwen3.7-Plus|--stream"
  "together/Qwen/Qwen3.7-Max|--stream"
  "together/google/gemma-4-31B-it|--stream"
  "together/nvidia/nemotron-3-ultra-550b-a55b|--stream"
  "together/zai-org/GLM-5.2|"
  "together/MiniMaxAI/MiniMax-M3|"
)

for entry in "${MODELS[@]}"; do
  model="${entry%%|*}"; flags="${entry##*|}"; leaf="${model##*/}"
  if compgen -G "$RUNS/${leaf}_*" > /dev/null 2>&1; then
    echo "SKIP $leaf (already has a folder under $RUNS)"; continue
  fi
  echo "=== RUN $leaf : ${flags:-(no stream)} ==="
  python src/run_probe.py --model "$model" $COMMON $flags \
    || echo "!! FAILED: $leaf (continuing to next model)"
done

echo "=== DK cohort series done. Folders under $RUNS/ ==="
ls -1 "$RUNS" 2>/dev/null
