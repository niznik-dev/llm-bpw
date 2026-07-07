#!/usr/bin/env bash
# Run the Denmark COHORT probe (thread B) across the 7-model subset into
# runs/<date>/dk_cohort_think{ON,OFF}/ (via --group). Idempotent + resumable: skips
# any model that already has a folder, so a failed/partial series can be re-run to
# fill gaps.
#
#   bash scripts/run_dk_cohort.sh              # thinking ON (default)
#   THINKING=off bash scripts/run_dk_cohort.sh # thinking OFF ablation
#
# THINKING defaults ON — it lowers RMSE vs HFD for all 7 models (the headline
# ablation), so it is the standard; OFF is the deliberate comparison arm. The state
# is ALWAYS in the folder name (never implicit) because the ablation IS the finding.
# --thinking on|off makes run_probe look up the per-model control in
# model_meta.yaml (thinking_on/thinking_off) and pass it as
# GenerateConfig(extra_body=...). Vendor surfaces differ:
#   Qwen / Gemma / Nemotron -> chat_template_kwargs.enable_thinking
#   GLM-5.2                 -> reasoning.enabled
#   MiniMax-M3              -> thinking.type   (NB: degenerates to token-salad OFF)
#
# STREAMING is driven by model_meta.yaml `stream:` (verified 2026-07-02: Qwen
# *-Plus/Max required, MiniMax-M3 forbidden) — run_probe applies it, so this runner
# no longer passes --stream per model.
#
# Cohort grid = data/grids/grid_denmark_cohort.csv (cohorts 1933/1945/1955/1974,
# Shared-A). Baseline for scoring = data/baselines/hfd_denmark_cohort_asfr.csv.
set -u
# THINKING=on (default) | off. Budget block is kept byte-identical to
# run_us_series.sh (one standard). ON keeps a generous 50k ceiling — most models
# finish ~1k; the tail past 50k is GLM's genuinely-unbounded ~2-3%, which retry_nulls
# owns at a higher ceiling. OFF clamps to 32 tokens: 6/7 models emit a bare decimal
# in <=6 tok (verified 2026-07-02 logs), so 32 is ~5x headroom with zero truncation
# on real answers — it just caps MiniMax's OFF token-salad early instead of letting
# it burn to the 512 default.
THINKING="${THINKING:-on}"
GRID="data/grids/grid_denmark_cohort.csv"
if [ "$THINKING" = "on" ]; then
  GROUP="dk_cohort_thinkON"
  BUDGET="--max-tokens 50000"   # GLM/nemotron tail; most finish ~1k
else
  GROUP="dk_cohort_thinkOFF"
  BUDGET="--max-tokens 32"      # bare decimal is <=6 tok; caps MiniMax salad early
fi
RUNS="data/runs/$(date +%Y%m%d)/$GROUP"   # for the skip-check + final listing
COMMON="--prompt cohort_baseline --grid $GRID --thinking $THINKING --group $GROUP $BUDGET"
echo "### thinking=$THINKING -> $RUNS ###"

# "<together id>|<extra flags>" — the per-model flag seam, same as run_us_series
# (none needed here: thinking + token budget are set globally above via $COMMON).
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
  echo "=== RUN $leaf : ${flags:-(no stream)} ==="
  python src/run_probe.py --model "$model" $COMMON $flags \
    || echo "!! FAILED: $leaf (continuing to next model)"
done

echo "=== DK cohort series done. Folders under $RUNS/ ==="
ls -1 "$RUNS" 2>/dev/null
