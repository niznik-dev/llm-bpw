# llm-bpw

**B**irths **p**er **w**oman: probing what language models believe about human
fertility, zero-shot.

Given a synthetic profile — birth cohort, age, sex, country — we ask a model to
estimate that profile's **age-specific fertility rate**: the expected number of
births per woman at that age in the coming year. Demographers call these
age-profiles "fertility schedules." The aim is to map what models believe across
cohorts and countries, with no training or fine-tuning.

This is a *rate*, not a yes/no probability — it can exceed 1 (twins, or births in
January and December) — so outputs are recorded as rates.

## Status

End-to-end pipeline working:

```
[1] generate grid → [2] prompt + query → [3] parse → [4] plot
    grid.csv  ✅       Inspect AI, hf/ ✅   results.csv ✅   *.png ✅
```

Models run through **Inspect AI's `hf/` provider** — a small Qwen on the laptop
(dev/smoke-test) and a large Qwen on della — so only the model changes between
runs. The prompt variants and answer parser live in `src/probe.py` so those
runs stay comparable.

## Input fields

The grid CSV holds exactly these four columns, the fields handed to the model:

| column | meaning | values |
|---|---|---|
| `year_of_birth` | birth cohort | user-supplied (default: decadal 1950–2000) |
| `age` | age in years | 0–55 inclusive (configurable) |
| `sex` | sex at birth | Male, Female |
| `country` | country | Denmark (expandable) |

The grid is the full cartesian product of these fields.

## Usage

Install deps (`pip install -r requirements.txt`): pandas + matplotlib for the
grid/plots, and Inspect AI + transformers/torch for the model probe.

```bash
# [1] Build the input grid (deterministic). Female-only by default.
python src/generate_grid.py --birth-years 1920 1960 1990 2024

# [2] Probe a model. Laptop dev on a small Qwen (auto-downloads from HF hub);
#     era_prior is the prompt variant that best surfaces cohort differences.
inspect eval src/inspect_task.py@bpw \
    --model hf/Qwen/Qwen3-4B \
    -T grid_path=$PWD/data/grid.csv -T prompt=era_prior \
    -M device=mps --log-dir logs

# [3] Turn the newest .eval log into a plot-ready results.csv
python src/inspect_to_csv.py --latest --out data/results.csv

# [4] Plot the fertility schedules (one curve per cohort)
python src/plot_results.py
```

The large-model run on della uses the same task via SLURM — see
`inspect/run_della.slurm`. To compare prompt framings, run step [2] per variant
into `data/sweep/<variant>.csv` and plot the small-multiples with
`src/plot_sweep.py`.

Qwen3 is a reasoning model; `inspect_task.py` disables thinking by default
(`disable_thinking=True`, which appends `/no_think`) so the answer isn't buried
in the reasoning channel.
