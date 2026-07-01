# llm-bpw

**B**irths **p**er **w**oman: probing what language models believe about human
fertility, zero-shot.

Given a synthetic profile — calendar year, age, sex, country — we ask a model to
estimate that profile's **age-specific fertility rate**: the expected number of
births per woman at that age during that year. Demographers call these
age-profiles "fertility schedules." This is **period** fertility — fix the year,
sweep age — not a birth cohort followed through time. The aim is to map what
models believe across years and countries, with no training or fine-tuning.

This is a *rate*, not a yes/no probability — it can exceed 1 (twins, or births in
January and December) — so outputs are recorded as rates.

## Status

End-to-end pipeline working:

```
[1] generate grid → [2] prompt + query → [3] parse → [4] plot → [5] compare vs HFD
    grid.csv  ✅       Inspect AI ✅        results.csv ✅   *.png ✅   leaderboard ✅
```

Models run through **Inspect AI's `hf/` provider** — a small Qwen on the laptop
(dev/smoke-test) and a large Qwen on della — so only the model changes between
runs. The prompt variants and answer parser live in `src/probe.py` so those
runs stay comparable.

## Input fields

The grid CSV holds exactly these four columns, the fields handed to the model:

| column | meaning | values |
|---|---|---|
| `year` | calendar (period) year | user-supplied (default: decadal 1950–2000) |
| `age` | age in years | 10–55 inclusive (configurable) |
| `sex` | sex at birth | Male, Female |
| `country` | country | Denmark (expandable) |

The grid is the full cartesian product of these fields.

## Usage

Install deps (`pip install -r requirements.txt`): pandas + matplotlib for the
grid/plots, and Inspect AI + transformers/torch for the model probe.

```bash
# [1] Build the input grid (deterministic). Female-only by default.
python src/generate_grid.py --years 1920 1960 1990 2024

# [2] Probe a model. Laptop dev on a small Qwen (auto-downloads from HF hub).
#     The prompt is bare by default; ask the model for a decimal rate directly.
inspect eval src/inspect_task.py@bpw \
    --model hf/Qwen/Qwen3-4B \
    -T grid_path=$PWD/data/grid.csv -T prompt=baseline \
    -M device=mps --log-dir logs

# [3] Turn the newest .eval log into a plot-ready results.csv
python src/inspect_to_csv.py --latest --out data/results.csv

# [4] Plot the fertility schedules (one curve per year)
python src/plot_results.py
```

The large-model run on della uses the same task via SLURM — see
`inspect/run_della.slurm`. To compare prompt framings, run step [2] per variant
(`baseline`, `year_explicit`, `era_prior`, `period_pure`) into
`data/sweep/<variant>.csv` and plot the small-multiples with `src/plot_sweep.py`.

To probe a hosted model via **Together.ai** (no local GPU), the same task runs
through Inspect's `together/` provider:

```bash
cp .env.example .env          # add your TOGETHER_API_KEY (.env is gitignored)
bash inspect/run_together.sh  # override with MODEL=… PROMPT=… as needed
```

Prompt scaffolding is opt-in, off by default — pass `-T give_hint=true` to add a
demographic-pattern hint, or `-T ask_decimal=false` for a fully bare prompt.

Qwen3 is a reasoning model; `inspect_task.py` disables thinking by default
(`disable_thinking=True`, which appends `/no_think`) so the answer isn't buried
in the reasoning channel.

`src/run_probe.py` wraps a run so its outputs land together in
`data/runs/<date>/<model>/` (results.csv + metadata.json + logs), which the
comparison tools below expect.

## Compare against real fertility (HFD)

The probe elicits *beliefs*; to score them, compare against real
[Human Fertility Database](https://www.humanfertility.org/) period ASFR — free to
use **with attribution** (cite HFD / MPIDR + VID; the raw files stay out of git).
Download a country's "year, age" ASFR file (e.g. `DNKasfrRR.txt`) into `data/`:

```bash
# Build the observed baseline (the anchor year is the first --years value)
python scripts/load_hfd.py                                   # -> data/hfd_denmark_asfr.csv
python scripts/load_hfd.py --src data/USAasfrRR.txt --country "United States" \
    --years 1933 1960 1990 2024 --out data/hfd_usa_asfr.csv  # US anchors on 1933 (1920 absent)

# Overlay the baseline (dashed), or plot model - observed residuals
python src/plot_compare.py --runs-dir data/runs/<date> --smooth 3          # schedules + baseline
python src/plot_compare.py --runs-dir data/runs/<date> --diff --smooth 3   # residual postage stamps

# Score: per-year RMSE + a per-country feature scorecard -> leaderboard.csv/.png
python scripts/score_models.py --runs-dir data/runs/<date> --real data/hfd_usa_asfr.csv \
    --years 1933 1960 1990 2024
python src/plot_leaderboard.py --csv data/runs/<date>/leaderboard.csv --country "United States"
```

The **feature scorecard is country-specific** (`src/features.py`): each country's
distinctive, well-posed signatures differ, so Denmark tests the subtle 1960≈1920
boom reversal while the US — whose boom dwarfs its anchor — tests boom magnitude
within a tolerance. Countries whose peak overflows the fixed axis take `--ymax`.

Recover any null (unparsed) rows by re-querying at a larger token budget, with
logs preserved and recovered rows tagged `backfilled`:

```bash
python scripts/retry_nulls.py data/runs/<date>            # 2x the run's tokens
python scripts/retry_nulls.py data/runs/<date> --factor 4 # chase stubborn reasoners
```
