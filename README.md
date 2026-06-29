# llm-bpw

**B**irths **p**er **w**oman: probing LLMs, zero-shot, for what they believe about
human fertility.

Given a short description of a "fake person" — birth cohort, age, sex, country — we
ask a language model to estimate that person's **age-specific fertility rate**: the
expected number of births per person in the coming year (births per woman, at that
age). Demographers plot these age-profiles ("fertility schedules"); the goal here is
to see what models believe across cohorts and countries, with no training or
fine-tuning.

This is a *rate*, not a yes/no probability — it can exceed what a single-birth
probability would allow (twins, or births in January and December) — so model
outputs will be recorded as a rate, not a probability.

## Status

Early prototype. Right now the repo builds **stage 1** of the pipeline:

```
[1] generate grid → [2] build prompts → [3] query model → [4] parse & join → [5] plot
    grid.csv  ✅       (next)             Ollama (Qwen)      results.csv         later
```

Stages 2–5 (prompting, querying a local Qwen via Ollama, parsing, plotting) are
future work.

## Input fields

The grid CSV contains exactly these four columns — the fields handed to the model:

| column | meaning | values |
|---|---|---|
| `year_of_birth` | birth cohort | user-supplied (default: decadal 1950–2000) |
| `age` | age in years | 0–55 inclusive (configurable) |
| `sex` | sex at birth | Male, Female |
| `country` | country | Denmark (expandable) |

The grid is the full cartesian product of these fields.

## Usage

Requires `pandas` (see `requirements.txt`).

```bash
# Defaults → data/grid.csv (6 cohorts × 56 ages × 2 sexes × 1 country = 672 rows)
python src/generate_grid.py

# Override anything from the command line
python src/generate_grid.py \
    --birth-years 1980 1990 2000 \
    --countries Denmark Japan \
    --max-age 50 \
    --out data/my_grid.csv
```

Output is deterministic — re-running with the same arguments yields the same file.
