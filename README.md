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

Early prototype. The repo currently implements **stage 1** of the pipeline:

```
[1] generate grid → [2] build prompts → [3] query model → [4] parse & join → [5] plot
    grid.csv  ✅       (next)             Ollama (Qwen)      results.csv         later
```

Stages 2–5 — prompting, querying a local Qwen via Ollama, parsing, and plotting —
are future work.

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

Requires `pandas` (see `requirements.txt`).

```bash
# Defaults → data/grid.csv (6 cohorts × 56 ages × 2 sexes × 1 country = 672 rows)
python src/generate_grid.py

# Override any field from the command line
python src/generate_grid.py \
    --birth-years 1980 1990 2000 \
    --countries Denmark Japan \
    --max-age 50 \
    --out data/my_grid.csv
```

Output is deterministic: the same arguments always produce the same file.
