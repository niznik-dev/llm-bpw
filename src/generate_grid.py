"""Generate the input grid of "fake people" for the llm-bpw probe.

Each row is one synthetic person description that we will later hand to an LLM,
asking it to estimate that person's births-per-woman (age-specific fertility
rate) in the coming year. This script does *no* model calls — it only produces
the clean input CSV.

The grid is the full cartesian product of:

    birth_years  x  ages(min..max)  x  sexes  x  countries

Columns (and only these, by design): year_of_birth, age, sex, country.

Example:
    python src/generate_grid.py
    python src/generate_grid.py --birth-years 1980 1990 2000 --countries Denmark Japan
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd

# Fixed CSV column order — the four fields the model will receive.
COLUMNS = ["year_of_birth", "age", "sex", "country"]

# Defaults, all overridable from the command line.
DEFAULT_BIRTH_YEARS = [1950, 1960, 1970, 1980, 1990, 2000]
DEFAULT_COUNTRIES = ["Denmark"]
DEFAULT_SEXES = ["Male", "Female"]
DEFAULT_MIN_AGE = 0
DEFAULT_MAX_AGE = 55
DEFAULT_OUT = Path("data/grid.csv")


def build_grid(birth_years, ages, sexes, countries):
    """Return a DataFrame of the cartesian product, in a deterministic order."""
    rows = itertools.product(birth_years, ages, sexes, countries)
    df = pd.DataFrame(rows, columns=COLUMNS)
    # Sort for human-readable, reproducible output (curve reads top-to-bottom).
    df = df.sort_values(["country", "year_of_birth", "sex", "age"]).reset_index(drop=True)
    return df


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--birth-years", type=int, nargs="+", default=DEFAULT_BIRTH_YEARS,
                   help="Birth-year cohorts to include (default: %(default)s).")
    p.add_argument("--countries", type=str, nargs="+", default=DEFAULT_COUNTRIES,
                   help="Countries to include (default: %(default)s).")
    p.add_argument("--sexes", type=str, nargs="+", default=DEFAULT_SEXES,
                   help="Sex (at birth) values (default: %(default)s).")
    p.add_argument("--min-age", type=int, default=DEFAULT_MIN_AGE,
                   help="Minimum age, inclusive (default: %(default)s).")
    p.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE,
                   help="Maximum age, inclusive (default: %(default)s).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output CSV path (default: %(default)s).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.min_age > args.max_age:
        raise SystemExit(f"--min-age ({args.min_age}) must be <= --max-age ({args.max_age}).")

    ages = list(range(args.min_age, args.max_age + 1))
    df = build_grid(args.birth_years, ages, args.sexes, args.countries)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
