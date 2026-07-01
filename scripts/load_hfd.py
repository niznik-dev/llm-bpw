"""Turn the Human Fertility Database Denmark file into our plot-ready baseline.

Reads the HFD period ASFR file (`DNKasfrRR.txt` — period fertility rates by
calendar year and single age, Lexis squares) and writes the same 5-column schema
the rest of the pipeline expects, so it drops in wherever a results.csv would.

This is the REAL observed baseline (it replaced an earlier hand-digitized
approximation read off the reference figure). HFD data is free to use with attribution
(Human Fertility Database, Max Planck Institute for Demographic Research and
Vienna Institute of Demography) — cite it wherever these curves appear.

    python scripts/load_hfd.py                      # -> data/hfd_denmark_asfr.csv
    python scripts/load_hfd.py --years 1920 1960 1990 2024
    python scripts/load_hfd.py --src data/USAasfrRR.txt --country "United States" \
        --years 1933 1960 1990 2024 --out data/hfd_usa_asfr.csv

The anchor (earliest) year is whatever you pass first in --years; pick each
country's earliest solid pre-boom year (Denmark 1920, USA 1933 = Depression low).
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_YEARS = [1920, 1960, 1990, 2024]
AGES = list(range(10, 56))  # match the model grid (10-55 inclusive)


def parse_age(token):
    """HFD ages: '12-' (open low) -> 12, '55+' (open high) -> 55, '30' -> 30."""
    return int(str(token).strip().rstrip("+-"))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=Path("data/DNKasfrRR.txt"),
                   help="HFD Denmark period ASFR file (default: %(default)s).")
    p.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS,
                   help="Calendar years to keep; the earliest is the anchor "
                        "(default: %(default)s).")
    p.add_argument("--country", default="Denmark",
                   help="Country label written to the CSV (default: %(default)s).")
    p.add_argument("--out", type=Path, default=Path("data/hfd_denmark_asfr.csv"),
                   help="Output CSV (default: %(default)s).")
    args = p.parse_args()

    # Two banner lines (title + "Last modified"), then a "Year Age ASFR" header.
    raw = pd.read_csv(args.src, skiprows=2, sep=r"\s+")
    raw.columns = [c.lower() for c in raw.columns]
    raw["age"] = raw["age"].map(parse_age)
    raw["year"] = raw["year"].astype(int)

    rows = []
    for year in args.years:
        by_age = raw[raw["year"] == year].set_index("age")["asfr"]
        if by_age.empty:
            raise SystemExit(f"Year {year} not found in {args.src} "
                             f"(HFD covers {raw['year'].min()}-{raw['year'].max()}).")
        for age in AGES:
            # Ages outside HFD's 12-55 window (10, 11) are genuinely zero.
            rows.append({"year": year, "age": age, "sex": "Female",
                         "country": args.country,
                         "births_per_woman": round(float(by_age.get(age, 0.0)), 5)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(rows)} rows, {len(args.years)} years — HFD observed)")


if __name__ == "__main__":
    main()
