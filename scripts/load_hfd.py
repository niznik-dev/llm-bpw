"""Turn a Human Fertility Database ASFR file into our plot-ready baseline.

Handles both HFD schedule types, auto-detected from the file's header row:
  * PERIOD  (`*asfrRR.txt`) — fertility by calendar YEAR and single age (Lexis
    squares). First column: `Year`.
  * COHORT  (`*asfrVH.txt`) — fertility by birth COHORT and age (horizontal
    parallelograms). First column: `Cohort`.

Either way it writes the same 5-column schema the rest of the pipeline expects,
with the dimension column named to match the source (`year` or `cohort`), so it
drops in wherever a results.csv would.

This is the REAL observed baseline. HFD data is free to use with attribution
(Human Fertility Database, Max Planck Institute for Demographic Research and
Vienna Institute of Demography) — cite it wherever these curves appear.

    # Period (default): Denmark calendar years
    python scripts/load_hfd.py               # -> data/baselines/hfd_denmark_period_asfr.csv
    python scripts/load_hfd.py --src data/hfd_raw/USAasfrRR.txt --country "United States" \
        --keys 1933 1960 1990 2024 --out data/baselines/hfd_usa_period_asfr.csv

    # Cohort: birth cohorts (pick completed ones)
    python scripts/load_hfd.py --src data/hfd_raw/DNKasfrVH.txt \
        --keys 1935 1945 1960 1975 --out data/baselines/hfd_denmark_cohort_asfr.csv

The anchor (earliest) key is whatever you pass first in --keys. For period, pick
each country's earliest solid pre-boom year (Denmark 1920, USA 1933). For cohort,
pick completed cohorts (recent cohorts still have missing rates at older ages).
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_KEYS = [1920, 1960, 1990, 2024]  # Denmark period years; override with --keys
AGES = list(range(10, 56))  # match the model grid (10-55 inclusive)
FERTILE = range(12, 50)     # ages that must be present for a "complete" schedule
                            # (through 49 ~ HFD's own completed-cohort convention)


def parse_age(token):
    """HFD ages: '12-' (open low) -> 12, '55+' (open high) -> 55, '30' -> 30."""
    return int(str(token).strip().rstrip("+-"))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=Path("data/hfd_raw/DNKasfrRR.txt"),
                   help="HFD ASFR file — period (asfrRR) or cohort (asfrVH) "
                        "(default: %(default)s).")
    p.add_argument("--keys", "--years", dest="keys", type=int, nargs="+",
                   default=DEFAULT_KEYS,
                   help="Values of the file's first dimension to keep — calendar "
                        "YEARS for period files, birth COHORTS for cohort files; "
                        "the earliest is the anchor (default: %(default)s).")
    p.add_argument("--country", default="Denmark",
                   help="Country label written to the CSV (default: %(default)s).")
    p.add_argument("--out", type=Path,
                   default=Path("data/baselines/hfd_denmark_period_asfr.csv"),
                   help="Output CSV (default: %(default)s).")
    args = p.parse_args()

    # Two banner lines (title + "Last modified"), then a "<dim> Age ASFR" header.
    raw = pd.read_csv(args.src, skiprows=2, sep=r"\s+")
    raw.columns = [c.lower() for c in raw.columns]
    dim = next(c for c in raw.columns if c not in ("age", "asfr"))  # 'year' | 'cohort'
    raw["age"] = raw["age"].map(parse_age)
    raw[dim] = raw[dim].astype(int)
    raw["asfr"] = pd.to_numeric(raw["asfr"], errors="coerce")  # HFD '.' missing -> NaN

    rows, incomplete = [], {}
    for key in args.keys:
        by_age = raw[raw[dim] == key].set_index("age")["asfr"]
        if by_age.empty:
            raise SystemExit(f"{dim.title()} {key} not found in {args.src} "
                             f"(HFD covers {raw[dim].min()}-{raw[dim].max()}).")
        gaps = [a for a in FERTILE if pd.isna(by_age.get(a))]
        if gaps:  # a recent cohort not yet done, or a sparse year — warn, don't drop
            incomplete[key] = gaps
        for age in AGES:
            # Ages outside HFD's 12-55 window (10, 11) are genuinely zero; a missing
            # in-range rate (NaN) is filled 0 so the row exists (see the warning).
            val = by_age.get(age, 0.0)
            val = 0.0 if pd.isna(val) else float(val)
            rows.append({dim: key, "age": age, "sex": "Female",
                         "country": args.country,
                         "births_per_woman": round(val, 5)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    kind = "cohort" if dim == "cohort" else "period"
    print(f"Wrote {args.out} ({len(rows)} rows, {len(args.keys)} {dim}s — "
          f"HFD observed, {kind})")
    if incomplete:
        print("  ⚠ incomplete (missing fertile-age rates → filled 0):")
        for key, gaps in incomplete.items():
            print(f"    {dim} {key}: ages {gaps}  — likely not a completed {dim}")


if __name__ == "__main__":
    main()
