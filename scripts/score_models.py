"""Score each model against the real HFD observed baseline (RMSE).

For each run under --runs-dir, computes the root-mean-square error of the model's
age-specific fertility schedule vs the HFD baseline — overall and per calendar
year — and writes a per-model metrics table (model_metrics.csv), sorted by
overall RMSE.

    python scripts/score_models.py --runs-dir data/runs/20260701 \
        --real data/hfd_usa_asfr.csv --years 1933 1960 1990 2024
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reference import (DEFAULT_REFERENCE, MAX_PLAUSIBLE, add_residual,  # noqa: E402
                       load_reference)

DEFAULT_YEARS = [1920, 1960, 1990, 2024]  # Denmark grid; override with --years


def load_runs(runs_dir, sex):
    """(model, year, age, births_per_woman) for every run under runs_dir."""
    frames = []
    for meta in sorted(Path(runs_dir).glob("*/metadata.json")):
        res = meta.parent / "results.csv"
        if not res.exists():
            continue
        df = pd.read_csv(res)
        if "year" not in df.columns or len(df) < 20:  # skip stubs / old framing
            continue
        df = df[df["sex"] == sex].copy()
        df["model"] = json.loads(meta.read_text()).get("model", "?").split("/")[-1]
        frames.append(df)
    if not frames:
        raise SystemExit(f"No usable runs under {runs_dir}.")
    return pd.concat(frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="data/runs/20260629")
    p.add_argument("--real", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--sex", default="Female")
    p.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS,
                   help="Grid years to report per-year RMSE for (default: %(default)s).")
    p.add_argument("--out", type=Path, default=None,
                   help="Metrics CSV (default: <runs-dir>/model_metrics.csv).")
    args = p.parse_args()
    years = sorted(args.years)

    ref = load_reference(args.real)
    if ref is None:
        raise SystemExit(f"Baseline {args.real} missing — run scripts/load_hfd.py.")
    country = ref["country"].iloc[0] if "country" in ref.columns else "?"
    big = load_runs(args.runs_dir, args.sex)
    big = big[big["births_per_woman"] <= MAX_PLAUSIBLE]  # drop parse leaks

    rows = []
    for model, mdf in big.groupby("model"):
        res = add_residual(mdf, ref)
        rmse = {f"rmse_{y}": np.sqrt((res[res.year == y]["residual"] ** 2).mean())
                for y in years}
        rows.append({
            "model": model,
            "rmse_overall": np.sqrt((res["residual"] ** 2).mean()),
            "mae_overall": res["residual"].abs().mean(),
            **rmse,
        })
    metrics = pd.DataFrame(rows).sort_values("rmse_overall").reset_index(drop=True)

    print(f"\nScored {len(metrics)} models vs HFD ({args.sex}, {country})  ·  "
          f"parse leaks > {MAX_PLAUSIBLE} dropped\n")
    print("=== Model metrics · sorted by RMSE (lower = better) ===")
    print(f"{'rank':>4}  {'model':<26}{'RMSE':>8}{'MAE':>8}   " +
          "".join(f"{y:>7}" for y in years))
    for i, r in metrics.iterrows():
        print(f"{i + 1:>4}  {r['model']:<26}{r['rmse_overall']:>8.4f}"
              f"{r['mae_overall']:>8.4f}   " +
              "".join(f"{r[f'rmse_{y}']:>7.4f}" for y in years))

    out = args.out or Path(args.runs_dir) / "model_metrics.csv"
    metrics.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
