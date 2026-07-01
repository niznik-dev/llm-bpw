"""Score each model against the real HFD observed baseline.

Two leaderboards, reported side by side because they DISAGREE (see the findings
snapshot): a pointwise RMSE table — which rewards hugging the generic mid-20s
hump — and a qualitative FEATURE scorecard, which rewards getting the
era-specific signatures that a squared-error metric quietly penalizes.

    python scripts/score_models.py                       # runs-dir default below
    python scripts/score_models.py --runs-dir data/runs/20260629 --out X.csv

Feature signatures (True/False per model, ground-truthed on HFD). Defined by the
four years' ROLES, not literal years, so a different anchor works unchanged:
sorted --years become [anchor, boom, modern, current] (Denmark 1920/1960/1990/2024,
USA 1933/1960/1990/2024):
  boom_magnitude : boom peak at least as tall as anchor (the baby-boom reversal)
  boom_timing    : boom peaks younger than anchor
  postponement   : current year peaks latest of all years
  tail_breadth   : anchor has the fattest old-age (38-44) tail
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import country_features, evaluate  # noqa: E402
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
                   help="Grid years; earliest is the anchor (default: %(default)s).")
    p.add_argument("--out", type=Path, default=None,
                   help="Leaderboard CSV (default: <runs-dir>/leaderboard.csv).")
    args = p.parse_args()
    years = sorted(args.years)

    ref = load_reference(args.real)
    if ref is None:
        raise SystemExit(f"Baseline {args.real} missing — run scripts/load_hfd.py.")
    country = ref["country"].iloc[0] if "country" in ref.columns else "?"
    feat_cols = [k for k, _, _ in country_features(country, years)]  # this country's set
    n_feat = len(feat_cols)
    big = load_runs(args.runs_dir, args.sex)
    big = big[big["births_per_woman"] <= MAX_PLAUSIBLE]  # drop parse leaks

    rows = []
    for model, mdf in big.groupby("model"):
        res = add_residual(mdf, ref)
        rmse = {f"rmse_{y}": np.sqrt((res[res.year == y]["residual"] ** 2).mean())
                for y in years}
        feat = evaluate(mdf, ref, years, country)
        rows.append({
            "model": model,
            "rmse_overall": np.sqrt((res["residual"] ** 2).mean()),
            "mae_overall": res["residual"].abs().mean(),
            **rmse, **feat,
            "features_hit": sum(feat.values()),
        })
    board = pd.DataFrame(rows).sort_values("rmse_overall").reset_index(drop=True)

    tick = {True: "✓", False: "✗"}

    print(f"\nScored {len(board)} models vs HFD ({args.sex}, {country})  ·  "
          f"parse leaks > {MAX_PLAUSIBLE} dropped\n")
    print("=== RMSE leaderboard (lower = better; rewards hugging the generic hump) ===")
    print(f"{'rank':>4}  {'model':<26}{'RMSE':>8}{'MAE':>8}   " +
          "".join(f"{y:>7}" for y in years))
    for i, r in board.iterrows():
        print(f"{i + 1:>4}  {r['model']:<26}{r['rmse_overall']:>8.4f}"
              f"{r['mae_overall']:>8.4f}   " +
              "".join(f"{r[f'rmse_{y}']:>7.4f}" for y in years))

    print("\n=== Feature scorecard (rewards the signatures RMSE penalizes) ===")
    print(f"{'model':<26}" + "".join(f"{c.split('_')[0][:5]:>7}" for c in feat_cols)
          + f"{'hit':>6}")
    for _, r in board.sort_values(["features_hit", "rmse_overall"],
                                  ascending=[False, True]).iterrows():
        print(f"{r['model']:<26}" + "".join(f"{tick[bool(r[c])]:>7}" for c in feat_cols)
              + f"{int(r['features_hit']):>4}/{n_feat}")
    real_feat = evaluate(ref, ref, years, country)
    print(f"{'HFD (ground truth)':<26}" +
          "".join(f"{tick[bool(real_feat[c])]:>7}" for c in feat_cols)
          + f"{sum(real_feat.values()):>4}/{n_feat}")

    out = args.out or Path(args.runs_dir) / "leaderboard.csv"
    board.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
