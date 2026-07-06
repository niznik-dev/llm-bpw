"""Plot the multi-model MEAN period fertility schedule (+ inter-model spread).

Averages births_per_woman across every model run under <runs-dir> — dropping
implausible parse-leak outliers (> 0.3) — into one mean curve per year, with a
shaded band (inter-model 25-75%) showing where the models agree vs disagree.

    python src/plot_ensemble.py --runs-dir data/runs/20260629/dk_period --smooth 3 \
        --out data/runs/20260629/dk_period/ensemble_mean_sm3.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_results import (AGE_LIMITS, RATE_LIMITS, RESID_LIMITS, YEAR_COLORS,
                          add_limit_args, limits_from_args, smoothing_note)
from reference import (DEFAULT_REFERENCE, DIM_KIND, DIM_LEGEND, MAX_PLAUSIBLE,
                       line_dim, resolve_baseline)


def load_all(runs_dir, sex):
    """Concatenate every plausible model run's results (tagged by model) + thinking."""
    frames, thinking = [], "?"
    for d in sorted(Path(runs_dir).glob("*/")):
        res, meta = d / "results.csv", d / "metadata.json"
        if not (res.exists() and meta.exists()):
            continue
        df = pd.read_csv(res)
        if not any(c in df.columns for c in ("year", "cohort")) or len(df) < 20:
            continue  # skip stubs / old framing (need a year or cohort dimension)
        m = json.loads(meta.read_text())
        df = df[df["sex"] == sex].copy()
        df["model"] = m.get("model", "?")
        frames.append(df)
        ta = m.get("task_args", {})
        th = ta.get("thinking")
        if isinstance(th, bool):
            thinking = "on" if th else "off"
        elif isinstance(th, str) and th:
            thinking = th
        elif thinking == "?":
            thinking = "off*" if ta.get("disable_thinking", True) else "on*"
    big = pd.concat(frames, ignore_index=True)
    big.loc[big["births_per_woman"] > MAX_PLAUSIBLE, "births_per_woman"] = pd.NA
    return big, thinking


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", required=True,
                   help="Run folder to average, e.g. data/runs/<date>/<group>.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output image (default: <runs-dir>/ensemble_mean.png).")
    p.add_argument("--sex", default="Female")
    p.add_argument("--smooth", type=int, default=1, help="Rolling-mean window; 1 = raw.")
    p.add_argument("--real", type=Path, default=DEFAULT_REFERENCE,
                   help="Observed baseline CSV to overlay dashed (default: %(default)s).")
    p.add_argument("--no-real", action="store_true", help="Hide the observed baseline overlay.")
    p.add_argument("--diff", action="store_true",
                   help="Plot ensemble-mean − observed residuals instead of raw schedules.")
    add_limit_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    args.out = args.out or Path(args.runs_dir) / "ensemble_mean.png"
    big, thinking = load_all(args.runs_dir, args.sex)
    n_models = big["model"].nunique()
    dim = line_dim(big)   # 'year' (period) or 'cohort'
    w = args.smooth

    keys = sorted(big[dim].unique())
    real = resolve_baseline(args.real, no_real=args.no_real, diff=args.diff)
    rate_limits, resid_limits = limits_from_args(args.ymax, args.resid_max)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, key in enumerate(keys):
        g = big[big[dim] == key].groupby("age")["births_per_woman"]
        mean = g.mean().rolling(w, center=True, min_periods=1).mean()
        lo = g.quantile(0.25).rolling(w, center=True, min_periods=1).mean()
        hi = g.quantile(0.75).rolling(w, center=True, min_periods=1).mean()
        c = YEAR_COLORS[i % len(YEAR_COLORS)]
        if args.diff:
            # Subtract the observed baseline age-aligned; the inter-model band
            # rides along, so it shows spread around the residual.
            obs = real[real[dim] == key].set_index("age")["births_per_woman"]
            mean, lo, hi = (s - s.index.map(obs) for s in (mean, lo, hi))
        ax.fill_between(mean.index, lo, hi, color=c, alpha=0.15)
        ax.plot(mean.index, mean, color=c, linewidth=2.2, label=str(key))
        if real is not None and not args.diff:
            rsub = real[real[dim] == key].sort_values("age")
            ax.plot(rsub["age"], rsub["births_per_woman"], color=c,
                    linestyle="--", linewidth=1.5, alpha=0.9)

    ax.set_xlim(*AGE_LIMITS)
    ax.set_xlabel("Age")
    ax.grid(True, alpha=0.3)
    kind = DIM_KIND[dim].lower()
    stamp = f"  ·  thinking {thinking.upper()}"
    if args.diff:
        ax.axhline(0, color="0.4", linewidth=0.8, zorder=1)
        ax.set_ylim(*resid_limits)
        ax.set_ylabel("Ensemble mean − observed (births per woman)")
        ax.set_title(f"Multi-model mean residual vs HFD — {n_models} models ({args.sex}){stamp}")
    else:
        ax.set_ylim(*rate_limits)
        ax.set_ylabel("Births per woman")
        ax.set_title(f"Multi-model mean {kind} fertility — {n_models} models ({args.sex}){stamp}")
    ax.legend(title=DIM_LEGEND[dim])

    cap = "band = inter-model 25–75%"
    if args.diff:
        cap = "solid = ensemble mean − observed · " + cap
    elif real is not None:
        cap = "solid = ensemble mean · dashed = observed (HFD) · " + cap
    note = smoothing_note(args.smooth)
    if note:
        cap += f"; {note}"
    ax.text(0.98, 0.02, cap, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, style="italic", color="0.4")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out} (mean of {n_models} models)")


if __name__ == "__main__":
    main()
