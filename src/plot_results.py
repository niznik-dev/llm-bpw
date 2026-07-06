"""Plot the period fertility schedules from a results CSV (pipeline stage 5).

One curve per calendar `year` (period fertility: fix the year, sweep age).
If a metadata.json sits beside the results CSV, the model name is read from it
and shown in the title.

    python src/plot_results.py --results data/runs/<run>/results.csv
    python src/plot_results.py --results <...> --smooth 1   # raw, unsmoothed

Year colors are assigned low->high, matching an external reference plot:
orange, green, teal, purple.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from reference import (DEFAULT_REFERENCE, DIM_KIND, DIM_LEGEND, add_residual,
                       line_dim, resolve_baseline)

# Year palette, applied in ascending year order (low -> high).
YEAR_COLORS = ["#E8731A", "#9AA63D", "#17BECF", "#B07CC6"]  # orange, sunburnt-grass green, teal, light purple

# Fixed axes so every panel/plot is directly comparable. Real ASFR peaks well
# under 0.25; values above it (e.g. parse-leak outliers) clip rather than rescale.
AGE_LIMITS = (10, 55)
RATE_LIMITS = (0, 0.25)
# Symmetric limits for residual (model - observed) plots, centered on zero.
RESID_LIMITS = (-0.15, 0.15)


def limits_from_args(ymax, resid_max):
    """Build (rate_limits, resid_limits) from optional --ymax / --resid-max.

    Shared by all three plotters so a high-peak country (US 1960 ~0.27) or a
    big-residual run can widen the fixed axes without editing code.
    """
    rate = (0, ymax) if ymax else RATE_LIMITS
    resid = (-resid_max, resid_max) if resid_max else RESID_LIMITS
    return rate, resid


def add_limit_args(p):
    """Attach --ymax / --resid-max to an argparse parser (shared by plotters)."""
    p.add_argument("--ymax", type=float, default=None,
                   help="Schedule y-axis ceiling (default %.2f; raise for the US "
                        "boom ~0.27)." % RATE_LIMITS[1])
    p.add_argument("--resid-max", type=float, default=None,
                   help="Residual y-axis half-range (default %.2f)." % RESID_LIMITS[1])


def short_model(model):
    """together/google/gemma-4-31B-it -> gemma-4-31B-it. Shared by plot_compare."""
    return model.split("/")[-1] if model else "?"


def smoothing_note(smooth):
    """Caption describing the smoothing, or None when raw (smooth <= 1)."""
    if smooth and smooth > 1:
        return f"smoothing: centered {smooth}-year moving average"
    return None


def model_from_results(results_path):
    """Read the model name from a metadata.json beside the results CSV, if any."""
    meta = Path(results_path).parent / "metadata.json"
    if meta.exists():
        try:
            return short_model(json.loads(meta.read_text()).get("model"))
        except (json.JSONDecodeError, OSError):
            pass
    return None


def draw_year_lines(ax, df, smooth=1, real=None, value_col="births_per_woman",
                    rate_limits=RATE_LIMITS, resid_limits=RESID_LIMITS):
    """Draw one curve per calendar year onto ``ax``.

    smooth : centered rolling-mean window (in years), applied per year to tame
        single-age jitter. Use 1 to disable. Cosmetic smoothing of the
        deterministic curve, not a re-measurement. Shared by plot_compare.py.
    real : optional observed-baseline df; its ``births_per_woman`` is overlaid as
        a dashed line in the matching year color. Ignored in residual mode.
    value_col : column to plot — ``births_per_woman`` for schedules, ``residual``
        for model - observed difference plots (draws a zero line, symmetric axis).
    rate_limits / resid_limits : y-limits for schedule / residual mode. Override
        the defaults for countries whose peak overflows 0.25 (e.g. the US 1960
        boom at ~0.27) or whose residuals exceed ±0.15.
    """
    resid = value_col == "residual"
    dim = line_dim(df)                    # 'year' (period) or 'cohort'
    keys = sorted(df[dim].unique())       # low -> high, to match palette
    for i, key in enumerate(keys):
        color = YEAR_COLORS[i % len(YEAR_COLORS)]
        sub = df[df[dim] == key].sort_values("age")
        y = sub[value_col].rolling(smooth, center=True, min_periods=1).mean()
        ax.plot(sub["age"], y, color=color, linewidth=2.0, label=str(key))
        if real is not None and not resid:
            rsub = real[real[dim] == key].sort_values("age")
            ax.plot(rsub["age"], rsub["births_per_woman"], color=color,
                    linestyle="--", linewidth=1.4, alpha=0.85)
    ax.set_xlim(*AGE_LIMITS)
    if resid:
        ax.axhline(0, color="0.4", linewidth=0.8, zorder=1)
        ax.set_ylim(*resid_limits)
    else:
        ax.set_ylim(*rate_limits)


def plot_schedules(df, sex="Female", smooth=1, model=None, real=None, diff=False,
                   rate_limits=RATE_LIMITS, resid_limits=RESID_LIMITS):
    """Draw one curve per year (schedules, or residuals if ``diff``). Returns Figure."""
    df = df[df["sex"] == sex]
    dim = line_dim(df)
    value_col = "births_per_woman"
    if diff:
        df = add_residual(df, real)
        value_col = "residual"

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_year_lines(ax, df, smooth=smooth, real=(None if diff else real),
                    value_col=value_col, rate_limits=rate_limits, resid_limits=resid_limits)

    country = df["country"].iloc[0] if len(df) else "?"
    who = f"{model} · {country}" if model else country
    ax.set_xlabel("Age")
    if diff:
        ax.set_ylabel("Model − observed (births per woman)")
        ax.set_title(f"Fertility residuals — {who} ({sex})")
    else:
        ax.set_ylabel("Births per woman")
        ax.set_title(f"{DIM_KIND[dim]} fertility schedules — {who} ({sex})")
    ax.legend(title=DIM_LEGEND[dim])
    ax.grid(True, alpha=0.3)

    caption = []
    if real is not None and not diff:
        caption.append("dashed = observed (HFD)")
    note = smoothing_note(smooth)
    if note:
        caption.append(note)
    if caption:
        ax.text(0.98, 0.02, " · ".join(caption), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, style="italic", color="0.4")
    fig.tight_layout()
    return fig


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", type=Path, default=Path("data/results.csv"),
                   help="Results CSV from inspect_to_csv.py (default: %(default)s).")
    p.add_argument("--out", type=Path, default=Path("data/fertility_schedules.png"),
                   help="Output image path (default: %(default)s).")
    p.add_argument("--sex", default="Female",
                   help="Which sex to plot (default: %(default)s).")
    p.add_argument("--smooth", type=int, default=1,
                   help="Centered rolling-mean window in years; 1 = raw "
                        "(default: %(default)s — raw, per Matt's preference).")
    p.add_argument("--real", type=Path, default=DEFAULT_REFERENCE,
                   help="Observed baseline CSV to overlay dashed (default: %(default)s).")
    p.add_argument("--no-real", action="store_true",
                   help="Hide the observed baseline overlay.")
    p.add_argument("--diff", action="store_true",
                   help="Plot model − observed residuals instead of raw schedules.")
    add_limit_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.results)
    real = resolve_baseline(args.real, no_real=args.no_real, diff=args.diff)
    rate_limits, resid_limits = limits_from_args(args.ymax, args.resid_max)
    fig = plot_schedules(df, sex=args.sex, smooth=args.smooth,
                         model=model_from_results(args.results),
                         real=real, diff=args.diff,
                         rate_limits=rate_limits, resid_limits=resid_limits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
