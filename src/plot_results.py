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

# Year palette, applied in ascending year order (low -> high).
YEAR_COLORS = ["#E8731A", "#9AA63D", "#17BECF", "#B07CC6"]  # orange, sunburnt-grass green, teal, light purple

# Fixed axes so every panel/plot is directly comparable. Real ASFR peaks well
# under 0.25; values above it (e.g. parse-leak outliers) clip rather than rescale.
AGE_LIMITS = (10, 55)
RATE_LIMITS = (0, 0.25)


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


def draw_year_lines(ax, df, smooth=1):
    """Draw one fertility curve per calendar year onto ``ax``.

    smooth : centered rolling-mean window (in years), applied per year to tame
        single-age jitter. Use 1 to disable. Cosmetic smoothing of the
        deterministic curve, not a re-measurement. Shared by plot_compare.py.
    """
    years = sorted(df["year"].unique())  # low -> high, to match palette
    for i, yr in enumerate(years):
        sub = df[df["year"] == yr].sort_values("age")
        y = sub["births_per_woman"].rolling(smooth, center=True, min_periods=1).mean()
        ax.plot(sub["age"], y, color=YEAR_COLORS[i % len(YEAR_COLORS)],
                linewidth=2.0, label=str(yr))
    ax.set_xlim(*AGE_LIMITS)
    ax.set_ylim(*RATE_LIMITS)


def plot_schedules(df, sex="Female", smooth=1, model=None):
    """Draw one fertility curve per year. Returns the matplotlib Figure."""
    df = df[df["sex"] == sex]
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_year_lines(ax, df, smooth=smooth)

    country = df["country"].iloc[0] if len(df) else "?"
    who = f"{model} · {country}" if model else country
    ax.set_xlabel("Age")
    ax.set_ylabel("Births per woman (this year)")
    ax.set_title(f"Period fertility schedules — {who} ({sex})")
    ax.legend(title="Year")
    ax.grid(True, alpha=0.3)
    note = smoothing_note(smooth)
    if note:
        ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, style="italic", color="0.4")
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
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.results)
    fig = plot_schedules(df, sex=args.sex, smooth=args.smooth,
                         model=model_from_results(args.results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
