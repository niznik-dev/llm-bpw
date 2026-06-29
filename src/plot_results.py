"""Plot the fertility schedules from a results CSV (pipeline stage 5).

Reads a results CSV (grid fields + ``births_per_woman``) and draws one
age-specific fertility curve per birth cohort: x = age, y = births per woman in
the coming year. By default it plots women only (the probe's target; male
profiles read ~0).

    python src/plot_results.py                       # data/results.csv -> data/fertility_schedules.png
    python src/plot_results.py --results data/results.csv --out fig.png

Cohort colors are assigned low->high birth year, matching an external reference
plot: orange, green, teal, purple.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Cohort palette, applied in ascending birth-year order (low year -> high year).
COHORT_COLORS = ["#E8731A", "#9AA63D", "#17BECF", "#B07CC6"]  # orange, sunburnt-grass green, teal, light purple


def draw_cohorts(ax, df, smooth=3):
    """Draw one cohort fertility curve per birth year onto ``ax``.

    smooth : centered rolling-mean window (in years), applied per cohort to tame
        single-age jitter. Use 1 to disable. Cosmetic smoothing of the
        deterministic curve, not a re-measurement. Shared by plot_sweep.py.
    """
    cohorts = sorted(df["year_of_birth"].unique())  # low -> high, to match palette
    for i, yob in enumerate(cohorts):
        sub = df[df["year_of_birth"] == yob].sort_values("age")
        y = sub["births_per_woman"].rolling(smooth, center=True, min_periods=1).mean()
        color = COHORT_COLORS[i % len(COHORT_COLORS)]
        ax.plot(sub["age"], y, color=color, linewidth=2.0, label=str(yob))


def plot_schedules(df, sex="Female", smooth=3):
    """Draw one fertility curve per cohort. Returns the matplotlib Figure."""
    df = df[df["sex"] == sex]
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_cohorts(ax, df, smooth=smooth)

    country = df["country"].iloc[0] if len(df) else "?"
    ax.set_xlabel("Age")
    ax.set_ylabel("Births per woman (next 12 months)")
    ax.set_title(f"Age-specific fertility schedules — {country} ({sex})")
    ax.legend(title="Birth cohort")
    ax.grid(True, alpha=0.3)
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
    p.add_argument("--smooth", type=int, default=3,
                   help="Centered rolling-mean window in years; 1 disables "
                        "(default: %(default)s).")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.results)
    fig = plot_schedules(df, sex=args.sex, smooth=args.smooth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
