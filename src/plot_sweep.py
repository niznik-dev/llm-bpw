"""Compare prompt variants side by side (small multiples).

Reads every CSV in a sweep directory — each tagged with a ``prompt`` column by
inspect_to_csv.py — and draws one panel per prompt variant, each showing the four
cohort fertility curves on shared axes. The point: see which prompt framing
makes the model actually differentiate birth cohorts.

    python src/plot_sweep.py                  # data/sweep/*.csv -> data/sweep_compare.png
    python src/plot_sweep.py --smooth 5
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_results import draw_year_lines, smoothing_note

# Preferred panel order: a cohort-salience gradient (least -> most explicit).
PROMPT_ORDER = ["baseline", "year_explicit", "era_prior", "period_pure"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sweep-dir", type=Path, default=Path("data/sweep"),
                   help="Directory of per-variant result CSVs (default: %(default)s).")
    p.add_argument("--out", type=Path, default=Path("data/sweep_compare.png"),
                   help="Output image path (default: %(default)s).")
    p.add_argument("--sex", default="Female",
                   help="Which sex to plot (default: %(default)s).")
    p.add_argument("--smooth", type=int, default=3,
                   help="Centered rolling-mean window; 1 disables (default: %(default)s).")
    return p.parse_args()


def main():
    args = parse_args()
    frames = {f.stem: pd.read_csv(f) for f in sorted(args.sweep_dir.glob("*.csv"))}
    if not frames:
        raise SystemExit(f"No CSVs found in {args.sweep_dir} (run the sweep first).")

    # Known variants in gradient order, then any extras alphabetically.
    names = ([n for n in PROMPT_ORDER if n in frames]
             + [n for n in frames if n not in PROMPT_ORDER])

    ncols = 2
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        draw_year_lines(ax, frames[name][frames[name]["sex"] == args.sex], smooth=args.smooth)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(names):]:  # hide any empty panels
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Year", loc="upper right")
    fig.supxlabel("Age")
    fig.supylabel("Births per woman (next 12 months)")
    title = "Prompt sweep — does framing make Qwen differentiate cohorts?"
    note = smoothing_note(args.smooth)
    if note:
        title += f"\n({note})"
    fig.suptitle(title)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
