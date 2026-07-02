"""Postage-stamp comparison of probe runs — one panel per model.

Point it at run folders (each holding results.csv + metadata.json, as produced
by run_probe.py). It reads the model from each metadata.json and draws all the
fertility schedules in a shared small-multiples grid, so models compare at a
glance on identical axes.

    python src/plot_compare.py data/runs/curious_counts_* data/runs/other_run
    python src/plot_compare.py                 # every run under data/runs/
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from model_meta import badge, best_rank
from plot_results import (add_limit_args, draw_year_lines, limits_from_args,
                          short_model, smoothing_note)
from reference import DEFAULT_REFERENCE, add_residual, resolve_baseline


def load_runs(dirs, min_rows):
    """Load (model, prompt, df) for each run dir that has both files + enough rows."""
    runs = []
    for d in dirs:
        d = Path(d)
        res, meta = d / "results.csv", d / "metadata.json"
        if not (res.exists() and meta.exists()):
            continue
        df = pd.read_csv(res)
        if len(df) < min_rows:  # skip backfill / single-sample test stubs
            continue
        m = json.loads(meta.read_text())
        runs.append({"model": m.get("model", "?"), "prompt": m.get("prompt", ""), "df": df})
    return runs


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="Run folders (default: all under --runs-dir).")
    p.add_argument("--runs-dir", default="data/runs", help="Default run source (default: %(default)s).")
    p.add_argument("--out", type=Path, default=Path("data/model_compare.png"),
                   help="Output image (default: %(default)s).")
    p.add_argument("--sex", default="Female", help="Which sex to plot (default: %(default)s).")
    p.add_argument("--smooth", type=int, default=1, help="Rolling-mean window; 1 = raw (default: %(default)s).")
    p.add_argument("--min-rows", type=int, default=20,
                   help="Skip runs with fewer data rows (drops backfill/test stubs).")
    p.add_argument("--real", type=Path, default=DEFAULT_REFERENCE,
                   help="Observed baseline CSV to overlay dashed (default: %(default)s).")
    p.add_argument("--no-real", action="store_true",
                   help="Hide the observed baseline overlay.")
    p.add_argument("--diff", action="store_true",
                   help="Plot model − observed residuals in each panel.")
    add_limit_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    # Recurse so runs nested under a datestamp folder are found; sentinels (which
    # also have results.csv) get dropped later by --min-rows.
    dirs = args.runs or sorted({str(p.parent) for p in Path(args.runs_dir).rglob("results.csv")})
    runs = load_runs(dirs, args.min_rows)
    if not runs:
        raise SystemExit("No run folders with results.csv + metadata.json (>= min-rows) found.")
    real = resolve_baseline(args.real, no_real=args.no_real, diff=args.diff)
    rate_limits, resid_limits = limits_from_args(args.ymax, args.resid_max)
    # Order panels best-available first, then the weaker siblings.
    runs.sort(key=lambda r: best_rank(short_model(r["model"])))

    n = len(runs)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, run in zip(axes, runs):
        panel = run["df"][run["df"]["sex"] == args.sex]
        value_col = "births_per_woman"
        if args.diff:
            panel = add_residual(panel, real)
            value_col = "residual"
        draw_year_lines(ax, panel, smooth=args.smooth,
                        real=(None if args.diff else real), value_col=value_col,
                        rate_limits=rate_limits, resid_limits=resid_limits)
        title = short_model(run["model"])
        if run["prompt"]:
            title += f"  ({run['prompt']})"
        title += f"\n{badge(short_model(run['model']))}"  # size tier · configured-thinking
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
    # Axes limits are fixed inside draw_year_lines (AGE_LIMITS / RATE_LIMITS),
    # so every panel is directly comparable and outliers clip rather than rescale.
    hidden = axes[n:]
    for ax in hidden:
        ax.set_visible(False)

    # Park the legend in the first empty cell if the grid isn't full; otherwise
    # anchor it just outside the top-right so it never sits on a panel.
    handles, labels = axes[0].get_legend_handles_labels()
    if len(hidden):
        legend_ax = hidden[0]
        legend_ax.set_visible(True)
        legend_ax.axis("off")
        legend_ax.legend(handles, labels, title="Year", loc="center")
    else:
        fig.legend(handles, labels, title="Year",
                   loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.supxlabel("Age")
    if args.diff:
        fig.supylabel("Model − observed (births per woman)")
        title = "Model residuals vs observed (HFD) — age-specific fertility"
    else:
        fig.supylabel("Births per woman (next 12 months)")
        title = "Model comparison — age-specific fertility schedules"
    subnotes = []
    if real is not None and not args.diff:
        subnotes.append("dashed = observed (HFD)")
    note = smoothing_note(args.smooth)
    if note:
        subnotes.append(note)
    subnotes.append("badge: ★ best-avail / ○ alt · thinking ● on / ○ off / ◐ amb")
    if subnotes:
        title += "\n(" + " · ".join(subnotes) + ")"
    fig.suptitle(title)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out} ({n} model{'s' if n != 1 else ''})")


if __name__ == "__main__":
    main()
