"""Boom capture: best-available models vs weaker siblings — 1960-RMSE by is_best.

Reads a model_metrics.csv (from score_models.py) and the model-metadata registry,
and plots each model's boom-year RMSE split into two groups: the best model each
provider offers on Together vs the weaker/older siblings we also ran. A simple
strip (raw points + per-group mean): is_best is a binary label, so there's
nothing continuous to fit — the comparison is exactly as strong as the two group
means, no more. Points are colored by *configured* thinking (a confound, not a
model property).

    python scripts/plot_boom_by_best.py --runs-dir data/runs/20260701/us_period --boom-col rmse_1960
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model_meta import load_registry, meta_for  # noqa: E402

# Configured-thinking palette (matches the ●/○/◐ badge semantics).
THINK_COLOR = {"on": "#2E8B57", "off": "#9AA0A6", "ambiguous": "#E8731A"}
# Group x-positions: best on the left, alt on the right.
GROUPS = [(True, 0, "best available\n(provider's top on Together)"),
          (False, 1, "alt\n(weaker / older sibling)")]


def ladder(ax, xs, ys, labels, x_label, ha, min_gap):
    """Fan labels into an evenly-spaced vertical column beside the dots, with a
    thin leader line from each dot to its label. Rungs are spaced >= min_gap and
    expanded past the data range when a cluster is too tight, so no two labels
    collide — deterministic, unlike an iterative repel. (Could be promoted to a
    shared helper in plot_results.py if other plots want it.)"""
    order = sorted(range(len(ys)), key=lambda i: ys[i])   # bottom->top, no crossings
    n = len(order)
    lo, hi = ys[order[0]], ys[order[-1]]
    span_needed = (n - 1) * min_gap
    if hi - lo < span_needed:                             # tight cluster -> expand
        mid = (lo + hi) / 2
        lo, hi = mid - span_needed / 2, mid + span_needed / 2
    rungs = ([lo + (hi - lo) * k / (n - 1) for k in range(n)] if n > 1
             else [ys[order[0]]])
    for rung, i in zip(rungs, order):
        ax.annotate(labels[i], xy=(xs[i], ys[i]), xytext=(x_label, rung),
                    textcoords="data", ha=ha, va="center", fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999",
                                    shrinkA=2, shrinkB=3))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="data/runs/20260701/us_period")
    p.add_argument("--metrics", type=Path, default=None,
                   help="Metrics CSV (default: <runs-dir>/model_metrics.csv).")
    p.add_argument("--boom-col", default="rmse_1960",
                   help="Which per-year RMSE column is the boom (default: %(default)s).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output image (default: <runs-dir>/boom_by_best.png).")
    args = p.parse_args()

    metrics_path = args.metrics or Path(args.runs_dir) / "model_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    if args.boom_col not in metrics.columns:
        raise SystemExit(f"{metrics_path} has no column {args.boom_col!r}; "
                         f"columns are {list(metrics.columns)}.")
    reg = load_registry()

    rows = []
    for _, r in metrics.iterrows():
        m = meta_for(r["model"], reg)
        rows.append({"model": r["model"], "is_best": bool(m.get("is_best")),
                     "boom": r[args.boom_col],
                     "thinking": m.get("thinking_configured", "ambiguous")})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No models found in metrics table.")

    fig, ax = plt.subplots(figsize=(9, 6))
    min_gap = 0.062 * (df["boom"].max() - df["boom"].min())  # rung spacing floor
    for is_best, x, _ in GROUPS:
        grp = df[df["is_best"] == is_best].sort_values("boom")
        n = len(grp)
        if not n:
            continue
        offs = [0.0] if n == 1 else [(-0.12 + 0.24 * i / (n - 1)) for i in range(n)]
        xs = [x + o for o in offs]
        ys = list(grp["boom"])
        labs = list(grp["model"])
        cols = [THINK_COLOR.get(t, "#999999") for t in grp["thinking"]]
        ax.scatter(xs, ys, s=110, zorder=3, color=cols, edgecolor="black", linewidth=0.6)
        # Labels fan to the outer side (best -> left, alt -> right); mean label inner.
        if is_best:
            ladder(ax, xs, ys, labs, x - 0.6, "right", min_gap)
            mean_x, mean_ha = x + 0.42, "left"
        else:
            ladder(ax, xs, ys, labs, x + 0.6, "left", min_gap)
            mean_x, mean_ha = x - 0.42, "right"
        mean = grp["boom"].mean()
        ax.plot([x - 0.28, x + 0.28], [mean, mean], color="black", lw=1.6, zorder=2)
        ax.annotate(f"mean {mean:.4f}", (mean_x, mean), fontsize=8, va="center", ha=mean_ha)

    ax.set_xticks([x for _, x, _ in GROUPS])
    ax.set_xticklabels([lab for _, _, lab in GROUPS])
    ax.set_xlim(-1.25, 2.25)
    ax.set_ylabel(f"{args.boom_col} — boom-year RMSE (lower = better boom capture)")
    ax.grid(True, axis="y", alpha=0.3)

    handles = [plt.Line2D([0], [0], marker="o", ls="", markersize=9,
                          markerfacecolor=THINK_COLOR[k], markeredgecolor="black",
                          label=f"thinking: {k}") for k in ["on", "off", "ambiguous"]]
    handles.append(plt.Line2D([0], [0], color="black", lw=1.6, label="group mean"))
    ax.legend(handles=handles, fontsize=8, loc="upper center")

    best_mean = df[df["is_best"]]["boom"].mean()
    alt_mean = df[~df["is_best"]]["boom"].mean()
    ax.set_title("Boom capture: best-available vs weaker siblings\n"
                 f"best mean {best_mean:.4f} < alt {alt_mean:.4f}, but overlapping — "
                 "top reader is an 'alt' (Qwen3.6-Plus)", fontsize=11)
    fig.tight_layout()
    out = args.out or Path(args.runs_dir) / "boom_by_best.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}  ·  best mean {best_mean:.4f} vs alt mean {alt_mean:.4f}")


if __name__ == "__main__":
    main()
