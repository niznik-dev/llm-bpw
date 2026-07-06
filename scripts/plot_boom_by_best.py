"""Boom capture: best-available models vs weaker siblings — 1960-RMSE by is_best.

Reads a model_metrics.csv (from score_models.py) and the model-metadata registry,
and plots each model's boom-year RMSE split into two groups: the best model each
provider offers on Together vs the weaker/older siblings we also ran. A simple
strip (raw points + per-group mean): is_best is a binary label, so there's
nothing continuous to fit — the comparison is exactly as strong as the two group
means, no more. Points are colored by *configured* thinking (a confound, not a
model property).

    python scripts/plot_boom_by_best.py --runs-dir data/runs/20260701 --boom-col rmse_1960
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


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="data/runs/20260701")
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
    for is_best, x, _ in GROUPS:
        grp = df[df["is_best"] == is_best].sort_values("boom")
        n = len(grp)
        if not n:
            continue
        offs = [0.0] if n == 1 else [(-0.2 + 0.4 * i / (n - 1)) for i in range(n)]
        for off, (_, row) in zip(offs, grp.iterrows()):
            ax.scatter(x + off, row["boom"], s=110, zorder=3,
                       color=THINK_COLOR.get(row["thinking"], "#999999"),
                       edgecolor="black", linewidth=0.6)
            ax.annotate(row["model"], (x + off, row["boom"]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7.5)
        mean = grp["boom"].mean()
        ax.plot([x - 0.28, x + 0.28], [mean, mean], color="black", lw=1.6, zorder=2)
        ax.annotate(f"mean {mean:.4f}", (x + 0.3, mean), fontsize=8, va="center")

    ax.set_xticks([x for _, x, _ in GROUPS])
    ax.set_xticklabels([lab for _, _, lab in GROUPS])
    ax.set_xlim(-0.6, 1.7)
    ax.set_ylabel(f"{args.boom_col} — boom-year RMSE (lower = better boom capture)")
    ax.grid(True, axis="y", alpha=0.3)

    handles = [plt.Line2D([0], [0], marker="o", ls="", markersize=9,
                          markerfacecolor=THINK_COLOR[k], markeredgecolor="black",
                          label=f"thinking: {k}") for k in ["on", "off", "ambiguous"]]
    handles.append(plt.Line2D([0], [0], color="black", lw=1.6, label="group mean"))
    ax.legend(handles=handles, fontsize=8, loc="upper left")

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
