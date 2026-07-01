"""Render the model leaderboard as one PNG: RMSE bars beside the feature scorecard.

Reads the leaderboard.csv written by scripts/score_models.py and draws two
aligned panels sharing model order (best RMSE at top):
  left  — horizontal RMSE bars, colored by how many era-signatures each model hit
  right — a ✓/✗ grid of the four features (+ an HFD ground-truth row)
The alignment is the point: a short (good) RMSE bar that is pale (few features)
sitting above a longer bar that is green (all features) *is* the scoring paradox.

    python src/plot_leaderboard.py                         # default paths
    python src/plot_leaderboard.py --csv data/runs/20260629/leaderboard.csv \
        --out data/runs/20260629/leaderboard.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from features import country_features

# Discrete red->green scale for features hit (0..N of N).
HIT_COLORS = {0: "#b2182b", 1: "#d73027", 2: "#fc8d59", 3: "#fee08b", 4: "#1a9850"}
GREEN, GRAY = "#1a9850", "#bbbbbb"


def draw_rmse_bars(ax, board, n_feat):
    y = range(len(board))
    ax.barh(y, board["rmse_overall"],
            color=[HIT_COLORS[int(h)] for h in board["features_hit"]],
            edgecolor="0.3", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(board["model"], fontsize=9)
    ax.invert_yaxis()  # rank 1 (best) on top
    ax.set_xlabel("RMSE vs HFD  (lower = better)")
    ax.set_title("Pointwise error", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    xmax = board["rmse_overall"].max()
    for i, r in enumerate(board.itertuples()):
        ax.text(r.rmse_overall + xmax * 0.01, i,
                f"{r.rmse_overall:.4f}  ({int(r.features_hit)}/{n_feat})",
                va="center", fontsize=8, color="0.25")
    ax.set_xlim(0, xmax * 1.22)
    ax.legend(handles=[Patch(facecolor=HIT_COLORS[h], edgecolor="0.3",
                             label=f"{h}/{n_feat} features") for h in range(n_feat, 0, -1)],
              title="signatures hit", fontsize=8, title_fontsize=8,
              loc="upper right", framealpha=0.9)


def years_from_board(board):
    """Recover the sorted grid years from the rmse_<year> column names."""
    return sorted(int(c[len("rmse_"):]) for c in board.columns
                  if c.startswith("rmse_") and c[len("rmse_"):].isdigit())


def draw_scorecard(ax, board, features):
    rows = list(board["model"]) + ["HFD (ground truth)"]
    ax.set_xlim(-0.5, len(features) - 0.5)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.invert_yaxis()
    for i, model_row in enumerate(board.itertuples()):
        for j, (col, _) in enumerate(features):
            hit = bool(getattr(model_row, col))
            ax.scatter(j, i, s=230, marker="o",
                       color=GREEN if hit else "white",
                       edgecolor=GREEN if hit else GRAY, linewidth=1.4, zorder=2)
            ax.text(j, i, "✓" if hit else "✗", ha="center", va="center",
                    color="white" if hit else GRAY, fontsize=10, zorder=3)
    gt = len(rows) - 1  # ground-truth row: all hit
    for j in range(len(features)):
        ax.scatter(j, gt, s=230, marker="o", color="0.35", edgecolor="0.2", zorder=2)
        ax.text(j, gt, "✓", ha="center", va="center", color="white", fontsize=10, zorder=3)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([label for _, label in features], fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Feature capture", fontsize=11)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("data/runs/20260629/leaderboard.csv"))
    p.add_argument("--out", type=Path, default=Path("data/runs/20260629/leaderboard.png"))
    p.add_argument("--country", default="Denmark", help="Country name for the title.")
    args = p.parse_args()

    board = pd.read_csv(args.csv).sort_values("rmse_overall").reset_index(drop=True)
    features = [(k, lab) for k, lab, _ in country_features(args.country, years_from_board(board))]
    feat_keys = [k for k, _ in features]
    n_feat = len(features)
    missing = [k for k in feat_keys if k not in board.columns]
    if missing:
        raise SystemExit(f"--country {args.country!r} expects features {missing} not in "
                         f"{args.csv}; did score_models.py run with a different country?")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.1),
                                   gridspec_kw={"width_ratios": [1.35, 1.2]})
    draw_rmse_bars(axL, board, n_feat)
    draw_scorecard(axR, board, features)

    # Call out the paradox explicitly.
    n = len(board)
    rmse_champ = board.iloc[0]
    feat_champ = board.sort_values(["features_hit", "rmse_overall"],
                                   ascending=[False, True]).iloc[0]
    # The rarest signature = the real discriminator (Denmark: boom magnitude 2/9;
    # US: boom timing 1/9 — derive it so the caption never lies for a new country).
    hits = {k: int(board[k].sum()) for k in feat_keys}
    disc_key = min(hits, key=hits.get)
    disc = f"{disc_key.replace('_', ' ')} ({hits[disc_key]}/{n})"
    # The two rankings may agree (US, with the right features) or not (Denmark).
    feat_rank = int(board.index[board["model"] == feat_champ["model"]][0]) + 1
    if rmse_champ["model"] == feat_champ["model"]:
        verdict = (f"The two rankings AGREE: {rmse_champ['model']} tops both — RMSE #1 "
                   f"and {int(rmse_champ['features_hit'])}/{n_feat} features.")
    else:
        verdict = (f"The two rankings disagree: RMSE champ = {rmse_champ['model']} "
                   f"({int(rmse_champ['features_hit'])}/{n_feat} features)  ≠  feature champ "
                   f"= {feat_champ['model']} ({int(feat_champ['features_hit'])}/{n_feat}, "
                   f"RMSE rank {feat_rank}).")
    fig.suptitle(f"Model leaderboard — RMSE vs feature capture  ·  {args.country} ASFR vs HFD",
                 fontsize=13, y=0.99)
    fig.text(0.5, 0.015,
             verdict + "  Squared error rewards hugging the generic hump; the rarest "
             f"signature ({disc}) is the discriminator. Parse leaks > 0.3 dropped.",
             ha="center", va="bottom", fontsize=8, style="italic", color="0.35", wrap=True)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
