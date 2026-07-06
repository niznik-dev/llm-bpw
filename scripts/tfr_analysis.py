"""Fertility conservation: is TFR conserved even when the schedule is mis-shaped?

TFR (total fertility rate) = the sum of age-specific births_per_woman over ages.
For each (model, year) we *derive* TFR from the existing probe runs — the model is
never asked for it — and compare to the HFD baseline, splitting the error into:

  * LEVEL — is the *total* right? TFR_model vs TFR_real per year.
  * SHAPE — normalize each schedule to unit area (divide by its own TFR) and
    compare the shapes alone, isolating distribution error from level error.

The residual dipole in the raw diff plots (deficit at the peak, surplus on the
flanks) hints that mass is *moved, not lost*. This quantifies whether the models'
peak-flattening conserves mass (redistributes it) or destroys it (undershoots the
total), and where displaced mass goes — younger, older, or both.

    python scripts/tfr_analysis.py --runs-dir data/runs/20260701 \
        --real data/hfd_usa_asfr.csv --years 1933 1960 1990 2024
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))            # scripts/ -> score_models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from score_models import load_runs  # noqa: E402
from reference import (DEFAULT_REFERENCE, add_residual,  # noqa: E402
                       load_reference)
from plot_results import YEAR_COLORS, short_model  # noqa: E402

DEFAULT_YEARS = [1920, 1960, 1990, 2024]  # Denmark grid; override with --years
PEAK_BAND = 2  # ages within +/- this of the observed peak count as "at the peak"
CONTROL = "(observed HFD)"  # label for the real-baseline control row
ENSEMBLE = "Multimodel mean"  # label for the mean-across-models aggregate row


def build_matched(runs_dir, real_path, sex, years):
    """Per-(model, year, age) frame on the shared grid, with TFR + normalized shapes.

    add_residual inner-joins each model to the baseline (and drops parse leaks), so
    TFR_model and TFR_real are summed over the *identical* age support per model —
    an apples-to-apples level comparison. Normalizing each schedule to its own TFR
    gives unit-area shapes whose residual (norm_model - norm_real) is pure shape.
    """
    ref = load_reference(real_path)
    if ref is None:
        raise SystemExit(f"Baseline {real_path} missing — run scripts/load_hfd.py.")
    ref = ref[ref["sex"] == sex]
    country = ref["country"].iloc[0] if "country" in ref.columns else "?"
    big = load_runs(runs_dir, sex)

    frames = []
    for model, mdf in big.groupby("model"):
        m = add_residual(mdf, ref)  # matched (year, age, sex); adds _observed, residual
        m = m[m["year"].isin(years)].copy()
        if m.empty:
            continue
        g = m.groupby("year")
        m["tfr_model"] = g["births_per_woman"].transform("sum")
        m["tfr_real"] = g["_observed"].transform("sum")
        m["norm_model"] = m["births_per_woman"] / m["tfr_model"]
        m["norm_real"] = m["_observed"] / m["tfr_real"]
        m["shape_resid"] = m["norm_model"] - m["norm_real"]  # sums to ~0 over ages
        m["model"] = model
        frames.append(m)
    if not frames:
        raise SystemExit(f"No runs under {runs_dir} cover years {years}.")
    return pd.concat(frames, ignore_index=True), country


def ensemble_frame(matched):
    """The mean-across-models schedule per (year, age), with TFR/norm/shape rebuilt.

    Averages each model's (parse-leak-cleaned) births_per_woman on the shared grid —
    the same 'ensemble mean schedule' the ensemble plots use — so summarize() can
    score the *crowd* exactly like an individual model (this is where the
    'tyranny of the majority' RMSE lives).
    """
    # Guard: the mean is only apples-to-apples if every (year, age) cell averages
    # the same models. add_residual joins each model independently, so a null drops
    # that model from an age, thinning the average there. Warn when that happens.
    n_models = matched["model"].nunique()
    models_per_cell = matched.groupby(["year", "age", "sex"])["model"].nunique()
    short = models_per_cell[models_per_cell < n_models]
    if len(short):
        print(f"⚠ ensemble: {len(short)}/{len(models_per_cell)} (year, age) cells "
              f"average < {n_models} models (min {int(short.min())}) — ragged "
              "coverage skews the mean across ages.", file=sys.stderr)

    ens = (matched.groupby(["year", "age", "sex"], as_index=False)
           .agg(births_per_woman=("births_per_woman", "mean"),
                _observed=("_observed", "first")))
    g = ens.groupby("year")
    ens["tfr_model"] = g["births_per_woman"].transform("sum")
    ens["tfr_real"] = g["_observed"].transform("sum")
    ens["norm_model"] = ens["births_per_woman"] / ens["tfr_model"]
    ens["norm_real"] = ens["_observed"] / ens["tfr_real"]
    ens["shape_resid"] = ens["norm_model"] - ens["norm_real"]
    ens["residual"] = ens["births_per_woman"] - ens["_observed"]
    ens["model"] = ENSEMBLE
    return ens


def summarize(matched):
    """One row per (model, year): level (TFR) + shape (displacement) diagnostics."""
    rows = []
    for (model, year), d in matched.groupby(["model", "year"]):
        tfr_m, tfr_r = d["tfr_model"].iloc[0], d["tfr_real"].iloc[0]
        peak_age = d.loc[d["norm_real"].idxmax(), "age"]
        at_peak = d["age"].between(peak_age - PEAK_BAND, peak_age + PEAK_BAND)
        younger = d["age"] < peak_age - PEAK_BAND
        older = d["age"] > peak_age + PEAK_BAND
        rows.append({
            "model": model, "year": year,
            "tfr_model": tfr_m, "tfr_real": tfr_r,
            "tfr_ratio": tfr_m / tfr_r, "tfr_diff": tfr_m - tfr_r,
            "rmse": np.sqrt((d["residual"] ** 2).mean()),             # vs observed, this year
            "mean_age": mean_age(d["age"], d["births_per_woman"]),     # model's center of mass
            "mean_age_real": mean_age(d["age"], d["_observed"]),       # observed center of mass
            "peak_age": int(peak_age),
            "peak_band_shift": d.loc[at_peak, "shape_resid"].sum(),   # <0 = flattened peak
            "young_shift": d.loc[younger, "shape_resid"].sum(),       # >0 = mass pushed younger
            "old_shift": d.loc[older, "shape_resid"].sum(),           # >0 = mass pushed older
            "shape_rmse": np.sqrt((d["shape_resid"] ** 2).mean()),
        })
    return pd.DataFrame(rows).sort_values(["year", "model"]).reset_index(drop=True)


def mean_age(ages, weights):
    """Mean age of childbearing: the schedule's center of mass, ∑age·w / ∑w."""
    return (ages * weights).sum() / weights.sum()


def pin_rank(model):
    """Sort key that pins the control row first, then the ensemble, then models."""
    return {CONTROL: 0, ENSEMBLE: 1}.get(model, 2)


def control_rows(summary):
    """One observed-HFD 'control' row per year, drawn from the matched baseline.

    tfr_real / mean_age_real are identical across models within a year (same
    baseline on the shared grid), so the first model's values stand in for the
    truth. RMSE and the shape shifts are zero by definition for the baseline.
    """
    rows = []
    for year, yd in summary.groupby("year"):
        r = yd.iloc[0]
        rows.append({
            "model": CONTROL, "year": year,
            "tfr_model": r["tfr_real"], "tfr_real": r["tfr_real"],
            "tfr_ratio": 1.0, "tfr_diff": 0.0, "rmse": 0.0,
            "mean_age": r["mean_age_real"], "mean_age_real": r["mean_age_real"],
            "peak_age": r["peak_age"], "peak_band_shift": 0.0,
            "young_shift": 0.0, "old_shift": 0.0, "shape_rmse": 0.0,
        })
    return pd.DataFrame(rows)


def report(summary, years):
    """Print the level table + a shape/displacement read for the focus year."""
    # Focus column: the 1960 boom in period runs, else the middle era (cohort).
    focus = 1960 if 1960 in years else years[len(years) // 2]

    print("\n=== LEVEL — is total fertility conserved? (TFR_model vs TFR_real) ===")
    print(f"{'model':<28}" + "".join(f"{y:>16}" for y in years))
    print(f"{'':28}" + "".join(f"{'(ratio)':>16}" for _ in years))
    for model, md in summary.groupby("model"):
        by_year = md.set_index("year")
        cells = []
        for y in years:
            if y in by_year.index:
                r = by_year.loc[y]
                cells.append(f"{r['tfr_model']:.2f}/{r['tfr_real']:.2f} {r['tfr_ratio']:.0%}")
            else:
                cells.append("--")
        print(f"{model:<28}" + "".join(f"{c:>16}" for c in cells))

    print(f"\n=== SHAPE — where does displaced mass go? ({focus}, unit-area) ===")
    print("  peak_band_shift <0 = flattened peak · young/old_shift >0 = mass moved there")
    print(f"{'model':<28}{'peak_age':>9}{'peak':>9}{'younger':>9}{'older':>9}{'shape_rmse':>12}")
    bd = summary[summary["year"] == focus]
    for _, r in bd.iterrows():
        print(f"{r['model']:<28}{r['peak_age']:>9}{r['peak_band_shift']:>+9.3f}"
              f"{r['young_shift']:>+9.3f}{r['old_shift']:>+9.3f}{r['shape_rmse']:>12.4f}")

    print(f"\n=== TFR · RMSE · mean age of childbearing, per era "
          f"({CONTROL} = observed) ===")
    for year in years:
        yd = summary[summary["year"] == year].copy()
        yd["_pin"] = yd["model"].map(pin_rank)  # control, ensemble, then best RMSE
        yd = yd.sort_values(["_pin", "rmse"], ascending=[True, True])
        print(f"\n  {year}")
        print(f"    {'model':<28}{'TFR':>7}{'RMSE':>9}{'mean age':>10}")
        for _, r in yd.iterrows():
            rmse = "  --  " if r["model"] == CONTROL else f"{r['rmse']:.4f}"
            print(f"    {r['model']:<28}{r['tfr_model']:>7.2f}{rmse:>9}{r['mean_age']:>10.1f}")

    ratios = (summary[~summary["model"].isin([CONTROL, ENSEMBLE])]
              .set_index(["model", "year"])["tfr_ratio"])
    print(f"\nTFR ratio across all (model, year): median {ratios.median():.0%}, "
          f"range {ratios.min():.0%}–{ratios.max():.0%}  "
          f"(100% = level perfectly conserved).")


def plot_level(summary, years, out, country="?"):
    """One panel per year: bar = each model's TFR, dashed line = observed TFR."""
    models = sorted(summary["model"].unique())
    ncols = math.ceil(math.sqrt(len(years)))
    nrows = math.ceil(len(years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.1 * len(models) * ncols / 2 + 2,
                                                    3.6 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax, year, color in zip(axes, sorted(years), YEAR_COLORS):
        d = summary[summary["year"] == year].set_index("model").reindex(models)
        ax.bar(range(len(models)), d["tfr_model"], color=color, alpha=0.85)
        real = d["tfr_real"].mean()
        ax.axhline(real, ls="--", color="black", lw=1.2, label=f"observed TFR = {real:.2f}")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([short_model(m) for m in models], rotation=90, fontsize=7)
        ax.set_title(str(year))
        ax.set_ylabel("TFR (∑ births/woman)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
    for ax in axes[len(years):]:
        ax.set_visible(False)
    fig.suptitle(f"{country} — fertility conservation: derived TFR vs observed (HFD)\n"
                 "(bar = model's summed schedule · dashed = real total)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nWrote {out}")


def plot_shape(matched, years, out, country="?"):
    """Postage stamps of normalized shape residual (norm_model - norm_real) vs age."""
    models = sorted(matched["model"].unique())
    ncols = math.ceil(math.sqrt(len(models)))
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()
    for ax, model in zip(axes, models):
        md = matched[matched["model"] == model]
        for i, year in enumerate(sorted(years)):
            yd = md[md["year"] == year].sort_values("age")
            if yd.empty:
                continue
            ax.plot(yd["age"], yd["shape_resid"], color=YEAR_COLORS[i % len(YEAR_COLORS)],
                    label=str(year), lw=1.4)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_ylim(-0.05, 0.05)
        ax.set_title(short_model(model), fontsize=9)
        ax.grid(True, alpha=0.3)
    hidden = axes[len(models):]
    for ax in hidden:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if len(hidden):
        hidden[0].set_visible(True)
        hidden[0].axis("off")
        hidden[0].legend(handles, labels, title="Year", loc="center")
    else:
        fig.legend(handles, labels, title="Year", loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.supxlabel("Age")
    fig.supylabel("Shape residual (model − observed, unit-area schedules)")
    fig.suptitle(f"{country} — fertility conservation: shape error with the level removed\n"
                 "(each schedule normalized to its own TFR · deficit at peak + surplus on flanks = mass moved, not lost)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def plot_table(full, years, out, country="?"):
    """Render the per-era TFR · RMSE · mean-age table as a readable image.

    One panel per era; models sorted by RMSE under a shaded observed-HFD control
    row. RMSE cells are heat-shaded (darker = worse) so the strong/weak split
    reads at a glance without hunting through numbers.
    """
    n_models = full[full["model"] != CONTROL]["model"].nunique()
    ncols = math.ceil(math.sqrt(len(years)))
    nrows = math.ceil(len(years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(6.6 * ncols, 0.34 * (n_models + 2) * nrows + 0.8))
    axes = axes.flatten()
    cmap = plt.cm.YlOrRd
    cols = ["model", "TFR", "RMSE", "mean age"]
    for ax, year in zip(axes, sorted(years)):
        ax.axis("off")
        yd = full[full["year"] == year].copy()
        yd["_pin"] = yd["model"].map(pin_rank)   # control, then ensemble, then models
        yd = yd.sort_values(["_pin", "rmse"], ascending=[True, True])
        vals = yd.loc[yd["_pin"] == 2, "rmse"]   # heat scale over the real models only
        lo, hi = vals.min(), vals.max()
        text, colours = [], []
        for _, r in yd.iterrows():
            ctrl = r["model"] == CONTROL
            text.append([r["model"], f"{r['tfr_model']:.2f}",
                         "—" if ctrl else f"{r['rmse']:.4f}", f"{r['mean_age']:.1f}"])
            if ctrl:
                colours.append(["#d7e3f4"] * 4)          # observed control — blue
            elif r["model"] == ENSEMBLE:
                colours.append(["#d9ead3"] * 4)          # crowd aggregate — green
            else:
                frac = 0.0 if hi == lo else (r["rmse"] - lo) / (hi - lo)
                colours.append(["white", "white", cmap(0.12 + 0.55 * frac), "white"])
        tbl = ax.table(cellText=text, colLabels=cols, cellColours=colours,
                       colColours=["#e8e8e8"] * 4, cellLoc="center", loc="center",
                       colWidths=[0.52, 0.15, 0.18, 0.17])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        for (rr, cc), cell in tbl.get_celld().items():
            if rr in (0, 1, 2):               # header + pinned control + ensemble rows
                cell.set_text_props(weight="bold")
            if cc == 0 and rr > 0:            # left-align model names
                cell.get_text().set_ha("left")
                cell.PAD = 0.04
        ax.set_title(str(year), fontweight="bold", fontsize=11)
    for ax in axes[len(years):]:
        ax.axis("off")
    fig.suptitle(f"{country} — derived TFR · RMSE · mean age of childbearing, per era\n"
                 "(blue = observed HFD control · green = multimodel mean · "
                 "RMSE shaded darker = worse · models sorted best→worst)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="data/runs/20260629")
    p.add_argument("--real", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--sex", default="Female")
    p.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS,
                   help="Grid years to analyze (default: %(default)s).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write outputs (default: <runs-dir>).")
    args = p.parse_args()
    years = sorted(args.years)
    out_dir = args.out_dir or Path(args.runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matched, country = build_matched(args.runs_dir, args.real, args.sex, years)
    summary = summarize(matched)
    ens_summary = summarize(ensemble_frame(matched))
    full = pd.concat([summary, ens_summary, control_rows(summary)], ignore_index=True)
    report(full, years)

    full.to_csv(out_dir / "tfr_summary.csv", index=False)
    print(f"\nWrote {out_dir / 'tfr_summary.csv'}")
    # Plots exclude the control: the observed baseline is already the dashed
    # line (level) and the zero line (shape).
    plot_level(summary, years, out_dir / "tfr_level.png", country)
    plot_shape(matched, years, out_dir / "tfr_shape.png", country)
    plot_table(full, years, out_dir / "tfr_table.png", country)


if __name__ == "__main__":
    main()
