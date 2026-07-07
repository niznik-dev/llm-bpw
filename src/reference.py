"""Observed-baseline helpers shared by the plotters.

The baseline is the real HFD Denmark ASFR (built by scripts/load_hfd.py). These
helpers load it and compute model - observed residuals on a matched (year, age,
sex) grid, so "how wrong is each model" is one call everywhere.
"""

from pathlib import Path

import pandas as pd

DEFAULT_REFERENCE = Path("data/baselines/hfd_denmark_period_asfr.csv")

# Analysis-layer plausibility filter for OUR scope: Europe + the Americas, all
# low-to-moderate fertility and all in HFD (the real single-year peak is US 1960 at
# ~0.267). A model value above this is implausible *for these countries*, so we drop
# it from residuals to keep a stray high value from dominating the axis. Distinct
# from probe.MAX_ASFR (0.5), the global physical parse-time reject: true leaks (>0.5)
# never reach here (they're nulled + retried at the source), so what this drops is
# the 0.3–0.5 "suspicious but not impossible" band — worth investigating, not a leak.
MAX_PLAUSIBLE = 0.3

# A schedule's "line dimension" — the field identifying one curve: calendar
# `year` (period) or birth `cohort` (cohort). Auto-detected everywhere so the
# same tooling serves both axes. Display labels for legends/titles alongside.
LINE_DIMS = ("year", "cohort")
DIM_LEGEND = {"year": "Year", "cohort": "Birth cohort"}
DIM_KIND = {"year": "Period", "cohort": "Cohort"}


def line_dim(df):
    """The line dimension present in df: 'cohort' if that column exists, else 'year'."""
    return next((d for d in LINE_DIMS if d in df.columns), "year")


def load_reference(path=DEFAULT_REFERENCE):
    """Load the observed baseline CSV, or None if it isn't there yet."""
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


def resolve_baseline(real_path, no_real=False, diff=False):
    """Load the baseline a plot should use, honoring --no-real / requiring --diff.

    Returns the baseline df (or None when hidden). Raises if ``diff`` is asked for
    but the baseline is missing, since there's nothing to difference against.
    """
    real = None if no_real else load_reference(real_path)
    if diff and real is None:
        raise SystemExit(f"--diff needs the observed baseline, but {real_path} is "
                         f"missing. Run scripts/load_hfd.py first.")
    return real


def add_residual(model_df, real_df, value="births_per_woman",
                 max_plausible=MAX_PLAUSIBLE):
    """Return model rows with a ``residual`` = model - observed column.

    Matched on (year, age, sex) via inner join, so ages the baseline doesn't
    cover are dropped rather than compared against nothing. Implausible model
    values (> ``max_plausible``) are removed first so a stray parse leak can't
    dominate the residual axis.
    """
    m = model_df.copy()
    if max_plausible is not None:
        m = m[m[value] <= max_plausible]
    keys = [line_dim(m), "age", "sex"]  # match on year|cohort + age + sex
    ref = real_df.rename(columns={value: "_observed"})[keys + ["_observed"]]
    m = m.merge(ref, on=keys, how="inner")
    m["residual"] = m[value] - m["_observed"]
    return m
