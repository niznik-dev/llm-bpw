"""Observed-baseline helpers shared by the plotters.

The baseline is the real HFD Denmark ASFR (built by scripts/load_hfd.py). These
helpers load it and compute model - observed residuals on a matched (year, age,
sex) grid, so "how wrong is each model" is one call everywhere.
"""

from pathlib import Path

import pandas as pd

DEFAULT_REFERENCE = Path("data/hfd_denmark_asfr.csv")

# Real ASFR peaks below ~0.2; a model value above this is a parse leak (e.g. a
# TFR that slipped out of the reasoning) and would otherwise blow up a residual.
MAX_PLAUSIBLE = 0.3


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
    ref = real_df.rename(columns={value: "_observed"})[["year", "age", "sex", "_observed"]]
    m = m.merge(ref, on=["year", "age", "sex"], how="inner")
    m["residual"] = m[value] - m["_observed"]
    return m
