"""Per-country 'key feature' definitions for the leaderboard scorecard.

Each country's distinctive, well-posed demographic signatures differ, so the
feature set is country-specific. A fixed set misleads: the US baby boom towers so
far over its 1933 anchor that an ORDINAL "1960 >= 1933" test is trivially true
(7/9) while measuring nothing, and a 1-year timing gap (22 vs 23) is ill-posed.
So the US swaps in a magnitude-tolerance test and a ratio test, and drops timing;
Denmark — where 1920 and 1960 nearly tie — keeps the ordinal reversal as its
sharp discriminator.

A small library of check types is composed per country; a country not in the
registry falls back to the role-based default (the original Denmark set).

Checks read peaks/tails precomputed from a schedule:
  peaks[year] = (peak_age, peak_height);  tails[year] = mean ASFR over ages 38-44.
Magnitude checks (peak_near) compare the MODEL's peak to the REAL peak.
"""

import numpy as np

PEAK_SMOOTH = 3  # light smoothing for peak detection, to dodge single-age spikes


def year_roles(years):
    """Sorted years -> {anchor, boom, modern, current} demographic roles."""
    ys = sorted(years)
    return {"anchor": ys[0], "boom": ys[1], "modern": ys[2], "current": ys[-1]}


def peak_age_height(df_year):
    """Peak (age, height) for one year's schedule, on a lightly smoothed curve."""
    s = df_year.sort_values("age")
    y = s["births_per_woman"].rolling(PEAK_SMOOTH, center=True, min_periods=1).mean().values
    i = int(np.argmax(y))
    return s["age"].values[i], y[i]


def tail_mass(df_year):
    """Mean fertility over the old-age tail (38-44) — how broad the schedule is."""
    return df_year[(df_year["age"] >= 38) & (df_year["age"] <= 44)]["births_per_woman"].mean()


# --- check library: each builder returns a predicate fn(ctx) -> bool ---
# ctx = {"mpk": model peaks, "mtail": model tails, "rpk": real peaks, "rtail": real tails}

def taller(a, b):                      # ordinal magnitude: peak height a >= b
    return lambda c: c["mpk"][a][1] >= c["mpk"][b][1]


def younger(a, b):                     # peak age a < b
    return lambda c: c["mpk"][a][0] < c["mpk"][b][0]


def latest(y):                         # peak age y is the latest of all years
    return lambda c: c["mpk"][y][0] == max(p[0] for p in c["mpk"].values())


def widest_tail(y):                    # tail y is the broadest of all years
    return lambda c: c["mtail"][y] == max(c["mtail"].values())


def peak_near(y, tol):                 # magnitude within tolerance of the REAL peak
    return lambda c: abs(c["mpk"][y][1] - c["rpk"][y][1]) <= tol * c["rpk"][y][1]


def dominates(a, b, k):                # peak a is at least k times peak b (ratio)
    return lambda c: c["mpk"][a][1] >= k * c["mpk"][b][1]


def young_peak(y, max_age):            # peak of year y falls at/under max_age
    return lambda c: c["mpk"][y][0] <= max_age


# --- per-country registry: country -> fn(roles) -> [(key, label, check)] ---
# Labels are multi-line for the scorecard columns and bake in the actual years.

def _denmark(Y):
    a, b, cur = Y["anchor"], Y["boom"], Y["current"]
    return [
        ("boom_magnitude", f"Boom\nmagnitude\n({b} ≥ {a})", taller(b, a)),
        ("boom_timing", f"Boom\ntiming\n({b} younger)", younger(b, a)),
        ("postponement", f"Postpone-\nment\n({cur} latest)", latest(cur)),
        ("tail_breadth", f"Tail\nbreadth\n({a} widest)", widest_tail(a)),
    ]


def _usa(Y):
    a, b, cur = Y["anchor"], Y["boom"], Y["current"]
    return [
        ("boom_magnitude", f"Boom\nmagnitude\n({b} within 20%)", peak_near(b, 0.20)),
        ("boom_dominance", f"Boom\ndominance\n({b} ≥ 1.8×{a})", dominates(b, a, 1.8)),
        ("young_boom", f"Young\nboom\n({b} peak ≤ 24)", young_peak(b, 24)),
        ("postponement", f"Postpone-\nment\n({cur} latest)", latest(cur)),
    ]


FEATURE_SETS = {"Denmark": _denmark, "United States": _usa}
DEFAULT = _denmark  # role-based fallback for unregistered countries


def country_features(country, years):
    """Ordered [(key, label, check_fn), ...] for a country (default: Denmark set)."""
    return FEATURE_SETS.get(country, DEFAULT)(year_roles(years))


def build_context(df, real, years):
    """Precompute model + real peaks/tails once for all checks."""
    return {
        "mpk": {y: peak_age_height(df[df["year"] == y]) for y in years},
        "mtail": {y: tail_mass(df[df["year"] == y]) for y in years},
        "rpk": {y: peak_age_height(real[real["year"] == y]) for y in years},
        "rtail": {y: tail_mass(real[real["year"] == y]) for y in years},
    }


def evaluate(df, real, years, country):
    """Return {feature_key: bool} for one schedule against the country's set."""
    ctx = build_context(df, real, years)
    return {key: bool(fn(ctx)) for key, _label, fn in country_features(country, years)}
