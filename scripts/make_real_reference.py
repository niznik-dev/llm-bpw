"""Build an APPROXIMATE real-Denmark ASFR reference, hand-digitized off the
figure Mattie shared ("Denmark: Age-Specific Fertility Rate by year").

This is an eyeballed approximation for VISUAL comparison only — read the anchor
points below as "roughly where the curve passes through," not official data.
Linear interpolation fills every integer age. Edit the anchors to refine.

    python scripts/make_real_reference.py   # -> data/real_denmark_asfr.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

# (age, births-per-woman) anchors read off the shared figure.
ANCHORS = {
    1920: [(13, 0), (15, 0.005), (17, 0.025), (19, 0.06), (21, 0.10), (23, 0.14),
           (25, 0.17), (26, 0.185), (27, 0.185), (28, 0.18), (30, 0.165),
           (32, 0.145), (34, 0.12), (36, 0.095), (38, 0.07), (40, 0.05),
           (42, 0.03), (44, 0.015), (46, 0.005), (48, 0)],
    1960: [(14, 0), (16, 0.015), (18, 0.05), (20, 0.10), (22, 0.155), (23, 0.185),
           (24, 0.19), (25, 0.185), (26, 0.17), (28, 0.13), (30, 0.10),
           (32, 0.07), (34, 0.05), (36, 0.033), (38, 0.02), (40, 0.012),
           (42, 0.006), (44, 0.002), (46, 0)],
    1990: [(15, 0), (18, 0.02), (20, 0.045), (22, 0.075), (24, 0.105), (26, 0.13),
           (27, 0.138), (28, 0.14), (29, 0.137), (30, 0.125), (32, 0.095),
           (34, 0.065), (36, 0.042), (38, 0.026), (40, 0.015), (42, 0.008),
           (44, 0.003), (46, 0)],
    2024: [(16, 0), (18, 0.005), (20, 0.018), (22, 0.035), (24, 0.052), (26, 0.075),
           (28, 0.10), (30, 0.125), (31, 0.13), (32, 0.125), (33, 0.115),
           (34, 0.10), (36, 0.072), (38, 0.048), (40, 0.03), (42, 0.016),
           (44, 0.007), (46, 0.002), (48, 0)],
}

AGES = list(range(10, 56))


def main():
    rows = []
    for year, pts in ANCHORS.items():
        xs, ys = zip(*pts)
        vals = np.interp(AGES, xs, ys, left=0.0, right=0.0)
        for age, v in zip(AGES, vals):
            rows.append({"year": year, "age": age, "sex": "Female",
                         "country": "Denmark", "births_per_woman": round(float(v), 4)})
    out = Path("data/real_denmark_asfr.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} rows, {len(ANCHORS)} years — DIGITIZED APPROXIMATION)")


if __name__ == "__main__":
    main()
