"""Convert an Inspect .eval log into a plot-ready results.csv.

Emits the canonical results.csv schema — year, age, sex, country,
births_per_woman, raw_reply, prompt — so plot_results.py and plot_sweep.py
consume Inspect output directly.

    python src/inspect_to_csv.py logs/2026-06-29-bpw.eval --out data/results.csv
    python src/inspect_to_csv.py --latest --out data/results.csv   # newest log in ./logs
"""

import argparse
import csv
from pathlib import Path

from inspect_ai.log import read_eval_log

from probe import PROFILE_FIELDS, parse_birth_rate, profile_schema

TRAILING = ["births_per_woman", "raw_reply", "prompt"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", nargs="?", type=Path, help="Path to an .eval log file.")
    p.add_argument("--latest", action="store_true",
                   help="Use the newest .eval log under --log-dir instead.")
    p.add_argument("--log-dir", type=Path, default=Path("logs"),
                   help="Where to look for logs with --latest (default: %(default)s).")
    p.add_argument("--out", type=Path, default=Path("data/results.csv"),
                   help="Output CSV path (default: %(default)s).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.latest:
        logs = sorted(args.log_dir.glob("*.eval"))
        if not logs:
            raise SystemExit(f"No .eval logs found in {args.log_dir}.")
        log_path = logs[-1]
    elif args.log:
        log_path = args.log
    else:
        raise SystemExit("Give a log path or pass --latest.")

    log = read_eval_log(str(log_path))
    samples = log.samples or []
    # Detect the line dimension (year vs cohort) from the first sample's metadata,
    # so a cohort run emits a `cohort` column instead of `year`.
    fields = list(profile_schema(samples[0].metadata) if samples else PROFILE_FIELDS)
    columns = fields + TRAILING

    rows = []
    for s in samples:
        md = s.metadata or {}
        completion = s.output.completion if s.output else ""
        row = {k: md.get(k) for k in fields}
        row["births_per_woman"] = parse_birth_rate(completion)
        row["raw_reply"] = completion
        row["prompt"] = md.get("prompt")
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out} (from {log_path})")


if __name__ == "__main__":
    main()
