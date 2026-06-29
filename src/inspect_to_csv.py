"""Convert an Inspect .eval log into a plot-ready results.csv.

Emits the canonical results.csv schema — year_of_birth, age, sex, country,
births_per_woman, raw_reply, prompt — so plot_results.py and plot_sweep.py
consume Inspect output directly.

    python src/inspect_to_csv.py logs/2026-06-29-bpw.eval --out data/results.csv
    python src/inspect_to_csv.py --latest --out data/results.csv   # newest log in ./logs
"""

import argparse
import csv
from pathlib import Path

from inspect_ai.log import read_eval_log

from probe import PROFILE_FIELDS, parse_birth_rate

COLUMNS = list(PROFILE_FIELDS) + ["births_per_woman", "raw_reply", "prompt"]


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
    rows = []
    for s in log.samples or []:
        md = s.metadata or {}
        completion = s.output.completion if s.output else ""
        rows.append({
            "year_of_birth": md.get("year_of_birth"),
            "age": md.get("age"),
            "sex": md.get("sex"),
            "country": md.get("country"),
            "births_per_woman": parse_birth_rate(completion),
            "raw_reply": completion,
            "prompt": md.get("prompt"),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out} (from {log_path})")


if __name__ == "__main__":
    main()
