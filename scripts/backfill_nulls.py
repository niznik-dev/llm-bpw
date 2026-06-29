"""Backfill null rows in period runs by re-querying at a larger token budget.

For each run under <runs-dir> with nulls, re-runs just the null profiles at 2x
the run's max_tokens (matching its thinking flag, with streaming), merges the
recovered values into results.csv, then re-plots. Finally rebuilds nulls_log.tsv
and the comparison wall from the post-backfill state.

    python scripts/backfill_nulls.py data/runs/20260629
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

KEY = ["year", "age", "sex", "country"]


def backfill_run(d):
    """Re-query d's nulls at 2x tokens and merge. Returns (before, after)."""
    df = pd.read_csv(d / "results.csv")
    nulls = df[df["births_per_woman"].isna()]
    if not len(nulls):
        return 0, 0
    m = json.loads((d / "metadata.json").read_text())
    targs = m.get("task_args") or {}
    cap = targs.get("max_tokens") or 512
    disable_thinking = targs.get("disable_thinking", True)
    grid = d / "backfill_grid.csv"
    nulls[KEY].to_csv(grid, index=False)

    tmp = d.parent / "_bf_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    cmd = ["python", "src/run_probe.py", "--model", m["model"], "--grid", str(grid),
           "--max-tokens", str(cap * 2), "--stream",
           "--run-name", f"{d.parent.name}/_bf_tmp"]
    if disable_thinking is False:
        cmd.append("--allow-thinking")
    subprocess.run(cmd, check=True)

    bf = pd.read_csv(tmp / "results.csv").set_index(KEY)
    o = df.set_index(KEY)
    o["raw_reply"] = o["raw_reply"].astype(object)
    o.loc[bf.index, "births_per_woman"] = bf["births_per_woman"].values
    o.loc[bf.index, "raw_reply"] = bf["raw_reply"].astype(str).values
    o.reset_index().to_csv(d / "results.csv", index=False)

    shutil.rmtree(tmp, ignore_errors=True)
    grid.unlink(missing_ok=True)
    after = int(pd.read_csv(d / "results.csv")["births_per_woman"].isna().sum())
    return len(nulls), after


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "data/runs/20260629")
    runs = [d for d in sorted(base.glob("*/")) if (d / "results.csv").exists()
            and not d.name.startswith("_")]

    for d in runs:
        (d / "backfill_grid.csv").unlink(missing_ok=True)  # clear any stale grid
    for d in runs:
        before, after = backfill_run(d)
        if before:
            print(f"{d.name}: {before} nulls -> {after} remaining", flush=True)

    print("re-plotting...", flush=True)
    for d in runs:
        subprocess.run(["python", "src/plot_results.py", "--results", str(d / "results.csv"),
                        "--out", str(d / "fertility_schedules.png")], check=True,
                       capture_output=True)
    subprocess.run(["python", "src/plot_compare.py", "--out", str(base / "model_compare.png")],
                   check=True)

    lines = ["timestamp\trun\tmodel\tyear\tage"]
    for d in runs:
        df = pd.read_csv(d / "results.csv")
        m = json.loads((d / "metadata.json").read_text())
        for r in df[df["births_per_woman"].isna()].itertuples():
            lines.append(f"current\t{d.name}\t{m['model']}\t{r.year}\t{r.age}")
    (base.parent / "nulls_log.tsv").write_text("\n".join(lines) + "\n")
    print(f"rebuilt nulls_log: {len(lines) - 1} remaining nulls total", flush=True)
    print("BACKFILL DONE", flush=True)


if __name__ == "__main__":
    main()
