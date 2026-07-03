"""Retry null rows by re-querying at DOUBLE the run's token budget.

Improves on the earlier CSV-only patch (backfill_nulls.py) so results are
paper-traceable:
  - PRESERVES the retry .eval log — copied into the run's own logs/ as
    backfill_<ts>_<name>.eval (the old script discarded it)
  - tags recovered rows with a `backfilled` column in results.csv
  - records each attempt in metadata.json under "backfills"

Recovery ONLY — it does not re-plot. Re-run the plot sweep afterward with the
right --ymax/--real flags. Idempotent: only still-null rows are re-queried, so a
second pass (optionally a bigger --factor) chips at whatever stayed null.

    python scripts/retry_nulls.py data/runs/20260701
    python scripts/retry_nulls.py data/runs/20260701 --factor 4   # quadruple tokens
    python scripts/retry_nulls.py data/runs/20260701 --only Qwen3.6-Plus_basalt
"""

import argparse
import datetime
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

# Stage retries OUTSIDE any runs-dir so the plotters' recursive globs never pick
# them up as extra "models"; it (and its own nulls_log) is deleted after use.
SCRATCH = Path("data/_retry")


def retry_run(d, factor, stream=True):
    """Re-query d's nulls at factor x tokens; merge, tag, preserve log. Returns
    (attempted, remaining). Matches the run's line dimension (year|cohort) and
    thinking control (explicit --thinking, else the legacy /no_think toggle)."""
    df = pd.read_csv(d / "results.csv")
    if "backfilled" not in df.columns:
        df["backfilled"] = False
    nulls = df[df["births_per_woman"].isna()]
    if nulls.empty:
        return 0, 0

    dim = "cohort" if "cohort" in df.columns else "year"
    key = [dim, "age", "sex", "country"]
    meta = json.loads((d / "metadata.json").read_text())
    targs = meta.get("task_args") or {}
    cap = int(targs.get("max_tokens") or 512)
    new_cap = cap * factor
    thinking = targs.get("thinking")
    if isinstance(thinking, bool):  # -T bool trap: off->False, on->True
        thinking = "on" if thinking else "off"

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    retry_name = f"{d.name}_bf_{ts}"
    grid = d / "retry_grid.csv"
    nulls[key].to_csv(grid, index=False)

    shutil.rmtree(SCRATCH, ignore_errors=True)
    cmd = ["python", "src/run_probe.py", "--model", meta["model"], "--grid", str(grid),
           "--prompt", meta.get("prompt") or "baseline",  # match the run's prompt variant
           "--max-tokens", str(new_cap),
           "--runs-dir", str(SCRATCH), "--run-name", retry_name]
    if stream:
        cmd.append("--stream")
    if isinstance(thinking, str) and thinking:      # explicit per-model control
        cmd += ["--thinking", thinking]
    elif targs.get("disable_thinking", True) is False:  # legacy /no_think toggle
        cmd.append("--allow-thinking")
    subprocess.run(cmd, check=True)

    retry_dir = SCRATCH / retry_name
    bf = pd.read_csv(retry_dir / "results.csv").set_index(key)
    o = df.set_index(key)
    o["raw_reply"] = o["raw_reply"].astype(object)
    idx = bf.index[bf["births_per_woman"].notna()]  # profiles the retry recovered
    o.loc[idx, "births_per_woman"] = bf.loc[idx, "births_per_woman"].values
    o.loc[idx, "raw_reply"] = bf.loc[idx, "raw_reply"].astype(str).values
    o.loc[idx, "backfilled"] = True
    o.reset_index().to_csv(d / "results.csv", index=False)

    # Preserve provenance: the retry log lands in the run's own logs/, and the
    # attempt is recorded in metadata (budget, counts) — no ghost run folder left.
    (d / "logs").mkdir(exist_ok=True)
    for ev in (retry_dir / "logs").glob("*.eval"):
        shutil.copy2(ev, d / "logs" / f"backfill_{ts}_{ev.name}")
    meta.setdefault("backfills", []).append({
        "timestamp": ts, "max_tokens": new_cap,
        "attempted": int(len(nulls)), "recovered": int(len(idx)),
    })
    (d / "metadata.json").write_text(json.dumps(meta, indent=2))

    grid.unlink(missing_ok=True)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    remaining = int(pd.read_csv(d / "results.csv")["births_per_woman"].isna().sum())
    return len(nulls), remaining


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs_dir")
    p.add_argument("--factor", type=int, default=2, help="Token-budget multiplier (default: 2x).")
    p.add_argument("--only", help="Restrict to one run folder name (for a cheap smoke test).")
    p.add_argument("--no-stream", action="store_true",
                   help="Retry without --stream (e.g. models that hang on it).")
    args = p.parse_args()

    base = Path(args.runs_dir)
    runs = [d for d in sorted(base.glob("*/")) if (d / "results.csv").exists()
            and not d.name.startswith("_")
            and (args.only is None or d.name == args.only)]

    grand_before = grand_after = 0
    for d in runs:
        before, after = retry_run(d, args.factor, stream=not args.no_stream)
        if before:
            grand_before += before
            grand_after += after
            print(f"{d.name}: {before} nulls -> {after} remaining "
                  f"({before - after} recovered)", flush=True)
    print(f"\nTotal: {grand_before} -> {grand_after} remaining "
          f"({grand_before - grand_after} recovered) across {len(runs)} run(s).")
    print("Re-run the plot sweep to refresh figures with the recovered values.")


if __name__ == "__main__":
    main()
