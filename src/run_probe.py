"""Run the probe end-to-end and bundle one run's outputs into its own folder.

Wraps `inspect eval` so each run lands in
data/runs/<adjective_verb_timestamp>/ with everything needed to tell it apart:

  - results.csv    parsed, plot-ready
  - metadata.json  model, prompt, params, grid, versions, git commit, log name
  - logs/          the run's .eval log

Backend-agnostic — laptop hf/, della hf/, or Together — because it just sets the
--model string Inspect is given.

    # one-sample accuracy check on a hosted model
    python src/run_probe.py --model together/<id> --grid data/grids/test_one.csv --limit 1
    # full local dev run
    python src/run_probe.py --model hf/Qwen/Qwen3-4B --device mps
"""

import argparse
import datetime
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling src modules
from model_meta import stream_for  # noqa: E402

# Run folders are <datestamp>/[<group>/]<model>_<noun> — the model says what, the
# random noun keeps same-model same-day runs apart, and an optional --group
# descriptor (e.g. us_period, dk_cohort) partitions same-day experiments.
_NOUNS = ["otter", "falcon", "cedar", "comet", "harbor", "lantern", "willow",
          "quartz", "raven", "meadow", "ember", "delta", "sparrow", "cobalt",
          "marlin", "juniper", "wren", "basalt", "pylon", "tundra"]


def short_model(model):
    """together/google/gemma-4-31B-it -> gemma-4-31B-it (filesystem-safe leaf)."""
    return model.split("/")[-1] if model else "model"


def make_run_name(model):
    return f"{short_model(model)}_{random.choice(_NOUNS)}"

# 4-profile sentinel: cheap de-risking before a full grid. (year, age, sex,
# country) — period framing. Spans the stress cases: historical-high,
# modern-peak, young-edge, and a future-year/older-age combo that exposes
# streaming, reasoning, and parsing problems.
SENTINEL_ROWS = [
    (1920, 28, "Female", "Denmark"),
    (1990, 28, "Female", "Denmark"),
    (1990, 16, "Female", "Denmark"),
    (2024, 45, "Female", "Denmark"),
]


def build_metadata(args, name, now, grid, log_dir, eval_log):
    """Collect run metadata — preferring the .eval log for effective params."""
    meta = {
        "run_name": name,
        "timestamp": now.isoformat(timespec="seconds"),
        "model": args.model,
        "prompt": args.prompt,
        "grid": str(grid),
        "limit": args.limit,
    }
    try:
        with open(grid) as f:
            meta["grid_rows"] = sum(1 for _ in f) - 1
    except OSError:
        pass
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        pass
    try:
        import inspect_ai
        meta["inspect_ai_version"] = getattr(inspect_ai, "__version__", "?")
    except Exception:
        pass

    if eval_log is not None:
        from inspect_ai.log import read_eval_log
        meta["eval_log"] = eval_log.name
        try:
            log = read_eval_log(str(eval_log))
            meta["effective_model"] = log.eval.model
            meta["task_args"] = dict(log.eval.task_args or {})
            meta["completed_samples"] = len(log.samples or [])
            usage = (log.stats.model_usage or {}) if log.stats else {}
            meta["token_usage"] = {
                m: {"input": u.input_tokens, "output": u.output_tokens}
                for m, u in usage.items()
            }
        except Exception as e:
            meta["metadata_note"] = f"partial log read: {e}"
    return meta


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Inspect model string, e.g. together/<id>.")
    p.add_argument("--prompt", default="baseline", help="Prompt variant (default: %(default)s).")
    p.add_argument("--grid", default="data/grids/grid.csv", help="Grid CSV (default: %(default)s).")
    p.add_argument("--limit", type=int, help="Cap number of samples (e.g. 1 for a test).")
    p.add_argument("--sentinel", action="store_true",
                   help="Run the built-in 4-profile sentinel instead of --grid "
                        "(de-risk a new model before the full run).")
    p.add_argument("--device", help="hf/ device, e.g. mps or auto (omit for hosted APIs).")
    p.add_argument("--stream", action="store_true",
                   help="Force streaming for models NOT in model_meta.yaml (e.g. hf/ "
                        "local). Registry models stream per their `stream:` entry.")
    p.add_argument("--allow-thinking", action="store_true",
                   help="Legacy /no_think toggle: don't append /no_think (hf/ "
                        "local models). Superseded by --thinking for Together.")
    p.add_argument("--thinking", choices=["off", "on"], default=None,
                   help="Explicit per-model thinking control via extra_body "
                        "(from model_meta.yaml — authoritative on Together). Omit "
                        "to use the legacy /no_think default.")
    p.add_argument("--give-hint", action="store_true", help="Add the demographic-pattern hint.")
    p.add_argument("--no-decimal", action="store_true", help="Drop the decimal-output instruction.")
    p.add_argument("--temperature", type=float, help="Override sampling temperature.")
    p.add_argument("--max-tokens", type=int, help="Override max output tokens.")
    p.add_argument("--run-name", help="Override the generated run-folder name.")
    p.add_argument("--runs-dir", default="data/runs", help="Parent of run folders (default: %(default)s).")
    p.add_argument("--group", help="Descriptor subfolder between the datestamp and the "
                   "model, e.g. us_period / dk_cohort. Layout: runs/<date>/<group>/<model>_<noun>/.")
    args = p.parse_args()

    now = datetime.datetime.now()
    # Default layout: runs/<datestamp>/<model>_<noun>/. --group inserts a descriptor
    # level (runs/<datestamp>/<group>/<model>_<noun>/) so same-day experiments don't
    # collide; --run-name overrides the whole subpath (sentinels under runs/sentinel/).
    if args.run_name:
        name = args.run_name
        run_dir = Path(args.runs_dir) / name
    else:
        name = make_run_name(args.model)
        base = Path(args.runs_dir) / now.strftime("%Y%m%d")
        run_dir = (base / args.group / name) if args.group else (base / name)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.sentinel:
        import csv as _csv
        grid = (run_dir / "sentinel_grid.csv").resolve()
        with open(grid, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["year", "age", "sex", "country"])
            w.writerows(SENTINEL_ROWS)
    else:
        grid = Path(args.grid).resolve()

    cmd = ["inspect", "eval", "src/inspect_task.py@bpw", "--model", args.model,
           "-T", f"grid_path={grid}", "-T", f"prompt={args.prompt}",
           "--log-dir", str(log_dir)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.device:
        cmd += ["-M", f"device={args.device}"]
    # Streaming is a per-model Together fact (model_meta.yaml `stream`): stream
    # unless the model is 'forbidden'; --stream only forces it for unlisted models.
    mode = stream_for(short_model(args.model))
    if mode == "forbidden":
        if args.stream:
            print(f"NB: --stream ignored — {short_model(args.model)} hangs when "
                  "streamed (stream: forbidden).")
        do_stream = False
    elif mode in ("required", "either"):
        do_stream = True
    else:  # unlisted (e.g. hf/ local) — honor the flag
        do_stream = args.stream
    if do_stream:
        cmd += ["-M", "stream=true"]
    if args.allow_thinking:
        cmd += ["-T", "disable_thinking=false"]
    if args.thinking:
        cmd += ["-T", f"thinking={args.thinking}",
                "-T", f"model_key={short_model(args.model)}"]
    if args.give_hint:
        cmd += ["-T", "give_hint=true"]
    if args.no_decimal:
        cmd += ["-T", "ask_decimal=false"]
    if args.temperature is not None:
        cmd += ["-T", f"temperature={args.temperature}"]
    if args.max_tokens is not None:
        cmd += ["-T", f"max_tokens={args.max_tokens}"]

    print(f"=== run {name} ===\n{' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)

    results = run_dir / "results.csv"
    subprocess.run(["python", "src/inspect_to_csv.py", "--latest",
                    "--log-dir", str(log_dir), "--out", str(results)], check=True)

    logs = sorted(log_dir.glob("*.eval"))
    meta = build_metadata(args, name, now, grid, log_dir, logs[-1] if logs else None)
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nBundled run -> {run_dir}/")
    print("  results.csv  metadata.json  logs/")

    # Central null log — appended to so nulls can be reviewed/backfilled later.
    import pandas as pd
    nulls = pd.read_csv(results)
    nulls = nulls[nulls["births_per_woman"].isna()]
    if len(nulls):
        # The line dimension is `year` (period) or `cohort` (cohort) — log whichever
        # this grid used so cohort runs don't trip on a missing `year` column.
        dim = "cohort" if "cohort" in nulls.columns else "year"
        log = Path(args.runs_dir) / "nulls_log.tsv"
        header = not log.exists()
        with open(log, "a") as fh:
            if header:
                fh.write("timestamp\trun\tmodel\tkey\tage\n")
            for r in nulls.itertuples():
                fh.write(f"{now:%Y%m%d_%H%M%S}\t{name}\t{args.model}\t"
                         f"{getattr(r, dim)}\t{r.age}\n")
        print(f"  logged {len(nulls)} null(s) to {log}")


if __name__ == "__main__":
    main()
