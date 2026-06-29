"""Run the probe end-to-end and bundle one run's outputs into its own folder.

Wraps `inspect eval` so each run lands in
data/runs/<adjective_verb_timestamp>/ with everything needed to tell it apart:

  - results.csv    parsed, plot-ready
  - metadata.json  model, prompt, params, grid, versions, git commit, log name
  - logs/          the run's .eval log

Backend-agnostic — laptop hf/, della hf/, or Together — because it just sets the
--model string Inspect is given.

    # one-sample accuracy check on a hosted model
    python src/run_probe.py --model together/<id> --grid data/test_one.csv --limit 1
    # full local dev run
    python src/run_probe.py --model hf/Qwen/Qwen3-4B --device mps
"""

import argparse
import datetime
import json
import random
import subprocess
from pathlib import Path

# adjective_verb run names — easy to say, hard to confuse at a glance.
_ADJECTIVES = ["brave", "curious", "gentle", "quiet", "eager", "clever", "calm",
               "bold", "wry", "keen", "merry", "lucid", "stoic", "nimble",
               "cosmic", "amber", "velvet", "patient"]
_VERBS = ["probes", "counts", "asks", "charts", "maps", "tallies", "queries",
          "samples", "gauges", "weighs", "reckons", "surveys"]


def make_run_name(now):
    return f"{random.choice(_ADJECTIVES)}_{random.choice(_VERBS)}_{now:%Y%m%d_%H%M%S}"


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
    p.add_argument("--grid", default="data/grid.csv", help="Grid CSV (default: %(default)s).")
    p.add_argument("--limit", type=int, help="Cap number of samples (e.g. 1 for a test).")
    p.add_argument("--device", help="hf/ device, e.g. mps or auto (omit for hosted APIs).")
    p.add_argument("--allow-thinking", action="store_true",
                   help="Don't append /no_think (use for non-Qwen reasoning "
                        "models like Gemma; pair with a larger --max-tokens).")
    p.add_argument("--give-hint", action="store_true", help="Add the demographic-pattern hint.")
    p.add_argument("--no-decimal", action="store_true", help="Drop the decimal-output instruction.")
    p.add_argument("--temperature", type=float, help="Override sampling temperature.")
    p.add_argument("--max-tokens", type=int, help="Override max output tokens.")
    p.add_argument("--run-name", help="Override the generated run-folder name.")
    p.add_argument("--runs-dir", default="data/runs", help="Parent of run folders (default: %(default)s).")
    args = p.parse_args()

    now = datetime.datetime.now()
    name = args.run_name or make_run_name(now)
    run_dir = Path(args.runs_dir) / name
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    grid = Path(args.grid).resolve()

    cmd = ["inspect", "eval", "src/inspect_task.py@bpw", "--model", args.model,
           "-T", f"grid_path={grid}", "-T", f"prompt={args.prompt}",
           "--log-dir", str(log_dir)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.device:
        cmd += ["-M", f"device={args.device}"]
    if args.allow_thinking:
        cmd += ["-T", "disable_thinking=false"]
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


if __name__ == "__main__":
    main()
