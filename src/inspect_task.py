"""Inspect AI task for the llm-bpw probe.

Runs everywhere through the hf/ provider (model loaded in-process via
transformers), so the laptop dev run and the della production run differ only
in model size. Reuses probe.py for the prompt and the answer parser.

Local dev/smoke-test on a small Qwen (auto-downloads from the HF hub):

    inspect eval src/inspect_task.py@bpw \
        --model hf/Qwen/Qwen3-4B \
        -T grid_path=$PWD/data/sweep_grid.csv -T prompt=era_prior --limit 8

On della (large local Qwen) — see inspect/run_della.slurm:

    inspect eval src/inspect_task.py@bpw \
        --model hf/Qwen3-14B \
        -M model_path=/scratch/gpfs/MSALGANIK/pretrained-llms/Qwen3-14B \
        -T grid_path=$PWD/data/grid.csv -T prompt=era_prior

Then turn the .eval log into a plot-ready results.csv with inspect_to_csv.py.
"""

import csv

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import chain, generate, system_message

from probe import coerce_profile, parse_birth_rate, system_prompt, user_prompt


def _load_samples(grid_path, prompt):
    """Read a grid CSV into Inspect Samples (one per profile row).

    The profile fields ride along as metadata so inspect_to_csv.py can rebuild
    the exact results.csv schema the plotters expect.
    """
    samples = []
    with open(grid_path, newline="") as f:
        for row in csv.DictReader(f):
            profile = coerce_profile(row)
            samples.append(Sample(
                input=user_prompt(profile, prompt),
                metadata={**profile, "prompt": prompt},
            ))
    return MemoryDataset(samples)


@scorer(metrics=[mean()])
def births_scorer():
    """Parse births-per-woman from the reply; expose it as score + metadata.

    There's no ground truth here — this is elicitation, not accuracy — so the
    'score' is simply the parsed rate, carried so it lands in the .eval log.
    """
    async def score(state, target: Target):
        rate = parse_birth_rate(state.output.completion)
        return Score(
            value=rate if rate is not None else float("nan"),
            answer=state.output.completion,
            metadata={"births_per_woman": rate},
        )
    return score


# Qwen3-series soft switch that turns off the model's thinking for the turn.
# We disable thinking to match the laptop path (which sets think=False on the
# Ollama API): a reasoning model otherwise buries the answer in its thinking
# channel and returns an empty completion. Appended at the solver level so
# probe.py's shared system prompt stays transport-neutral. Harmless on
# non-reasoning models (e.g. Qwen2.5-Instruct), which simply ignore it.
_NO_THINK = "/no_think"


@task
def bpw(grid_path: str = "data/grid.csv",
        prompt: str = "era_prior",
        temperature: float = 1e-7,   # near-greedy; hf/transformers rejects 0.0
        max_tokens: int = 128,
        disable_thinking: bool = True) -> Task:
    """Births-per-woman probe over a profile grid, using one prompt variant."""
    system = system_prompt(prompt)
    if disable_thinking:
        system = f"{system}\n{_NO_THINK}"
    return Task(
        dataset=_load_samples(grid_path, prompt),
        solver=chain(
            system_message(system),
            generate(),
        ),
        scorer=births_scorer(),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
    )
