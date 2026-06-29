"""Shared scientific core for the llm-bpw probe.

The probe runs everywhere through Inspect AI's hf/ provider — a small Qwen on
the laptop (dev/smoke-test) and a large Qwen on della — so the only thing that
changes between runs is the model. This module holds the parts that must stay
identical for those runs to be comparable: the prompt variants and the answer
parser. inspect_task.py imports them.
"""

import json
import re

# JSON-output instruction appended to every variant's system prompt. The hf
# provider can't hard-enforce a schema, so we instruct the format here and rely
# on parse_birth_rate() below to read it back tolerantly.
_JSON_TAIL = (
    " Answer with your best point estimate as JSON of the form "
    "{\"births_per_1000\": <integer>}. Do not explain."
)

# Demographic-pattern hint shared by the cohort-implicit variants.
_PATTERN_HINT = (
    "Use real-world demographic patterns: births are concentrated in the late "
    "20s to early 30s and fall to roughly zero in the teens and after the "
    "mid-40s."
)


def _noun(sex):
    """Plural group noun for natural phrasing ('women'/'men'/'people')."""
    return {"Female": "women", "Male": "men"}.get(sex, "people")


# --- Prompt variants, ordered along a cohort-salience gradient ---------------
# Each entry maps a name to a {system, user(profile)->str} pair. The science we
# vary is ONLY how visible the birth cohort / calendar year is; scale anchoring
# (per 1000, one year) is held fixed via _JSON_TAIL.

PROMPTS = {
    # Current prompt: cohort given only as a separate "year of birth" field.
    "baseline": {
        "system": (
            "You are a demographer estimating age-specific fertility rates. "
            "Consider a hypothetical group of 1000 people who all share the "
            "given profile. Estimate how many of them will give birth during "
            "the next 12 months — a single-year count, not a lifetime total. "
            + _PATTERN_HINT + _JSON_TAIL
        ),
        "user": lambda p: (
            "Of 1000 people with the following profile, how many give birth in "
            "the next 12 months?\n"
            f"- Year of birth: {p['year_of_birth']}\n"
            f"- Current age: {p['age']}\n"
            f"- Sex: {p['sex']}\n"
            f"- Country: {p['country']}"
        ),
    },
    # State the actual calendar year (year_of_birth + age) so the historical
    # period is explicit rather than something the model must infer.
    "year_explicit": {
        "system": (
            "You are a demographer estimating age-specific fertility rates for "
            "a specific country and calendar year. Estimate, out of 1000 people "
            "of the stated age, how many give birth during that year. "
            + _PATTERN_HINT + _JSON_TAIL
        ),
        "user": lambda p: (
            f"The calendar year is {p['year_of_birth'] + p['age']}. In "
            f"{p['country']}, of 1000 {_noun(p['sex'])} who are {p['age']} years "
            f"old (born in {p['year_of_birth']}), how many give birth during "
            f"that year?"
        ),
    },
    # As year_explicit, but prime the model that fertility has changed over time.
    "era_prior": {
        "system": (
            "You are a demographer estimating age-specific fertility rates for "
            "a specific country and calendar year. Fertility rates have changed "
            "enormously over time: they were substantially higher in the early "
            "and mid 20th century and have fallen markedly in many countries "
            "since. Take the specific historical period into account. Estimate, "
            "out of 1000 people of the stated age, how many give birth during "
            "that year." + _JSON_TAIL
        ),
        "user": lambda p: (
            f"The calendar year is {p['year_of_birth'] + p['age']}. In "
            f"{p['country']}, of 1000 {_noun(p['sex'])} who are {p['age']} years "
            f"old (born in {p['year_of_birth']}), how many give birth during "
            f"that year?"
        ),
    },
    # Pure period question: no cohort framing at all, just country/year/age.
    "period_pure": {
        "system": (
            "You are a demographer. Report the actual historical age-specific "
            "fertility rate for the given country, calendar year, and age, "
            "expressed as the number of live births per 1000 women of that age "
            "during that year." + _JSON_TAIL
        ),
        "user": lambda p: (
            f"Country: {p['country']}\n"
            f"Calendar year: {p['year_of_birth'] + p['age']}\n"
            f"Age of women: {p['age']}\n"
            "Births per 1000 women of this age during this year?"
        ),
    },
}

DEFAULT_PROMPT = "baseline"

# Fields every profile row carries, and their types (for CSV coercion).
PROFILE_FIELDS = {"year_of_birth": int, "age": int, "sex": str, "country": str}


def coerce_profile(row):
    """Coerce a raw CSV/dict row into a typed profile dict (ints where needed)."""
    return {k: cast(row[k]) for k, cast in PROFILE_FIELDS.items()}


def system_prompt(prompt=DEFAULT_PROMPT):
    """Return the system prompt text for the named variant."""
    return PROMPTS[prompt]["system"]


def user_prompt(profile, prompt=DEFAULT_PROMPT):
    """Return the user message text for a profile under the named variant."""
    return PROMPTS[prompt]["user"](profile)


_PER_1000_RE = re.compile(r"births_per_1000\D+(\d+(?:\.\d+)?)")
_FIRST_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_birth_rate(text):
    """Parse a model reply into births-per-woman (rate), or None if unparseable.

    Tolerant by design: the hf provider returns free text that merely *contains*
    the requested JSON (sometimes fenced, multi-line, or with stray prose). We
    try, in order:
      1. strict JSON object with births_per_1000,
      2. a `births_per_1000: N` substring,
      3. the first number anywhere in the reply.
    Whatever we find is a count per 1000, so we divide by 1000.
    """
    if text is None:
        return None
    candidate = None
    try:
        candidate = json.loads(text)["births_per_1000"]
    except (json.JSONDecodeError, KeyError, TypeError):
        m = _PER_1000_RE.search(text) or _FIRST_NUM_RE.search(text)
        if m:
            candidate = m.group(1) if m.re is _PER_1000_RE else m.group(0)
    if candidate is None:
        return None
    try:
        return float(candidate) / 1000.0
    except (TypeError, ValueError):
        return None
