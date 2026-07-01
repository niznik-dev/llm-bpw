"""Shared scientific core for the llm-bpw probe.

The probe runs everywhere through Inspect AI's hf/ provider — a small Qwen on
the laptop (dev/smoke-test) and a large Qwen on della — so the only thing that
changes between runs is the model. This module holds the parts that must stay
identical for those runs to be comparable: the prompt variants and the answer
parser. inspect_task.py imports them.

The default prompt is deliberately bare: it asks a capable model for the
age-specific fertility rate as a plain decimal (e.g. 0.1) and trusts it to
answer. Scaffolding is opt-in via system_prompt() knobs — dial it up only if a
model underperforms without it:
  - give_hint   : add a demographic-pattern hint (off by default — it biases)
  - ask_decimal : add a one-line "reply with only the decimal" instruction
"""

import re

# Optional one-line nudge toward a parseable answer. Minimal output hygiene, not
# the heavy JSON schema we started with.
_DECIMAL_TAIL = " Respond with only the rate as a decimal number (for example, 0.1)."

# Optional demographic-pattern hint. Opt-in: a capable model shouldn't need it,
# and handing it over biases the elicitation toward a known shape.
_PATTERN_HINT = (
    " Real fertility is concentrated in the late 20s to early 30s and falls to "
    "roughly zero in the teens and after the mid-40s."
)


def _noun(sex):
    """Plural group noun for natural phrasing ('women'/'men'/'people')."""
    return {"Female": "women", "Male": "men"}.get(sex, "people")


# --- Prompt variants (phrasings of one period-fertility question) ------------
# Each entry is a {system, user(profile)->str} pair. All ask for the
# age-specific fertility rate in a given country, calendar YEAR, and age — this
# is PERIOD fertility: fix the year, vary age (not a birth cohort followed
# through time). The variants differ only in phrasing / priming; hints and
# output format are opt-in (see system_prompt).

PROMPTS = {
    # Profile listed as fields.
    "baseline": {
        "system": (
            "You are a demographer estimating age-specific fertility rates. "
            "Estimate the age-specific fertility rate for the profile below — "
            "the expected number of births per woman of this age, in this "
            "country, during this calendar year."
        ),
        "user": lambda p: (
            "Profile:\n"
            f"- Year: {p['year']}\n"
            f"- Age: {p['age']}\n"
            f"- Sex: {p['sex']}\n"
            f"- Country: {p['country']}\n"
            "Age-specific fertility rate (births per woman)?"
        ),
    },
    # Phrased as a sentence.
    "year_explicit": {
        "system": (
            "You are a demographer estimating age-specific fertility rates for "
            "a specific country and calendar year."
        ),
        "user": lambda p: (
            f"In {p['country']}, in the year {p['year']}, what is the "
            f"age-specific fertility rate (births per woman) for "
            f"{_noun(p['sex'])} aged {p['age']}?"
        ),
    },
    # As year_explicit, but prime that fertility changes over time.
    "era_prior": {
        "system": (
            "You are a demographer estimating age-specific fertility rates for "
            "a specific country and calendar year. Fertility rates have changed "
            "substantially over time; take the specific historical period into "
            "account."
        ),
        "user": lambda p: (
            f"In {p['country']}, in the year {p['year']}, what is the "
            f"age-specific fertility rate (births per woman) for "
            f"{_noun(p['sex'])} aged {p['age']}?"
        ),
    },
    # Terse field list.
    "period_pure": {
        "system": (
            "You are a demographer reporting historical age-specific fertility "
            "rates."
        ),
        "user": lambda p: (
            f"Country: {p['country']}\n"
            f"Year: {p['year']}\n"
            f"Age: {p['age']}\n"
            "Age-specific fertility rate (births per woman):"
        ),
    },
}

DEFAULT_PROMPT = "baseline"

# Fields every profile row carries, and their types (for CSV coercion).
# `year` is the calendar (period) year — the line identity in a period schedule.
PROFILE_FIELDS = {"year": int, "age": int, "sex": str, "country": str}


def coerce_profile(row):
    """Coerce a raw CSV/dict row into a typed profile dict (ints where needed)."""
    return {k: cast(row[k]) for k, cast in PROFILE_FIELDS.items()}


def system_prompt(prompt=DEFAULT_PROMPT, give_hint=False, ask_decimal=True):
    """System prompt for a variant, with optional opt-in scaffolding.

    give_hint   : append the demographic-pattern hint (default off — biasing).
    ask_decimal : append a one-line decimal-output instruction (default on, for
                  parse reliability; turn off for a fully bare prompt).
    """
    text = PROMPTS[prompt]["system"]
    if give_hint:
        text += _PATTERN_HINT
    if ask_decimal:
        text += _DECIMAL_TAIL
    return text


def user_prompt(profile, prompt=DEFAULT_PROMPT):
    """Return the user message text for a profile under the named variant."""
    return PROMPTS[prompt]["user"](profile)


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_birth_rate(text):
    """Parse a model reply into an age-specific fertility rate (decimal).

    The prompt asks for a bare decimal (e.g. 0.1), but models sometimes wrap it
    in prose. Rates are fractional, so we prefer the first decimal-looking
    number (avoids grabbing an echoed age or year), falling back to the first
    integer (e.g. a flat "0"). Returns None if no number is present.
    """
    if text is None:
        return None
    # Reasoning models may inline a <think>...</think> block before the answer
    # (and /no_think doesn't always suppress it). Parse only the final segment so
    # we don't grab a number from the reasoning (e.g. a TFR mentioned mid-thought).
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    nums = _NUM_RE.findall(text)
    if not nums:
        return None
    decimals = [n for n in nums if "." in n]
    pick = decimals[0] if decimals else nums[0]
    try:
        return float(pick)
    except ValueError:
        return None
