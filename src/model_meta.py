"""Load the model-metadata registry (model_meta.yaml) for the plotters.

The registry itself is YAML (editable without touching code); this module is the
thin access layer — load it, look models up by short name, and order/annotate them
by ``is_best`` (best-available-on-Together vs a weaker sibling). See the YAML header
for why we dropped the arbitrary S/M/L/XL size tiers in favor of this binary.
"""

from pathlib import Path

import yaml

DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / "model_meta.yaml"

# Compact, font-safe glyphs for panel badges (keyed on thinking_configured):
# filled = reasoning on, empty = off, half = switch-ignored/ambiguous.
THINKING_GLYPH = {"on": "● on", "off": "○ off", "ambiguous": "◐ amb"}

# YAML parses bare on/off as booleans, so `thinking_configured: on` arrives as
# True. Map it back to the string states the glyphs expect.
_THINKING_FROM_BOOL = {True: "on", False: "off"}


def load_registry(path=DEFAULT_REGISTRY):
    """Return {short_model_name: {...meta...}} or {} if the registry is absent."""
    path = Path(path)
    if not path.exists():
        return {}
    reg = yaml.safe_load(path.read_text()) or {}
    for meta in reg.values():
        if isinstance(meta.get("thinking_configured"), bool):
            meta["thinking_configured"] = _THINKING_FROM_BOOL[meta["thinking_configured"]]
    return reg


def meta_for(model, registry=None):
    """Metadata dict for a short model name, or a safe unknown-stub if unlisted."""
    registry = load_registry() if registry is None else registry
    return registry.get(model, {
        "is_best": False, "thinking_configured": "ambiguous", "provider": "?",
        "params_total_b": None, "params_active_b": None,
        "cost_output_per_mtok": None,
    })


def cost_rank(model, registry=None):
    """Sort key: cheapest output price first; unknown/None costs sort last."""
    cost = meta_for(model, registry).get("cost_output_per_mtok")
    return (float("inf") if cost is None else cost, model)


def best_rank(model, registry=None):
    """Sort key: best-available models first, then by name (legacy ordering)."""
    return (0 if meta_for(model, registry).get("is_best") else 1, model)


def badge(model, registry=None):
    """One-line annotation for a panel title: output cost · best-flag."""
    m = meta_for(model, registry)
    cost = m.get("cost_output_per_mtok")
    price = f"${cost:.2f}/M" if cost is not None else "$?/M"
    flag = "★ best" if m.get("is_best") else "○ alt"
    return f"{price} · {flag}"


def thinking_extra_body(model, state, registry=None):
    """`extra_body` dict to force a model's reasoning `state` ('off'|'on').

    Returns the per-model control from the registry (Qwen/Gemma/Nemotron use
    chat_template_kwargs, GLM uses reasoning, MiniMax uses thinking) for Inspect's
    GenerateConfig(extra_body=...). Returns {} when the model isn't configured
    (e.g. an hf/ local model), so the caller can fall back to the /no_think path.
    """
    return meta_for(model, registry).get(f"thinking_{state}") or {}


def stream_for(model, registry=None):
    """Together streaming constraint: 'required' | 'forbidden' | 'either', or None
    if the model isn't in the registry (e.g. an hf/ local model). run_probe streams
    unless 'forbidden'; None means fall back to the --stream flag."""
    return meta_for(model, registry).get("stream")
