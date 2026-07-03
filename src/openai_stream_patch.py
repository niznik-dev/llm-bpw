"""Tolerate Together's non-spec streamed `logprobs` on Qwen thinking-ON responses.

On the Qwen-thinking-ON *streamed* path, Together returns `logprobs` as a bare
number (e.g. `0`, `-1.79e-06`) instead of the OpenAI-spec object (`{content: …}`)
or `null`. The openai SDK's ChatCompletionStreamState guards logprobs with
`is not None` but then does `choice.logprobs.content` — so `0.content` raises
`AttributeError` and aborts the whole eval (inspect just uses this accumulator).

We coerce any non-object `logprobs` -> None on each incoming chunk choice before
the accumulator touches it, so both `is not None` guards skip cleanly. Import this
module for its side effect (inspect_task does). Fully guarded: if a future openai
layout change breaks the patch, it degrades to a no-op rather than an error.

Root cause is Together's API (verified: the field arrives as int/float), not
inspect or our code; this is a client-side tolerance shim until they fix it.
"""


def _apply():
    try:
        from openai.lib.streaming.chat import _completions as sc
    except Exception:
        return
    state = sc.ChatCompletionStreamState
    if getattr(state, "_together_logprobs_shim", False):
        return
    original = state.handle_chunk

    def handle_chunk(self, chunk):
        for choice in (getattr(chunk, "choices", None) or []):
            lp = getattr(choice, "logprobs", None)
            if lp is not None and not hasattr(lp, "content"):
                try:
                    choice.logprobs = None   # vendor junk (int/float) -> None
                except Exception:
                    pass
        return original(self, chunk)

    state.handle_chunk = handle_chunk
    state._together_logprobs_shim = True


_apply()
