# Model queue (Together.ai)

Working list for the cross-model probe. Run each with:

```bash
python src/run_probe.py --model together/<id> [--stream] [--allow-thinking --max-tokens 1024]
```

Single-test each new model first (`--grid data/test_one.csv --limit 1`) — it
surfaces streaming-only, reasoning, accuracy, and a bad id in one cheap call.

**Handling notes**
- *Streaming-only* models (`*-Plus`, `*-Max`, big tiers) → add `--stream` or you
  get `400 streaming_required`.
- *Reasoning* models inline `<think>…</think>`; the parser now reads only the
  final answer after `</think>`, so leaked mid-thought numbers are ignored. If a
  completion comes back empty (reasoning truncated), raise `--max-tokens`.

## Done

| Model | Together id | Reasoning | Stream | Run |
|---|---|---|---|---|
| Gemma 4 31B Instruct | `google/gemma-4-31B-it` | yes | no | `curious_counts` (`--allow-thinking --max-tokens 1024`) |
| Qwen 3.6 Plus | `Qwen/Qwen3.6-Plus` | yes | **yes** | `lucid_counts` (`--stream`; parser-fix re-extract) |

## Queued (ids confirmed; sentinel-characterized)

All 7 work over `--stream` (none hard-failed). The heavy reasoners need a big
token budget or they truncate before emitting the answer.

| # | Model | Together id | Sentinel | Run flags |
|---|---|---|---|---|
| 1 | Qwen 3.7 Plus | `Qwen/Qwen3.7-Plus` | ✅ clean | `--stream` |
| 2 | Qwen 3.7 Max | `Qwen/Qwen3.7-Max` | ✅ clean | `--stream` |
| 3 | MiniMax M3 | `MiniMaxAI/MiniMax-M3` | ✅ clean | `--stream` |
| 4 | ~~DeepSeek V4 Pro~~ | `deepseek-ai/DeepSeek-V4-Pro` | ❌ **dropped** | bottomless reasoner — never closes `</think>` even at 16k tokens |
| 4b | GPT-OSS 20B | `openai/gpt-oss-20b` | (testing) | swapped in for DeepSeek |
| 5 | Nemotron 3 Ultra 550B | `nvidia/nemotron-3-ultra-550b-a55b` | ⚠️ token-starved | `--stream --allow-thinking --max-tokens 4096` |
| 6 | GLM 5.2 | `zai-org/GLM-5.2` | ⚠️ token-starved | `--stream --allow-thinking --max-tokens 4096` |
| 7 | GPT-OSS 120B | `openai/gpt-oss-120b` | ✅ clean | `--stream` |

Workflow per model: **sentinel (4 calls) → eyeball → full grid.**
