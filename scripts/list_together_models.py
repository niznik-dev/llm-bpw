"""List Together.ai chat models — run it yourself (uses TOGETHER_API_KEY).

Reads the key from the environment or a .env in the current dir (same way
Inspect does), hits the Together models endpoint, and prints chat model ids so
you can pick one to pass as MODEL=together/<id>. The probe never calls this;
it's just a convenience for model discovery.

    python scripts/list_together_models.py                # all chat models
    python scripts/list_together_models.py qwen gemma     # filter by substring(s)
"""

import json
import os
import sys
import urllib.request


def load_dotenv():
    """Minimal .env loader so the key flows the same way Inspect loads it."""
    if os.path.exists(".env"):
        for line in open(".env"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    load_dotenv()
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        sys.exit("Set TOGETHER_API_KEY (or put it in .env).")

    req = urllib.request.Request(
        "https://api.together.xyz/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    payload = json.load(urllib.request.urlopen(req))
    models = payload if isinstance(payload, list) else payload.get("data", [])

    filters = [s.lower() for s in sys.argv[1:]]
    rows = []
    for m in models:
        mid = m.get("id") or m.get("name", "")
        mtype = (m.get("type") or "").lower()
        if mtype and mtype not in ("chat", "language"):
            continue
        if filters and not any(f in mid.lower() for f in filters):
            continue
        ctx = m.get("context_length") or m.get("context_window") or "?"
        rows.append((mid, ctx))

    for mid, ctx in sorted(rows):
        print(f"{mid}\t(ctx {ctx})")
    print(f"\n{len(rows)} models", file=sys.stderr)


if __name__ == "__main__":
    main()
