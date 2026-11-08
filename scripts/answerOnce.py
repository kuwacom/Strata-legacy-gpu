#!/usr/bin/env python3
"""One deterministic 24K answer, for output-identity checks (PATTN_SM60 etc.)."""
import glob
import json
import os
import sys
import urllib.request

base = os.environ.get("STRATA_URL", "http://127.0.0.1:18080")
key = os.environ["STRATA_API_KEY"]
n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128

corpus = "\n\n".join(
    open(p, encoding="utf-8", errors="ignore").read()
    for p in sorted(glob.glob(os.path.expanduser("~/Strata/docs/*.md")))
)
n_chars = int(n_tokens * 3.6)
body = corpus[:n_chars]
prompt = body + "\n\n---\nList the first five markdown headings that appear in the documents above, in order, one per line."

req = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=3600) as res:
    d = json.loads(res.read().decode())
answer = d["choices"][0]["message"].get("content") or ""
print(json.dumps({
    "prompt": d.get("usage", {}).get("prompt_tokens", 0),
    "answer": answer,
}, ensure_ascii=False), flush=True)
