#!/usr/bin/env python3
"""128K needle retrieval: one unique code buried mid-prompt.

Used to compare KVA off/on (and PATTN output identity) on the same prompt.
"""
import glob
import json
import os
import sys
import time
import urllib.request

base = os.environ.get("STRATA_URL", "http://127.0.0.1:18080")
key = os.environ["STRATA_API_KEY"]
n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 128000
code = os.environ.get("NEEDLE_CODE", "QV-4821-ZT")
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 32

corpus = "\n\n".join(
    open(p, encoding="utf-8", errors="ignore").read()
    for p in sorted(glob.glob(os.path.expanduser("~/Strata/docs/*.md")))
)
while len(corpus) < 2_000_000:
    corpus = corpus + "\n\n" + corpus

n_chars = int(n_tokens * 3.6)
body = corpus[:n_chars]
mid = len(body) // 2
line = f"\n\nIMPORTANT: The access code for vault 7 is {code}.\n\n"
prompt = body[:mid] + line + body[mid:] + (
    "\n\n---\nWhat is the access code for vault 7? Reply with only the code, nothing else."
)

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
t0 = time.time()
with urllib.request.urlopen(req, timeout=3600) as res:
    d = json.loads(res.read().decode())
t1 = time.time()
answer = d["choices"][0]["message"].get("content") or ""
usage = d.get("usage", {})
found = code.lower() in answer.lower()
print(json.dumps({
    "needle": code,
    "prompt": usage.get("prompt_tokens", 0),
    "wall_s": round(t1 - t0, 2),
    "found": found,
    "answer": answer[:400],
}, ensure_ascii=False), flush=True)
