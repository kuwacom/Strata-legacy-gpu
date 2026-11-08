#!/usr/bin/env python3
"""prefill TTFT bench: fixed-seed prompt built from the repo docs.

Same seeding scheme as ~/bench.py so two arms get the same prompt per run index.
A per-run salt (deterministic from BENCH_SEED / size / kind / index) keeps the
conversation cache from reusing a prefix across runs: without it the engine
reports "16384 reused + N read" and the TTFT is not a prefill time.
Prints one JSON line per request.
"""
import glob
import hashlib
import json
import os
import random
import sys
import time
import urllib.request

base = os.environ.get("STRATA_URL", "http://127.0.0.1:18080")
key = os.environ["STRATA_API_KEY"]
sizes = [int(x) for x in sys.argv[1].split(",")]
repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 1
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 1
kind = sys.argv[4] if len(sys.argv) > 4 else "prose"
seed_env = os.environ.get("BENCH_SEED", "42")

corpus = "\n\n".join(
    open(p, encoding="utf-8", errors="ignore").read()
    for p in sorted(glob.glob(os.path.expanduser("~/Strata/docs/*.md")))
)
# 480K トークン級でも足りるよう、コーパスを繰り返して長くする
while len(corpus) < 2_000_000:
    corpus = corpus + "\n\n" + corpus

tasks = {
    "code": "Based on the documents above, write a Python script that parses the markdown tables in them into JSON, with type hints and a few unit tests.",
    "prose": "Based on the documents above, explain in detail how this system works and what its main trade-offs are.",
}
seed_rnd = random.Random(int(seed_env))


def salt_of(n_tokens: int, kind: str, idx: int) -> str:
    """Prefix unique to this run index; identical in every arm (BENCH_SEED 依存)"""
    h = hashlib.sha256(f"{seed_env}:{n_tokens}:{kind}:{idx}".encode()).digest()
    return "".join(chr(ord("a") + (b % 26)) for b in h[:64])


def make_prompt(n_tokens: int, kind: str, idx: int) -> str:
    rnd = random.Random(seed_rnd.random())
    n_chars = int(n_tokens * 3.6)
    start = rnd.randint(0, max(0, len(corpus) - n_chars - 1))
    start = corpus.find("\n", start) + 1
    return salt_of(n_tokens, kind, idx) + "\n\n" + corpus[start:start + n_chars] + "\n\n---\n" + tasks[kind]


def run(n_tokens: int, kind: str, idx: int) -> dict:
    prompt = make_prompt(n_tokens, kind, idx)
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    t_first = None
    n_chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as res:
        for line in res:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")):
                if t_first is None:
                    t_first = time.time()
                n_chunks += 1
    t_end = time.time()
    p = usage.get("prompt_tokens", 0)
    c = usage.get("completion_tokens", n_chunks)
    ttft = (t_first or t_end) - t0
    return {
        "kind": kind,
        "run": idx,
        "target_tokens": n_tokens,
        "prompt": p,
        "ttft_s": round(ttft, 2),
        "prefill_tps": round(p / ttft, 1) if ttft else 0,
        "completion": c,
    }


for n in sizes:
    for i in range(repeat):
        print(json.dumps(run(n, kind, i)), flush=True)
