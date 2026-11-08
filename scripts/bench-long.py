"""1 本の長文プロンプトを、アーム間で同じ位置から作って投げる（128K / 480K 用）。

オフセットは BENCH_SEED と n_tokens から決まるので、どのアームでも同じ本文になる。
送る前に /v1/messages/count_tokens で実トークン数を測り、max_context を超えないよう縮める
（bench.py の 3.6 chars/token は markdown では甘く、実測は約 2.86 chars/token）。
"""
import json
import os
import random
import sys
import time
import urllib.request

base = os.environ.get("STRATA_URL", "http://127.0.0.1:8080")
key = os.environ["STRATA_API_KEY"]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 480000
kind = sys.argv[2] if len(sys.argv) > 2 else "prose"
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128
label = os.environ.get("BENCH_LABEL", "")
limit = int(os.environ.get("PROMPT_TOKEN_LIMIT", "515000"))

cf = os.environ["CORPUS_FILE"]
corpus = open(cf, encoding="utf-8", errors="ignore").read()

tasks = {
    "code": "Based on the documents above, write a Python script that parses the markdown tables in them into JSON, with type hints and a few unit tests.",
    "prose": "Based on the documents above, explain in detail how this system works and what its main trade-offs are.",
}


def count_tokens(text):
    body = {"model": "default", "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(base + "/v1/messages/count_tokens", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as res:
        return int(json.load(res)["input_tokens"])


n_chars = int(n * float(os.environ.get("CHARS_PER_TOKEN", "2.86")))
if n_chars >= len(corpus):
    raise SystemExit(f"corpus too small: {len(corpus)} chars, need {n_chars}")
rnd = random.Random(int(os.environ.get("BENCH_SEED", "42")) * 100000 + n)
start = rnd.randint(0, len(corpus) - n_chars - 1)
start = corpus.find("\n", start) + 1
prompt = corpus[start:start + n_chars] + "\n\n---\n" + tasks[kind]
counted = count_tokens(prompt)
for _ in range(6):
    if counted <= limit:
        break
    n_chars = int(n_chars * 0.93)
    start = rnd.randint(0, len(corpus) - n_chars - 1)
    start = corpus.find("\n", start) + 1
    prompt = corpus[start:start + n_chars] + "\n\n---\n" + tasks[kind]
    counted = count_tokens(prompt)
print(f"count_tokens {counted} (limit {limit}), {len(prompt)} chars", flush=True)

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
dec = t_end - (t_first or t_end)
print(json.dumps({
    "label": label,
    "kind": kind,
    "requested_tokens": n,
    "counted_tokens": counted,
    "prompt": p,
    "ttft_s": round(ttft, 2),
    "prefill_tps": round(p / ttft, 1) if ttft else 0,
    "completion": c,
    "decode_tps": round((c - 1) / dec, 2) if dec > 0 and c > 1 else 0,
}), flush=True)

