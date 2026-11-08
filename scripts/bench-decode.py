"""decode A/B bench: fixed seed, corpus override, one JSON line per request.

~/bench.py と同じプロンプト生成だが、コーパスを CORPUS_FILE で差し替えられる。
BENCH_SEED を固定すると、リクエスト順に同じ乱数列を使う（アーム間でプロンプトを揃える）。
"""
import glob
import json
import os
import random
import sys
import time
import urllib.request

base = os.environ.get("STRATA_URL", "http://127.0.0.1:8080")
key = os.environ["STRATA_API_KEY"]
sizes = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [4000, 24000]
repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 3
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 256
kinds = sys.argv[4].split(",") if len(sys.argv) > 4 else ["code", "prose"]
label = os.environ.get("BENCH_LABEL", "")

cf = os.environ.get("CORPUS_FILE")
paths = [cf] if cf else sorted(glob.glob(os.path.expanduser("~/Strata/docs/*.md")))
corpus = "\n\n".join(open(p, encoding="utf-8", errors="ignore").read() for p in paths)

tasks = {
    "code": "Based on the documents above, write a Python script that parses the markdown tables in them into JSON, with type hints and a few unit tests.",
    "prose": "Based on the documents above, explain in detail how this system works and what its main trade-offs are.",
}

seed_env = os.environ.get("BENCH_SEED")
seed_rnd = random.Random(int(seed_env)) if seed_env else None


def make_prompt(n_tokens: int, kind: str) -> str:
    rnd = random.Random(seed_rnd.random()) if seed_rnd else random.Random(time.time_ns())
    n_chars = int(n_tokens * 3.6)
    start = rnd.randint(0, max(0, len(corpus) - n_chars - 1))
    start = corpus.find("\n", start) + 1
    return corpus[start:start + n_chars] + "\n\n---\n" + tasks[kind]


def run(n_tokens: int, kind: str) -> dict:
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({
            "model": "default",
            "messages": [{"role": "user", "content": make_prompt(n_tokens, kind)}],
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
    with urllib.request.urlopen(req, timeout=1800) as res:
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
    return {
        "label": label,
        "kind": kind,
        "prompt": p,
        "ttft_s": round(ttft, 2),
        "prefill_tps": round(p / ttft, 1) if ttft else 0,
        "completion": c,
        "decode_tps": round((c - 1) / dec, 2) if dec > 0 and c > 1 else 0,
    }


for n in sizes:
    for kind in kinds:
        for i in range(repeat):
            print(json.dumps(run(n, kind)), flush=True)

