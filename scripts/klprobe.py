"""固定 4 プロンプトの greedy 24 token について top-5 logprobs を保存する（候補間の KL 用）。

answers.py と同じ 4 プロンプト。非ストリームで logprobs=true / top_logprobs=5。
--parallel 1・単一 GPU なので split-engine の logprobs 拒否には当たらない。
"""
import json
import os
import sys
import urllib.request

base = os.environ.get("STRATA_URL", "http://127.0.0.1:8080")
key = os.environ["STRATA_API_KEY"]
prompts = [
    "フランスの首都はどこですか。また17*23はいくつですか。短く答えてください。",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain in three paragraphs why the sky is blue.",
    "List five prime numbers greater than 100 and explain how you checked each one.",
]
out = []
for p in prompts:
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({
            "model": "default",
            "messages": [{"role": "user", "content": p}],
            "max_tokens": 24,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as res:
        d = json.load(res)
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    out.append({
        "content": ch["message"]["content"],
        "tokens": lp.get("content") or [],
    })
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
n = sum(len(o["tokens"]) for o in out)
print(f"klprobe -> {sys.argv[1]} ({n} positions, {sum(1 for o in out if not o['tokens'])} prompts with no logprobs)")
