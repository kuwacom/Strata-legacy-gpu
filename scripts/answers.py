"""固定プロンプトの greedy 回答を保存する。アーム間で回答本文がどこまで一致するかを見るため。

~/opt-work/src-build/answers.py と同じ 4 プロンプト。
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
            "max_tokens": 200,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as res:
        out.append(json.load(res)["choices"][0]["message"]["content"])
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"answers -> {sys.argv[1]}")

