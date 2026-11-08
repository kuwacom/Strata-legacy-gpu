#!/usr/bin/env python3
"""Fixed 4 prompts at one prompt length, deterministic, for output-identity checks.

Prints one JSON line per prompt (id, prompt tokens, answer). Same corpus slice for
every arm, so two arms differ only by the knob under test.
"""
import glob
import json
import os
import sys
import urllib.request

base = os.environ.get("STRATA_URL", "http://127.0.0.1:18080")
key = os.environ["STRATA_API_KEY"]
n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
ids = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "2", "3", "4"])
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128
code = os.environ.get("NEEDLE_CODE", "QV-4821-ZT")

corpus = "\n\n".join(
    open(p, encoding="utf-8", errors="ignore").read()
    for p in sorted(glob.glob(os.path.expanduser("~/Strata/docs/*.md")))
)
n_chars = int(n_tokens * 3.6)
body = corpus[:n_chars]
mid = len(body) // 2

# プロンプトごとの salt で conversation cache の prefix 再利用を防ぐ（両アームで同一）
SALTS = {pid: f"salt-{pid}-" + "z" * 48 + "\n\n" for pid in "1234"}

prompts = {
    # 見出し列挙: 局所的な順序と書式の再現を見る
    "1": body + "\n\n---\nList the first five markdown headings that appear in the documents above, in order, one per line.",
    # 要約: 長文の内容理解と語彙選択の揺れを見る
    "2": body + "\n\n---\nExplain in three short paragraphs how this system stores its key-value cache and what its trade-offs are.",
    # 埋め込み検索: 中盤の 1 行を取り出せるか
    "3": body[:mid] + f"\n\nIMPORTANT: The access code for vault 7 is {code}.\n\n" + body[mid:] + "\n\n---\nWhat is the access code for vault 7? Reply with only the code, nothing else.",
    # 生成: コードの書式と型注釈の再現性を見る
    "4": body + "\n\n---\nWrite a Python function parse_md_tables(text: str) -> list[dict] that converts markdown tables into dicts. Code only.",
}


def once(pid: str) -> dict:
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({
            "model": "default",
            "messages": [{"role": "user", "content": SALTS[pid] + prompts[pid]}],
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as res:
        d = json.loads(res.read().decode())
    return {
        "id": pid,
        "prompt": d.get("usage", {}).get("prompt_tokens", 0),
        "answer": d["choices"][0]["message"].get("content") or "",
    }


for pid in ids:
    print(json.dumps(once(pid), ensure_ascii=False), flush=True)
