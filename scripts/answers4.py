"""固定 4 プロンプト (2K/8K/16K/24K) の greedy 回答を保存する。A/B の出力一致と A 同士の揺れを見るため。
プロンプトは bench.py と同じ corpus と固定シードで作るので、アーム間で本文が同一になる。
"""
import glob, json, os, random, sys, urllib.request

base = os.environ.get("STRATA_URL", "http://127.0.0.1:8080")
key = os.environ["STRATA_API_KEY"]
corpus = "\n\n".join(open(p, encoding="utf-8", errors="ignore").read()
                      for p in sorted(glob.glob(os.path.expanduser("~/Strata/docs/*.md"))))
seed_rnd = random.Random(42)   # BENCH_SEED=42 と同じ
sizes = [2000, 8000, 16000, 24000]
task = ("Based on the documents above, answer in one short paragraph: "
        "what are the three most important trade-offs this system makes?")
out = []
for n in sizes:
    rnd = random.Random(seed_rnd.random())
    n_chars = int(n * 3.6)
    start = rnd.randint(0, max(0, len(corpus) - n_chars - 1))
    start = corpus.find("\n", start) + 1
    prompt = corpus[start:start + n_chars] + "\n\n---\n" + task
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as res:
        d = json.load(res)
    out.append({"size": n, "prompt_tokens": d.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": d.get("usage", {}).get("completion_tokens", 0),
                "content": d["choices"][0]["message"].get("content", "")})
    print(f"  size {n}: prompt {out[-1]['prompt_tokens']} tok, answer {out[-1]['completion_tokens']} tok", flush=True)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
