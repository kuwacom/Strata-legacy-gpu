"""logits dump 用の 2 リクエスト (4K と 24K)。bench.py と同じ corpus・固定シードでプロンプトを作る。
使い方: logits-two.py <port> <api_key>
"""
import glob, json, os, random, sys, urllib.request

port, key = sys.argv[1], sys.argv[2]
corpus = "\n\n".join(open(p, encoding="utf-8", errors="ignore").read()
                      for p in sorted(glob.glob(os.path.expanduser("~/Strata/docs/*.md"))))
seed_rnd = random.Random(42)
task = "Based on the documents above, write a Python script that parses the markdown tables in them into JSON."

out = []
for n in (4000, 24000):
    rnd = random.Random(seed_rnd.random())
    n_chars = int(n * 3.6)
    start = rnd.randint(0, max(0, len(corpus) - n_chars - 1))
    start = corpus.find("\n", start) + 1
    prompt = corpus[start:start + n_chars] + "\n\n---\n" + task
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": "default", "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": 1, "stream": False, "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    out.append({"size": n, "usage": d.get("usage"), "content": d["choices"][0]["message"].get("content", "")})
    print(json.dumps(out[-1]), flush=True)
