"""480K 相当の TTFT を 1 回測る。corpus (~/Strata/docs/*.md) は 1.36M 文字しかないため、
480K トークンぶん (約 3.2 文字/トークンの実測から 1.54M 文字) まで繰り返して使う。
使い方: longprefill.py <port> <api_key> <target_tokens>
"""
import glob, json, os, sys, time, urllib.request

port, key, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
corpus = "\n\n".join(open(p, encoding="utf-8", errors="ignore").read()
                      for p in sorted(glob.glob(os.path.expanduser("~/Strata/docs/*.md"))))
n_chars = int(target * 3.21)   # 実測 3.21 文字/トークン (100K 要求 -> 112K 実測)
reps = (n_chars // len(corpus)) + 2
text = (corpus + "\n\n") * reps
start = max(0, (len(text) - n_chars) // 2)
prompt = text[start:start + n_chars] + "\n\n---\nSummarise the documents above in one short paragraph."
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps({"model": "default", "messages": [{"role": "user", "content": prompt}],
                     "max_tokens": 1, "stream": True, "stream_options": {"include_usage": True},
                     "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
t0 = time.time()
t_first = None
usage = {}
with urllib.request.urlopen(req, timeout=7200) as res:
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
t_end = time.time()
p = usage.get("prompt_tokens", 0)
ttft = (t_first or t_end) - t0
print(json.dumps({"target": target, "chars": len(prompt), "prompt": p,
                  "ttft_s": round(ttft, 2), "prefill_tps": round(p / ttft, 1) if ttft else 0,
                  "completion": usage.get("completion_tokens", 0)}), flush=True)
