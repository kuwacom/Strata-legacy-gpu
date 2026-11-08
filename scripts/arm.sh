#!/bin/bash
# 1 アーム: 本番は呼び出し側が止める。GPU ロックも呼び出し側が持つ
# 使い方: arm.sh <label> <image> <sizes> <reps> <max_tokens> <kinds> [extra-env-file|-] [answers 0/1] [kl 0/1]
# sizes は "24000:3,128000:2" のようにサイズごとの回数を持てる（:N が無ければ reps を使う）
set -u
label=$1; img=$2; sizes=${3:-24000}; reps=${4:-1}; maxt=${5:-1}; kinds=${6:-code,prose}; extra=${7:--}; ans=${8:-0}; kl=${9:-0}
res=~/work/results/pgdn
out=$res/arms/$label
vol=/var/lib/docker/volumes/compose_strata-data/_data
port=18080
mkdir -p "$out"
K=$(grep ^API_KEY= ~/opt-work/prod.env | cut -d= -f2)

sudo docker rm -f strata-pgdn >/dev/null 2>&1
sudo rm -rf "$out/state"
sudo cp -a "$vol/state" "$out/state"
envargs=()
if [ "$extra" != "-" ]; then
  while IFS= read -r line; do [ -n "$line" ] && envargs+=(-e "$line"); done < "$extra"
fi
[ "$kl" = 1 ] && envargs+=(-e "STRATA_LOGITS_DUMP=/out/logits.bin")

t0=$(date +%s)
sudo docker run -d --name strata-pgdn --gpus all --cpuset-cpus 36-47 \
  --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$port:8080 \
  -e MODEL=IQ3_XXS -e FAMILY=qwen -e CONTEXT=512k -e PARALLEL=1 -e VISION=off -e TUNE=skip \
  -e API_KEY="$K" -e PORT=8080 \
  -e STRATA_SET="ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2" \
  -e STRATA_DENSE_F32=1 \
  -e STRATA_PREFILL_TIMING=1 \
  "${envargs[@]}" \
  -v "$vol":/data -v "$out/state":/data/state -v "$out":/out \
  --restart no "$img" > "$out/docker-run.log" 2>&1
[ $? -eq 0 ] || { echo "container failed $label"; exit 3; }

s=""
for i in $(seq 1 120); do
  s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata-pgdn 2>/dev/null)
  [ "$s" = healthy ] && break
  sudo docker inspect -f "{{.State.Status}}" strata-pgdn 2>/dev/null | grep -q exited && break
  sleep 5
done
t1=$(date +%s)
echo "$label health=$s after $((t1-t0))s"
sudo docker ps -a --filter name=strata-pgdn --format "{{.Status}}"
if [ "$s" != healthy ]; then
  sudo docker logs strata-pgdn > "$out/docker.log" 2>&1
  tail -40 "$out/docker.log"
  sudo docker rm -f strata-pgdn >/dev/null 2>&1
  exit 4
fi
sudo docker logs strata-pgdn > "$out/docker.log" 2>&1
PID=$(pgrep -f "strata --serve" | head -1)
sudo cat /proc/$PID/cmdline | tr "\0" " " > "$out/engine-cmdline.txt"; echo >> "$out/engine-cmdline.txt"
sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" | grep -v "KEY" > "$out/engine-environ.txt"
nvidia-smi --query-gpu=name,memory.used,memory.total,clocks.sm --format=csv > "$out/gpu.txt"

if [ "$kl" = 1 ]; then
  python3 - "$port" "$K" > "$out/kl-request.json" <<'PY'
import glob, json, os, sys, urllib.request
port, key = sys.argv[1], sys.argv[2]
corpus = "\n\n".join(open(p, encoding="utf-8", errors="ignore").read() for p in sorted(glob.glob(os.path.expanduser("~/Strata/docs/*.md"))))
n_chars = int(4000 * 3.6)
text = corpus[0:n_chars] + "\n\n---\nWrite a Python function that adds two integers."
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps({"model":"default","messages":[{"role":"user","content":text}],"max_tokens":1,
                     "stream":False,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    d = json.load(r)
print(json.dumps({"prompt_chars": len(text), "usage": d.get("usage"), "content": d["choices"][0]["message"]["content"]}))
PY
  echo "kl request done"
fi

ts=$(date +%m%d-%H%M%S)
: > "$out/bench-$ts.jsonl"
IFS=, read -ra specs <<< "$sizes"
for spec in "${specs[@]}"; do
  sz=${spec%%:*}; rp=${spec#*:}
  [ "$rp" = "$spec" ] && rp=$reps
  STRATA_URL="http://127.0.0.1:$port" BENCH_SEED=42 STRATA_API_KEY=$K python3 ~/bench.py "$sz" "$rp" "$maxt" "$kinds" >> "$out/bench-$ts.jsonl" 2>&1
done
cat "$out/bench-$ts.jsonl"

if [ "$ans" = 1 ]; then
  STRATA_URL="http://127.0.0.1:$port" STRATA_API_KEY=$K python3 ~/opt-work/src-build/answers.py "$out/answers.json" 2>/dev/null || true
  echo "answers done"
fi

sleep 2
sudo docker exec strata-pgdn sh -c "cat /data/state/strata/launch/docker.engine.log" > "$out/engine.log" 2>/dev/null || sudo docker logs strata-pgdn > "$out/engine.log" 2>&1
sudo docker rm -f strata-pgdn >/dev/null 2>&1
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 1000 ] && break
  sleep 2
done
echo "$label done vram=$used"
