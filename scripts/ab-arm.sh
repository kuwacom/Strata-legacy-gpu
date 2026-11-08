#!/bin/bash
# MMQ I=128 A/B: one arm. Starts an image on the production data volume, answers, benches, stops the container.
# The GPU lock, the production stop and the production restore are the session script's job (block.sh).
#   ab-arm.sh <label> <image> <outdir> [sizes] [repeat] [max_tokens] [kinds]
set -u
label=$1; tag=$2; out=$3
sizes=${4:-4000,24000}; rep=${5:-3}; maxt=${6:-128}; kinds=${7:-code,prose}
vol=/var/lib/docker/volumes/compose_strata-data/_data
port=${PORT:-18080}
mkdir -p "$out"
K=$(grep ^API_KEY= ~/opt-work/prod.env | cut -d= -f2)
SSET=${SSET:-"ple-io=ram prefill=auto STRATA_DENSE_F32=1 spec=4 spec-min-p=0.5"}
extra=""
[ "${DUMP:-0}" = 1 ] && extra="-e STRATA_LOGITS_DUMP=/dump/logits.bin"
[ "${TIMING:-0}" = 1 ] && extra="$extra -e STRATA_PREFILL_TIMING=1"
[ "${MTP_F16:-0}" = 1 ] && extra="$extra -e STRATA_MTP_BATCH_F16=1"

sudo docker rm -f strata-fork >/dev/null 2>&1
sudo rm -rf "$out/state"
sudo cp -a "$vol/state" "$out/state"
t0=$(date +%s)
# shellcheck disable=SC2086
sudo docker run -d --name strata-fork --gpus all --cpuset-cpus 36-47 \
  --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$port:8080 \
  -e MODEL=IQ3_XXS -e FAMILY=qwen -e CONTEXT=128k -e PARALLEL=1 -e VISION=off -e TUNE=skip \
  -e API_KEY="$K" -e STRATA_SET="$SSET" $extra \
  -v "$vol":/data -v "$out/state":/data/state -v "$out":/dump \
  --restart no "$tag" > "$out/docker-run.log" 2>&1
[ $? -eq 0 ] || { cat "$out/docker-run.log"; sudo docker rm -f strata-fork >/dev/null 2>&1; exit 3; }

s=""
for i in $(seq 1 120); do
  s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata-fork 2>/dev/null)
  [ "$s" = healthy ] && break
  sudo docker inspect -f "{{.State.Status}}" strata-fork 2>/dev/null | grep -q exited && break
  sleep 5
done
t1=$(date +%s)
echo "$label health=$s after $((t1-t0))s"
if [ "$s" != healthy ]; then
  sudo docker logs strata-fork > "$out/docker.log" 2>&1
  tail -40 "$out/docker.log"
  sudo docker rm -f strata-fork >/dev/null 2>&1
  exit 4
fi

sudo docker logs strata-fork > "$out/docker.log" 2>&1
PID=$(pgrep -f "strata --serve" | head -1)
sudo cat /proc/$PID/cmdline | tr "\0" " " > "$out/engine-cmdline.txt"; echo >> "$out/engine-cmdline.txt"
sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$out/engine-environ.txt"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,clocks.sm --format=csv > "$out/gpu.txt"
free -m > "$out/free.txt"
grep -E "engine started|VRAM|RAM|expert cache|draft layer|KV streaming|PLE table|memory guard" "$out/state/strata/launch/docker.engine.log" | tail -40 > "$out/startup.log"

curl -s "http://127.0.0.1:$port/v1/chat/completions" -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"What is the capital of France? Then compute 17*23."}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" > "$out/answer.txt" 2>&1
cat "$out/answer.txt"

ts=$(date +%m%d-%H%M%S)
STRATA_URL="http://127.0.0.1:$port" BENCH_SEED=42 STRATA_API_KEY=$K python3 ~/bench.py "$sizes" "$rep" "$maxt" "$kinds" > "$out/bench-$ts.jsonl"
cat "$out/bench-$ts.jsonl"

# the prompt path's own breakdown (STRATA_PREFILL_TIMING): the engine log keeps the last chunk's phase lines
grep -E "strata prefill timing|gemm gate|gemm down|dequant|expert|gather|wait copy|host grouping" "$out/state/strata/launch/docker.engine.log" | tail -40 > "$out/prefill-timing-$ts.txt"
[ -f "$out/logits.bin" ] && cp "$out/logits.bin" "$out/logits-$ts.bin"

sudo docker rm -f strata-fork >/dev/null 2>&1
echo "$label done $(date -Is)"
