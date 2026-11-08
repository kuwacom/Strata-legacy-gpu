#!/bin/bash
# 統合イメージの 1 アーム計測。本番は呼び出し側が止め、GPU ロックも呼び出し側が持つ
# 使い方: arm-integrated.sh <label> <image> <mode>   mode: decode|prefill|quality|long
set -u
label=$1; img=$2; mode=${3:-decode}
res=~/work/results/integrated
out=$res/arms/$label-$mode
vol=/var/lib/docker/volumes/compose_strata-data/_data
port=18080
mkdir -p "$out"
K=$(grep ^API_KEY= ~/opt-work/prod.env | cut -d= -f2)

sudo docker rm -f strata-integrated >/dev/null 2>&1
sudo rm -rf "$out/state"
sudo cp -a "$vol/state" "$out/state"

envargs=(-e "STRATA_DECODE_TIMING=1" -e "STRATA_PREFILL_TIMING=1")
[ "$mode" = quality ] && envargs+=(-e "STRATA_LOGITS_DUMP=/out/logits.bin")

t0=$(date +%s)
sudo docker run -d --name strata-integrated --gpus all --cpuset-cpus 36-47 \
  --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$port:8080 \
  -e MODEL=IQ3_XXS -e FAMILY=qwen -e CONTEXT=512k -e PARALLEL=1 -e VISION=off -e TUNE=skip \
  -e API_KEY="$K" -e PORT=8080 \
  -e STRATA_SET="ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 kv=int8 kv-resident=32768 spec=4 spec-min-p=0.5" \
  -e STRATA_DENSE_F32=1 \
  "${envargs[@]}" \
  -v "$vol":/data -v "$out/state":/data/state -v "$out":/out \
  --restart no "$img" > "$out/docker-run.log" 2>&1
[ $? -eq 0 ] || { echo "container failed $label"; exit 3; }

s=""
for i in $(seq 1 120); do
  s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata-integrated 2>/dev/null)
  [ "$s" = healthy ] && break
  sudo docker inspect -f "{{.State.Status}}" strata-integrated 2>/dev/null | grep -q exited && break
  sleep 5
done
t1=$(date +%s)
echo "$label $mode health=$s after $((t1-t0))s"
sudo docker ps -a --filter name=strata-integrated --format '{{.Status}}'
if [ "$s" != healthy ]; then
  sudo docker logs strata-integrated > "$out/docker.log" 2>&1
  tail -40 "$out/docker.log"
  sudo docker rm -f strata-integrated >/dev/null 2>&1
  exit 4
fi
sudo docker logs strata-integrated > "$out/docker.log" 2>&1
PID=$(pgrep -f "strata --serve" | head -1)
sudo cat /proc/$PID/cmdline | tr "\0" " " > "$out/engine-cmdline.txt"; echo >> "$out/engine-cmdline.txt"
sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" | grep -v KEY > "$out/engine-environ.txt"
nvidia-smi --query-gpu=name,memory.used,memory.total,clocks.sm --format=csv > "$out/gpu.txt"

case "$mode" in
  decode|speed)
    ts=$(date +%m%d-%H%M%S)
    STRATA_URL="http://127.0.0.1:$port" BENCH_SEED=42 STRATA_API_KEY=$K python3 ~/bench.py 4000,24000 3 256 code,prose > "$out/bench-$ts.jsonl" 2>&1
    cat "$out/bench-$ts.jsonl"
    if [ "$mode" = speed ]; then
      STRATA_URL="http://127.0.0.1:$port" BENCH_SEED=42 STRATA_API_KEY=$K python3 ~/bench.py 24000 3 1 code > "$out/prefill-24k-$ts.jsonl" 2>&1
      cat "$out/prefill-24k-$ts.jsonl"
      # 128K は最初の 1 本が KV streaming の初期化で極端に遅いので、暖機 1 本を捨てて 2 本を測る
      # 暖機は別シードにして、計測 1 本目と同じプロンプト (プロンプトキャッシュに当たる) を避ける
      STRATA_URL="http://127.0.0.1:$port" BENCH_SEED=43 STRATA_API_KEY=$K python3 ~/bench.py 114000 1 1 code > "$out/prefill-128k-warmup-$ts.jsonl" 2>&1
      cat "$out/prefill-128k-warmup-$ts.jsonl"
      STRATA_URL="http://127.0.0.1:$port" BENCH_SEED=42 STRATA_API_KEY=$K python3 ~/bench.py 114000 2 1 code > "$out/prefill-128k-$ts.jsonl" 2>&1
      cat "$out/prefill-128k-$ts.jsonl"
    fi
    ;;
  quality)
    # logits dump: 4K と 24K の各 1 リクエスト (max_tokens 1)
    python3 ~/work/wt/integrated/scripts/logits-two.py "$port" "$K" > "$out/logits-requests.json" 2>&1
    cat "$out/logits-requests.json"
    STRATA_URL="http://127.0.0.1:$port" STRATA_API_KEY=$K python3 ~/work/wt/integrated/scripts/answers4.py "$out/answers.json" > "$out/answers.out" 2>&1
    cat "$out/answers.out"
    ;;
  long)
    # 480K TTFT 1 回。corpus を繰り返して長いプロンプトを作る
    python3 ~/work/wt/integrated/scripts/longprefill.py "$port" "$K" 480000 > "$out/long-480k.jsonl" 2>&1
    cat "$out/long-480k.jsonl"
    ;;
  *)
    echo "unknown mode $mode" >&2; exit 2 ;;
esac

sleep 2
sudo docker exec strata-integrated sh -c "cat /data/state/strata/launch/docker.engine.log" > "$out/engine.log" 2>/dev/null || sudo docker logs strata-integrated > "$out/engine.log" 2>&1
sudo docker rm -f strata-integrated >/dev/null 2>&1
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 1000 ] && break
  sleep 2
done
echo "$label $mode done vram=$used"
