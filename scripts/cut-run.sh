#!/bin/bash
# 切り分け実験: base の expert-cache auto と 60% で waitCPU を測る
# GPU ロック -> 本番停止 -> fork 起動 -> 計測 -> 本番復元 -> ロック解放
set -u
res=~/work/results/lookahead
vol=/var/lib/docker/volumes/compose_strata-data/_data
port=${PORT:-18081}
tag=strata:fork-9f7d977-base
mkdir -p "$res"

t=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $t -ge 600 ] && { echo "GPU lock busy" >&2; exit 2; }
  sleep 3; t=$((t+1))
done
echo "lookahead-cut $(date -Is)" > /tmp/strata-gpu.lock/owner

restore_prod() {
  cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
  sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
  cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1
  s=""
  for i in $(seq 1 90); do s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null); [ "$s" = healthy ] && break; sleep 10; done
  echo "prod health: $s image: $(sudo docker inspect -f '{{.Config.Image}}' strata)"
  PID=$(pgrep -f "strata --serve" | head -1)
  sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$res/prod-environ.txt"
  cat "$res/prod-environ.txt"
  sudo docker ps --format '{{.Names}} {{.Status}}'
  if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
    rm -rf /tmp/strata-gpu.lock && echo "gpu lock released $(date -Is)" && return 0
  fi
  echo "gpu lock NOT released (restore failed)"; return 1
}

K=$(grep ^API_KEY= ~/opt-work/prod.env | cut -d= -f2)

sudo docker exec strata sh -c "strata launch --stop --preset docker" >/dev/null 2>&1
sudo docker stop -t 120 strata >/dev/null 2>&1
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  hp=$(awk '/HugePages_Free/{print $2}' /proc/meminfo)
  alive=$(pgrep -f "strata --serve" | wc -l)
  [ "$used" -lt 3000 ] && [ "$hp" -ge 20464 ] && [ "$alive" -eq 0 ] && break
  sleep 3
done
echo "prod stopped: VRAM ${used} MiB, HugePages_Free ${hp}, engines ${alive}"

start_fork() {   # $1 = arm name, $2 = STRATA_SET, $3 = extra env ("A=1 B=2")
  local arm=$1 sset=$2 extra=$3
  sudo rm -rf "$res/state-$arm"
  sudo cp -a "$vol/state" "$res/state-$arm"
  sudo docker rm -f strata-fork >/dev/null 2>&1
  local eargs=()
  for kv in $extra; do eargs+=(-e "$kv"); done
  sudo docker run -d --name strata-fork --gpus all --cpuset-cpus 36-47 \
    --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$port:8080 \
    -e MODEL=IQ3_XXS -e FAMILY=qwen -e CONTEXT=512k -e PARALLEL=1 -e VISION=off -e TUNE=skip \
    -e API_KEY="$K" -e STRATA_SET="$sset" "${eargs[@]}" \
    -v "$vol":/data -v "$res/state-$arm":/data/state \
    --restart no "$tag" > "$res/docker-run-$arm.log" 2>&1
  [ $? -eq 0 ] || { cat "$res/docker-run-$arm.log"; return 3; }
  local s=""
  for i in $(seq 1 90); do
    s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata-fork 2>/dev/null)
    [ "$s" = healthy ] && break
    sudo docker inspect -f "{{.State.Status}}" strata-fork 2>/dev/null | grep -q exited && break
    sleep 5
  done
  echo "$arm health=$s"
  sudo docker logs strata-fork > "$res/docker-$arm.log" 2>&1
  if [ "$s" != healthy ]; then tail -40 "$res/docker-$arm.log"; return 4; fi
  sudo docker exec strata-fork sh -c 'cat /data/state/strata/launch/docker.engine.log' > "$res/enginelog-$arm.log" 2>&1 || true
  return 0
}

bench() {  # $1 = arm, $2 = tag suffix, $3 = sizes
  local arm=$1 sfx=$2 sizes=$3 ts=$(date +%m%d-%H%M%S)
  STRATA_URL="http://127.0.0.1:$port" BENCH_SEED=42 STRATA_API_KEY=$K \
    python3 ~/bench.py "$sizes" 1 256 prose > "$res/bench-$arm-$sfx-$ts.jsonl" 2>&1
  cat "$res/bench-$arm-$sfx-$ts.jsonl"
  curl -s "http://127.0.0.1:$port/v1/chat/completions" -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"What is the capital of France? Then compute 17*23."}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" > "$res/answer-$arm-$sfx.txt" 2>&1
}

SSET_PROF="ple-io=ram prefill=auto STRATA_DENSE_F32=1 spec=4 spec-min-p=0.5 rope-scaling=yarn rope-scale=2 STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1 STRATA_LOOKAHEAD_STATS=1"
SSET_QUIET="ple-io=ram prefill=auto STRATA_DENSE_F32=1 spec=4 spec-min-p=0.5 rope-scaling=yarn rope-scale=2 STRATA_DECODE_TIMING=1"

for arm in A B; do
  if [ "$arm" = A ]; then sset_base=$SSET_PROF; else sset_base="$SSET_PROF expert-cache=4800"; fi
  start_fork "$arm-prof" "$sset_base" "" || { restore_prod; exit 4; }
  bench "$arm" prof 4000
  bench "$arm" prof 4000
  sudo docker rm -f strata-fork >/dev/null 2>&1
  sleep 5

  if [ "$arm" = A ]; then sset_q=$SSET_QUIET; else sset_q="$SSET_QUIET expert-cache=4800"; fi
  start_fork "$arm-quiet" "$sset_q" "" || { restore_prod; exit 4; }
  bench "$arm" quiet 4000
  bench "$arm" quiet 4000
  sudo docker rm -f strata-fork >/dev/null 2>&1
  sleep 5
done

restore_prod || exit 5
echo CUT_ALL_OK
