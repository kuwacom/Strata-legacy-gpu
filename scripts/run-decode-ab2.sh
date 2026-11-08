#!/bin/bash
# dknobs decode A-B (fix 2): fork base (engine 0.1.57) を 512K 本番相当で起動し、
#   base (MTP spec4) -> df2 (dflash2 spec6 conf0.5) を測る。
# 起動は entrypoint の settings 経路ではなく `strata launch` を直接叩く（--drafting を明示するため）。
# モデルは IQ3_XXS を明示し、既定モデル (IQ3_S) の setup/download を起こさない。
#
# 使い方: scripts/run-decode-ab2.sh <DEADLINE_S> <arm...>
set -u
RES=~/work/results/dknobs
VOL=/var/lib/docker/volumes/compose_strata-data/_data
IMG=strata:fork-9f7d977-base
PORT=${PORT:-18080}
DEADLINE_S=${1:-2100}
shift
ARMS=${*:-"base:3 df2:1"}
K=$(grep ^API_KEY= ~/opt-work/prod.env | cut -d= -f2)
mkdir -p "$RES"
exec > >(tee -a "$RES/run.log") 2>&1

echo "=== dknobs decode A-B (fix 2) start $(date -Is) deadline ${DEADLINE_S}s arms: $ARMS"

# GPU ロック（他人が持っている場合は待つ。自分の owner のときだけ再利用する）
me="dknobs-decode"
if [ -d /tmp/strata-gpu.lock ] && grep -q "^$me " /tmp/strata-gpu.lock/owner 2>/dev/null; then
  echo "lock already mine: $(cat /tmp/strata-gpu.lock/owner)"
else
  t=0
  until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
    [ $t -ge 600 ] && { echo "GPU lock busy after 30 min, exit"; exit 2; }
    sleep 3; t=$((t+1))
  done
  echo "$me $(date -Is)" > /tmp/strata-gpu.lock/owner
  echo "lock acquired $(date -Is)"
fi
T0=$(date +%s)
left() { echo $(( DEADLINE_S - ( $(date +%s) - T0 ) )); }
now()  { date -Is; }

restore_prod() {
  cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
  sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
  cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1
  s=""
  for i in $(seq 1 90); do s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null); [ "$s" = healthy ] && break; sleep 10; done
  echo "prod health: $s image: $(sudo docker inspect -f '{{.Config.Image}}' strata)"
  PID=$(pgrep -f "strata --serve" | head -1)
  sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$RES/prod-environ.txt"
  cat "$RES/prod-environ.txt"
  sudo docker ps --format '{{.Names}} {{.Status}}'
  if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
    rm -rf /tmp/strata-gpu.lock && echo "gpu lock released $(date -Is)" && return 0
  fi
  echo "gpu lock NOT released (restore failed)"; return 1
}

stop_prod() {
  sudo docker exec strata sh -c "strata launch --stop --preset docker" >/dev/null 2>&1
  sudo docker stop -t 120 strata >/dev/null 2>&1
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    alive=$(pgrep -f "strata --serve" | wc -l)
    [ "$used" -lt 3000 ] && [ "$avail" -ge 50 ] && [ "$alive" -eq 0 ] && break
    sleep 3
  done
  echo "prod stopped: VRAM ${used} MiB, MemAvailable ${avail} GiB, engines ${alive} $(now)"
}

arm_settings() {
  case "$1" in
    base)  echo "--drafting mtp --set spec=4 --set spec-min-p=0.5 --set kv-resident=32768 --set expert-cache=auto" ;;
    df2)   echo "--drafting dflash2 --set spec=6 --set draft-conf=0.5 --set kv-resident=32768 --set expert-cache=auto" ;;
    kv64)  echo "--drafting mtp --set spec=4 --set spec-min-p=0.5 --set kv-resident=65536 --set expert-cache=auto" ;;
    kv4k)  echo "--drafting mtp --set spec=4 --set spec-min-p=0.5 --set kv-resident=4096 --set expert-cache=auto" ;;
    *) echo "" ;;
  esac
}

start_arm() {
  arm=$1
  echo "gpu before $arm: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)"
  sudo rm -rf "$RES/state-$arm"
  sudo cp -a "$VOL/state" "$RES/state-$arm"
  sudo docker rm -f strata-fork >/dev/null 2>&1
  sets="$(arm_settings "$arm") --set STRATA_DENSE_F32=1 --set STRATA_DECODE_TIMING=1 --set rope-scaling=yarn --set rope-scale=2"
  echo "sets: $sets"
  sudo docker run -d --name strata-fork --gpus all --cpuset-cpus 36-47 \
    --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$PORT:8080 \
    -e STRATA_LAUNCH_KEY="$K" \
    -v "$VOL":/data -v "$RES/state-$arm":/data/state \
    --restart no "$IMG" \
    strata launch --no-tui --access lan --port 8080 --data-dir /data/Strata-data \
      --model IQ3_XXS --family qwen --context 512k --parallel 1 --vision off --kv int8 --tune skip \
      --save-preset "$arm" --api-key-env STRATA_LAUNCH_KEY $sets \
    > "$RES/docker-run-$arm.log" 2>&1
  [ $? -eq 0 ] || { cat "$RES/docker-run-$arm.log"; return 1; }
  # download guard: 新しい .part が出たら即中止（モデルの再 DL を避ける）
  sleep 45
  newpart=$(sudo find "$VOL/Strata-data/models" -name "*.part" -newermt "-3 minutes" 2>/dev/null | head -3)
  if [ -n "$newpart" ]; then
    echo "ABORT: a model download started: $newpart"
    sudo docker rm -f strata-fork >/dev/null 2>&1
    return 2
  fi
  s=""
  for i in $(seq 1 180); do
    s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata-fork 2>/dev/null)
    [ "$s" = healthy ] && break
    sudo docker inspect -f "{{.State.Status}}" strata-fork 2>/dev/null | grep -q exited && break
    sleep 5
  done
  echo "arm $arm health=$s $(now)"
  sudo docker ps -a --filter name=strata-fork --format '{{.Status}}'
  if [ "$s" != healthy ]; then
    sudo docker logs strata-fork > "$RES/docker-$arm.log" 2>&1
    tail -30 "$RES/docker-$arm.log"
    return 1
  fi
  PID=$(pgrep -f "strata --serve" | head -1)
  sudo cat /proc/$PID/cmdline | tr "\0" " " > "$RES/cmdline-$arm.txt"; echo >> "$RES/cmdline-$arm.txt"
  sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$RES/env-$arm.txt"
  sudo docker logs strata-fork > "$RES/docker-$arm.log" 2>&1
  grep -E "engine started|expert cache|KV streaming|window up to|dflash2|mtp:|memory guard|draft" \
    "$RES/state-$arm/strata/launch/$arm.engine.log" | tail -30 > "$RES/startup-$arm.txt"
  cat "$RES/startup-$arm.txt"
  return 0
}

stop_arm() {
  sudo docker rm -f strata-fork >/dev/null 2>&1
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 3000 ] && break
    sleep 2
  done
  echo "arm stopped: VRAM ${used} MiB $(now)"
}

run_short() {
  arm=$1; reps=$2; maxt=${3:-256}
  STRATA_URL="http://127.0.0.1:$PORT" BENCH_SEED=42 BENCH_LABEL="$arm" \
    CORPUS_FILE="$RES/corpus/dedup.md" STRATA_API_KEY="$K" \
    python3 ~/work/wt/dknobs/scripts/bench-decode.py 4000,24000 "$reps" "$maxt" \
    > "$RES/bench-$arm.jsonl"
  cat "$RES/bench-$arm.jsonl"
  echo "gpu after $arm: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)"
}

collect_timing() {
  arm=$1
  grep "strata decode timing:" "$RES/state-$arm/strata/launch/$arm.engine.log" > "$RES/timing-$arm.txt" 2>/dev/null
  grep -E "decode expert cache hit rate|KV streaming:|suffix drafts|drafts accepted" \
    "$RES/state-$arm/strata/launch/$arm.engine.log" > "$RES/hits-$arm.txt" 2>/dev/null
  echo "--- timing $arm ---"; cat "$RES/timing-$arm.txt"
  echo "--- hits $arm ---"; cat "$RES/hits-$arm.txt"
}

stop_prod

for spec in $ARMS; do
  arm=${spec%%:*}; reps=${spec##*:}
  [ "$reps" = "$arm" ] && reps=3
  if [ $(left) -lt 600 ]; then echo "skip $arm: left $(left)s"; continue; fi
  start_arm "$arm"
  rc=$?
  if [ "$rc" = 0 ]; then
    STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" python3 ~/work/wt/dknobs/scripts/answers.py "$RES/answers-$arm.json"
    run_short "$arm" "$reps" 256
    collect_timing "$arm"
  fi
  stop_arm
  [ "$rc" = 2 ] && { echo "download guard tripped: aborting arms"; break; }
done

echo "=== measured, restoring prod $(now) left $(left)s"
restore_prod || exit 5
echo "DKNODES_DECODE_OK $(now)"

