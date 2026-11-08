#!/bin/bash
# dknobs decode A-B: fork base (engine 0.1.57) を 512K 本番相当で起動し、
#   base (MTP spec4) -> df2 (dflash2 spec6 conf0.5) -> kv64 (kv-resident 65536) + 480K
# を 1 ブロック (60 分) で測る。GPU ロックを取り、最後に本番を prod.env から復元する。
#
# 使い方: scripts/run-decode-ab.sh [DEADLINE_S]
set -u
RES=~/work/results/dknobs
VOL=/var/lib/docker/volumes/compose_strata-data/_data
IMG=strata:fork-9f7d977-base
PORT=${PORT:-18080}
DEADLINE_S=${1:-3480}
K=$(grep ^API_KEY= ~/opt-work/prod.env | cut -d= -f2)
mkdir -p "$RES"
exec > >(tee -a "$RES/run.log") 2>&1

echo "=== dknobs decode A-B start $(date -Is) deadline ${DEADLINE_S}s"

# GPU ロック（30 分で諦める）
t=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $t -ge 600 ] && { echo "GPU lock busy after 30 min, exit"; exit 2; }
  sleep 3; t=$((t+1))
done
echo "dknobs-decode $(date -Is)" > /tmp/strata-gpu.lock/owner
T0=$(date +%s)
echo "lock acquired $(date -Is)"

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

gpu_state() {
  nvidia-smi --query-gpu=name,memory.used,temperature.gpu,clocks.sm,clocks_throttle_reasons.active \
    --format=csv,noheader >> "$RES/gpu-$1.txt" 2>/dev/null || true
  tail -1 "$RES/gpu-$1.txt"
}

start_arm() {
  arm=$1
  echo "gpu before $arm: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)"
  gpu_state "$arm"
  sudo rm -rf "$RES/state-$arm"
  sudo cp -a "$VOL/state" "$RES/state-$arm"
  sudo python3 ~/work/wt/dknobs/scripts/mk-preset.py /tmp/prod-preset.json "$RES/state-$arm/strata/launch/$arm.json" "$arm" || return 1
  sudo docker rm -f strata-fork >/dev/null 2>&1
  sudo docker run -d --name strata-fork --gpus all --cpuset-cpus 36-47 \
    --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$PORT:8080 \
    -e PRESET="$arm" -e API_KEY="$K" -e TUNE=skip \
    -v "$VOL":/data -v "$RES/state-$arm":/data/state \
    --restart no "$IMG" > "$RES/docker-run-$arm.log" 2>&1
  [ $? -eq 0 ] || { cat "$RES/docker-run-$arm.log"; return 1; }
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
  grep -E "engine started|expert cache|KV streaming|window up to|dflash2|mtp:|memory guard" \
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
  gpu_state "$arm-after"
}

run_long() {
  arm=$1; n=$2
  STRATA_URL="http://127.0.0.1:$PORT" BENCH_SEED=42 BENCH_LABEL="$arm" \
    CORPUS_FILE="$RES/corpus/dedup.md" STRATA_API_KEY="$K" \
    python3 ~/work/wt/dknobs/scripts/bench-long.py "$n" prose 128 \
    > "$RES/long-$arm-$n.jsonl"
  cat "$RES/long-$arm-$n.jsonl"
}

collect_timing() {
  arm=$1
  grep "strata decode timing:" "$RES/state-$arm/strata/launch/$arm.engine.log" > "$RES/timing-$arm.txt" 2>/dev/null
  grep -E "decode expert cache hit rate|KV streaming:|suffix drafts|drafts accepted" \
    "$RES/state-$arm/strata/launch/$arm.engine.log" > "$RES/hits-$arm.txt" 2>/dev/null
  echo "--- timing $arm ---"; cat "$RES/timing-$arm.txt"
  echo "--- hits $arm ---"; cat "$RES/hits-$arm.txt"
}

# コーパス（重複 md を除いて 1 本に）
python3 ~/work/wt/dknobs/scripts/mk-corpus.py "$RES/corpus/dedup.md"

stop_prod

# ---- Phase 1: base (MTP spec 4, kv 32768) ----
if [ $(left) -ge 900 ]; then
  if start_arm base; then
    STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" python3 ~/work/wt/dknobs/scripts/answers.py "$RES/answers-base.json"
    run_short base 3 256
    collect_timing base
  fi
  stop_arm
fi

# ---- Phase 2: df2 (dflash2 spec 6, draft-conf 0.5) ----
if [ $(left) -ge 900 ]; then
  if start_arm df2; then
    STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" python3 ~/work/wt/dknobs/scripts/answers.py "$RES/answers-df2.json"
    run_short df2 3 256
    collect_timing df2
  fi
  stop_arm
fi

# ---- Phase 3: kv64 (kv-resident 65536) + 480K ----
if [ $(left) -ge 1500 ]; then
  if start_arm kv64; then
    STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" python3 ~/work/wt/dknobs/scripts/answers.py "$RES/answers-kv64.json"
    if [ $(left) -ge 1700 ]; then run_long kv64 480000; fi
    if [ $(left) -ge 300 ]; then run_short kv64 1 128; fi
    collect_timing kv64
  fi
  stop_arm
fi

echo "=== measured, restoring prod $(now) left $(left)s"
restore_prod || exit 5
echo "DKNODES_DECODE_OK $(now)"

