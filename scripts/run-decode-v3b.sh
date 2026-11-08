#!/bin/bash
# dknobs v3b: draft-conf 検証ブロック（run-decode-ab4.sh を A-B-A 用に再構成）
# 使い方: run-decode-v3b.sh <block-id> "<label>:<reps>:<mode>" [...]
#   label: a1/a2 = conf 0.7（本番）、b = conf 0.5、spec3/spec5 = spec 変更（1 パス用）
#   mode: full = bench+answers+klprobe、bench = bench のみ
# 生ログ: ~/work/results/dknobs/v3b/。cpuset は付けない（decode は CPU experts 律速）
set -u
BLOCK=${1:-d1}; shift
RES=~/work/results/dknobs/v3b
SC=~/work/wt/dknobs/scripts
CORPUS=~/work/results/dknobs/corpus/dedup.md
VOL=/var/lib/docker/volumes/compose_strata-data/_data
IMG=strata:fork-v3
PORT=18080
DEADLINE=${DEADLINE:-2700}
ANS2=${ANS2:-1}
K=$(grep ^API_KEY= ~/opt-work/prod.env | cut -d= -f2)
mkdir -p "$RES"
exec > >(tee -a "$RES/block-$BLOCK.log") 2>&1
echo "=== dknobs v3b block $BLOCK start $(date -Is) ANS2=$ANS2 specs: $*"

T0=$(date +%s)
left() { echo $(( DEADLINE - ( $(date +%s) - T0 ) )); }

me="dknobs-v3b-$BLOCK"
lw=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $lw -ge 90 ] && { echo "GPU lock busy after 45 min, exit"; exit 2; }
  sleep 30; lw=$((lw+1))
done
echo "$me $(date -Is)" > /tmp/strata-gpu.lock/owner
T0=$(date +%s)
echo "lock acquired $(date -Is)"

restore_prod() {
  cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
  sudo docker rm -f strata-fork >/dev/null 2>&1
  sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
  ( cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1 )
  s=""
  for i in $(seq 1 90); do s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null); [ "$s" = healthy ] && break; sleep 10; done
  {
    echo "prod health: $s image: $(sudo docker inspect -f '{{.Config.Image}}' strata)"
    PID=$(pgrep -f "strata --serve" | head -1)
    if [ -n "${PID:-}" ]; then
      sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_"
      echo "--- cmdline ---"
      sudo cat /proc/$PID/cmdline | tr "\0" " "; echo
    fi
    sudo docker ps --format '{{.Names}} {{.Status}}'
  } > "$RES/prod-restore-$BLOCK.txt" 2>&1
  sudo docker exec strata sh -c "cat /data/config/strata/launch-presets/docker.json" > "$RES/prod-preset-$BLOCK.json" 2>&1
  if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
    rm -rf /tmp/strata-gpu.lock
    echo "gpu lock released $(date -Is)"
    return 0
  fi
  echo "RESTORE FAILED - lock kept"
  return 1
}
trap restore_prod EXIT

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
  echo "prod stopped: VRAM ${used} MiB, MemAvailable ${avail} GiB, engines ${alive} left=$(left)"
}

# 本番 STRATA_SET（prod.env と同一）+ アーム差分
PROD_SET="ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 spec=4"
set_of() {
  case "$1" in
    a1|a2) echo "$PROD_SET STRATA_DECODE_TIMING=1" ;;
    b)     echo "$PROD_SET draft-conf=0.5 STRATA_DECODE_TIMING=1" ;;
    spec3) echo "${PROD_SET/spec=4/spec=3} STRATA_DECODE_TIMING=1" ;;
    spec5) echo "${PROD_SET/spec=4/spec=5} STRATA_DECODE_TIMING=1" ;;
    *) echo "" ;;
  esac
}
expect_flags() {
  case "$1" in
    a1|a2)        echo "--draft-conf 0.7|--spec 4|--kv-resident 32768|--draft dflash2" ;;
    b)            echo "--draft-conf 0.5|--spec 4" ;;
    spec3)        echo "--spec 3" ;;
    spec5)        echo "--spec 5" ;;
    *) echo "" ;;
  esac
}

start_arm() {
  arm=$1
  setv=$(set_of "$arm")
  [ -n "$setv" ] || { echo "unknown arm $arm"; return 1; }
  echo "gpu before $arm: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)"
  sudo rm -rf "$RES/state-$arm"
  sudo cp -a "$VOL/state" "$RES/state-$arm"
  sudo docker rm -f strata-fork >/dev/null 2>&1
  printf 'API_KEY=%s\n' "$K" > "$RES/env-$arm.launch"; chmod 600 "$RES/env-$arm.launch"
  echo "STRATA_SET: $setv"
  sudo docker run -d --name strata-fork --gpus all \
    --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$PORT:8080 \
    --env-file "$RES/env-$arm.launch" \
    -e MODEL=IQ3_XXS -e FAMILY=qwen -e CONTEXT=512k -e VISION=off -e PARALLEL=1 -e TUNE=skip \
    -e PRESET="$arm" -e STRATA_SET="$setv" \
    -v "$VOL":/data -v "$RES/state-$arm":/data/state \
    --restart no "$IMG" > "$RES/docker-run-$arm.log" 2>&1
  [ $? -eq 0 ] || { cat "$RES/docker-run-$arm.log"; return 1; }
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
  echo "arm $arm health=$s $(date -Is) left=$(left)"
  if [ "$s" != healthy ]; then
    sudo docker logs strata-fork > "$RES/docker-$arm.log" 2>&1
    tail -40 "$RES/docker-$arm.log"
    return 1
  fi
  PID=$(pgrep -f "strata --serve" | head -1)
  sudo cat /proc/$PID/cmdline | tr "\0" " " > "$RES/cmdline-$arm.txt"; echo >> "$RES/cmdline-$arm.txt"
  sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" | grep -v KEY > "$RES/env-$arm.txt"
  miss=""
  IFS='|' read -ra want <<< "$(expect_flags "$arm")"
  for w in "${want[@]}"; do
    grep -q -- "$w" "$RES/cmdline-$arm.txt" || miss="$miss [$w]"
  done
  if [ -n "$miss" ]; then echo "WARN $arm: flags not in cmdline:$miss"; else echo "flags ok $arm"; fi
  echo "workers: $(grep -aoE '[0-9]+ expert-pool workers' "$RES/state-$arm/strata/launch/$arm.engine.log" 2>/dev/null | tail -1)"
  return 0
}

stop_arm() {
  sudo docker rm -f strata-fork >/dev/null 2>&1
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 3000 ] && break
    sleep 2
  done
  echo "arm stopped: VRAM ${used} MiB left=$(left)"
}

run_short() {
  arm=$1; reps=$2; maxt=${3:-256}
  STRATA_URL="http://127.0.0.1:$PORT" BENCH_SEED=42 BENCH_LABEL="$arm" \
    CORPUS_FILE="$CORPUS" STRATA_API_KEY="$K" \
    python3 "$SC/bench-decode.py" 4000,24000 "$reps" "$maxt" \
    > "$RES/bench-$arm.jsonl"
  cat "$RES/bench-$arm.jsonl"
  echo "gpu after $arm: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)"
}

answers_once() {
  arm=$1; pass=$2
  STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" \
    python3 "$SC/answers.py" "$RES/answers-$arm-pass$pass.json" >/dev/null
  echo "answers $arm pass$pass done left=$(left)"
}

kl_once() {
  arm=$1; pass=$2
  STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" \
    python3 "$SC/klprobe.py" "$RES/kl-$arm-pass$pass.json" >/dev/null
  echo "klprobe $arm pass$pass done left=$(left)"
}

collect_timing() {
  arm=$1
  log="$RES/state-$arm/strata/launch/$arm.engine.log"
  grep -a "strata decode timing:" "$log" > "$RES/timing-$arm.txt" 2>/dev/null
  grep -aE "decode expert cache hit rate|KV streaming:|drafts accepted" "$log" > "$RES/hits-$arm.txt" 2>/dev/null
  echo "--- timing $arm ($(wc -l < "$RES/timing-$arm.txt") lines) ---"
  tail -2 "$RES/timing-$arm.txt"
}

stop_prod

for spec in "$@"; do
  arm=${spec%%:*}; rest=${spec#*:}; reps=${rest%%:*}; mode=${rest##*:}
  [ "$reps" = "$arm" ] && reps=3
  need=700; [ "$arm" = spec3 -o "$arm" = spec5 ] && need=450
  if [ "$(left)" -lt "$need" ]; then echo "skip $arm: left $(left)s"; continue; fi
  start_arm "$arm" || { echo "ARM $arm FAILED"; continue; }
  case "$mode" in
    full)
      n=1; [ "$ANS2" = 2 ] && n=2
      for p in $(seq 1 $n); do answers_once "$arm" "$p"; kl_once "$arm" "$p"; done
      run_short "$arm" "$reps" 256
      ;;
    bench)
      run_short "$arm" "$reps" 256
      ;;
  esac
  collect_timing "$arm"
  stop_arm
done

echo "=== measured, restoring prod $(date -Is) left $(left)s"
restore_prod || { echo "RESTORE FAILED - lock kept"; exit 5; }

echo "=== final check $(date -Is)"
sudo docker inspect -f '{{.State.Health.Status}} {{.Config.Image}}' strata
sudo docker exec strata sh -c 'cat /data/state/strata/launch/docker.json' | head -20
rm -rf /tmp/strata-gpu.lock && echo "gpu lock released $(date -Is)"
echo "DKNOBS_V3B_OK $(date -Is)"
