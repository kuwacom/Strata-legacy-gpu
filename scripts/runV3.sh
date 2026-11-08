#!/bin/bash
# pknobs v3: prefill knobs A-B-A on strata:fork-v3 with the 512K production shape.
# Usage: runV3.sh chunk|pattn|w16iq|big
# Writes only under ~/work/results/pknobs/v3.
set -u
phase=${1:-chunk}
RES=~/work/results/pknobs/v3
WT=~/work/wt/pknobs
TAG=strata:fork-v3
VOL=/var/lib/docker/volumes/compose_strata-data/_data
PORT=${PORT:-18080}
BASE_SET="kv=int8 kv-resident=32768 ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 spec=4"
mkdir -p "$RES"

# API キーは表示しない
K=$(grep '^API_KEY=' ~/opt-work/prod.env | cut -d= -f2)

lock_wait=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $lock_wait -ge 600 ] && { echo "GPU lock busy" >&2; exit 2; }
  sleep 3; lock_wait=$((lock_wait+1))
done
echo "pknobs-v3-$phase $(date -Is)" > /tmp/strata-gpu.lock/owner
T0=$(date +%s)
say() { echo "[$(date +%H:%M:%S) +$(( ($(date +%s) - T0) / 60 ))m] $*"; }

restore_prod() {
  cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
  # fork が RAM の expert arena を握ったままだと本番が OOM で再起動を繰り返すため、先に消す
  sudo docker rm -f strata-fork >/dev/null 2>&1
  sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
  ( cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1 )
  s=""
  for i in $(seq 1 90); do
    s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null)
    [ "$s" = healthy ] && break
    sleep 10
  done
  {
    echo "prod health: $s"
    PID=$(pgrep -f "strata --serve" | head -1)
    if [ -n "${PID:-}" ]; then
      sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_"
      echo "--- cmdline ---"
      sudo cat /proc/$PID/cmdline | tr "\0" " "; echo
    fi
    sudo docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
  } > "$RES/prod-restore.txt" 2>&1
  sudo docker exec strata sh -c "cat /data/config/strata/launch-presets/docker.json" > "$RES/prod-preset.json" 2>&1
  if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
    rm -rf /tmp/strata-gpu.lock
    say "gpu lock released"
    return 0
  fi
  echo "restore failed, lock kept" >&2
  return 1
}
trap restore_prod EXIT

# 本番を止めて VRAM と RAM を空ける
sudo docker exec strata sh -c "strata launch --stop --preset docker" >/dev/null 2>&1
sudo docker stop -t 120 strata >/dev/null 2>&1
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  hp=$(awk '/HugePages_Free/{print $2}' /proc/meminfo)
  alive=$(pgrep -f "strata --serve" | wc -l)
  [ "$used" -lt 3000 ] && [ "$hp" -ge 20464 ] && [ "$alive" -eq 0 ] && break
  sleep 5
done
say "prod stopped: VRAM ${used} MiB, HugePages_Free ${hp}, engines ${alive}" | tee "$RES/stop.txt"

# 本番の state/config をスナップショット。config も overlay して共有 preset を汚さない
sudo rm -rf "$RES/pristine"
sudo mkdir -p "$RES/pristine"
sudo cp -a "$VOL/state" "$RES/pristine/state"
sudo cp -a "$VOL/config" "$RES/pristine/config"

start_arm() {
  label=$1; sset=$2; shift 2
  d="$RES/$label"
  mkdir -p "$d"
  sudo rm -rf "$d/state" "$d/config"
  sudo cp -a "$RES/pristine/state" "$d/state"
  sudo cp -a "$RES/pristine/config" "$d/config"
  sudo docker rm -f strata-fork >/dev/null 2>&1
  t0=$(date +%s)
  sudo docker run -d --name strata-fork --gpus all --cpuset-cpus 36-47 \
    --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$PORT:8080 \
    -e MODEL=IQ3_XXS -e FAMILY=qwen -e CONTEXT=512k -e PARALLEL=1 -e VISION=off -e TUNE=skip \
    -e API_KEY="$K" -e STRATA_SET="$sset" "$@" \
    -v "$VOL":/data -v "$d/state":/data/state -v "$d/config":/data/config \
    --restart no "$TAG" > "$d/docker-run.log" 2>&1
  s=""
  for i in $(seq 1 120); do
    s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata-fork 2>/dev/null)
    [ "$s" = healthy ] && break
    sudo docker inspect -f "{{.State.Status}}" strata-fork 2>/dev/null | grep -q exited && break
    sleep 5
  done
  t1=$(date +%s)
  sudo docker logs strata-fork > "$d/docker.log" 2>&1
  PID=$(pgrep -f "strata --serve" | head -1)
  if [ -n "${PID:-}" ]; then
    sudo cat /proc/$PID/cmdline | tr "\0" " " > "$d/engine-cmdline.txt"; echo >> "$d/engine-cmdline.txt"
    sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$d/engine-environ.txt"
  fi
  nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,clocks.sm --format=csv > "$d/gpu-start.txt"
  LOG="$d/state/strata/launch/docker.engine.log"
  grep -nE "prompt chunk|QSA layers|pattn|FP16 weight cache|KVA|dense MMQ|K-quant prompt|do not fit|does not fit|out of memory|staged" "$LOG" 2>/dev/null | tail -20 > "$d/keylines.txt"
  say "arm $label health=$s after $((t1-t0))s" | tee -a "$RES/arms.txt"
  cp "$d/keylines.txt" "$RES/$label-keylines.txt" 2>/dev/null
  if [ "$s" != healthy ]; then
    echo "arm $label NOT healthy" >&2
    tail -30 "$d/docker.log" >&2
    return 1
  fi
  return 0
}

bench() {
  label=$1; shift
  STRATA_URL="http://127.0.0.1:$PORT" BENCH_SEED=42 STRATA_API_KEY="$K" \
    python3 "$WT/scripts/prefillBench.py" "$@" | tee -a "$RES/$label-bench.jsonl"
}

answers() {
  label=$1; shift
  STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" \
    python3 "$WT/scripts/answers4.py" "$@" | tee -a "$RES/$label-answers.jsonl"
}

arm_bench() {
  label=$1
  bench "$label" 24000 3 1 prose
  bench "$label" 24000 3 1 code
  bench "$label" 128000 2 1 prose
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader > "$RES/$label-gpu-end.txt"
}

case "$phase" in
chunk)
  # A (auto 8192) / B (--prefill 16384) / A'
  start_arm A1 "$BASE_SET" || { restore_prod; trap - EXIT; exit 3; }
  arm_bench A1
  answers A1 24000 1,2,3,4
  say "A1 done"

  start_arm B1 "${BASE_SET/prefill=auto/prefill=16384}" || { restore_prod; trap - EXIT; exit 3; }
  arm_bench B1
  answers B1 24000 1,2,3,4
  say "B1 done"

  start_arm A2 "$BASE_SET" || { restore_prod; trap - EXIT; exit 3; }
  arm_bench A2
  answers A2 24000 1
  answers A2 24000 1
  say "A2 done"
  ;;
pattn)
  # A (PATTN off) / B (PATTN on) / A'
  start_arm C1 "$BASE_SET" || { restore_prod; trap - EXIT; exit 3; }
  arm_bench C1
  answers C1 24000 1,2,3,4
  answers C1 24000 1,2,3,4
  say "C1 done"

  start_arm C2 "$BASE_SET" -e STRATA_PATTN_SM60=1 || { restore_prod; trap - EXIT; exit 3; }
  arm_bench C2
  answers C2 24000 1,2,3,4
  say "C2 done"

  start_arm C3 "$BASE_SET" || { restore_prod; trap - EXIT; exit 3; }
  arm_bench C3
  answers C3 24000 1
  say "C3 done"
  ;;
w16iq)
  # W1 baseline / W2 FP16 weight cache / I1 staged IQ tables / W1b baseline
  start_arm W1 "$BASE_SET" || { restore_prod; trap - EXIT; exit 3; }
  bench W1 24000 3 1 prose
  bench W1 24000 3 1 code
  say "W1 done"

  start_arm W2 "$BASE_SET" -e STRATA_PF_W16_CACHE_MIB=auto || { restore_prod; trap - EXIT; exit 3; }
  bench W2 24000 3 1 prose
  bench W2 24000 3 1 code
  say "W2 done"

  start_arm I1 "$BASE_SET" -e STRATA_IQ_STAGE_TABLES=1 || { restore_prod; trap - EXIT; exit 3; }
  bench I1 24000 3 1 prose
  bench I1 24000 3 1 code
  say "I1 done"

  start_arm W1b "$BASE_SET" || { restore_prod; trap - EXIT; exit 3; }
  bench W1b 24000 3 1 prose
  bench W1b 24000 3 1 code
  say "W1b done"
  ;;
big)
  # 480K one run + 128K two runs on the chosen arm, then a 4096 chunk arm
  SSET_BIG=${BIG_SET:-$BASE_SET}
  start_arm G1 "$SSET_BIG" || { restore_prod; trap - EXIT; exit 3; }
  bench G1 480000 1 1 prose
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader > "$RES/G1-gpu-end.txt"
  say "G1 480K done"
  bench G1 128000 2 1 prose
  say "G1 done"

  start_arm D1 "${BASE_SET/prefill=auto/prefill=4096}" || { restore_prod; trap - EXIT; exit 3; }
  bench D1 24000 3 1 prose
  bench D1 24000 3 1 code
  say "D1 done"
  ;;
esac

restore_prod
trap - EXIT
echo "PHASE_DONE $phase"
