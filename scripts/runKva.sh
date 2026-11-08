#!/bin/bash
# KVA quality/speed with the MTP drafter (DFlash2 turns KVA off, so the first quality run could not test it).
# Usage: runKva.sh
set -u
RES=~/work/results/pknobs
WT=~/work/wt/pknobs
TAG=strata:fork-9f7d977-base
VOL=/var/lib/docker/volumes/compose_strata-data/_data
PORT=${PORT:-18080}
SSET="kv=int8 kv-resident=32768 ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 spec=4"
mkdir -p "$RES"
K=$(grep '^API_KEY=' ~/opt-work/prod.env | cut -d= -f2)

lock_wait=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $lock_wait -ge 600 ] && { echo "GPU lock busy" >&2; exit 2; }
  sleep 3; lock_wait=$((lock_wait+1))
done
echo "pknobs-kva $(date -Is)" > /tmp/strata-gpu.lock/owner
T0=$(date +%s)

restore_prod() {
  cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
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
    fi
    sudo docker ps --format '{{.Names}} {{.Status}}'
  } > "$RES/prod-restore-kva.txt"
  sudo docker exec strata sh -c "cat /data/config/strata/launch-presets/docker.json" > "$RES/prod-preset-kva.json" 2>&1
  if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
    rm -rf /tmp/strata-gpu.lock
    echo "gpu lock released $(date -Is) elapsed $(( $(date +%s) - T0 ))s"
    return 0
  fi
  echo "restore failed, lock kept"
  return 1
}
trap restore_prod EXIT

sudo docker exec strata sh -c "strata launch --stop --preset docker" >/dev/null 2>&1
sudo docker stop -t 120 strata >/dev/null 2>&1
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  hp=$(awk '/HugePages_Free/{print $2}' /proc/meminfo)
  alive=$(pgrep -f "strata --serve" | wc -l)
  [ "$used" -lt 3000 ] && [ "$hp" -ge 20464 ] && [ "$alive" -eq 0 ] && break
  sleep 3
done
echo "prod stopped: VRAM ${used} MiB, HugePages_Free ${hp}, engines ${alive}" | tee "$RES/stop-kva.txt"

sudo rm -rf "$RES/pristine-kva"
sudo mkdir -p "$RES/pristine-kva"
sudo cp -a "$VOL/state" "$RES/pristine-kva/state"
sudo cp -a "$VOL/config" "$RES/pristine-kva/config"

start_arm() {
  label=$1; shift
  d="$RES/$label"
  mkdir -p "$d"
  sudo rm -rf "$d/state" "$d/config"
  sudo cp -a "$RES/pristine-kva/state" "$d/state"
  sudo cp -a "$RES/pristine-kva/config" "$d/config"
  # DFlash2 は late layer の hidden state を読むため KVA を止める。MTP に固定する
  sudo sed -i 's/"drafting": null/"drafting": "mtp"/' "$d/config/strata/launch-presets/docker.json"
  sudo docker rm -f strata-fork >/dev/null 2>&1
  t0=$(date +%s)
  sudo docker run -d --name strata-fork --gpus all --cpuset-cpus 36-47 \
    --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$PORT:8080 \
    -e MODEL=IQ3_XXS -e FAMILY=qwen -e CONTEXT=512k -e PARALLEL=1 -e VISION=off -e TUNE=skip \
    -e API_KEY="$K" -e STRATA_SET="$SSET" "$@" \
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
  echo "arm $label health=$s after $((t1-t0))s" | tee -a "$RES/arms.txt"
  sudo docker logs strata-fork > "$d/docker.log" 2>&1
  PID=$(pgrep -f "strata --serve" | head -1)
  if [ -n "${PID:-}" ]; then
    sudo cat /proc/$PID/cmdline | tr "\0" " " > "$d/engine-cmdline.txt"; echo >> "$d/engine-cmdline.txt"
    sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$d/engine-environ.txt"
  fi
  if [ "$s" != healthy ]; then
    echo "arm $label NOT healthy" >&2
    tail -40 "$d/docker.log"
    return 1
  fi
  return 0
}

bench() {
  STRATA_URL="http://127.0.0.1:$PORT" BENCH_SEED=42 STRATA_API_KEY="$K" \
    python3 "$WT/scripts/prefillBench.py" "$@"
}
needle() {
  STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" \
    python3 "$WT/scripts/needleTest.py" "$@"
}
answer() {
  STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" \
    python3 "$WT/scripts/answerOnce.py" "$@"
}

start_arm K0 || { restore_prod; trap - EXIT; exit 3; }
bench 24000 2 1 prose | tee "$RES/K0-24k.jsonl"
needle 128000 32 | tee "$RES/K0-needle.jsonl"
answer 24000 128 | tee "$RES/K0-answer.jsonl"

start_arm K1 -e STRATA_KVA=1 || { restore_prod; trap - EXIT; exit 3; }
bench 24000 2 1 prose | tee "$RES/K1-24k.jsonl"
needle 128000 32 | tee "$RES/K1-needle.jsonl"
answer 24000 128 | tee "$RES/K1-answer.jsonl"

restore_prod
trap - EXIT
echo "PHASE_DONE kva"
