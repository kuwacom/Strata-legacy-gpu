#!/bin/bash
# pknobs v3 (block 2+): PATTN speed, FP16 cache check, 480K, chunk 4096.
# Usage: runV3b.sh pattn|big480|iq4096
# Writes only under ~/work/results/pknobs/v3.
set -u
phase=${1:-pattn}
RES=~/work/results/pknobs/v3
WT=~/work/wt/pknobs
TAG=strata:fork-v3
VOL=/var/lib/docker/volumes/compose_strata-data/_data
PORT=${PORT:-18080}
BASE_SET="kv=int8 kv-resident=32768 ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 spec=4"
mkdir -p "$RES"

# API キーは表示しない
K=$(grep '^API_KEY=' ~/opt-work/prod.env | cut -d= -f2)

# 他タスクのブロックが続くため、空くまで最大 60 分待ってから取る
lock_wait=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $lock_wait -ge 1200 ] && { echo "GPU lock busy" >&2; exit 2; }
  sleep 3; lock_wait=$((lock_wait+1))
done
echo "pknobs-v3b-$phase $(date -Is)" > /tmp/strata-gpu.lock/owner
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
  } > "$RES/prod-restore-$phase.txt" 2>&1
  sudo docker exec strata sh -c "cat /data/config/strata/launch-presets/docker.json" > "$RES/prod-preset-$phase.json" 2>&1
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
say "prod stopped: VRAM ${used} MiB, HugePages_Free ${hp}, engines ${alive}" | tee "$RES/stop-$phase.txt"

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
  say "arm $label health=$s after $((t1-t0))s" | tee -a "$RES/arms-$phase.txt"
  if [ "$s" != healthy ]; then
    echo "arm $label NOT healthy" >&2
    tail -30 "$d/docker.log" >&2
    return 1
  fi
  return 0
}

keyline() {
  label=$1
  grep -nE "prompt chunk|QSA layers|pattn|FP16 weight cache|KVA|dense MMQ|K-quant prompt|does not fit|out of memory|staged|borrows" \
    "$RES/$label/state/strata/launch/docker.engine.log" 2>/dev/null | tail -8 > "$RES/$label-keylines.txt"
  cat "$RES/$label-keylines.txt"
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
pattn)
  # C1 (PATTN off) / C2 (PATTN on) / C3 (off), plus a W16 cache arm (1 run)
  start_arm C1 "$BASE_SET" || { restore_prod; trap - EXIT; exit 3; }
  arm_bench C1
  answers C1 24000 1,2,3,4
  answers C1 24000 1,2,3,4   # A 同士の揺れ
  say "C1 done"

  start_arm C2 "$BASE_SET" -e STRATA_PATTN_SM60=1 || { restore_prod; trap - EXIT; exit 3; }
  arm_bench C2
  answers C2 24000 1,2,3,4
  say "C2 done"

  start_arm C3 "$BASE_SET" || { restore_prod; trap - EXIT; exit 3; }
  arm_bench C3
  answers C3 24000 1
  say "C3 done"

  # FP16 weight cache は VRAM の空きが要る。1 run + ログで採否が分かる
  start_arm W2 "$BASE_SET" -e STRATA_PF_W16_CACHE_MIB=auto || { restore_prod; trap - EXIT; exit 3; }
  bench W2 24000 1 1 prose
  keyline W2
  say "W2 done"
  ;;
big480)
  # 480K を 1 回（25 分以内）。BIG_SET / BIG_ENV で候補を選ぶ
  SSET_BIG=${BIG_SET:-$BASE_SET}
  if [ -n "${BIG_ENV:-}" ]; then
    start_arm G1 "$SSET_BIG" -e "$BIG_ENV" || { restore_prod; trap - EXIT; exit 3; }
  else
    start_arm G1 "$SSET_BIG" || { restore_prod; trap - EXIT; exit 3; }
  fi
  bench G1 480000 1 1 prose
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader > "$RES/G1-gpu-end.txt"
  say "G1 480K done"
  bench G1 128000 1 1 prose
  say "G1 done"
  ;;
iq4096)
  # IQ_STAGE_TABLES (bitwise) と chunk 4096。24K だけで見る
  start_arm K1 "$BASE_SET" || { restore_prod; trap - EXIT; exit 3; }
  bench K1 24000 3 1 prose
  bench K1 24000 3 1 code
  say "K1 done"

  start_arm I1 "$BASE_SET" -e STRATA_IQ_STAGE_TABLES=1 || { restore_prod; trap - EXIT; exit 3; }
  bench I1 24000 3 1 prose
  bench I1 24000 3 1 code
  say "I1 done"

  start_arm D1 "${BASE_SET/prefill=auto/prefill=4096}" || { restore_prod; trap - EXIT; exit 3; }
  bench D1 24000 3 1 prose
  bench D1 24000 3 1 code
  say "D1 done"
  ;;
esac

restore_prod
trap - EXIT
echo "PHASE_DONE $phase"
