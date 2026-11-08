#!/bin/bash
# pknobs v3b: PATTN 検証ブロック（runV3b.sh の pattn 部分を A-B-A 用に再構成）
# 使い方: runV3c.sh <block-id> "<label>:<mode>" [...]
#   label: A1/A2 = PATTN off、B/B480 = STRATA_PATTN_SM60=1
#   mode: full = bench+answers+needle50/90+QA、qa = bench+QA、bench = bench のみ、n480 = needle 480K のみ
# 生ログ: ~/work/results/pknobs/v3b/。ソース変更なし、モデルは IQ3_XXS を明示
set -u
BLOCK=${1:-p1}; shift
RES=~/work/results/pknobs/v3b
WT=~/work/wt/pknobs
SC=~/work/results/v3/scripts
TAG=strata:fork-v3
VOL=/var/lib/docker/volumes/compose_strata-data/_data
PORT=18080
BASE_SET="kv=int8 kv-resident=32768 ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 spec=4"
K=$(grep '^API_KEY=' ~/opt-work/prod.env | cut -d= -f2)
DEADLINE=${DEADLINE:-2700}
ANS2=${ANS2:-1}
mkdir -p "$RES"
exec > >(tee -a "$RES/block-$BLOCK.log") 2>&1
echo "=== pknobs v3b block $BLOCK start $(date -Is) ANS2=$ANS2 specs: $*"

T0=$(date +%s)
left() { echo $(( DEADLINE - ( $(date +%s) - T0 ) )); }

# GPU ロック: 30 秒間隔で最大 45 分再試行（1 回の失敗で諦めない）。
# DEADLINE はロック取得後に起算する（待ち時間を計測予算から引かない）
lw=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $lw -ge 90 ] && { echo "GPU lock busy after 45 min" >&2; exit 2; }
  sleep 30; lw=$((lw+1))
done
echo "pknobs-v3b-$BLOCK $(date -Is)" > /tmp/strata-gpu.lock/owner
T0=$(date +%s)
echo "lock acquired $(date -Is)"

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
  } > "$RES/prod-restore-$BLOCK.txt" 2>&1
  sudo docker exec strata sh -c "cat /data/config/strata/launch-presets/docker.json" > "$RES/prod-preset-$BLOCK.json" 2>&1
  if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
    rm -rf /tmp/strata-gpu.lock
    echo "gpu lock released $(date -Is)"
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
echo "prod stopped: VRAM ${used} MiB, HugePages_Free ${hp}, engines ${alive} left=$(left)"

sudo rm -rf "$RES/pristine"
sudo mkdir -p "$RES/pristine"
sudo cp -a "$VOL/state" "$RES/pristine/state"
sudo cp -a "$VOL/config" "$RES/pristine/config"

start_arm() {
  label=$1; sset=$2; shift 2
  d="$RES/state-$label"
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
    --restart no "$TAG" > "$RES/docker-run-$label.log" 2>&1
  s=""
  for i in $(seq 1 120); do
    s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata-fork 2>/dev/null)
    [ "$s" = healthy ] && break
    sudo docker inspect -f "{{.State.Status}}" strata-fork 2>/dev/null | grep -q exited && break
    sleep 5
  done
  t1=$(date +%s)
  sudo docker logs strata-fork > "$RES/docker-$label.log" 2>&1
  PID=$(pgrep -f "strata --serve" | head -1)
  if [ -n "${PID:-}" ]; then
    sudo cat /proc/$PID/cmdline | tr "\0" " " > "$RES/engine-cmdline-$label.txt"; echo >> "$RES/engine-cmdline-$label.txt"
    sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" | grep -v KEY > "$RES/engine-env-$label.txt"
  fi
  echo "arm $label health=$s after $((t1-t0))s left=$(left)"
  if [ "$s" != healthy ]; then
    echo "arm $label NOT healthy"; tail -30 "$RES/docker-$label.log"
    return 1
  fi
  return 0
}

stop_arm() {
  sudo docker rm -f strata-fork >/dev/null 2>&1
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 3000 ] && break
    sleep 2
  done
  echo "arm stopped: VRAM ${used} MiB left=$(left)"
}

bench() {
  label=$1; shift
  STRATA_URL="http://127.0.0.1:$PORT" BENCH_SEED=42 STRATA_API_KEY="$K" \
    python3 "$WT/scripts/prefillBench.py" "$@" | tee -a "$RES/$label-bench.jsonl"
}

ans4() {
  label=$1; pass=$2
  STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" \
    python3 "$WT/scripts/answers4.py" 24000 1,2,3,4 > "$RES/ans4-$label-pass$pass.json"
  echo "ans4 $label pass$pass done left=$(left)"
}

needle() {
  label=$1; tok=$2; pct=$3
  STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" \
    python3 "$SC/needle.py" "$tok" "$pct" "${label}" 24 "$RES/needle-$label-$tok.jsonl" 3.25 retrieval | tail -1
  echo "needle $label ${tok}k $pct% done left=$(left)"
}

qa() {
  label=$1
  STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" \
    python3 "$SC/qa.py" 128000 "$RES/qa-$label.json" > /dev/null
  python3 "$SC/score_qa.py" "$RES/qa-$label.json" | tee -a "$RES/qa-score.jsonl" > /dev/null
  echo "qa $label done: $(python3 -c "import json;d=json.load(open('$RES/qa-$label.json'));print(d['score'],'/',d['total'])" 2>/dev/null || echo '?') left=$(left)"
}

PATTN_ENV=()
for spec in "$@"; do
  label=${spec%%:*}; mode=${spec##*:}
  case "$label" in
    B*) PATTN_ENV=(-e STRATA_PATTN_SM60=1);;
    *)  PATTN_ENV=();;
  esac
  echo "=== arm $label mode=$mode $(date -Is)"
  start_arm "$label" "$BASE_SET" "${PATTN_ENV[@]}" || { echo "ARM $label FAILED"; continue; }
  grep -q "STRATA_PATTN_SM60=1" "$RES/engine-env-$label.txt" && echo "pattn env ok" || echo "pattn env MISSING"

  case "$mode" in
    full|qa|bench)
      bench "$label" 24000 3 1 prose
      bench "$label" 24000 3 1 code
      bench "$label" 128000 3 1 prose
      nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader > "$RES/gpu-end-$label.txt"
      echo "bench $label done left=$(left)"
      ;;
  esac

  case "$mode" in
    full)
      n=1; [ "$ANS2" = 2 ] && n=2
      for p in $(seq 1 $n); do ans4 "$label" "$p"; done
      [ "$(left)" -gt 300 ] && needle "$label" 128000 50
      [ "$(left)" -gt 300 ] && needle "$label" 128000 90
      [ "$(left)" -gt 340 ] && qa "$label"
      ;;
    qa)
      [ "$(left)" -gt 340 ] && qa "$label"
      [ "$(left)" -gt 470 ] && { bench "$label" 128000 2 1 code; echo "bench128c $label done left=$(left)"; }
      ;;
    n480)
      [ "$(left)" -gt 920 ] && needle "$label" 480000 50
      ;;
  esac
  stop_arm
done

restore_prod
trap - EXIT
echo "BLOCK_DONE $BLOCK $(date -Is)"
