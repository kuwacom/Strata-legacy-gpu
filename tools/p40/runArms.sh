#!/bin/bash
# 複数アームを 1 回の GPU ロックで順に測る（本番の停止・復旧は前後 1 回だけ）
#   MANIFEST=arms.txt IMG=strata:fork-x TAG=x bash runArms.sh <block>
# MANIFEST の各行: <label>|<追加の SET（空可）>|<tokens（空なら 128000）>
set -u
BLOCK=${1:-a1}
IMG=${IMG:?set IMG}
TAG=${TAG:-arms}
MANIFEST=${MANIFEST:?set MANIFEST}
RES=${RES:-$HOME/work/results/$TAG}
VOL=/var/lib/docker/volumes/compose_strata-data/_data
PORT=18080
BASE_SET=${BASE_SET:-"ple-io=ram prefill=32768 kv-resident=32768 draft-conf=0.7 STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 spec=4 STRATA_PATTN_SM60=1 STRATA_SELECT_SIMT=1 STRATA_HC_DOWN_BF16=1 STRATA_MMQ_SQ=1"}
K=$(grep '^API_KEY=' $HOME/opt-work/prod.env | cut -d= -f2)
mkdir -p "$RES"
exec > >(tee -a "$RES/block-$BLOCK.log") 2>&1
echo "=== arms $BLOCK start $(date -Is) IMG=$IMG"

lw=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $lw -ge 120 ] && { echo "GPU lock busy after 60 min" >&2; exit 2; }
  sleep 30; lw=$((lw+1))
done
echo "$TAG-$BLOCK $(date -Is)" > /tmp/strata-gpu.lock/owner
echo "lock acquired $(date -Is)"

restore_prod() {
  cp $HOME/opt-work/prod.env $HOME/Strata/docker/compose/.env
  sudo docker rm -f strata-fork >/dev/null 2>&1
  sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
  ( cd $HOME/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1 )
  s=""
  for i in $(seq 1 90); do
    s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null)
    [ "$s" = healthy ] && break
    sleep 10
  done
  echo "prod health: $s" > "$RES/prod-restore-$BLOCK.txt" 2>&1
  if [ "$s" = healthy ] && diff -q $HOME/Strata/docker/compose/.env $HOME/opt-work/prod.env >/dev/null; then
    rm -rf /tmp/strata-gpu.lock
    echo "gpu lock released $(date -Is)"
    return 0
  fi
  echo "restore failed, lock kept" >&2
  return 1
}
trap restore_prod EXIT

sudo docker exec strata sh -c "strata launch --stop --preset docker" >/dev/null 2>&1
sudo docker stop -t 120 strata >/dev/null 2>&1
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  hp=$(awk '/HugePages_Free/{print $2}' /proc/meminfo)
  alive=$(pgrep -f "strata --serve" | wc -l)
  [ "$used" -lt 3000 ] && [ "$hp" -ge 20464 ] && [ "$alive" -eq 0 ] && break
  sleep 5
done
echo "prod stopped: VRAM ${used} MiB"

sudo rm -rf "$RES/pristine"
sudo mkdir -p "$RES/pristine"
sudo cp -a "$VOL/state" "$RES/pristine/state"
sudo cp -a "$VOL/config" "$RES/pristine/config"

while IFS='|' read -r label extra toks; do
  [ -z "${label:-}" ] && continue
  case "$label" in \#*) continue ;; esac
  toks=${toks:-128000}
  set_eff="$BASE_SET"
  [ -n "${extra:-}" ] && set_eff="$BASE_SET $extra"
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
    -e API_KEY="$K" -e STRATA_SET="$set_eff" -e STRATA_PREFILL_TIMING=1 \
    -v "$VOL":/data -v "$d/state":/data/state -v "$d/config":/data/config \
    --restart no "$IMG" > "$RES/docker-run-$label.log" 2>&1
  s=""
  for i in $(seq 1 120); do
    s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata-fork 2>/dev/null)
    [ "$s" = healthy ] && break
    sudo docker inspect -f "{{.State.Status}}" strata-fork 2>/dev/null | grep -q exited && break
    sleep 5
  done
  t1=$(date +%s)
  echo "arm $label health=$s after $((t1-t0))s tokens=$toks extra='${extra:-}'"
  if [ "$s" != healthy ]; then
    sudo docker logs strata-fork > "$RES/docker-$label.log" 2>&1
    tail -25 "$RES/docker-$label.log"
    sudo docker rm -f strata-fork >/dev/null 2>&1
    continue
  fi
  export STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" BENCH_SEED=42
  for t in $toks; do
    python3 $HOME/work/wt/pknobs/scripts/prefillBench.py "$t" 1 1 prose >> "$RES/bench-$label.jsonl" 2>&1
  done
  echo "== $label bench =="; tail -2 "$RES/bench-$label.jsonl"
  sudo docker logs strata-fork > "$RES/docker-$label.log" 2>&1
  sudo cp "$d/state/strata/launch/docker.engine.log" "$RES/engine-$label.log" 2>/dev/null || true
  sudo chown $(id -un) "$RES/engine-$label.log" 2>/dev/null || true
  sudo docker rm -f strata-fork >/dev/null 2>&1
  sleep 15
done < "$MANIFEST"
echo "=== arms $BLOCK done $(date -Is)"
exit 0
