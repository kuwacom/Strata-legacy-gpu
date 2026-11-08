#!/bin/bash
# 本番相当（expert-cache auto、profiler off）で A-B-A を取る: 補足の 2 アーム比較は順序の交絡があるため
set -u
res=~/work/results/lookahead
vol=/var/lib/docker/volumes/compose_strata-data/_data
port=${PORT:-18081}
imgA=${IMGA:-strata:fork-9f7d977-base}
imgB=${IMGB:-strata:fork-9f7d977-lookahead}
mkdir -p "$res"

t=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $t -ge 900 ] && { echo "GPU lock busy" >&2; exit 2; }
  sleep 5; t=$((t+1))
done
echo "lookahead-aba2 $(date -Is)" > /tmp/strata-gpu.lock/owner

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
  if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null && \
     ! grep -qE "STRATA_VERIFY_PROFILE|STRATA_POOL_PREFETCH|STRATA_DECODE_TIMING|STRATA_LOOKAHEAD_STATS" "$res/prod-environ.txt"; then
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

start_fork() {
  local arm=$1 tag=$2 sset=$3
  sudo rm -rf "$res/state-$arm"; sudo cp -a "$vol/state" "$res/state-$arm"
  sudo docker rm -f strata-fork >/dev/null 2>&1
  sudo docker run -d --name strata-fork --gpus all --cpuset-cpus 36-47 \
    --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$port:8080 \
    -e MODEL=IQ3_XXS -e FAMILY=qwen -e CONTEXT=512k -e PARALLEL=1 -e VISION=off -e TUNE=skip \
    -e API_KEY="$K" -e STRATA_SET="$sset" \
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
  echo "$arm health=$s image=$tag $(date -Is)"
  if [ "$s" != healthy ]; then sudo docker logs strata-fork > "$res/docker-$arm.log" 2>&1; tail -30 "$res/docker-$arm.log"; return 4; fi
  PID=$(pgrep -f "strata --serve" | head -1)
  sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$res/engine-environ-$arm.txt"
  grep -E "STRATA_POOL_PREFETCH|STRATA_VERIFY_PROFILE|STRATA_LOOKAHEAD_STATS" "$res/engine-environ-$arm.txt" || echo "  (none)"
  return 0
}

stop_fork() {
  sudo docker exec strata-fork sh -c 'cat /data/state/strata/launch/docker.engine.log' > "$res/enginelog-$1.log" 2>&1 || true
  sudo docker rm -f strata-fork >/dev/null 2>&1
  sleep 5
}

SET="ple-io=ram prefill=auto STRATA_DENSE_F32=1 spec=4 spec-min-p=0.5 rope-scaling=yarn rope-scale=2 STRATA_DECODE_TIMING=1"

for spec in "P-A1 $imgA " "P-B $imgB STRATA_POOL_PREFETCH=1" "P-A2 $imgA "; do
  set -- $spec
  arm=$1; tag=$2; extra=${3:-}
  start_fork "$arm" "$tag" "$SET $extra" || { restore_prod; exit 4; }
  ts=$(date +%m%d-%H%M%S)
  STRATA_URL="http://127.0.0.1:$port" BENCH_SEED=42 STRATA_API_KEY=$K \
    python3 ~/bench.py 4000 3 256 prose > "$res/bench-$arm-$ts.jsonl" 2>&1
  cat "$res/bench-$arm-$ts.jsonl"
  stop_fork "$arm-$ts"
done

restore_prod || exit 5
echo ABA2_ALL_OK
