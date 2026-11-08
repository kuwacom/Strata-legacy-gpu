#!/bin/bash
# 汚染された A2 の再測: base イメージ (A) だけを 4 reps で走らせる。A-B-A の 3 本目に使う
set -u
res=~/work/results/defaults
A=${A_IMAGE:-strata:fork-9f7d977-base}
vol=/var/lib/docker/volumes/compose_strata-data/_data
port=${PORT:-18080}
mkdir -p "$res"

t=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $t -ge 1200 ] && { echo "GPU lock busy" >&2; exit 2; }
  sleep 3; t=$((t+1))
done
echo "defaults-a2only $(date -Is)" > /tmp/strata-gpu.lock/owner

restore_prod() {
  cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
  sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
  cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1
  s=""
  for i in $(seq 1 90); do s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null); [ "$s" = healthy ] && break; sleep 10; done
  echo "prod health: $s"
  sudo docker ps --format '{{.Names}} {{.Status}}'
  if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
    rm -rf /tmp/strata-gpu.lock && echo "gpu lock released $(date -Is)" && return 0
  fi
  echo "gpu lock NOT released (restore failed)"; return 1
}

K=$(grep ^API_KEY= ~/opt-work/prod.env | cut -d= -f2)
SSET=${SSET:-"ple-io=ram prefill=auto STRATA_DENSE_F32=1 spec=4 spec-min-p=0.5"}

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

label=A2b
st="$res/state-$label"
sudo rm -rf "$st"; sudo cp -a "$vol/state" "$st"
sudo docker rm -f strata-fork >/dev/null 2>&1
sleep 5
sudo docker run -d --name strata-fork --gpus all --cpuset-cpus 36-47 \
  --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$port:8080 \
  -e MODEL=IQ3_XXS -e FAMILY=qwen -e CONTEXT=128k -e PARALLEL=1 -e VISION=off -e TUNE=skip \
  -e API_KEY="$K" -e STRATA_SET="$SSET" \
  -v "$vol":/data -v "$st":/data/state \
  --restart no "$A" > "$res/docker-run-$label.log" 2>&1
s=""
for i in $(seq 1 90); do
  s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata-fork 2>/dev/null)
  [ "$s" = healthy ] && break
  sudo docker inspect -f "{{.State.Status}}" strata-fork 2>/dev/null | grep -q exited && break
  sleep 5
done
echo "arm $label health=$s"
sudo docker logs strata-fork > "$res/docker-$label.log" 2>&1
ts=$(date +%m%d-%H%M%S)
STRATA_URL="http://127.0.0.1:$port" BENCH_SEED=42 STRATA_API_KEY=$K python3 ~/bench.py 4000,24000 4 256 \
  > "$res/bench-a2only-$label-$ts.jsonl"
cat "$res/bench-a2only-$label-$ts.jsonl"
STRATA_URL="http://127.0.0.1:$port" STRATA_API_KEY=$K python3 ~/work/wt/defaults/scripts/answers4.py "$res/answers-a2only-$label-$ts.json" \
  > "$res/answers-a2only-$label-$ts.out" 2>&1 || true

sudo docker rm -f strata-fork >/dev/null 2>&1
sleep 5
restore_prod || exit 5
echo A2ONLY_ALL_OK
