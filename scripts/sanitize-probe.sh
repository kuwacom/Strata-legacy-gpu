#!/bin/bash
# I=96 probe under compute-sanitizer: stop production, run the MMQ screen with memcheck, restore production, release the lock.
set -u
outroot=${1:-/home/manager/work/results/mmqv3/sanitizer}
wt=/home/manager/work/wt/mmq
mkdir -p "$outroot"

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

echo "=== compute-sanitizer memcheck $(date -Is)"
sudo docker run --rm --gpus all -v "$wt/build-native":/b -w /b nvidia/cuda:12.9.1-devel-ubuntu24.04 \
  compute-sanitizer --tool memcheck --launch-timeout 180 ./prefill_mmq_v31_test > "$outroot/memcheck.log" 2>&1
echo "sanitizer exit=$?"
grep -c "Invalid __global__ write" "$outroot/memcheck.log" || true
tail -20 "$outroot/memcheck.log"

cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1
s=""
for i in $(seq 1 120); do s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null); [ "$s" = healthy ] && break; sleep 10; done
echo "prod health: $s image: $(sudo docker inspect -f '{{.Config.Image}}' strata)"
PID=$(pgrep -f "strata --serve" | head -1)
if [ -n "$PID" ]; then sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$outroot/prod-environ.txt"; fi
sudo docker ps --format '{{.Names}} {{.Status}}'
if sudo docker ps -a --format '{{.Names}}' | grep -qE "strata-fork"; then echo "RESIDUE: strata-fork present"; else echo "no residue"; fi
if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
  echo "prod restored and .env matches ~/opt-work/prod.env"
else
  echo "RESTORE_FAILED"; exit 5
fi
rm -rf /tmp/strata-gpu.lock && echo "gpu lock released $(date -Is)"
echo SANITIZE_OK
