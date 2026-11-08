#!/bin/bash
# Restore production from ~/opt-work/prod.env (re-read now) and release the GPU lock if it is healthy.
set -u
outroot=${1:-/home/manager/work/results/mmq}
cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1
s=""
for i in $(seq 1 120); do s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null); [ "$s" = healthy ] && break; sleep 10; done
echo "prod health: $s image: $(sudo docker inspect -f '{{.Config.Image}}' strata)"
PID=$(pgrep -f "strata --serve" | head -1)
if [ -n "$PID" ]; then sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" | tee "$outroot/prod-environ.txt"; fi
sudo docker ps --format '{{.Names}} {{.Status}}'
if sudo docker ps -a --format '{{.Names}}' | grep -q "strata-fork"; then echo "RESIDUE: strata-fork present"; else echo "no residue"; fi
if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
  echo "prod restored, .env matches ~/opt-work/prod.env"
  if [ "${KEEP_LOCK:-0}" != 1 ]; then rm -rf /tmp/strata-gpu.lock && echo "gpu lock released $(date -Is)"; fi
else
  echo "RESTORE_FAILED"; exit 5
fi
