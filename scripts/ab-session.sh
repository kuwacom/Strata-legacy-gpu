#!/bin/bash
# MMQ I=128 A/B session: GPU lock -> stop production -> arms -> restore production -> release the lock.
# The arms are "<label>:<image>" pairs; they run in the order given (A-B-A).
#   ab-session.sh <outroot> <label>:<image> [<label>:<image> ...]
# One block stays under 60 minutes: the deadline check skips an arm that would not fit, and the production
# restore always runs (~/opt-work/prod.env is re-read at that moment, so a 512K setting that landed meanwhile is kept).
set -u
outroot=$1; shift
vol=/var/lib/docker/volumes/compose_strata-data/_data
deadline_s=${DEADLINE_S:-2700}
start=$(date +%s)
mkdir -p "$outroot"

# GPU lock: 3-5 s interval, give up after 30 minutes and say so
t=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $t -ge 600 ] && { echo "GPU lock busy (owner: $(cat /tmp/strata-gpu.lock/owner 2>/dev/null))" >&2; exit 2; }
  sleep 4; t=$((t+1))
done
echo "mmq-i128 $(date -Is)" > /tmp/strata-gpu.lock/owner

restore_prod() {
  cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
  sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
  cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1
  s=""
  for i in $(seq 1 120); do s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null); [ "$s" = healthy ] && break; sleep 10; done
  echo "prod health: $s image: $(sudo docker inspect -f '{{.Config.Image}}' strata)"
  PID=$(pgrep -f "strata --serve" | head -1)
  [ -n "$PID" ] && sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$outroot/prod-environ.txt" && cat "$outroot/prod-environ.txt"
  sudo docker ps --format '{{.Names}} {{.Status}}'
  sudo docker ps -a --format '{{.Names}} {{.Status}}' | grep -E "strata-fork|mmq" && echo "RESIDUE" || echo "no residue"
  if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
    rm -rf /tmp/strata-gpu.lock && echo "gpu lock released $(date -Is)" && return 0
  fi
  echo "gpu lock NOT released (restore failed)"; return 1
}

# production off: the container too, so the Terminal cannot restart the engine behind us
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

for arm in "$@"; do
  label=${arm%%:*}; image=${arm#*:}
  now=$(date +%s); left=$((deadline_s - (now - start)))
  if [ $left -lt 900 ]; then echo "skip $label: only ${left}s left"; continue; fi
  bash ~/work/wt/mmq/scripts/ab-arm.sh "$label" "$image" "$outroot/$label" "${SIZES:-4000,24000}" "${REP:-3}" "${MAXT:-128}" "${KINDS:-code,prose}"
done

restore_prod || exit 5
echo SESSION_OK
