#!/bin/bash
# One GPU block for the MMQ I=128 A/B. Assumes the GPU lock is already held (owner file written).
# Stops production, runs the parity screens, the bitwise logits pair (A/B) and/or one more dump arm,
# the timing arms in the order given, restores production from ~/opt-work/prod.env (re-read now) and releases the lock.
#   block.sh <outroot> <label>:<image> [<label>:<image> ...]
# Env: SKIP_PARITY=1, SKIP_DUMP=1, DUMP_ARM, DUMP_IMAGE, DUMP_F16=1 (drafter FP16 GEMMs in the dump arms),
#      SIZES, REP, MAXT, KINDS, DEADLINE_S, KEEP_LOCK=1
set -u
outroot=$1; shift
vol=/var/lib/docker/volumes/compose_strata-data/_data
wt=/home/manager/work/wt/mmq
deadline_s=${DEADLINE_S:-3000}
start=$(date +%s)
mkdir -p "$outroot"

restore_prod() {
  cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
  sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
  cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1
  s=""
  for i in $(seq 1 120); do s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null); [ "$s" = healthy ] && break; sleep 10; done
  echo "prod health: $s image: $(sudo docker inspect -f '{{.Config.Image}}' strata)"
  PID=$(pgrep -f "strata --serve" | head -1)
  if [ -n "$PID" ]; then sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$outroot/prod-environ.txt"; cat "$outroot/prod-environ.txt"; fi
  sudo docker ps --format '{{.Names}} {{.Status}}'
  if sudo docker ps -a --format '{{.Names}}' | grep -qE "strata-fork"; then echo "RESIDUE: strata-fork present"; else echo "no residue"; fi
  if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
    echo "prod restored and .env matches ~/opt-work/prod.env"
    return 0
  fi
  echo "RESTORE_FAILED"; return 1
}

# production off (the container too: the Terminal would restart the engine)
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

f16=""
[ "${DUMP_F16:-0}" = 1 ] && f16="MTP_F16=1"

if [ "${SKIP_PARITY:-0}" != 1 ]; then
  echo "=== parity $(date -Is)"
  bash "$wt/scripts/parity.sh" "$outroot/parity"
fi

if [ "${SKIP_DUMP:-0}" != 1 ]; then
  echo "=== logits dump pair $(date -Is)${f16:+ (drafter FP16)}"
  env $f16 DUMP=1 bash "$wt/scripts/ab-arm.sh" dump-A "${DUMP_A:-strata:fork-9f7d977-base}" "$outroot/dump-A" 4000 1 128 code
  env $f16 DUMP=1 bash "$wt/scripts/ab-arm.sh" dump-B "${DUMP_B:-strata:fork-mmq-i128-v2}" "$outroot/dump-B" 4000 1 128 code
fi

if [ -n "${DUMP_ARM:-}" ]; then
  echo "=== logits dump $DUMP_ARM $(date -Is)${f16:+ (drafter FP16)}"
  env $f16 DUMP=1 bash "$wt/scripts/ab-arm.sh" "$DUMP_ARM" "${DUMP_IMAGE:-strata:fork-9f7d977-base}" "$outroot/$DUMP_ARM" 4000 1 128 code
fi

for arm in "$@"; do
  label=${arm%%:*}; image=${arm#*:}
  now=$(date +%s); left=$((deadline_s - (now - start)))
  if [ $left -lt 900 ]; then echo "skip $label: only ${left}s left"; continue; fi
  echo "=== arm $label $(date -Is)"
  TIMING=1 bash "$wt/scripts/ab-arm.sh" "$label" "$image" "$outroot/$label" "${SIZES:-4000,24000,30000}" "${REP:-3}" "${MAXT:-128}" "${KINDS:-code}"
done

restore_prod || exit 5
if [ "${KEEP_LOCK:-0}" = 1 ]; then
  echo "lock kept for the next block"
else
  rm -rf /tmp/strata-gpu.lock && echo "gpu lock released $(date -Is)"
fi
echo BLOCK_OK
