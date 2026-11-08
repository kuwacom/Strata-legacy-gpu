#!/bin/bash
# 1 ブロック: GPU ロック -> 本番停止 -> 引数のアームを順に計測 -> 本番復元 -> ロック解放
# 使い方: block-integrated.sh "label image mode" "label image mode" ...
set -u
res=~/work/results/integrated
mkdir -p "$res"
if [ $# -eq 0 ]; then echo "usage: block-integrated.sh 'label image mode' ..." >&2; exit 2; fi

t=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $t -ge 3600 ] && { echo "GPU lock busy" >&2; exit 2; }
  sleep 3; t=$((t+1))
done
echo "integrated-block $(date -Is)" > /tmp/strata-gpu.lock/owner
trap 'echo "trap: restoring prod $(date -Is)"; ~/work/wt/integrated/scripts/restore-prod.sh' EXIT

sudo docker exec strata sh -c "strata launch --stop --preset docker" >/dev/null 2>&1
sudo docker stop -t 120 strata >/dev/null 2>&1
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  hp=$(awk '/HugePages_Free/{print $2}' /proc/meminfo)
  alive=$(pgrep -f "strata --serve" | wc -l)
  [ "$used" -lt 3000 ] && [ "$hp" -ge 20464 ] && [ "$alive" -eq 0 ] && break
  sleep 3
done
echo "prod stopped: VRAM ${used} MiB, HugePages_Free ${hp}, engines ${alive}"

for spec in "$@"; do
  set -- $spec
  label=$1; img=$2; mode=$3
  echo "=== arm $label $mode $(date -Is) ==="
  ~/work/wt/integrated/scripts/arm-integrated.sh "$label" "$img" "$mode" || echo "ARM $label $mode FAILED"
done
echo BLOCK_DONE elapsed=$SECONDS
