#!/bin/bash
# 1 ブロック: GPU ロック -> 本番停止 -> (任意で parity / algo sweep) -> アーム計測 -> 復元 -> ロック解放
# 使い方: block-measure.sh <label>=<image> ...
# env: SIZES REPS MAXT KINDS EXTRA ANS KL SWEEP SWEEP_T PARITY SKIP_LOCK
set -u
res=~/work/results/pgdn
mkdir -p "$res"
if [ "${SKIP_LOCK:-0}" != 1 ]; then
  t=0
  until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
    [ $t -ge 600 ] && { echo "GPU lock busy" >&2; exit 2; }
    sleep 3; t=$((t+1))
  done
  echo "pgdn-block $(date -Is)" > /tmp/strata-gpu.lock/owner
fi
trap 'echo "trap: restoring prod $(date -Is)"; ~/work/wt/pgdn/scripts/restore.sh' EXIT

sudo docker exec strata sh -c "strata launch --stop --preset docker" >/dev/null 2>&1
sudo docker stop -t 120 strata >/dev/null 2>&1
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  hp=$(awk '/HugePages_Free/{print $2}' /proc/meminfo)
  # the prod container's own engine is what holds VRAM/hugepages; a host-wide pgrep also matches the caller's shell
  alive=$(sudo docker ps --format '{{.Names}}' | grep -c '^strata$')
  [ "$used" -lt 3000 ] && [ "$hp" -ge 20464 ] && [ "$alive" -eq 0 ] && break
  sleep 3
done
echo "prod stopped: VRAM ${used} MiB, HugePages_Free ${hp}, prod container ${alive}"

if [ "${PARITY:-0}" = 1 ]; then
  sudo docker run --rm --gpus all --cpuset-cpus 36-47 -v ~/work/wt/pgdn:/work/strata \
    nvidia/cuda:12.9.1-devel-ubuntu24.04 sh -c '
      cd /work/strata/build-native
      echo "==== gemm_f32_parity --quick"
      ./gemm_f32_parity --quick 2>&1 | tail -12; echo "exit=$?"
      echo "==== gdn_parity --selftest"
      ./gdn_parity --selftest 2>&1 | tail -12; echo "exit=$?"
      echo "==== gdn_chunk_parity --selftest"
      ./gdn_chunk_parity --selftest 2>&1 | tail -12; echo "exit=$?"
      echo "==== gdn_rec_parity"
      ./gdn_rec_parity 2>&1 | tail -8; echo "exit=$?"
      echo "==== prefill_mmq_kquant_test"
      STRATA_MMQ_KQ=1 ./prefill_mmq_kquant_test 2>&1 | tail -30; echo "exit=$?"
    ' > "$res/parity.log" 2>&1
  echo "parity done rc=$?"
  tail -40 "$res/parity.log"
fi

if [ "${SWEEP:-0}" = 1 ]; then
  sudo docker run --rm --gpus all --cpuset-cpus 36-47 -v ~/work/wt/pgdn/scripts:/w -w /w \
    nvidia/cuda:12.9.1-devel-ubuntu24.04 ./gemm_algo_sweep "${SWEEP_T:-8192}" > "$res/algo-sweep-T${SWEEP_T:-8192}.txt" 2>&1
  echo "algo sweep done rc=$?"
  cat "$res/algo-sweep-T${SWEEP_T:-8192}.txt"
fi

for spec in "$@"; do
  label=${spec%%=*}; img=${spec#*=}
  ~/work/wt/pgdn/scripts/arm.sh "$label" "$img" "${SIZES:-24000,128000}" "${REPS:-1}" "${MAXT:-1}" "${KINDS:-code,prose}" "${EXTRA:--}" "${ANS:-0}" "${KL:-0}" || echo "ARM $label FAILED"
done
echo BLOCK_DONE elapsed=$SECONDS
