#!/bin/bash
# 統合イメージのパリティ。GPU ロック -> 本番停止 -> まとめて実行 -> 本番復元 -> ロック解放
set -u
res=~/work/results/integrated
wt=~/work/wt/integrated
bd=/work/strata/build-native
img=nvidia/cuda:12.9.1-devel-ubuntu24.04
mkdir -p "$res"

t=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $t -ge 3600 ] && { echo "GPU lock busy" >&2; exit 2; }
  sleep 3; t=$((t+1))
done
echo "integrated-parity $(date -Is)" > /tmp/strata-gpu.lock/owner
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

run() {   # $1 name, rest: command line inside the build dir
  local name=$1; shift
  sudo docker run --rm --gpus all --cpuset-cpus 36-47 -v "$wt":/work/strata -w "$bd" \
    "$img" sh -c "export LD_LIBRARY_PATH=$bd:/usr/local/cuda/lib64; cd $bd && $*" \
    > "$res/parity-$name.log" 2>&1
  local rc=$?
  echo "parity $name exit $rc"
  tail -4 "$res/parity-$name.log"
  return $rc
}

rc=0
run nv_defaults_test ./nv_defaults_test || rc=1
run gemm_f32_parity ./gemm_f32_parity || rc=1
run prefill_mmq_v31_test ./prefill_mmq_v31_test || rc=1
run prefill_mmq_scatter_test ./prefill_mmq_scatter_test || rc=1
run prefill_swiglu_quant_test ./prefill_swiglu_quant_test || rc=1
run native_grouped_parity ./native_grouped_parity || rc=1
run qsa_merge_daily_default './qsa_merge_daily_parity --device 0' || rc=1
run qsa_merge_daily_daily 'STRATA_ATTN_DAILY=1 ./qsa_merge_daily_parity --device 0' || rc=1
run qsa_rows_rows './qsa_rows_bench --device 0 --iters 20 --rows' || rc=1
run mmvq_multi ./mmvq_multi_parity || rc=1
run b6_mmvq './b6_mmvq_bench 50 0 1' || rc=1
run s2_expert_grouped './s2_expert_grouped_parity --selftest' || rc=1
run mmvq_il ./mmvq_il_parity || rc=1
run dequant_wide ./dequant_wide_parity || rc=1
run verify_parity './verify_parity --selftest' || rc=1
run router_top10 './router_top10_parity --selftest' || rc=1
run router_dot ./router_dot_parity || rc=1
run router_chain './router_chain_bench 2 20' || rc=1
run native_multi ./native_multi_parity || rc=1
run gr_parity './gr_parity --selftest' || rc=1
run shared_expert './shared_expert_parity --selftest' || rc=1

echo "PARITY_RC=$rc"
[ $rc -eq 0 ] && echo PARITY_ALL_OK || echo PARITY_SOME_FAILED
