#!/bin/bash
# sm_61 に関係するパリティテストを GPU ロック下で実行する。本番を止めて VRAM を空け、終了時に復元する
set -u
res=~/work/results/defaults
wt=~/work/wt/defaults
bd=/work/strata/build-native
img=nvidia/cuda:12.9.1-devel-ubuntu24.04
mkdir -p "$res"

t=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $t -ge 600 ] && { echo "GPU lock busy" >&2; exit 2; }
  sleep 3; t=$((t+1))
done
echo "defaults-parity $(date -Is)" > /tmp/strata-gpu.lock/owner

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

run_gpu() {   # $1 name, rest: command line inside the build dir
  local name=$1; shift
  sudo docker run --rm --gpus all --cpuset-cpus 36-47 -v "$wt":/work/strata -w "$bd" \
    "$img" sh -c "export LD_LIBRARY_PATH=$bd:/usr/local/cuda/lib64; cd $bd && $*" \
    > "$res/parity-$name.log" 2>&1
  local rc=$?
  echo "parity $name exit $rc"
  tail -6 "$res/parity-$name.log"
  return $rc
}

run_cpu() {   # $1 name, rest: command line inside the build dir
  local name=$1; shift
  sudo docker run --rm --cpuset-cpus 36-47 -v "$wt":/work/strata -w "$bd" \
    "$img" sh -c "export LD_LIBRARY_PATH=$bd:/usr/local/cuda/lib64; cd $bd && $*" \
    > "$res/parity-$name.log" 2>&1
  local rc=$?
  echo "parity $name exit $rc"
  tail -6 "$res/parity-$name.log"
  return $rc
}

rc=0
run_cpu nv_defaults_test ./nv_defaults_test || rc=1
run_gpu qsa_merge_daily_default ./qsa_merge_daily_parity --device 0 || rc=1
run_gpu qsa_merge_daily_daily 'STRATA_ATTN_DAILY=1 ./qsa_merge_daily_parity --device 0' || rc=1
run_gpu qsa_rows_rows './qsa_rows_bench --device 0 --iters 20 --rows' || rc=1
run_gpu mmvq_multi ./mmvq_multi_parity || rc=1
run_gpu b6_mmvq './b6_mmvq_bench 50 0 1' || rc=1
run_gpu native_grouped ./native_grouped_parity || rc=1
run_gpu s2_expert_grouped './s2_expert_grouped_parity --selftest' || rc=1
run_gpu mmvq_il './mmvq_il_parity' || rc=1
run_gpu dequant_wide ./dequant_wide_parity || rc=1

echo "PARITY_RC=$rc"
restore_prod || exit 5
[ $rc -eq 0 ] && echo PARITY_ALL_OK || echo PARITY_SOME_FAILED
