#!/bin/bash
# v7 の総合検証: パリティ -> prefill 24K x2 / 128K -> decode -> needle 24K/32K/128K -> QA 8 -> 本番復旧
#   IMG=strata:fork-v7 TAG=v7 SET_EXTRA="..." bash verifyV7.sh <block>
set -u
BLOCK=${1:-v7}
IMG=${IMG:?set IMG}
TAG=${TAG:-v7}
RES=${RES:-$HOME/work/results/$TAG}
WT=${WT:-$HOME/work/wt/v7}
VOL=/var/lib/docker/volumes/compose_strata-data/_data
PORT=18080
SET_EXTRA=${SET_EXTRA:-STRATA_MMQ_SQ=1}
SET="ple-io=ram prefill=32768 kv-resident=32768 draft-conf=0.7 STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 spec=4 STRATA_PATTN_SM60=1 STRATA_SELECT_SIMT=1 STRATA_HC_DOWN_BF16=1 $SET_EXTRA"
K=$(grep '^API_KEY=' $HOME/opt-work/prod.env | cut -d= -f2)
mkdir -p "$RES"
exec > >(tee -a "$RES/block-$BLOCK.log") 2>&1
echo "=== verify $BLOCK start $(date -Is) IMG=$IMG extra='$SET_EXTRA'"

lw=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $lw -ge 120 ] && { echo "GPU lock busy after 60 min" >&2; exit 2; }
  sleep 30; lw=$((lw+1))
done
echo "$TAG-$BLOCK $(date -Is)" > /tmp/strata-gpu.lock/owner
echo "lock acquired $(date -Is)"

restore_prod() {
  cp $HOME/opt-work/prod.env $HOME/Strata/docker/compose/.env
  sudo docker rm -f strata-fork >/dev/null 2>&1
  sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
  ( cd $HOME/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1 )
  s=""
  for i in $(seq 1 90); do
    s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null)
    [ "$s" = healthy ] && break
    sleep 10
  done
  echo "prod health: $s" > "$RES/prod-restore-$BLOCK.txt" 2>&1
  if [ "$s" = healthy ] && diff -q $HOME/Strata/docker/compose/.env $HOME/opt-work/prod.env >/dev/null; then
    rm -rf /tmp/strata-gpu.lock
    echo "gpu lock released $(date -Is)"
    return 0
  fi
  echo "restore failed, lock kept" >&2
  return 1
}
trap restore_prod EXIT

sudo docker exec strata sh -c "strata launch --stop --preset docker" >/dev/null 2>&1
sudo docker stop -t 120 strata >/dev/null 2>&1
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  hp=$(awk '/HugePages_Free/{print $2}' /proc/meminfo)
  alive=$(pgrep -f "strata --serve" | wc -l)
  [ "$used" -lt 3000 ] && [ "$hp" -ge 20464 ] && [ "$alive" -eq 0 ] && break
  sleep 5
done
echo "prod stopped: VRAM ${used} MiB"

# ---- パリティ ----
CIMG=nvidia/cuda:12.9.1-devel-ubuntu24.04
BD=/work/strata/build-native
for t in gemm_f32_parity gemm_bf16_parity qsa_pattn_parity prefill_swiglu_quant_test prefill_mmq_tab_test; do
  if sudo docker run --rm -v "$WT":/work/strata -w "$BD" "$CIMG" test -x "$BD/$t"; then
    sudo docker run --rm --gpus all --cpuset-cpus 36-47 -v "$WT":/work/strata -w "$BD" "$CIMG" \
      sh -c "export LD_LIBRARY_PATH=$BD:/usr/local/cuda/lib64; ./$t --device 0 || ./$t" \
      > "$RES/parity-$t.log" 2>&1
    echo "parity $t exit $? : $(tail -1 "$RES/parity-$t.log" | cut -c1-160)"
  else
    echo "parity $t : not built"
  fi
done

# ---- アーム ----
sudo rm -rf "$RES/pristine"
sudo mkdir -p "$RES/pristine"
sudo cp -a "$VOL/state" "$RES/pristine/state"
sudo cp -a "$VOL/config" "$RES/pristine/config"
d="$RES/state-$BLOCK"
mkdir -p "$d"
sudo rm -rf "$d/state" "$d/config"
sudo cp -a "$RES/pristine/state" "$d/state"
sudo cp -a "$RES/pristine/config" "$d/config"
sudo docker rm -f strata-fork >/dev/null 2>&1
sudo docker run -d --name strata-fork --gpus all --cpuset-cpus 36-47 \
  --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$PORT:8080 \
  -e MODEL=IQ3_XXS -e FAMILY=qwen -e CONTEXT=512k -e PARALLEL=1 -e VISION=off -e TUNE=skip \
  -e API_KEY="$K" -e STRATA_SET="$SET" -e STRATA_PREFILL_TIMING=1 \
  -v "$VOL":/data -v "$d/state":/data/state -v "$d/config":/data/config \
  --restart no "$IMG" > "$RES/docker-run-$BLOCK.log" 2>&1
s=""
for i in $(seq 1 120); do
  s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata-fork 2>/dev/null)
  [ "$s" = healthy ] && break
  sudo docker inspect -f "{{.State.Status}}" strata-fork 2>/dev/null | grep -q exited && break
  sleep 5
done
echo "arm health=$s"
if [ "$s" != healthy ]; then
  sudo docker logs strata-fork > "$RES/docker-$BLOCK.log" 2>&1; tail -25 "$RES/docker-$BLOCK.log"; exit 4
fi
export STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" BENCH_SEED=42
python3 $HOME/work/wt/pknobs/scripts/prefillBench.py 24000 2 1 prose >> "$RES/bench-$BLOCK.jsonl" 2>&1
python3 $HOME/work/wt/pknobs/scripts/prefillBench.py 128000 1 1 prose >> "$RES/bench-$BLOCK.jsonl" 2>&1
echo "== prefill =="; cat "$RES/bench-$BLOCK.jsonl"
python3 $HOME/bench.py 4000,24000 2 256 >> "$RES/decode-$BLOCK.jsonl" 2>&1
echo "== decode =="; cat "$RES/decode-$BLOCK.jsonl"
for t in 24000 32000 128000; do
  python3 $HOME/work/results/v3/scripts/needle.py $t 50 V7-R2K9 24 "$RES/needle$t-$BLOCK.jsonl" 2>&1 | tail -1
done
echo "== needle =="; cat "$RES"/needle*"-$BLOCK.jsonl" 2>/dev/null | python3 -c "import sys,json
for l in sys.stdin:
    try:
        d=json.loads(l); print(d.get('target_tokens'), d.get('found'), d.get('prefill_tps'))
    except Exception: pass"
python3 $HOME/work/results/v3/scripts/qa.py 128000 "$RES/qa-$BLOCK.jsonl" 2>&1 | tail -3
echo "== qa =="; tail -2 "$RES/qa-$BLOCK.jsonl" 2>/dev/null
sudo docker logs strata-fork > "$RES/docker-$BLOCK.log" 2>&1
sudo cp "$d/state/strata/launch/docker.engine.log" "$RES/engine-$BLOCK.log" 2>/dev/null || true
sudo chown $(id -un) "$RES/engine-$BLOCK.log" 2>/dev/null || true
sudo docker rm -f strata-fork >/dev/null 2>&1
echo "=== verify $BLOCK done $(date -Is)"
exit 0
