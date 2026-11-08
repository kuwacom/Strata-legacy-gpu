#!/bin/bash
# pfw v3: PF_W16 cache / prefill chunk 4096+16384 / SELECT_SIMT の A-B-A 計測
# Usage: runPfw.sh w16a|w16b|w16a2|chunk4096|chunk16384|simt|simt480
# 書き込み先は ~/work/results/pfw のみ。ソース変更なし。commit/push なし。
set -u
phase=${1:-w16a}
RES=~/work/results/pfw
# 雛形スクリプトは results/pfw/scripts にコピー済み（pqsa の A1 失敗教訓: 参照先に置く）
SC=~/work/results/pfw/scripts
TAG=strata:fork-v3
VOL=/var/lib/docker/volumes/compose_strata-data/_data
PORT=18080
BASE_SET="kv=int8 kv-resident=32768 ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 spec=4"
mkdir -p "$RES/scripts"

# API キーは変数に入れるだけ（表示しない）
K=$(grep '^API_KEY=' ~/opt-work/prod.env | cut -d= -f2)

# GPU ロック: 50 秒間隔で最大 45 分待つ（1 回の失敗で諦めない）
lock_wait=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $lock_wait -ge 54 ] && { echo "GPU lock busy after 45min" >&2; exit 2; }
  sleep 50; lock_wait=$((lock_wait+1))
done
echo "pfw-$phase $(date -Is)" > /tmp/strata-gpu.lock/owner
T0=$(date +%s)
say() { echo "[$(date +%H:%M:%S) +$(( ($(date +%s) - T0) / 60 ))m] $*" | tee -a "$RES/block-$phase.out"; }

{
  echo "disk before: $(df -h / | tail -1)"
  echo "volume: $(sudo du -sh $VOL | cut -f1)"
} >> "$RES/block-$phase.out"

restore_prod() {
  cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
  # fork が RAM の expert arena を握ったままだと本番が OOM で再起動を繰り返すため、先に消す
  sudo docker rm -f strata-fork >/dev/null 2>&1
  sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
  ( cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1 )
  s=""
  for i in $(seq 1 90); do
    s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null)
    [ "$s" = healthy ] && break
    sleep 10
  done
  {
    echo "prod health: $s"
    PID=$(pgrep -f "strata --serve" | head -1)
    if [ -n "${PID:-}" ]; then
      sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_"
      echo "--- cmdline ---"
      sudo cat /proc/$PID/cmdline | tr "\0" " "; echo
    fi
    sudo docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
  } > "$RES/prod-restore-$phase.txt" 2>&1
  sudo docker exec strata sh -c "cat /data/config/strata/launch/docker.json" > "$RES/prod-preset-$phase.json" 2>&1
  if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
    rm -rf /tmp/strata-gpu.lock
    say "gpu lock released"
    return 0
  fi
  echo "restore failed, lock kept" >&2
  return 1
}
trap restore_prod EXIT

stop_prod() {
  # 本番を止めて VRAM と RAM を空ける
  sudo docker exec strata sh -c "strata launch --stop --preset docker" >/dev/null 2>&1
  sudo docker stop -t 120 strata >/dev/null 2>&1
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    hp=$(awk '/HugePages_Free/{print $2}' /proc/meminfo)
    alive=$(pgrep -f "strata --serve" | wc -l)
    [ "$used" -lt 3000 ] && [ "$hp" -ge 20464 ] && [ "$alive" -eq 0 ] && break
    sleep 5
  done
  say "prod stopped: VRAM ${used} MiB, HugePages_Free ${hp}, engines ${alive}"

  sudo rm -rf "$RES/pristine"
  sudo mkdir -p "$RES/pristine"
  sudo cp -a "$VOL/state" "$RES/pristine/state"
  sudo cp -a "$VOL/config" "$RES/pristine/config"
}

start_arm() {
  label=$1; sset=$2; shift 2
  d="$RES/$label"
  mkdir -p "$d"
  sudo rm -rf "$d/state" "$d/config"
  sudo cp -a "$RES/pristine/state" "$d/state"
  sudo cp -a "$RES/pristine/config" "$d/config"
  sudo docker rm -f strata-fork >/dev/null 2>&1
  t0=$(date +%s)
  # STRATA_ARENA_RAM_WAIT_S=120: 前エンジン退出待ちで 20 秒で蹴られた W2 の失敗を避ける
  # （起動待ちのみに影響し計測値には無関係。全アーム共通）
  sudo docker run -d --name strata-fork --gpus all --cpuset-cpus 36-47 \
    --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$PORT:8080 \
    -e MODEL=IQ3_XXS -e FAMILY=qwen -e CONTEXT=512k -e PARALLEL=1 -e VISION=off -e TUNE=skip \
    -e API_KEY="$K" -e STRATA_SET="$sset" -e STRATA_ARENA_RAM_WAIT_S=120 "$@" \
    -v "$VOL":/data -v "$d/state":/data/state -v "$d/config":/data/config \
    --restart no "$TAG" > "$d/docker-run.log" 2>&1
  s=""
  for i in $(seq 1 120); do
    s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata-fork 2>/dev/null)
    [ "$s" = healthy ] && break
    sudo docker inspect -f "{{.State.Status}}" strata-fork 2>/dev/null | grep -q exited && break
    sleep 5
  done
  t1=$(date +%s)
  sudo docker logs strata-fork > "$d/docker.log" 2>&1
  PID=$(pgrep -f "strata --serve" | head -1)
  if [ -n "${PID:-}" ]; then
    sudo cat /proc/$PID/cmdline | tr "\0" " " > "$d/engine-cmdline.txt"; echo >> "$d/engine-cmdline.txt"
    sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$d/engine-environ.txt"
  fi
  nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,clocks.sm --format=csv > "$d/gpu-start.txt"
  say "arm $label health=$s after $((t1-t0))s"
  if [ "$s" != healthy ]; then
    echo "arm $label NOT healthy" >&2
    tail -30 "$d/docker.log" >&2
    return 1
  fi
  return 0
}

keylines() {
  label=$1
  grep -nE "prompt chunk|expert cache|prompt path borrows|FP16|W16|w16|SIMT|qsa select|QSA layers|does not fit|out of memory|not bit-exact|K-quant|dense MMQ" \
    "$RES/$label/state/strata/launch/docker.engine.log" 2>/dev/null | tail -14 > "$RES/$label-keylines.txt"
  cat "$RES/$label-keylines.txt"
}

bench() {
  label=$1; shift
  STRATA_URL="http://127.0.0.1:$PORT" BENCH_SEED=42 STRATA_API_KEY="$K" \
    python3 "$SC/prefillBench.py" "$@" | tee -a "$RES/$label-bench.jsonl"
}

answers() {
  label=$1; shift
  STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" \
    python3 "$SC/answers4.py" "$@" | tee -a "$RES/$label-answers.jsonl"
}

klreq() {
  label=$1; tokens=$2
  STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" \
    python3 "$SC/klreq.py" "$tokens" "$RES/$label-kl.json" | tee -a "$RES/block-$phase.out"
}

collect() {
  label=$1
  keylines "$label"
  # logits dump を回収して volume を掃除する
  if sudo test -f "$VOL/pfw-logits-$label.bin"; then
    sudo cp "$VOL/pfw-logits-$label.bin" "$RES/$label-logits.bin"
    sudo rm -f "$VOL/pfw-logits-$label.bin"
    say "$label logits saved: $(sudo du -sh $RES/$label-logits.bin | cut -f1)"
  fi
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader > "$RES/$label-gpu-end.txt"
}

fail() {
  say "PHASE_FAILED $phase"
  restore_prod
  trap - EXIT
  exit 3
}

# フルベンチ: 24K prose/code x3 + 128K prose/code x$1
bench_full() {
  label=$1; n128=$2
  bench "$label" 24000 3 1 prose
  bench "$label" 24000 3 1 code
  bench "$label" 128000 "$n128" 1 prose
  bench "$label" 128000 "$n128" 1 code
}

case "$phase" in
w16a)
  # A1: ベース（KL/bitwise 比較の基準側。klreq を最初のリクエストにして rec 0 を揃える）
  stop_prod
  start_arm A1 "$BASE_SET" -e STRATA_LOGITS_DUMP=/data/pfw-logits-A1.bin || fail
  klreq A1 24000
  bench_full A1 3
  answers A1 24000 1,2,3,4
  collect A1
  say "A1 done"
  ;;
w16b)
  # B: STRATA_PF_W16_CACHE_MIB=auto（bitwise の見込みなので logits/回答一致を確認）
  stop_prod
  start_arm B1 "$BASE_SET" -e STRATA_PF_W16_CACHE_MIB=auto -e STRATA_LOGITS_DUMP=/data/pfw-logits-B1.bin || fail
  klreq B1 24000
  bench_full B1 3
  answers B1 24000 1,2,3,4
  collect B1
  say "B1 done"
  ;;
w16a2)
  # A2: A-B-A の 3 本目（24K 全種 + 128K prose のみで時間を削る）
  stop_prod
  start_arm A2 "$BASE_SET" -e STRATA_LOGITS_DUMP=/data/pfw-logits-A2.bin || fail
  # A1 との rec 0 比較（klreq を最初のリクエストに）でクロス再起動のノイズフロアを測る
  klreq A2 24000
  bench A2 24000 3 1 prose
  bench A2 24000 3 1 code
  bench A2 128000 2 1 prose
  answers A2 24000 1,2,3,4
  collect A2
  say "A2 done"
  ;;
chunk4096)
  # chunk 4096（bitwise の見込み）。固定 4 プロンプトの回答一致を確認
  stop_prod
  start_arm D4096 "${BASE_SET/prefill=auto/prefill=4096}" -e STRATA_LOGITS_DUMP=/data/pfw-logits-D4096.bin || fail
  klreq D4096 24000
  bench_full D4096 2
  answers D4096 24000 1,2,3,4
  collect D4096
  say "D4096 done"
  ;;
chunk16384)
  # chunk 16384（8192 超は not bit-exact の可能性）。回答一致を確認
  stop_prod
  start_arm D16384 "${BASE_SET/prefill=auto/prefill=16384}" -e STRATA_LOGITS_DUMP=/data/pfw-logits-D16384.bin || fail
  klreq D16384 24000
  bench_full D16384 2
  answers D16384 24000 1,2,3,4
  collect D16384
  say "D16384 done"
  ;;
simt)
  # STRATA_SELECT_SIMT=1（非 bitwise）。KL 用に klreq を最初に回す
  stop_prod
  start_arm S1 "$BASE_SET" -e STRATA_SELECT_SIMT=1 -e STRATA_LOGITS_DUMP=/data/pfw-logits-S1.bin || fail
  klreq S1 24000
  bench S1 128000 3 1 prose
  bench S1 128000 3 1 code
  collect S1
  say "S1 done"
  ;;
simt480)
  # SIMT の品質（needle 32K/128K）と 480K を 1 回
  stop_prod
  start_arm S2 "$BASE_SET" -e STRATA_SELECT_SIMT=1 || fail
  STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" \
    python3 "$SC/needle.py" 32000 50 PFW-7734-KM 96 "$RES/S2-needle.jsonl" | tee -a "$RES/block-$phase.out"
  STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" \
    python3 "$SC/needle.py" 128000 50 PFW-7734-KM 96 "$RES/S2-needle.jsonl" | tee -a "$RES/block-$phase.out"
  bench S2 480000 1 1 prose
  collect S2
  say "S2 done"
  ;;
*)
  echo "unknown phase $phase" >&2
  rm -rf /tmp/strata-gpu.lock
  trap - EXIT
  exit 1
  ;;
esac

restore_prod
trap - EXIT
say "PHASE_DONE $phase"

