#!/bin/bash
# dknobs decode A-B-A (v3): 本番 strata:fork-v3 (engine 0.1.57, 512K, dflash2 --spec 4 --draft-conf 0.7,
#   --kv-resident 32768) を A とし、起動フラグ候補を B として同一ブロックで交互に測る。
#   A の 2 回目は本番復元後の prod コンテナに対して回す（帯域とロード時間の節約、復元の検証も兼ねる）。
# 起動は entrypoint の settings 経路ではなく `strata launch` を直接叩く（--drafting を明示するため）。
# モデルは IQ3_XXS を明示し、既定モデル (IQ3_S) の setup/download を起こさない。
#
# 使い方: scripts/run-decode-ab3.sh <DEADLINE_S> <arm:reps ...>
#   env: PROD_AB=1     復元後に prod へ同じ bench を回して A の 2 回目にする
#        LONG_KEYS     long プロンプトも測るアーム名（空白区切り）
#        LONG_TOKENS   128000 など（カンマ区切り）
#        LONG_480_MIN_LEFT 480000 を始めるのに必要な残り秒（既定 2100）
set -u
RES=~/work/results/dknobs/v3
CORPUS=~/work/results/dknobs/corpus/dedup.md
VOL=/var/lib/docker/volumes/compose_strata-data/_data
IMG=strata:fork-v3
PORT=${PORT:-18080}
DEADLINE_S=${1:-3300}
shift
ARMS=${*:-"base:3"}
K=$(grep ^API_KEY= ~/opt-work/prod.env | cut -d= -f2)
PROD_AB=${PROD_AB:-0}
LONG_KEYS=${LONG_KEYS:-}
LONG_TOKENS=${LONG_TOKENS:-128000}
LONG_480_MIN_LEFT=${LONG_480_MIN_LEFT:-2100}
mkdir -p "$RES"
exec > >(tee -a "$RES/run.log") 2>&1

echo "=== dknobs decode A-B-A v3 start $(date -Is) deadline ${DEADLINE_S}s arms: $ARMS prod_ab=$PROD_AB long=$LONG_KEYS:$LONG_TOKENS"

# GPU ロック（他人が持っている場合は待つ。自分の owner のときだけ再利用する）
me="dknobs-decode-v3"
if [ -d /tmp/strata-gpu.lock ] && grep -q "^$me " /tmp/strata-gpu.lock/owner 2>/dev/null; then
  echo "lock already mine: $(cat /tmp/strata-gpu.lock/owner)"
else
  t=0
  until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
    [ $t -ge 2000 ] && { echo "GPU lock busy after 100 min, exit"; exit 2; }
    sleep 3; t=$((t+1))
  done
  echo "$me $(date -Is)" > /tmp/strata-gpu.lock/owner
  echo "lock acquired $(date -Is)"
fi
own=$(cat /tmp/strata-gpu.lock/owner 2>/dev/null)
case "$own" in "$me "*) ;; *) echo "lock owner is not me ($own) - exit"; exit 2;; esac
T0=$(date +%s)
left() { echo $(( DEADLINE_S - ( $(date +%s) - T0 ) )); }
now()  { date -Is; }

restore_prod() {
  cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
  sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
  cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1
  s=""
  for i in $(seq 1 90); do s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null); [ "$s" = healthy ] && break; sleep 10; done
  echo "prod health: $s image: $(sudo docker inspect -f '{{.Config.Image}}' strata)"
  PID=$(pgrep -f "strata --serve" | head -1)
  sudo cat /proc/$PID/cmdline | tr "\0" " " > "$RES/prod-cmdline.txt"; echo >> "$RES/prod-cmdline.txt"
  sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$RES/prod-environ.txt"
  cat "$RES/prod-environ.txt" | grep -v -i key
  sudo docker ps --format '{{.Names}} {{.Status}}'
  if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
    return 0
  fi
  return 1
}

stop_prod() {
  sudo docker exec strata sh -c "strata launch --stop --preset docker" >/dev/null 2>&1
  sudo docker stop -t 120 strata >/dev/null 2>&1
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    alive=$(pgrep -f "strata --serve" | wc -l)
    [ "$used" -lt 3000 ] && [ "$avail" -ge 50 ] && [ "$alive" -eq 0 ] && break
    sleep 3
  done
  echo "prod stopped: VRAM ${used} MiB, MemAvailable ${avail} GiB, engines ${alive} $(now)"
}

# アーム = 本番 v3 の構成 + 差分。STRATA_SET の本番値は base と同一にそろえる
arm_settings() {
  case "$1" in
    base)  echo "--set spec=4 --set draft-conf=0.7 --set kv-resident=32768" ;;
    dc05)  echo "--set spec=4 --set draft-conf=0.5 --set kv-resident=32768" ;;
    dc08)  echo "--set spec=4 --set draft-conf=0.8 --set kv-resident=32768" ;;
    kv48)  echo "--set spec=4 --set draft-conf=0.7 --set kv-resident=49152" ;;
    kv64)  echo "--set spec=4 --set draft-conf=0.7 --set kv-resident=65536" ;;
    spec3) echo "--set spec=3 --set draft-conf=0.7 --set kv-resident=32768" ;;
    spec5) echo "--set spec=5 --set draft-conf=0.7 --set kv-resident=32768" ;;
    *) echo "" ;;
  esac
}
expect_flags() {
  case "$1" in
    base)  echo "--draft-conf 0.7|--spec 4|--kv-resident 32768" ;;
    dc05)  echo "--draft-conf 0.5" ;;
    dc08)  echo "--draft-conf 0.8" ;;
    kv48)  echo "--kv-resident 49152" ;;
    kv64)  echo "--kv-resident 65536" ;;
    spec3) echo "--spec 3" ;;
    spec5) echo "--spec 5" ;;
    *) echo "" ;;
  esac
}

start_arm() {
  arm=$1
  echo "gpu before $arm: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)"
  sudo rm -rf "$RES/state-$arm"
  sudo cp -a "$VOL/state" "$RES/state-$arm"
  sudo docker rm -f strata-fork >/dev/null 2>&1
  sets="$(arm_settings "$arm") --set ple-io=ram --set prefill=auto --set rope-scaling=yarn --set rope-scale=2"
  sets="$sets --set STRATA_DENSE_F32=1 --set STRATA_DENSE_MMQ=1 --set STRATA_MMQ_KQ=1 --set STRATA_DECODE_TIMING=1"
  printf "STRATA_LAUNCH_KEY=%s\n" "$K" > "$RES/env-launch"; chmod 600 "$RES/env-launch"
  echo "sets: $sets"
  sudo docker run -d --name strata-fork --gpus all --cpuset-cpus 36-47 \
    --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$PORT:8080 \
    --env-file "$RES/env-launch" \
    -v "$VOL":/data -v "$RES/state-$arm":/data/state \
    --restart no "$IMG" \
    strata launch --no-tui --access lan --port 8080 --data-dir /data/Strata-data \
      --model IQ3_XXS --family qwen --context 512k --parallel 1 --vision off --kv int8 --tune skip \
      --drafting auto --save-preset "$arm" --api-key-env STRATA_LAUNCH_KEY $sets \
    > "$RES/docker-run-$arm.log" 2>&1
  [ $? -eq 0 ] || { cat "$RES/docker-run-$arm.log"; return 1; }
  # download guard: 新しい .part が出たら即中止（モデルの再 DL を避ける）
  sleep 45
  newpart=$(sudo find "$VOL/Strata-data/models" -name "*.part" -newermt "-3 minutes" 2>/dev/null | head -3)
  if [ -n "$newpart" ]; then
    echo "ABORT: a model download started: $newpart"
    sudo docker rm -f strata-fork >/dev/null 2>&1
    return 2
  fi
  s=""
  for i in $(seq 1 180); do
    s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata-fork 2>/dev/null)
    [ "$s" = healthy ] && break
    sudo docker inspect -f "{{.State.Status}}" strata-fork 2>/dev/null | grep -q exited && break
    sleep 5
  done
  echo "arm $arm health=$s $(now) load_s=$(( $(date +%s) - T0 ))"
  sudo docker ps -a --filter name=strata-fork --format '{{.Status}}'
  if [ "$s" != healthy ]; then
    sudo docker logs strata-fork > "$RES/docker-$arm.log" 2>&1
    tail -30 "$RES/docker-$arm.log"
    return 1
  fi
  PID=$(pgrep -f "strata --serve" | head -1)
  sudo cat /proc/$PID/cmdline | tr "\0" " " > "$RES/cmdline-$arm.txt"; echo >> "$RES/cmdline-$arm.txt"
  sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$RES/env-$arm.txt"
  # 起動フラグが意図どおり効いたか（--set が無視されると A と同じ構成を測ってしまう）
  miss=""
  IFS='|' read -ra want <<< "$(expect_flags "$arm")"
  for w in "${want[@]}"; do
    grep -q -- "$w" "$RES/cmdline-$arm.txt" || miss="$miss [$w]"
  done
  if [ -n "$miss" ]; then echo "WARN $arm: flags not in cmdline:$miss"; else echo "flags ok $arm"; fi
  sudo docker logs strata-fork > "$RES/docker-$arm.log" 2>&1
  grep -aE "expert cache [0-9]+ slots|window up to|DFlash2 drafter on|KV streaming:|pcie_frac|lowest|free with everything" \
    "$RES/state-$arm/strata/launch/$arm.engine.log" 2>/dev/null | tail -12 > "$RES/startup-$arm.txt"
  cat "$RES/startup-$arm.txt"
  return 0
}

stop_arm() {
  sudo docker rm -f strata-fork >/dev/null 2>&1
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 3000 ] && break
    sleep 2
  done
  echo "arm stopped: VRAM ${used} MiB $(now)"
}

run_short() {
  arm=$1; reps=$2; maxt=${3:-256}
  STRATA_URL="http://127.0.0.1:$PORT" BENCH_SEED=42 BENCH_LABEL="$arm" \
    CORPUS_FILE="$CORPUS" STRATA_API_KEY="$K" \
    python3 ~/work/wt/dknobs/scripts/bench-decode.py 4000,24000 "$reps" "$maxt" \
    > "$RES/bench-$arm.jsonl"
  cat "$RES/bench-$arm.jsonl"
  echo "gpu after $arm: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)"
}

run_long() {
  arm=$1; tok=$2
  local lbl="$arm-$((tok/1000))k"
    CORPUS_FILE="$CORPUS" STRATA_API_KEY="$K" PROMPT_TOKEN_LIMIT=515000 \
    python3 ~/work/wt/dknobs/scripts/bench-long.py "$tok" prose 128 2>&1 | tee "$RES/long-$lbl.txt"
}

collect_timing() {
  arm=$1
  log="$RES/state-$arm/strata/launch/$arm.engine.log"
  grep -a "strata decode timing:" "$log" > "$RES/timing-$arm.txt" 2>/dev/null
  grep -aE "decode expert cache hit rate|KV streaming:|drafts accepted|window" "$log" > "$RES/hits-$arm.txt" 2>/dev/null
  echo "--- timing $arm ($(wc -l < "$RES/timing-$arm.txt") lines) ---"; tail -6 "$RES/timing-$arm.txt"
  echo "--- hits $arm ---"; tail -8 "$RES/hits-$arm.txt"
}

stop_prod

for spec in $ARMS; do
  arm=${spec%%:*}; reps=${spec##*:}
  [ "$reps" = "$arm" ] && reps=3
  if [ $(left) -lt 700 ]; then echo "skip $arm: left $(left)s"; continue; fi
  start_arm "$arm"
  rc=$?
  if [ "$rc" = 0 ]; then
    STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" python3 ~/work/wt/dknobs/scripts/answers.py "$RES/answers-$arm.json" 2>&1 | tail -2
    STRATA_URL="http://127.0.0.1:$PORT" STRATA_API_KEY="$K" python3 ~/work/wt/dknobs/scripts/klprobe.py "$RES/kl-$arm.json" 2>&1 | tail -2
    [ "$reps" -gt 0 ] && run_short "$arm" "$reps" 256
    case " $LONG_KEYS " in *" $arm "*)
      for tok in ${LONG_TOKENS//,/ }; do
        need=700; [ "$tok" -ge 400000 ] && need=$LONG_480_MIN_LEFT
        if [ $(left) -lt "$need" ]; then echo "skip long $tok for $arm: left $(left)s"; continue; fi
        run_long "$arm" "$tok"
      done ;;
    esac
    collect_timing "$arm"
  fi
  stop_arm
  [ "$rc" = 2 ] && { echo "download guard tripped: aborting arms"; break; }
done

echo "=== measured, restoring prod $(now) left $(left)s"
restore_prod || { echo "RESTORE FAILED - lock kept"; exit 5; }
echo "prod restored $(now) left $(left)s"

if [ "$PROD_AB" = 1 ] && [ $(left) -gt 500 ]; then
  echo "=== prod A (2nd) $(now)"
  STRATA_URL="http://127.0.0.1:8080" STRATA_API_KEY="$K" python3 ~/work/wt/dknobs/scripts/answers.py "$RES/answers-prod.json" 2>&1 | tail -2
  STRATA_URL="http://127.0.0.1:8080" STRATA_API_KEY="$K" python3 ~/work/wt/dknobs/scripts/klprobe.py "$RES/kl-prod.json" 2>&1 | tail -2
  STRATA_URL="http://127.0.0.1:8080" BENCH_SEED=42 BENCH_LABEL=prod CORPUS_FILE="$CORPUS" STRATA_API_KEY="$K" \
    python3 ~/work/wt/dknobs/scripts/bench-decode.py 4000,24000 3 256 > "$RES/bench-prod.jsonl"
  cat "$RES/bench-prod.jsonl"
  sudo docker exec strata sh -c "grep -a 'strata decode timing:' /data/state/strata/launch/docker.engine.log" > "$RES/timing-prod.txt" 2>/dev/null
  sudo docker exec strata sh -c "grep -aE 'decode expert cache hit rate|KV streaming:|drafts accepted' /data/state/strata/launch/docker.engine.log" > "$RES/hits-prod.txt" 2>/dev/null
  tail -4 "$RES/timing-prod.txt"; tail -4 "$RES/hits-prod.txt"
  for tok in ${LONG_TOKENS//,/ }; do
    need=700; [ "$tok" -ge 400000 ] && need=$LONG_480_MIN_LEFT
    if [ $(left) -lt "$need" ]; then echo "skip long $tok for prod: left $(left)s"; continue; fi
    STRATA_URL="http://127.0.0.1:8080" BENCH_SEED=42 BENCH_LABEL="prod-$((tok/1000))k" \
      CORPUS_FILE="$CORPUS" STRATA_API_KEY="$K" PROMPT_TOKEN_LIMIT=515000 \
      python3 ~/work/wt/dknobs/scripts/bench-long.py "$tok" prose 128 2>&1 | tee "$RES/long-prod-$((tok/1000))k.txt"
  done
fi

# 最終確認: healthy / イメージ / preset 汚染なし
echo "=== final check $(now)"
sudo docker inspect -f '{{.State.Health.Status}} {{.Config.Image}}' strata
sudo docker exec strata sh -c 'python3 - <<EOF
import json
d=json.load(open("/data/state/strata/launch/docker.json"))
print("image_preset_env", d.get("env"))
print("launch_set", d.get("launch_set"))
print("args", " ".join(d.get("args", [])))
EOF'
rm -rf /tmp/strata-gpu.lock && echo "gpu lock released $(date -Is)"
echo "DKNODES_DECODE_V3_OK $(now)"
