#!/bin/bash
# GPU ロックを取って本番エンジンを止め、フォークイメージを本番相当の設定で起動し、
# 回答確認 -> bench.py -> 本番復元までを1ブロックで行う（60 分以内）
# 使い方: scripts/run-fork.sh <label> <image> [sizes] [repeat] [max_tokens]
#
# モデルデータは本番と同じボリュームを rw でマウントする。Terminal は起動時に /data/Strata-data へ
# 書けることを要求するため（ro だと "cannot write to /data/Strata-data" で即終了）、ro にはできない。
# 代わりに state（preset・ログ・キー）だけをコピーして /data/state に重ね、本番の状態を触らない
set -u
label=$1; tag=$2
sizes=${3:-4000,24000}; rep=${4:-1}; maxt=${5:-256}
res=~/work/results/build/run-$label
vol=/var/lib/docker/volumes/compose_strata-data/_data
port=${PORT:-18080}
mkdir -p "$res"

# GPU ロック（取れなければ 3 秒間隔で待つ。30 分で諦める）
t=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $t -ge 600 ] && { echo "GPU lock busy" >&2; exit 2; }
  sleep 3; t=$((t+1))
done
echo "run $label $(date -Is)" > /tmp/strata-gpu.lock/owner

restore_prod() {
  cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
  # 本番コンテナが止まっている場合もあるので、preset 削除は動いているときだけ行う
  sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
  cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1
  s=""
  for i in $(seq 1 90); do s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null); [ "$s" = healthy ] && break; sleep 10; done
  echo "prod health: $s image: $(sudo docker inspect -f '{{.Config.Image}}' strata)"
  PID=$(pgrep -f "strata --serve" | head -1)
  sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$res/prod-environ.txt"
  cat "$res/prod-environ.txt"
  sudo docker ps --format '{{.Names}} {{.Status}}'
  if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null; then
    rm -rf /tmp/strata-gpu.lock && echo "gpu lock released $(date -Is)" && return 0
  fi
  echo "gpu lock NOT released (restore failed)"; return 1
}

K=$(grep ^API_KEY= ~/opt-work/prod.env | cut -d= -f2)
# 本番と同じドラフタ・窓にするための上書き（既定はフォーク Terminal が選ぶ dflash2 + spec 6）
SSET=${SSET:-"ple-io=ram prefill=auto STRATA_DENSE_F32=1"}

# 本番のエンジンを止めて VRAM と hugepage を空ける。コンテナごと止めるのは、Terminal が
# エンジンを勝手に起こし直して 40 GiB の arena を二重に確保し OOM になるのを避けるため
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

# Terminal の状態は prod のコピーを使い、本番の preset には触らない
# 所有者は root のままにする: runtime ディレクトリは「自分だけが読める (0700)」ことを要求し、
# コンテナ内は root で動くため、manager 所有にすると起動が止まる
sudo rm -rf "$res/state"
sudo cp -a "$vol/state" "$res/state"

sudo docker rm -f strata-fork >/dev/null 2>&1
t0=$(date +%s)
sudo docker run -d --name strata-fork --gpus all --cpuset-cpus 36-47 \
  --ulimit memlock=-1:-1 --shm-size 2g -p 127.0.0.1:$port:8080 \
  -e MODEL=IQ3_XXS -e FAMILY=qwen -e CONTEXT=128k -e PARALLEL=1 -e VISION=off -e TUNE=skip \
  -e API_KEY="$K" -e STRATA_SET="$SSET" \
  -v "$vol":/data -v "$res/state":/data/state \
  --restart no "$tag" > "$res/docker-run.log" 2>&1
[ $? -eq 0 ] || { cat "$res/docker-run.log"; sudo docker rm -f strata-fork >/dev/null 2>&1; restore_prod; exit 3; }

s=""
for i in $(seq 1 90); do
  s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata-fork 2>/dev/null)
  [ "$s" = healthy ] && break
  sudo docker inspect -f "{{.State.Status}}" strata-fork 2>/dev/null | grep -q exited && break
  sleep 5
done
t1=$(date +%s)
echo "fork health=$s after $((t1-t0))s"
sudo docker ps -a --filter name=strata-fork --format '{{.Status}}'
if [ "$s" != healthy ]; then
  sudo docker logs strata-fork > "$res/docker.log" 2>&1
  tail -40 "$res/docker.log"
  sudo docker rm -f strata-fork >/dev/null 2>&1
  restore_prod; exit 4
fi

# 起動ログ・メモリ・フラグの記録
sudo docker logs strata-fork > "$res/docker.log" 2>&1
PID=$(pgrep -f "strata --serve" | head -1)
sudo cat /proc/$PID/cmdline | tr "\0" " " > "$res/engine-cmdline.txt"; echo >> "$res/engine-cmdline.txt"
sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$res/engine-environ.txt"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,clocks.sm --format=csv > "$res/gpu.txt"
free -m > "$res/free.txt"
grep -E "engine started|VRAM|RAM|expert cache|draft layer|KV streaming|PLE table|loaded|MiB|GiB|memory guard" "$res/state/strata/launch/docker.engine.log" \
  | tail -40 > "$res/startup.log"

# 回答確認（英問。temperature 0）
curl -s "http://127.0.0.1:$port/v1/chat/completions" -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"What is the capital of France? Then compute 17*23."}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" > "$res/answer.txt" 2>&1
cat "$res/answer.txt"

# ベンチ（ウォームアップ込みでよいので repeat は既定 1）
ts=$(date +%m%d-%H%M%S)
STRATA_URL="http://127.0.0.1:$port" BENCH_SEED=42 STRATA_API_KEY=$K python3 ~/bench.py "$sizes" "$rep" "$maxt" > "$res/bench-$ts.jsonl"
cat "$res/bench-$ts.jsonl"
STRATA_URL="http://127.0.0.1:$port" STRATA_API_KEY=$K python3 ~/opt-work/src-build/answers.py "$res/answers-$ts.json" 2>/dev/null || true

# エンジンログの計測行
awk '/engine started/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$res/state/strata/launch/docker.engine.log" \
  | grep -E "engine started|prompt .* tokens|decode expert cache|route tail|KV streaming" > "$res/englog-$ts.txt"

sudo docker rm -f strata-fork >/dev/null 2>&1
sleep 5
restore_prod || exit 5
echo RUN_ALL_OK
