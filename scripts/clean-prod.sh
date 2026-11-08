#!/bin/bash
# 本番 state の汚染（docker.choices.json の extra と docker.json の env）を除去して本番を 512K で作り直す
# GPU ロック -> backup -> 除去 -> preset 削除 + compose up -> healthy/残留確認 -> ロック解放
set -u
res=~/work/results/lookahead
vol=/var/lib/docker/volumes/compose_strata-data/_data
mkdir -p "$res"

t=0
until mkdir /tmp/strata-gpu.lock 2>/dev/null; do
  [ $t -ge 600 ] && { echo "GPU lock busy" >&2; exit 2; }
  sleep 5; t=$((t+1))
done
echo "lookahead-clean $(date -Is)" > /tmp/strata-gpu.lock/owner

ts=$(date +%m%d-%H%M%S)
sudo cp -a "$vol/state/strata/launch/docker.choices.json" "$res/docker.choices.json.bak-$ts"
sudo cp -a "$vol/state/strata/launch/docker.json" "$res/docker.json.bak-$ts"
echo "backups: $res/docker.choices.json.bak-$ts $res/docker.json.bak-$ts"

sudo python3 - "$vol/state/strata/launch/docker.choices.json" "$vol/state/strata/launch/docker.json" <<'EOF'
import json, sys
BAD = ["--expert-cache", "STRATA_DECODE_TIMING", "STRATA_LOOKAHEAD_STATS", "STRATA_POOL_PREFETCH",
       "STRATA_VERIFY_PROFILE"]
ch = json.load(open(sys.argv[1]))
for k in BAD:
    ch.get("extra", {}).pop(k, None)
json.dump(ch, open(sys.argv[1], "w"), indent=1)
print("choices extra now:", ch.get("extra"))
dj = json.load(open(sys.argv[2]))
for k in BAD:
    dj.get("env", {}).pop(k, None)
# 実効 args の expert-cache を auto に戻す（prod.env の既定）
a = dj.get("args", [])
for i, v in enumerate(a):
    if v == "--expert-cache" and i + 1 < len(a):
        a[i + 1] = "auto"
json.dump(dj, open(sys.argv[2], "w"), indent=1)
print("docker.json env now:", dj.get("env"))
EOF

cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1
s=""
for i in $(seq 1 90); do s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null); [ "$s" = healthy ] && break; sleep 10; done
echo "prod health: $s image: $(sudo docker inspect -f '{{.Config.Image}}' strata)"
PID=$(pgrep -f "strata --serve" | head -1)
sudo cat /proc/$PID/cmdline | tr "\0" " " > "$res/prod-cmdline.txt"; echo >> "$res/prod-cmdline.txt"
sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" > "$res/prod-environ.txt"
cat "$res/prod-environ.txt"
grep -o "expert-cache [a-z0-9]*" "$res/prod-cmdline.txt" || true
sudo docker ps --format '{{.Names}} {{.Status}}'
sudo grep -E "expert cache auto|expert cache [0-9]+ slots|KV streaming: 32768 of 524288|decode expert cache hit rate" \
  "$vol/state/strata/launch/docker.engine.log" | tail -4
if [ "$s" = healthy ] && diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null && ! grep -qE "STRATA_VERIFY_PROFILE|STRATA_POOL_PREFETCH|STRATA_DECODE_TIMING|STRATA_LOOKAHEAD_STATS" "$res/prod-environ.txt"; then
  rm -rf /tmp/strata-gpu.lock && echo "CLEAN_OK gpu lock released $(date -Is)"
else
  echo "CLEAN_FAILED - lock kept"
  exit 1
fi
