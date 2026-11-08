#!/bin/bash
# 本番を ~/opt-work/prod.env から復元し、healthy・512K・preset 汚染なし・残留なしを確認して GPU ロックを解放する
set -u
res=~/work/results/pgdn
mkdir -p "$res"
{
cp ~/opt-work/prod.env ~/Strata/docker/compose/.env
sudo docker exec strata sh -c "strata launch --stop --preset docker >/dev/null 2>&1; strata launch --delete-preset docker" >/dev/null 2>&1
cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate >/dev/null 2>&1
s=""
for i in $(seq 1 90); do s=$(sudo docker inspect -f "{{.State.Health.Status}}" strata 2>/dev/null); [ "$s" = healthy ] && break; sleep 10; done
echo "prod health: $s image: $(sudo docker inspect -f '{{.Config.Image}}' strata)"
PID=$(pgrep -f "strata --serve" | head -1)
sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "^STRATA_" | grep -v KEY
sudo cat /proc/$PID/cmdline | tr "\0" " "; echo
sudo docker ps --format '{{.Names}} {{.Status}}'
# cmdline は NUL 区切り: 文字列にしてから照合する
if sudo cat /proc/$PID/cmdline | tr "\0" " " | grep -q "max-context 524288"; then echo "max-context 524288 OK"; else echo "max-context NOT 524288"; fi
diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null && echo ".env matches prod.env"
echo "--- preset ---"
sudo docker exec strata cat /data/config/strata/launch-presets/docker.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('context:', d.get('context'), 'extra:', d.get('extra'))" || echo "no preset"
echo "--- leftovers ---"
sudo docker ps -a --format '{{.Names}} {{.Status}}' | grep -E 'strata-pgdn|strata-fork' || echo "no arm containers"
} 2>&1 | tee "$res/restore.txt"
if grep -q "prod health: healthy" "$res/restore.txt" && grep -q ".env matches prod.env" "$res/restore.txt" \
   && grep -q "max-context 524288 OK" "$res/restore.txt"; then
  rm -rf /tmp/strata-gpu.lock && echo "gpu lock released $(date -Is)"
else
  echo "RESTORE FAILED - lock kept"; exit 1
fi
