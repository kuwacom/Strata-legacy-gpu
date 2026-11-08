#!/bin/bash
# 本番を ~/opt-work/prod.env から復元し、healthy・512K・残留なし・preset 汚染なしを確認して GPU ロックを解放する
set -u
res=~/work/results/integrated
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
grep -q "max-context 524288" /proc/$PID/cmdline 2>/dev/null || true
diff -q ~/Strata/docker/compose/.env ~/opt-work/prod.env >/dev/null && echo ".env matches prod.env"
echo "--- preset (shared) ---"
for p in /data/state/strata/launch/docker.json /data/config/strata/launch-presets/docker.json; do
  sudo docker exec strata sh -c "cat $p" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('$p args:', ' '.join(d.get('args',[]))); print('$p env:', d.get('env',{}))" 2>/dev/null || true
done
echo "--- leftovers (mine) ---"
sudo docker ps -a --format '{{.Names}} {{.Status}}' | grep -E 'strata-integrated' || echo "no integrated arm containers"
echo "--- others ---"
sudo docker ps --format '{{.Names}} {{.Status}}'
} 2>&1 | tee "$res/restore-$(date +%H%M%S).txt"
last=$(ls -t "$res"/restore-*.txt | head -1)
if grep -q "prod health: healthy" "$last" && grep -q ".env matches prod.env" "$last" \
   && grep -q "max-context 524288" "$last" && grep -q "no integrated arm containers" "$last"; then
  rm -rf /tmp/strata-gpu.lock && echo "gpu lock released $(date -Is)"
else
  echo "RESTORE FAILED - lock kept"; exit 1
fi
