#!/bin/bash
# 待機中の run-decode-ab3.sh を止めて起動し直す（自分の bash -c を巻き込まないよう cmdline 先頭で一致させる）
for p in $(pgrep -f "^bash scripts/run-decode-ab3\.sh"); do kill "$p" 2>/dev/null; done
sleep 2
cd ~/work/wt/dknobs || exit 1
setsid nohup env DEADLINE_S=${DEADLINE_S:-3300} PROD_AB=${PROD_AB:-1} LONG_KEYS="${LONG_KEYS:-}" LONG_TOKENS="${LONG_TOKENS:-128000}" \
  bash scripts/run-decode-ab3.sh "${DEADLINE_S:-3300}" ${ARMS:-base:3} </dev/null >> ~/work/results/dknobs/v3/driver.log 2>&1 &
sleep 2
pgrep -af "^bash scripts/run-decode-ab3" | head -2
