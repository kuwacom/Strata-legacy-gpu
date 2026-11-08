#!/bin/bash
# run-decode-ab3.sh を切り離して起動する（待機中の kill/再起動を安全にするため）
cd ~/work/wt/dknobs || exit 1
exec setsid nohup env DEADLINE_S=${DEADLINE_S:-3300} PROD_AB=${PROD_AB:-1} LONG_KEYS="${LONG_KEYS:-}" LONG_TOKENS="${LONG_TOKENS:-128000}" \
  bash scripts/run-decode-ab3.sh "${DEADLINE_S:-3300}" "$@" </dev/null >> ~/work/results/dknobs/v3/driver.log 2>&1
