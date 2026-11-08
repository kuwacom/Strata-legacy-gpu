# Qwen3.8-Flash-Next IQ3_XXS の 512K context 本番適用

## 結論

- 本番コンテナ `strata` を 512K 構成（`--max-context 524288`、int8 KV、`--kv-resident 32768`、`--rope-scaling yarn --rope-scale 2`）で起動し、healthy、短い質問（Paris / 391）に正答した
- `CONTEXT=512k` だけで setup が rope を yarn factor 2 に解決し、`STRATA_SET` の `rope-scaling=yarn rope-scale=2` も同じ値で上書きされる。engine の起動行にフラグが 1 回だけ入ることを確認した
- KV は int8 のまま 12,672 B/token、512K で 6.19 GiB の pinned RAM。VRAM 側は 32,768 cells のままで、expert cache は 10,315 slots / 16.73 GiB
- 100K は TTFT 221.4 秒・prefill 451.4 tok/s、300K は TTFT 630.7 秒・prefill 482.3 tok/s。OOM も #2065 警告も出ていない
- 500K は 60 分ロックに収まる見込みが立たないため未実施（理由は後述）

## 最終構成

`~/Strata/docker/compose/.env`（`~/opt-work/prod.env` と同一。API キーは非表示）

```
STRATA_IMAGE=ghcr.io/niko1221/strata:0.2.7-cuda12
MODEL=IQ3_XXS
CONTEXT=512k
VISION=off
STRATA_BIND=0.0.0.0
API_KEY=***
PARALLEL=1
TUNE=skip
STRATA_SET="ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2"
```

128K から変えたのは `CONTEXT` と `STRATA_SET` の rope 2 項目だけ。`PARALLEL=1`、`MODEL=IQ3_XXS`、image、`TUNE=skip`、`VISION=off`、`STRATA_BIND` は据え置き。

## rope を `STRATA_SET` で渡せるか

- `docker/entrypoint.sh:193` が `CONTEXT` を `--context 512k` にし、`docker/entrypoint.sh:225` が `STRATA_SET` の各 `KEY=VALUE` を `--set KEY=VALUE` に展開する
- `strata_cli/launch/settings.py` の `split_setting` が `rope-scaling` -> `--rope-scaling`、`rope-scale` -> `--rope-scale` に正規化し、`as_argv_changes` が engine 引数へ反映する
- さらに `setup.py` の `resolve_rope` は 262,144 超の context で method 未指定なら yarn、factor 未指定なら `ctx / 262144`（512K なら 2.0）を自動で入れる。`strata launch --context 512k` は setup を `--yes` で再実行するため、`STRATA_SET` に rope を書かなくても同じフラグが入る
- 実際の起動行（`~/work/results/ctx512prod/engine-512k.log:18683`）:

```
--max-context 524288 --kv int8 --kv-resident 32768 --ple-io ram --prefill auto --rope-scaling yarn --rope-scale 2
```

- `STRATA_SET` 経由の値は setup が書いた値を `apply_to_argv` が一度外して末尾に付け直すため、重複はしない（起動行に 1 回だけ現れる）
- ログの rope 行: `rope scaling yarn, factor 2 (freq_scale 0.5, base 1e+07, mscale 1.069315), --max-context 524288 against a trained context of 262144`

## 起動とメモリ

- コンテナ再作成 18:16:50 -> healthy 18:19:43（173 秒。setup 再実行とモデルロードを含む）。engine 開始は 18:17:30 なので engine 起動から healthy まで約 133 秒
- `KV streaming: 32768 of 524288 cells per QSA layer in VRAM, the K/V in 6.19 GiB of pinned RAM`
- `expert cache 10315 slots, 16.73 GiB of VRAM`
- `conversation cache sized from the free RAM: 7168 MiB`
- idle: VRAM 23,945 MiB 使用 / 544 MiB 空き、RAM 77 GiB 使用 / 20 GiB 利用可能
- 起動時に `the expert arena needs 39.97 GiB of RAM but 21.68 GiB is available` の警告は出るが、page cache を追い出して 39.97 GiB を pin でき、起動は完了する（128K 本番でも同じ警告）
- 起動時のその他の警告・エラーなし

## 速度（512K 構成）

`~/bench.py`、`BENCH_SEED=42`、temperature 0。4K/24K は各 1 回。

| プロンプト tokens | 種別 | prefill tok/s | decode tok/s | TTFT 秒 |
| ---: | --- | ---: | ---: | ---: |
| 3,176 | prose | 434.3 | 43.95 | 7.31 |
| 4,793 | code | 307.3 | 43.47 | 15.60 |
| 27,634 | prose | 484.3 | 42.83 | 57.06 |
| 30,247 | code | 491.7 | 51.46 | 61.52 |
| 99,964 | prose | 451.4 | 24.2（2 token のみ） | 221.43 |
| 304,175 | prose | 482.3 | 21.9（2 token のみ） | 630.72 |

- 4K code の 307 tok/s は起動後最初のリクエストで warmup を含む値。同じ 4K でも prose（2 本目）は 434 tok/s
- 100K / 300K の decode は max_tokens を 128 にしたが `ok` の 2 token で stop したため、decode 値は参考外
- 128K 本番の基準（prefill 450-480、decode 45-50）と比べ、24K/100K/300K の prefill は同等、decode は 43-51 tok/s で同等

## 長文プロンプトの VRAM / RAM

合成プロンプトは `~/Strata/docs/*.md` を繰り返して作った（前タスクと同じ方法）。VRAM/RAM は 5 秒間隔の `nvidia-smi` と `free -m` で記録。

| プロンプト tokens | VRAM 使用 最大 | VRAM 空き 最小 | RAM 使用 最大 | RAM 利用可能 最小 |
| ---: | ---: | ---: | ---: | ---: |
| 99,964 | 24,149 MiB | 291 MiB | 81,105 MiB | 19,464 MiB |
| 304,175 | 24,149 MiB | 291 MiB | 88,102 MiB | 12,467 MiB |

- 300K の 630 秒は engine の `reading the prompt` 行でも裏が取れる（最後 304,168/304,175 tokens, 629 秒）
- 終了時（ベンチ後）: VRAM 24,167 MiB 使用 / 273 MiB 空き、RAM 86.6 GiB 使用 / 13.9 GiB 利用可能
- **OOM なし**。`cudaMalloc` 失敗、stall watchdog、リクエストの error 終了はいずれも今回の実行区間には出ていない
- **#2065 の VRAM 警告なし**。前タスクは 327K 中に空き 273 MiB で警告が出たが、今回は nvidia-smi 上の空きが 291 MiB まで下がっても engine は警告を出さなかった。engine の reserve（既定 700 MiB）込みの判定が 512 MiB を上回っていた可能性がある（推定。ログに判定値は出ない）
- KV streaming のヒット率は今回の長文では記録できていない（engine ログの集計行がリクエスト単位で出る形式ではなく、今回の区間には該当行がない）。前タスクの 327K は 89.3%

## 500K を実施しなかった理由

- GPU ロックは 18:13:15 開始の 60 分（19:13:15 まで）。300K の計測が 18:38:30 に終わり、残り約 35 分
- 500K の prefill は、今回の 300K 実測 482 tok/s なら約 1,088 秒（18 分）、前タスクの 327K 実測 351 tok/s なら約 1,494 秒（25 分）。深い位置での KV streaming 低下を考えると後者に寄る可能性が高い
- 18 分で終わっても、その後の docs 作成・最終確認・ロック解放を含めると 60 分に余裕がなく、25 分かかると確実に超過する。途中で打ち切ると engine の状態が不定になり、本番を healthy で残す要件を満たせない恐れがある
- 以上から、指示の「60 分ロック内に収まらない場合は 300K までで打ち切り、理由を書く」に従い 500K は未実施。max-context 524288 の確保自体は起動ログで確認済み

## `--vram-reserve-mib 896` の検討

- 今回は OOM も #2065 警告も出なかったため**採用しなかった**。既定の 700 MiB reserve のまま
- 採用すると expert slots が約 120 減る見込み（前タスクの推定）。decode のヒット率を少し下げる可能性がある
- 273 MiB まで空きが減る状態が常態化しているため、将来 OOM や decode 低下が出たら `STRATA_SET` に `vram-reserve-mib=896` を足すのが第一候補

## 採用判断

- 512K を恒久設定として採用する。理由:
  - 起動・healthy・短答・100K・300K がすべて成功し、OOM も警告も出ていない
  - 24K/100K/300K の prefill は 128K 基準と同等、decode も 43-51 tok/s を維持
  - KV は int8 のままなので 128K との品質差を持ち込まない
  - 変更は `.env` の `CONTEXT` と rope 2 項目だけで、他は据え置き
- リスク: YaRN は experimental で長文の品質（recall・一貫性）は未評価。VRAM 余裕は活動後に 273 MiB と小さい

## 再現手順

```sh
# GPU ロック
mkdir /tmp/strata-gpu.lock && echo "ctx512prod $(date -Is)" > /tmp/strata-gpu.lock/owner

# 本番を止め、.env を 512K に変更（API キー行は触らない）
sudo docker stop strata
cd ~/Strata/docker/compose
sed -i 's/^CONTEXT=.*/CONTEXT=512k/' .env
sed -i 's|^STRATA_SET=.*|STRATA_SET="ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2"|' .env
cp .env ~/opt-work/prod.env

# プリセット削除（停止中は image を一時コンテナで使う）
sudo docker run --rm --entrypoint /usr/bin/tini -v compose_strata-data:/data \
  -e HOME=/data/home -e XDG_STATE_HOME=/data/state -e XDG_CONFIG_HOME=/data/config \
  -e STRATA_IN_DOCKER=1 -e STRATA_COMPONENTS_DIR=/opt/strata/components \
  ghcr.io/niko1221/strata:0.2.7-cuda12 -- /opt/strata/terminal/bin/strata launch --delete-preset docker

# 起動と healthy 待ち
sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate
sudo docker inspect -f '{{.State.Health.Status}}' strata   # healthy まで約 3 分

# 起動フラグとメモリの確認
sudo docker exec strata sh -c 'cat /data/install/strata-iq3_xxs.json' | grep -A1 -E 'max-context|rope-scaling|rope-scale|kv-resident'
sudo docker exec strata sh -c 'grep -E "rope scaling|KV streaming|expert cache|VRAM free" /data/state/strata/launch/docker.engine.log | tail -5'
free -g; nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# 短答テスト（キーは表示しない）
KEY=$(sudo docker exec strata /opt/strata/terminal/bin/strata launch --show-key --preset docker 2>/dev/null | tail -1)
# Paris / 391 は ~/work/results/ctx512prod/shorttest.sh で確認

# 長文（100K / 300K 相当）
STRATA_API_KEY=$KEY python3 ~/work/results/ctx512/longbench.py 330000 128
STRATA_API_KEY=$KEY python3 ~/work/results/ctx512/longbench.py 990000 128

# 4K/24K 基準
BENCH_SEED=42 STRATA_API_KEY=$KEY python3 ~/bench.py 4000,24000 1 256 code,prose

# 終了確認とロック解放
sudo docker inspect -f '{{.State.Health.Status}}' strata
diff ~/Strata/docker/compose/.env ~/opt-work/prod.env && echo identical
PID=$(pgrep -f "strata --serve" | head -1); sudo cat /proc/$PID/environ | tr '\0' '\n' | grep -E '^STRATA_'
rm -rf /tmp/strata-gpu.lock
```

## 成果物

- 生ログ: `~/work/results/ctx512prod/engine-512k.log`、`serve-512k.log`、`startup-512k.log`
- 構成: `~/work/results/ctx512prod/ctx512-prod-config.json`、`env-before-512k.txt`
- ベンチ: `bench-100k.jsonl`、`bench-300k.jsonl`、`bench-4k-24k.jsonl`
- 監視: `monitor-100k.log`、`monitor-300k.log`、`monitor-bench.log`
- スクリプト: `shorttest.sh`、`monitor.sh`
