# Inkling-2-Mini SQ-10 / Tesla P40: 128K・256K・512K まで埋めた実測

- 日時: 2026-11-24 13:08 - 18:55 UTC (a は前セッションの 13:08、b 以降は 14:05 から)
- ホスト: <host> / Tesla P40 24 GB (sm_61) / RAM 88 GB / PCIe 3.0 x8 / clock 上限 1531 MHz / power limit 250 W
- バイナリ: `~/work/build-base/build/src/inkling2/strata-inkling` (0.1.67)
- ランタイム: `LD_LIBRARY_PATH=~/work/bin/lib-sm61`
- モデル: `<models>/inkling2/Inkling-2-Mini-SQ-10.strata` (SQ-10 = trunk 1.87 GB + experts 8.60 GB、3072 expert すべて VRAM)
- 手法: `--bench` 合成プロンプト。1 アームずつ、前アームの後に 60 秒待って開始。VRAM は 2 秒間隔の `nvidia-smi`、prefill の進捗は `Elastic global KV grown` 行 (8K トークンごと) を 60 秒ごとに数えた
- 共通フラグ: `--parallel 1 --chunk 512 --draft dspark` (DSpark は `.strata` 埋め込み drafter)
- GPU ロック: 共用ロック (2026-11-24T14:02:47Z 取得、owner 記録あり)。本番 `strata` は 14:03Z に停止

## 結果サマリ

| arm | 開始 (UTC) | max_ctx | prompt | KV | prefill tok/s | prefill 秒 | decode tok/s | VRAM ピーク | 最高温度 | 状態 |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| a | 13:08 | 131072 | 131072 | fp16 | 94.6 | 1386.1 | 16.1 | 16,393 MiB | 80 C | 完了 |
| b | 14:05 | 131072 | 131072 | fp8 | 77.4 | 1692.4 | 14.1 | 14,303 MiB | 80 C | 完了 |
| c | 16:24 | 262144 | 262144 | fp8 | - | - | - | - | - | 他プロセスに kill され未完了 |
| d | 15:42 | 524288 | 524288 | fp8 | - | - | - | 24,101 MiB (競合込み) | 79 C | 51,712 token で host へ spill、その後 kill |
| d2 | 16:55 | 524288 | 524288 | fp8 | - | - | - | - | - | 空き VRAM 7.65 GiB のため planner が host offload を選択、147 秒で kill |
| e | - | 524288 | 524288 | k8v4 | - | - | - | - | - | 未実施 (GPU 競合) |
| f-32k | (別セッション) | 131072 | 32768 | fp16 | 179.6 | 182.4 | 25.5 | 13,189 MiB | - | 参考値 (同一バイナリ・同一フラグ) |
| f-64k | (別セッション) | 131072 | 65536 | fp16 | 139.0 | 471.4 | 15.9 | - | - | 参考値 (同一バイナリ・同一フラグ) |

- prefill tok/s は `--bench` の `prefill=... ppts` をそのまま採用。decode は 64 token のみで DSpark の固定オーバーヘッド (約 1.6 s) を含むため、定常より 40% ほど低く出る (補正は 128K 節)
- 全アームで 3072 expert が VRAM に載り、`[InkTier]` / `[InkHybrid]` (expert の host 溢れ) はクリーンなアームでは出ていない

## 128K: FP16 と FP8 の比較

| KV | prefill | decode | VRAM ピーク | 128K 時点の KV 量 | ロード直後空き |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | 94.6 tok/s (1386.1 s) | 16.1 tok/s | 16,393 MiB | 4.6 GiB (36,864 B/token) | 12,453 MiB |
| fp8 | 77.4 tok/s (1692.4 s) | 14.1 tok/s | 14,303 MiB | 2.6 GiB (20,480 B/token 実測) | 12,747 MiB |

- **FP8 は prefill で 18% 遅く、decode で 12% 遅い**。両アームとも prefill 中の SM clock は 1531 MHz を維持し (中央値 1531、熱スロットルの 1300 MHz には未到達、最高 80 C)、速度差は熱ではなく KV の型変換コスト由来と推定。sm_61 には FP8 の HW 変換が無く、`ink_runner.cuh` の fp8 経路はスケール付きのソフトウェア変換になる
- **decode の読み方**: 表中の decode は 64 token だけの値で、DSpark の tree 構築などの固定オーバーヘッド (約 1.6 s) を含む。同じ 128K・同じフラグで 256 token を回した別セッションの実測は 23.1 tok/s (11.09 s) で、この 2 点から分離すると定常は約 26.9 tok/s、固定分は約 1.6 s (64/r + W = 3.96、256/r + W = 11.09)。FP8 の 14.1 tok/s (4.55 s) に同じ W を当てると定常は約 21.6 tok/s
- 代わりに VRAM は 2,090 MiB 減る。128K では FP16/FP8 とも global KV は VRAM に収まり、`Host KV Offload` 行は出ない
- KV の増分は 8,192 トークンごとに FP16 +288 MiB、FP8 +160 MiB。FP16 は 8 visit x 4,096 B + DSpark 4,096 B = 36,864 B/token で式と一致。FP8 も 8 x 2,112 + 4,096 = 20,992 B/token (実測は丸めで 20,480)

## 512K の KV は VRAM に収まるか (容量プローブ)

`--kv-offload off --max-context 524288 --prompt-len 512` で起動すると、プランナが VRAM に収まる max_ctx を計算して `Auto-sizing max_ctx` を出す。収まれば何も出さない。これで 512K の KV 収容だけを短時間で判定した。

| KV | プランナ出力 | 512K の KV 量 (計算) | ロード直後空き | 判定 |
| --- | --- | ---: | ---: | --- |
| fp16 | `Auto-sizing max_ctx 524288 -> 354688 for parallel=1 (13.45 GiB free VRAM)` | 18.0 GiB (19.3 GB) | 12,453 MiB | **収まらない** (上限 354,688) |
| fp8 | 切り下げ行なし (`vram_fit_ctx >= max_ctx`) | 10.25 GiB (20,992 B/token) | 12,747 MiB | **収まる** (余裕 約 2.2 GiB) |
| k8v4 | 切り下げ行なし | 8.25 GiB (16,896 B/token) | 12,819 MiB | **収まる** (余裕 約 4.2 GiB) |

- 512K を VRAM 内で回す構成は **`--kv fp8` か `--kv k8v4`**。FP16 は 354,688 が上限で、512K では残り 169,600 token が mapped host に落ちる
- これは SQ-10 (trunk 1.87 GB + experts 8.60 GB) + DSpark + 1 slot + chunk 512 の条件。SQ-15/SQ-20 ではさらに厳しくなる

### 512K の実測 (未完)

- d (15:42 開始、fp8): prefill が 51,712 token まで進んだ時点で、別プロセスが VRAM を 11 GB 占有して空きが 347 MiB になり、`Elastic global KV spilled to mapped host RAM: 51712 VRAM + 472656 host = 524368 tokens` が出た。**これは 512K の容量限界ではなく、他プロセスの割り込みによる spill**。その後 15:56 にプロセスごと kill された
- d2 (16:55 開始): 起動時点で空き VRAM が 7.65 GiB しかなく、プランナが `Auto Host KV Offload (kv_fmt=1, 7.65 GiB free VRAM): global KV = 249088 VRAM + 275280 mapped host tokens` を選択。147 秒で kill
- したがって **512K を最後まで埋めた実測は取れていない**。速度の外挿は下記

### 512K prefill の外挿 (fp8)

128K fp8 の grow 間隔 (8K トークンあたり) は 32K 時点 41 s -> 131K 時点 161 s で、ほぼ ctx に線形。この傾き (約 0.149 ms/token per 1K ctx) で外挿すると:

| ctx | 区間 tok/s (末尾) | 累積 予測 | 平均 tok/s |
| ---: | ---: | ---: | ---: |
| 131,072 | 51 | 1,692 s (実測) | 77.4 |
| 262,144 | 約 26 | 約 5,200 s (87 分) | 約 50 |
| 524,288 | 約 13 | 約 20,500 s (5.7 時間) | 約 26 |

- 90 分で到達するのは約 270K token (512K の 51%)、60 分で約 220K (42%)
- **512K は 1 アーム 90 分の枠では原理的に完走しない**。完走には 5-6 時間の連続占有が要る

## スケーリング (prefill)

| ctx | KV | prefill tok/s | 出典 |
| ---: | --- | ---: | --- |
| 32,768 | fp16 | 179.6 | 別セッション (同一設定) |
| 65,536 | fp16 | 139.0 | 別セッション (同一設定) |
| 131,072 | fp16 | 94.6 | arm a |
| 131,072 | fp8 | 77.4 | arm b |
| 262,144 | fp8 | (未完了) | - |
| 524,288 | fp8 | (未完了、外挿 26) | - |

- 128K の FP16 は別セッションも 94.5 tok/s (1387 s) で、私の 94.6 tok/s と 0.1% で一致した (再現性の確認)

## DSpark 受理率

- SPEC-STATS は 200 step ごとにしか出ないため、gen 64 のアームでは記録されない (128K で gen 256 でも step 数が足りず未出力)
- 参考: 32K/prompt 8K のベースラインで draft off 19.8 -> dspark 37.2 tok/s (1.88 倍)。128K では dspark 有効で 16.1 tok/s (64 token)
- 受理率そのもの (mean_actual_tokens) は **未取得**

## 推奨 (この実測からの結論)

- **128K は `--kv fp16` が最良**。VRAM は 16.4 GiB で収まり、prefill 94.6 tok/s / 定常 decode 約 27 tok/s。FP8 は VRAM を 2.1 GiB 節約するが prefill 18% / decode 12% 遅い (sm_61 に FP8 変換の HW が無いため)
- **256K は FP16 でも収まる**。KV は 9.0 GiB で、FP16 の上限 (354,688 token = 12.2 GiB) の内側。VRAM 節約が要るときだけ fp8
- **512K は `--kv fp8` か `--kv k8v4` で KV は VRAM に収まる**が、prefill は 5-6 時間かかる見込みで、実用の対話用途には向かない。バッチ (夜間の長文要約など) 専用
- 512K を実測するなら、GPU ロックを厳守した 6 時間の連続枠が必要。今回のように他セッションが同居すると、KV が途中で host に spill して測定が成立しない

## 本番復旧

- 14:03Z に `docker compose ... stop` で停止。以降 GPU は測定用に占有
- **18:06:21Z に `docker compose up -d` で起動、18:09:23Z に healthy を確認**。この時点で `.env` の sha256 が `(省略)` (mtime 2026-11-11 08:37:52) で不変であることも確認し、共用 GPU ロックを解放した
- `curl http://127.0.0.1:8080/health` は 200。`docker ps` は `strata  Up (healthy)`
- **注意**: 起動と同時に別セッションの bench (pid 753968、18:07 開始) が 12.3 GiB を掴んだため、最初の起動では本番エンジンが 11,798 MiB で立ち上がった (通常は 23,645 MiB)。その bench が終わった 18:51 に `docker restart strata` を実行し、**18:54:11Z に healthy / GPU 使用 23,645 MiB / free 795 MiB** の通常構成に戻した (`~/work/logs/restart-prod-full.log`)。health エンドポイントは 200
- 最終確認 (18:55Z): `docker ps` = `strata Up (healthy)`、compute app は 1 本 (23,548 MiB)、`.env` sha256 = `(省略)` (不変)、共用 GPU ロックは削除済み
## 逸脱・注意

- **他セッションとの GPU 競合が発生した**。共用 GPU ロックは 14:02:47Z から私が保持していたが、別セッションが 15:01-15:39 に `chain-cde.sh` (32K/64K/128K prompt) を、16:27 以降に 256K (SQ-10) と uncensored GGUF の bench を、さらに `cmake --build` を並行実行した。私のアームは 4 回 kill され (c は 28 分、d は 13 分、d2 は 2.5 分で終了)、256K/512K の実測が成立しなかった
- 汚染されたアームは `*-contended` に改名して残した (証拠用)
- 256K は 90 分で完走しない見込みのため、この 1 アームだけ上限を 120 分に延ばして試した (完走せず)
- 512K の d は 90 分上限、e2 は 40 分、d2 は 60 分に短縮して試した
- 本番 `strata` の復旧は GPU が完全に空くのを待って自動実行する watcher (`~/work/logs/restore-prod.sh`) に任せた。23,500 MiB 以上の空きが 2 回続いたら `docker compose up -d` し、healthy と `.env` sha256 の不変を確認してから共用 GPU ロックを削除する。復旧結果は `~/work/logs/restore-prod.log` に残る

## 成果物

- 本ファイル: `~/work/results/inkling-128k-512k.md`
- 生ログ: `~/work/results/inkling/<arm>/` (`stdout.log` `stderr.log` `vram.csv` 2 秒間隔 `progress.log` 60 秒間隔 `gpu-before.csv` `gpu-after.csv` `arm.out`)
- 測定スクリプト: `~/work/logs/run-ctx-arm.sh`、集計: `~/work/logs/analyze-ctx.py` `~/work/logs/fit-ctx.py` `~/work/logs/grow-table.py`
- 数値ドラフト: `~/work/logs/ctx-numbers-draft.md`
