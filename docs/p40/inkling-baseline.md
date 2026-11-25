# Inkling-2-Mini Tesla P40 ベースライン実測 (最適化前)

- 日時: 2026-11-24 10:35-12:0x UTC
- ホスト: <host> / Tesla P40 24GB (sm_61) / RAM 88GB / PCIe x8 / driver 既定
- バイナリ: `~/work/build-base/build/src/inkling2/strata-inkling` (0.1.67)
- ランタイム: `LD_LIBRARY_PATH=~/work/bin/lib-sm61`
- 手法: `--bench` 合成プロンプト、1 アームずつ実行。前アームの後に 60 秒待ってから開始し、開始時温度・クロックを記録。VRAM は 2 秒間隔の `nvidia-smi` サンプリングの最大値
- 本番 `strata` は全アーム停止、終了後に復旧済み (healthy)

## 1. 結果サマリ

### 1.1 速度

| arm | モデル | max_ctx | chunk | draft | prompt | gen | prefill tok/s | prefill 時間 | decode tok/s | decode 時間 | TTFT (計算) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a | SQ-10 | 32768 | 512 | off | 512 | 64 | 88.4 | 5.79 s | 20.6 | 3.11 s | 5.84 s |
| b | SQ-15 | 32768 | 512 | off | 512 | 64 | 32.8 | 15.59 s | 13.4 | 4.76 s | 15.66 s |
| c | SQ-10 | 131072 | 512 | off | 8192 | 64 | 223.1 | 36.71 s | 19.8 | 3.23 s | 36.76 s |
| d | SQ-10 | 131072 | 512 | dspark | 8192 | 64 | 223.1 | 36.72 s | **37.2** | 1.72 s | 36.75 s |
| e1 | SQ-10 | 131072 | 1024 | off | 8192 | 64 | 223.6 | 36.63 s | 19.9 | 3.22 s | 36.68 s |
| e2 | SQ-10 | 131072 | 2048 | off | 8192 | 64 | 223.7 | 36.63 s | 19.8 | 3.23 s | 36.68 s |
| f | SQ-15 | 131072 | 512 | off | 8192 | 64 | 37.8 | 216.55 s | 13.7 | 4.69 s | 216.62 s |

- TTFT は直接測定ではなく `prefill 時間 + 1 decode step (1 / decode tok/s)` の計算値。`--bench` は SSE を出さないため HTTP の実 TTFT は未取得
- prefill tok/s は `--bench` の `prefill=NNN ppts` をそのまま採用
- a と c は prompt 長が違うため prefill tok/s を直接比較できない (512 token は固定オーバーヘッドの比率が大きい)

### 1.2 VRAM・状態

| arm | VRAM ピーク (MiB) | ロード直後 free (MiB) | runner+KV (MiB) | expert 配置 | host 溢れ | 開始温度 / クロック | 最高温度 | 所要 (ロード込) |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| a | 11,487 | 14,293 | 1,320 (max_batch=512) | 3072 VRAM / 0 host | なし | 44 C / 1531 MHz | 55 C | 147.7 s |
| b | 15,185 | 10,595 | 1,320 (512) | 3072 VRAM / 0 host | なし | 44 C / 1531 MHz | 55 C | 115.9 s |
| c | 11,743 | 14,293 | 1,320 (512) | 3072 VRAM / 0 host | なし | 45 C / 1531 MHz | 61 C | 95.8 s |
| d | 12,315 | 13,851 | 1,320 (512) + DSpark 78 | 3072 VRAM / 0 host | なし | 47 C / 1531 MHz | 63 C | 107.6 s |
| e1 | 12,111 | 14,293 | 1,684 (1024) | 3072 VRAM / 0 host | なし | 49 C / 1531 MHz | 63 C | 82.0 s |
| e2 | 13,011 | 14,293 | 2,576 (2048) | 3072 VRAM / 0 host | なし | 50 C / 1531 MHz | 64 C | 81.7 s |
| f | 15,441 | 10,595 | 1,320 (512) | 3072 VRAM / 0 host | なし | 48 C / 1531 MHz | 68 C | 259.4 s |

- 全アームで `[InkCUDA] Loaded ... experts=3072 VRAM / 0 host (hyb=0, promo=0)` -> **expert は全 3072 個が VRAM に全載り、host 溢れなし**
- 全アームで `Host KV Offload` 行が出ていない -> **global KV も VRAM 内**。`[InkCUDA] prefill chunk` 行は `--chunk` を明示したため出ない (`chunk_auto=false`)
- `[InkTier]` / `[InkHybrid]` 行も全アームで出ていない (溢れていないため)
- 温度は全アーム 68 C 以下、測定中に 1531 MHz から低下した記録なし (89 C の熱スロットルには未到達)

## 2. アーム別の生ログ要点

### a. SQ-10 / 32K / chunk512 / draft off / prompt512
```
[bench] backend=cuda format=strata parallel=1 gpus=1 kv=fp16 prompt=512 gen=64 | prefill=88.4 ppts (5.79s) | decode=20.6 tps agg (20.6 tps/slot, 3.11s)
[InkCUDA] Loaded ... (SQ-10): trunk=1.87 GB (16.62s), experts=3072 VRAM / 0 host (86.36s, hyb=0, promo=0, pf_buf=0MB), sm120_tc=0, screen=0
[InkCUDA] memory: 14293 MiB free after load, runner+KV took 1320 MiB (max_batch=512), DSpark took 0 MiB, 12973 MiB free now
[InkCUDA] expert codecs (gate_up/down code: count, mean MB): 2/2: 585, 2.18; 3/2: 1814, 2.85; 3/3: 658, 3.18; 15/15: 15, 4.48;
```

### b. SQ-15 / 32K / chunk512 / draft off / prompt512
```
[bench] ... prompt=512 gen=64 | prefill=32.8 ppts (15.59s) | decode=13.4 tps agg (13.4 tps/slot, 4.76s)
[InkCUDA] Loaded ... (SQ-15): trunk=2.07 GB (11.91s), experts=3072 VRAM / 0 host (73.03s, hyb=0, promo=0, pf_buf=0MB), sm120_tc=0, screen=0
[InkCUDA] memory: 10595 MiB free after load, runner+KV took 1320 MiB (max_batch=512), DSpark took 0 MiB, 9275 MiB free now
[InkCUDA] expert codecs: 3/3: 1022, 3.18; 4/4: 509, 4.17; 5/4: 1, 4.84; 5/5: 2, 5.17; 15/15: 1538, 4.48;
```

### c. SQ-10 / 128K / chunk512 / draft off / prompt8192
```
[bench] ... prompt=8192 gen=64 | prefill=223.1 ppts (36.71s) | decode=19.8 tps agg (19.8 tps/slot, 3.23s)
[InkCUDA] Loaded ... (SQ-10): trunk=1.87 GB (8.57s), experts=3072 VRAM / 0 host (45.34s, ...), screen=0
[InkCUDA] memory: 14293 MiB free after load, runner+KV took 1320 MiB (max_batch=512), DSpark took 0 MiB, 12973 MiB free now
```

### d. SQ-10 / 128K / chunk512 / draft dspark / prompt8192
```
[bench] ... prompt=8192 gen=64 | prefill=223.1 ppts (36.72s) | decode=37.2 tps agg (37.2 tps/slot, 1.72s)
[InkCUDA] Loaded ... (SQ-10): trunk=1.87 GB (9.04s), experts=3072 VRAM / 0 host (57.82s, ...), sm120_tc=0, screen=1
[InkCUDA] memory: 13851 MiB free after load, runner+KV took 1320 MiB (max_batch=512), DSpark took 78 MiB, 12453 MiB free now
```
DSpark は `.strata` 埋め込みの drafter を使用 (`--dspark PATH` は未指定)。`--info` の `drafter=dspark` で確認

### e1 / e2. SQ-10 / 128K / chunk1024・2048 / draft off
```
chunk1024: prefill=223.6 ppts (36.63s) | decode=19.9 tps (3.22s) | runner+KV took 1684 MiB (max_batch=1024), peak 12111 MiB
chunk2048: prefill=223.7 ppts (36.63s) | decode=19.8 tps (3.23s) | runner+KV took 2576 MiB (max_batch=2048), peak 13011 MiB
```

### f. SQ-15 / 128K / chunk512 / draft off / prompt8192
```
[bench] ... prompt=8192 gen=64 | prefill=37.8 ppts (216.55s) | decode=13.7 tps agg (13.7 tps/slot, 4.69s)
[InkCUDA] Loaded ... (SQ-15): trunk=2.07 GB (11.47s), experts=3072 VRAM / 0 host (24.61s, ...), sm61, screen=0
[InkCUDA] memory: 10595 MiB free after load, runner+KV took 1320 MiB (max_batch=512), DSpark took 0 MiB, 9275 MiB free now
```

## 3. 所見

- **SQ-15 は SQ-10 より大幅に遅い**。8K prompt で prefill 223.1 -> 37.8 tok/s (5.9 倍差)、decode 19.8 -> 13.7 tok/s (1.45 倍差)。VRAM ピークは +3.7 GiB (11.7 -> 15.4 GiB)
- 原因の推定: ログの expert codec 内訳で SQ-15 は 1538 expert が code `15/15`、SQ-10 は 15 個のみ。runbook の記載どおり sm_61 では `15/15` (SF4/MX8 系) が LUT デコード + FFMA 経路になり、dp4a の `2/2` `3/2` `3/3` より遅いためと推定 (コード経路の確認は未実施)
- **DSpark は decode を 1.88 倍に高速化** (19.8 -> 37.2 tok/s)。prefill は不変 (223.1)。VRAM 増は +572 MiB (ピーク 11,743 -> 12,315)
- **chunk 512/1024/2048 は prefill に影響しない** (223.1 / 223.6 / 223.7 tok/s)。8K prompt では 512 で既に飽和。chunk を上げると workspace が増えるだけ (runner+KV 1,320 -> 1,684 -> 2,576 MiB)
- 24GB カードに対し全アームで余裕があり、最大ピークは f の 15,441 MiB。SQ-10 は 12 GiB 強で 12 GiB 以上の空きを残す

## 4. 未取得・注意

- **真の 128K コンテキストは未測定**。`--max-context 131072` でも起動時に確保される global KV は 8192 pos のみで、prompt 8192 では KV が 8K までしか伸びない。128K まで埋めたときの VRAM・decode は未取得 (この場合 SQ-15 は全載りぎりぎり、SQ-20 は `--kv fp8` 必須という runbook の推定は未検証)
- HTTP の実 TTFT (SSE 初回 delta) は未取得。表中の TTFT は計算値
- SQ-20 は今回未測定 (アーム指示外)。Vision も未測定
- 品質ゲート (`--spec-check` / `--ring-check` / needle) は未実施。DSpark の lossless 性は未検証
- 所要時間にはモデルロードが含まれる。ロード時間はページキャッシュで大きく変わる (例: SQ-10 trunk 16.62 s -> 1.06 s) ため、アーム間の所要時間比較に意味はない
- 温度は全アーム 68 C 以下で、熱スロットル (89 C / 1300 MHz) は発生していない

## 5. runbook / help との差分

- `--help` の `--bench` 行には `--max-context` / `--chunk` / `--draft` が列挙されていないが、パーサは全モード共通で受け付ける (`ink_main.cpp:872-912`)。runbook の表が正しく、help が不足
- バイナリの実パスは `~/work/build-base/build/src/inkling2/strata-inkling`。runbook の `build/engine/src/inkling2/...` は存在しない
- SQ-15 は 14,348,316,672 B で完全な状態に再取得済み (runbook の「5.63 GB で不完全」は解消)。SQ-20 も取得済み (19,346,685,952 B)
- `[InkCUDA] prefill chunk` 行は `--chunk` 明示時には出ない (`chunk_auto=false`)。runbook の「起動ログで必ず読む行」は auto 時のみ有効

## 6. 成果物

- 本ファイル: `~/work/results/inkling-baseline.md`
- 生ログ: `~/work/results/inkling/<arm>/` に `stdout.log` `stderr.log` `vram.csv` (2 秒間隔) `gpu-before.csv` `gpu-after.csv`
- 測定スクリプト: `~/work/logs/run-arm.sh`、集計: `~/work/logs/analyze.py`
- 復旧確認: `strata` healthy、`~/Strata/docker/compose/.env` は sha256 `(省略)` / mtime 2026-11-11 08:37:52 で不変、GPU 使用 23,645 MiB
