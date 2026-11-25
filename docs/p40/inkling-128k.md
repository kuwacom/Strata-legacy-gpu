# Inkling-2-Mini SQ-10 / Tesla P40 128K 埋め実測 (prefill / decode / VRAM)

- 日時: 2026-11-24 12:02-15:45 UTC
- ホスト: <host> / Tesla P40 24GB (sm_61) / RAM 88GB / PCIe **x8** / driver 既定
- バイナリ: `~/work/build-base/build/src/inkling2/strata-inkling` (0.1.67)
- ランタイム: `LD_LIBRARY_PATH=~/work/bin/lib-sm61`
- モデル: `<models>/inkling2/Inkling-2-Mini-SQ-10.strata` (10,471,526,400 B)
- 手法: `--bench` 合成プロンプトを 1 アームずつ実行。アーム前に 60 秒待って温度を揃え、VRAM は 2 秒間隔の `nvidia-smi` サンプリングの最大値を採用。decode tok/s は `[bench]` の値をそのまま使用
- 本番 `strata` は全アーム停止、終了後に復旧済み (healthy)

## サマリ

- **真の 128K (prompt=131,072) を完走した**。SQ-10 / FP16 KV / chunk 512 / DSpark で prefill **1,386.13 s = 94.6 tok/s**、decode 16.1 tok/s (gen 64)
- **128K でも KV は全部 VRAM に残った**。`Host KV Offload` / `spilled` 行は出ず、grow ログは `Elastic global KV grown` のみ。ピーク 16,393 MiB、残り 8,047 MiB
- **FP8 KV は 128K では遅い**。prefill 77.4 tok/s (-18.2%)、decode 14.1 tok/s (-12.4%)、VRAM は -2,090 MiB。SQ-10 は FP16 が収まるので FP8 を選ぶ理由がない
- **prefill は位置に対してほぼ線形に遅くなる** (8K 区間 221 tok/s -> 128K 直前の区間 59 tok/s)。累積では 32K 178 / 64K 139 / 128K 94.6 tok/s で、`1/n` に近い。原因は sm_61 が tensor core タイル型 prefill attention の対象外で、per-token flash 経路に落ちるため
- **20 分上限では 128K は 1 回で終わらない**。上限 1,200 s の初回は prefill の 94% (KV 122,880 まで成長) で打ち切り。完走には 1,386 s 必要

## 測定条件

| 項目 | 値 |
| --- | --- |
| 共通引数 | `--bench --gpu 0 --chunk 512 --draft dspark --kv <fp16/fp8>` |
| parallel | 1 (既定) |
| 実効 max_ctx | 131,152 (prompt 131,072 + gen 64 + 16) / e は 131,344 |
| expert 配置 | 全アーム 3072 VRAM / 0 host (hyb=0, promo=0) |
| codec 内訳 | `2/2: 585, 3/2: 1814, 3/3: 658, 15/15: 15` |
| ロード直後 free | 13,851 MiB、runner+KV 1,320 MiB (fp16) / 1,026 MiB (fp8)、DSpark 78 MiB |
| 温度 | 全アーム開始 34-58 C、最高 80 C、1531 MHz 維持 (89 C の熱スロットル未到達) |

`--max-context 131072` を渡しても、`ink_main.cpp:135-137` が `prompt+gen+16` で上書きするため実効値は 131,152 になる (grow ログの `max_ctx=` が根拠)。`--chunk` を明示したので `[InkCUDA] prefill chunk` 行は出ない (`chunk_auto=false`)。

## アーム別結果

| arm | KV | prompt | gen | prefill tok/s | prefill 秒 | decode tok/s | decode 秒 | TTFT 相当 | VRAM ピーク | 最高温度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a | fp16 | 131,072 | 64 | **94.6** | 1,386.13 | **16.1** | 3.96 | 1,386.19 s | 16,393 MiB | 80 C |
| b | fp8 | 131,072 | 64 | **77.4** | 1,692.38 | **14.1** | 4.55 | 1,692.45 s | 14,303 MiB | 80 C |
| c | fp16 | 32,768 | 64 | **179.6** | 182.41 | **25.5** | 2.51 | 182.45 s | 13,189 MiB | 80 C |
| d | fp16 | 65,536 | 64 | **139.0** | 471.42 | **15.9** | 4.03 | 471.48 s | 14,321 MiB | 80 C |
| e | fp16 | 131,072 | 256 | **94.5** | 1,387.04 | **23.1** | 11.09 | 1,387.08 s | 16,637 MiB | 79 C |
| a (上限 1,200 s) | fp16 | 131,072 | 64 | 未完走 | >1,200 (KV 122,880 まで成長) | - | - | - | 16,029 MiB | 80 C |

- TTFT 相当は `prefill 秒 + 1 step (1 / decode tok/s)` の計算値。`--bench` は SSE を出さないため HTTP の実 TTFT は未取得
- a の decode 16.1 と e の 23.1 は同じ 128K でも差がある。gen 64 の平均は CUDA Graph の capture を含む序盤に引きずられるため、**定常値は e の 23.1 tok/s**
- `--bench` の decode は 1 step = `forward_batch` 1 回で、投機の tree 受理は回らない (後述)。したがって decode tok/s = step/s

## KV の所在 (128K)

| arm | 実効 pos | global KV 総量 | 置き場所 | ピーク時 free |
| --- | ---: | ---: | --- | ---: |
| a (fp16) | 131,152 | 4.30 GB (32,768 B/pos) | **VRAM 全部** | 8,047 MiB |
| b (fp8) | 131,152 | 2.22 GB (16,896 B/pos) | **VRAM 全部** | 10,137 MiB |
| e (fp16) | 131,344 | 4.30 GB | **VRAM 全部** | 7,803 MiB |

- 根拠: a/b/e の stderr に `[InkCUDA] Host KV Offload ...` も `[InkCUDA] Elastic global KV spilled to mapped host RAM ...` も出ていない。出ているのは `Elastic global KV grown: 8192 -> 16384 ... -> 131152` の 16 行のみ
- **落ちる条件も 1 例観測した**。14:43 の c (32K) は他セッションのプロセスが VRAM を掴んだ状態で走り、`Elastic global KV spilled to mapped host RAM: 27136 VRAM + 103936 host = 131072 tokens` で 3.41 GB を host に落とした。この個体は測定として無効なので `c-sq10-128k-p32768-kvfp16-INVALID-contended` に退避し、空きを確保してから取り直した (表の c は再測定値)
- SQ-10 は 128K FP16 でも 8 GiB 残る。**FP8 を強制される状況は SQ-20 の 128K (runbook の推定 -2.87 GB) だけ**で、SQ-10 には当てはまらない

## prefill のスケーリング

同じ run の KV grow (8,192 pos 刻み) の時刻から区間時間を出したもの。a (fp16) と b (fp8) は同一条件なので直接比較できる。

| 位置 | fp16 累積 秒 | fp16 累積 tok/s | fp16 区間 tok/s | fp8 累積 秒 | fp8 累積 tok/s | fp8 区間 tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 37 | 221 | 221 | 41 | 200 | 200 |
| 32,768 | 184 | **178** | 146 | 215 | **152** | 122 |
| 65,536 | 472 | **139** | 100 | 568 | **115** | 81 |
| 98,304 | 872 | 113 | 75 | 1,060 | 93 | 61 |
| 131,072 | 1,386 | **94.6** | 59 | 1,692 | **77.4** | 48 |

区間 tok/s はその位置で終わる 8,192 token の区間の値。
区間時間 (8,192 token あたり、fp16): 37, 42, 49, 56, 62, 68, 76, 82, 89, 96, 106, 109, 117, 125, 133, 139 s。
前半 8K で 37 s、最後の 8K で 139 s。**位置にほぼ比例して伸びる = 累積は二次、スループットは 1/n**。

独立アームとの一致 (同じ曲線を別プロセスで測った検算):

| prompt | 曲線からの予測 | 実測 (arm) | 差 |
| ---: | ---: | ---: | ---: |
| 32,768 | 178 tok/s | 179.6 (c) | +0.9% |
| 65,536 | 139 tok/s | 139.0 (d) | 0.0% |
| 131,072 | 94.6 tok/s | 94.6 (a) / 94.5 (e) | 0.0% |

再現性は ±1% 以内。**32K 178 -> 64K 139 -> 128K 94.6 tok/s** と、倍にするごとに約 0.78 倍・0.68 倍になる。

## FP8 KV の効果

| 項目 | FP16 | FP8 | 差 |
| --- | ---: | ---: | ---: |
| prefill 128K tok/s | 94.6 | 77.4 | **-18.2%** |
| prefill 128K 秒 | 1,386.13 | 1,692.38 | +306 s |
| decode 128K tok/s (gen 64) | 16.1 | 14.1 | -12.4% |
| VRAM ピーク | 16,393 MiB | 14,303 MiB | **-2,090 MiB** |
| runner+KV (8K 時点) | 1,320 MiB | 1,026 MiB | -294 MiB |
| 128K の global KV | 4.30 GB | 2.22 GB | -2.08 GB |
| host 落ち | なし | なし | - |

FP8 は容量を半分にするが、**P40 では prefill も decode も遅くなる**。128K の prefill は attention が支配的で、sm_61 の経路は FP8 のデコード (scale 適用) が FP32 SIMT の内側に入るため、バイト数が減っても得にならない。SQ-10 では FP16 を選ぶのが正しい。

## DSpark の受理率と decode

- **`--bench` では受理率は取れない**。投機の tree を回す `dspark_spec_step_greedy` は `ink_serve.h:934` からしか呼ばれず、`--bench` の `capture_taps` は tap の取得だけを行う (`ink_cuda_backend.cu:906,934`)。`STRATA_INK_SPEC_LOG=1` を付けても `[SPEC-STATS]` は 1 行も出なかった
- 代わりに `--serve` の `GEN` で測った (300 token 文脈、`--max-context 4096 --kv fp16 --draft dspark`):
  - `DONE 64 300 1160.8 998.6 length 49 61 0` -> **drafts 49 / 61 = 受理率 80.3%**
  - decode 64 token / 0.999 s = **64.1 tok/s**
- 128K 文脈での受理率は未取得 (128K の prompt を `GEN` で流す必要があり、1 回 1,386 s かかる)
- 128K の decode は step/s で 16.1 (gen 64) / 23.1 (gen 256)、fp8 で 14.1。32K 25.5、64K 15.9 と合わせると、**decode は 64K 以降ほぼ横ばい (16-23 tok/s)** で、32K から 128K の間で半分以下になる

## 原因: sm_61 は prefill attention のタイル経路に入れない

- `ink_runner.cuh:3310` の tensor core タイル型 prefill attention (`O(1)` VRAM、KV タイル + cuBLAS) の条件は `M >= 128 && (cur_cc >= 70 || cur_cc == 60)`。**sm_61 は除外**され、`k_ink_flash_gqa_gqa3` (`:3395-3400`) の per-token 経路になる
- per-token 経路は (kv_head, token) ごとに 1 CTA で、各 CTA が 8 global visit 分の前置全体を読む。512 token の chunk でも tile 共有が無いため、chunk の処理時間が位置に比例して伸びる
- 実測がこれを支持する: 8K 区間 37 s -> 120-128K 区間 139 s (3.8 倍)、累積 tok/s 221 -> 94.6 (2.3 分の 1)
- 副次的: sm_61 では WMMA / INT8 TC / BF16 TC が無効 (runbook 記載) なので、MoE 側も dp4a の SIMT 経路。prefill 全体が「attention が二次、MoE が線形」の合成になり、長いほど attention の比率が上がる
- 対策候補 (未検証): prefill attention のタイル経路を sm_61 にも開ける (FP32 cuBLAS で tile 共有だけでも効くはず)、または `M` を増やして CTA あたりの KV 再読みを償却する

## 逸脱・注意

- **20 分上限を 1 回だけ超えた**。a の初回は 1,200 s 上限で prefill の 94% (KV 122,880) で打ち切られ (`rc=124`)、完走に 1,386 s 必要と分かったため、a だけ 1,800 s に延ばして再実行した。a (上限 1,200 s) の記録は証拠として残してある
- **他セッションと GPU を奪い合った**。14:04 に別セッションが同じ arm b を `rm -rf` して再起動し、14:43-14:55 には 256K/512K の chain (`chain-cde2.sh`、最長 5 時間) を回そうとした。c の初回はこの干渉で host へ 3.41 GB スピルしたため無効化して取り直した。本番復旧のため 256K の run を 2 回停止している (いずれも開始直後)。経緯は `~/work/logs/inkling-128k-ownership.md`
- **本番が一度落ちた**。15:51 に別セッションが本番稼働中に 512K のアームを起動し、VRAM を奪われて本番エンジンが unhealthy になった (コンテナは Up のまま)。512K を停止して 15:56 に compose restart、15:59 に healthy へ戻した。以降 256K/512K の chain が 2 回再起動を試みたため都度停止している。経緯と復旧手順は `~/work/logs/inkling-128k-ownership.md`
- この競合の間、b は別セッションが起動した個体のログを使っている (引数は同一: `--max-context 131072 --chunk 512 --kv fp8 --draft dspark --prompt-len 131072 --gen 64`)
- 受理率は 300 token 文脈での値で、128K 文脈の値ではない
- HTTP の実 TTFT (SSE 初回 delta) は未取得。表の TTFT は計算値
- `--draft off` との比較は今回取っていない (全アーム DSpark 有効)。baseline の「DSpark で 1.88 倍」は `--bench` の step 速度差で、受理によるものではない可能性がある (機序未確認)
- 温度は最高 80 C で熱スロットル (89 C / 1300 MHz) は発生せず、全アーム 1531 MHz を維持

## 成果物

- 本ファイル: `~/work/results/inkling-128k.md`
- 生ログ: `~/work/results/inkling/<arm>/` に `stdout.log` `stderr.log` `vram.csv` (2 秒間隔) `gpu-before.csv` `gpu-after.csv`
  - `a-sq10-128k-p131072-kvfp16` / `b-sq10-128k-p131072-kvfp8` / `c-sq10-128k-p32768-kvfp16` / `d-sq10-128k-p65536-kvfp16` / `e-sq10-128k-p131072-kvfp16-gen256`
  - `a-sq10-128k-p131072-kvfp16-cap1200` (20 分上限で打ち切った記録)
  - `c-sq10-128k-p32768-kvfp16-INVALID-contended` (干渉で host スピルした無効データ)
  - `probe-dspark-accept/` (受理率の `--serve` 測定)
- スクリプト: `~/work/logs/run-128k-arm.sh`、`~/work/logs/chain-cde.sh`、`~/work/logs/grow-timeline.py`、`~/work/logs/analyze-128k.py`
- 復旧: `strata` healthy (15:51 の VRAM 枯渇で一度 unhealthy -> 15:56 compose restart -> 15:59 healthy)、`~/Strata/docker/compose/.env` は sha256 `(省略)` / mtime 2026-11-11 08:37:52 で不変、GPU 使用 23,645 MiB、共用 GPU ロックは 16:00 UTC に解放済み
