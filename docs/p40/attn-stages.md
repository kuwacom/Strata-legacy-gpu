# Inkling タイル型 prefill attention のステージ別内訳 (sm_61 / FP32 SGEMM 経路)

調査日: 2026-11-25 / 対象: `~/work/ink-ap` (branch `perf/ink-attn-prof`、`1893427af`、base `perf/ink-sk-occ-final` `f25fdc067`) / バイナリ sha256 `2abd34bc…` / Tesla P40 (sm_61) 単機 / SQ-10 / 本番は停止 -> 復旧済み (healthy、`.env` sha256 `(省略)` 不変、GPU/ビルド両ロック解放)

## 結論

- 128K (M=512 x 256 chunk) の attention 位相 (`attn=`) は **921.28 ms/chunk、合計 235.85 s** で prefill 全体 (671.88 s) の **35.1%**。ステージ別の和は attn= と **-0.01%** (chunk ごとの最大ずれ 0.24%) で閉じた
- 最大区間は **PV SGEMM 80.50 s (34.15%)**。次いで QK SGEMM 59.55 s (25.29%)、softmax 55.68 s (23.60%)、pack_kv 29.73 s (12.61%)、accum/unpack/gate 5.20 s (2.21%)、rel_table 4.43 s (1.88%)、pack_q 0.57 s (0.24%)、online_init 0.06 s (0.02%)
- global 経路 (K_len > 2048、visit 8 本) が **207.80 s (88.1%)**、SWA 経路 (K_len <= 2048、visit 40 本) が **28.01 s (11.9%)**。SWA は文脈長に依存せず **109.41 ms/chunk 一定** (128K の 11.9% を常に占める)
- 文脈長で伸びるのは global だけ。chunk 位置別の attention は 序盤 **387.12** -> 中盤 **924.77** -> 末尾 **1,457.83 ms/chunk** で、末尾では global が 92.5% を占める
- QK+PV は 928 TFLOP (実測 K_t から計算) を 140.10 s で処理し **6.63 TFLOPS** = P40 FP32 ピーク 11.76 の **56%**。PV (5.77 TFLOPS) は QK (7.79 TFLOPS) より 26% 効率が低い
- 計測の影響は小さい: 同一バイナリで `STRATA_INK_ATTN_PROF` 有無の差は prefill **-0.26%** (195.1 vs 195.6 ppts)、attn= 位相 +0.42% (917.39 -> 921.28 ms/chunk)。基準 197.2 ppts に対しては prof 有りで **-1.07%**、無しで **-0.81%** (合格 ±3%)
- 512K (k8v4) も完走 (prefill 5,220.52 s / 100.4 ppts)。attention は **3,430.15 s = prefill の 65.9%** で、ステージ順位は 128K と同じ (PV 35.6% > QK 25.9% > softmax 23.6% > pack_kv 12.2%)。和は attn= と -0.00% で閉じた

## 測定条件

| 項目 | 内容 |
|---|---|
| バイナリ | `~/work/ink-ap/build/src/inkling2/strata-inkling` sha256 `2abd34bc52ebcac94ac2e6032f76b9e34b41f442fee74d5cf6c638b0eca976ed` (実行時は `~/work/bin/strata-inkling-attn-prof` に複製) |
| ランタイム | `LD_LIBRARY_PATH=~/work/bin/lib-sm61`、`CUDA_VISIBLE_DEVICES=0` |
| モデル | `<models>/inkling2/Inkling-2-Mini-SQ-10.strata` |
| 共通 env | `STRATA_INK_PF_TILE=1 STRATA_INK_PROF=2`、prof アームのみ `STRATA_INK_ATTN_PROF=1` |
| 共通引数 | `--bench --native --gpu 0 --parallel 1 --chunk 512 --draft off` |
| 128K | `--max-context 131072 --kv fp16 --prompt-len 131072 --gen 16` (256 chunk) |
| 512K | `--max-context 524288 --kv k8v4 --kv-offload off --prompt-len 524288 --gen 16` (1,024 chunk) |
| 順序 | smoke (4K) -> 60 s -> p128-prof -> 60 s -> p128-off -> 60 s -> p512-prof、各アーム前に 60 s 待機 |
| 生データ | `~/work/results/attn-stages/{smoke,p128-prof,p128-off,p512-prof}.{out,err,gpu}`、ログ `~/work/logs/attn-prof-ab.log` |
| スクリプト | `~/work/bin/attn-prof-ab.sh` (計測)、`attn-prof-parse.py` (集計)、`attn-phase.py` (位相平均)、`attn-prof-build.sh` (ビルド) |
| GPU 状態 | 各アーム終了時の clock 1531 MHz (P40 上限)、温度 73-74 C、消費 64 W で throttle の兆候なし |

- 計測は `STRATA_INK_PROF=2` (eager、CUDA Graph 無効、1 chunk 1 行) の上で取った。`attn=` 位相は runner の `prof_mark_end(3)` 区間 (rel_table の前から attention 本体の終わりまで)
- 起動ログに `tiled prefill attention is off` が無いことを確認済み (タイル経路が動いている)
- visit 構成は 48 本で、全 chunk で global 8 本 / SWA 40 本 (prof 出力の `global_visits` / `swa_visits` で確認)

## 変更 (計測コードのみ、+109 行)

| ファイル | 箇所 | 内容 |
|---|---|---|
| `engine/src/inkling2/cuda/ink_runner.cuh` | `:1980` 付近 | ステージ別 prof の状態と `attn_begin/mark/abort/report` を追加。event はリングで事前確保し、区間ごとに同期しない |
| 同上 | `:2212` | `STRATA_INK_ATTN_PROF=1` の読み取り (既定 off、無効時は mark が bool 1 個で戻る) |
| 同上 | `:3415-3540` (run_visit の 7 番) | visit ごとに rel_table の前から event を張り、pack_q / pack_kv / QK / softmax / PV / init / accum の終端で mark |
| 同上 | `:3849` | chunk 終了時に `attn_report()` を 1 回だけ呼び、`attn=` 位相と突き合わせて 3 行出す |

- 同期は chunk 終了時の 1 回 (`cudaStreamSynchronize`) だけ。測っている区間の中では 1 度も同期しない
- ステージ境界の区間 (event 対) は chunk あたり最大 5,592 個 (128K 末尾) / 20,952 個 (512K 末尾)。無効時はイベントを作らず、`attn_begin` 等は即 return する
- 演算・カーネル・引数は変えていない (追加のみ)

## 128K のステージ別内訳

ms/chunk (M=512 の 1 chunk = 1 step)。占率は 235.85 s に対する比。位置別は p0 を 3 等分 (序盤 p0 0-43,520 / 中盤 44,032-87,040 / 末尾 87,552-130,560)。

| 位置 | chunk | rel_table | pack_q | pack_kv | QK | softmax | PV | init | accum/unpack/gate | 合計 | attn= | 差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 序盤 | 86 | 17.32 | 2.22 | 43.74 | 92.80 | 90.88 | 130.96 | 0.21 | 8.98 | 387.12 | 387.25 | -0.04% |
| 中盤 | 85 | 17.31 | 2.22 | 116.61 | 233.86 | 218.28 | 315.87 | 0.22 | 20.40 | 924.77 | 924.90 | -0.01% |
| 末尾 | 85 | 17.31 | 2.22 | 188.88 | 373.87 | 344.61 | 499.02 | 0.22 | 31.70 | 1,457.83 | 1,457.96 | -0.01% |
| **全体** | 256 | **17.31** | **2.22** | **116.12** | **232.96** | **217.43** | **314.56** | **0.22** | **20.31** | **921.15** | **921.28** | **-0.02%** |

合計 (s) と占率:

| ステージ | 合計 s | 占率 | 何を測っているか |
|---|---:|---:|---|
| PV SGEMM | 80.50 | 34.15% | `cublasGemmStridedBatchedEx` (OP_N, 128 x C x K_t、batch 24) |
| QK SGEMM | 59.55 | 25.29% | 同 (OP_T, K_t x C x 128、batch 24、scale 0.0884 込み) |
| softmax | 55.68 | 23.60% | `k_ink_pf_softmax_tile_online` (rel bias + causal/SWA mask + online m/l) |
| pack_kv | 29.73 | 12.61% | `k_ink_pf_pack_kv<2>` (FP16 KV -> FP32、GQA 3 複製込み) |
| accum/unpack/gate | 5.20 | 2.21% | `k_ink_pf_accum_tile` (online 結合 + 最終 gate) / `k_ink_pf_unpack_o_gate` |
| rel_table | 4.43 | 1.88% | `k_ink_rel_table` (48 visit 分、visit あたり 0.36 ms) |
| pack_q | 0.57 | 0.24% | `k_ink_pf_pack_q<2>` |
| online_init | 0.06 | 0.02% | `k_ink_pf_online_init` (m/l/o の初期化、2 q-block x 8 global visit) |
| **合計** | **235.81** | **100%** | attn= は 235.85 s (差 -0.01%) |

## global 経路と SWA 経路の別

| 経路 | visit/ chunk | 合計 s | 占率 | 1 chunk 平均 | 伸び方 |
|---|---:|---:|---:|---:|---|
| global (長い経路、K_len > 2048) | 8 | 207.80 | 88.1% | 811.73 ms | p0 に比例 (K_t 平均 1,969 -> 2,032) |
| SWA (短い経路、K_len <= 2048) | 40 | 28.01 | 11.9% | 109.41 ms | 一定 (K_t 平均 1,529、window 1024 + chunk 512) |

global の内訳 (ms/chunk):

| 位置 | pack_kv | QK | softmax | PV | accum | rel_table+pack_q | 合計 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 序盤 | 36.90 | 71.48 | 65.98 | 94.33 | 6.26 | 3.25 | 278.41 |
| 中盤 | 109.68 | 212.34 | 193.12 | 278.71 | 17.68 | 3.25 | 815.00 |
| 末尾 | 181.96 | 352.34 | 319.45 | 461.86 | 28.98 | 3.25 | 1,348.06 |
| 全体 | 109.23 | 211.51 | 192.35 | 277.58 | 17.60 | 3.25 | 811.73 |

SWA の内訳 (ms/chunk、全位置でほぼ同値):

| pack_kv | QK | softmax | PV | accum | rel_table+pack_q | 合計 |
|---:|---:|---:|---:|---:|---:|---:|
| 6.90 | 21.46 | 25.08 | 36.98 | 2.72 | 16.28 | 109.41 |

呼び出し回数と平均 C / K_t (全体、[n, C, K_t]):

| ステージ | global | SWA |
|---|---|---|
| pack_kv | 519 [256, 2,021] | 40 [512, 1,529] |
| QK | 520 [256, 2,020] | 80 [256, 1,274] |
| softmax | 520 [256, 2,020] | 80 [256, 1,274] |
| PV | 520 [256, 2,020] | 80 [256, 1,274] |
| accum/unpack/gate | 520 [256, 2,020] | 80 [256, 1,274] |
| online_init | 15 [256, 0] | 0 |

- 序盤の chunk (K_len <= 2048) は短い経路に入り、pack_kv が visit あたり 1 回 (C=512) になるため、global の pack_kv だけ n と C がわずかにずれる (519 / 256.1)
- SWA は 1 visit あたり 2 q-block (C=256) を回すので QK 以降は n=80。pack_kv は visit あたり 1 回 (C=512) で n=40

## 演算効率と律速 (実測値からの計算)

K_t と C は上の実測平均、FLOP は 24 head x 2 x C x K_t x 128 で計算。

| ステージ | FLOP | 時間 | 実効 | P40 ピーク比 | 転送量/呼び出し | 実効帯域 | FLOP/byte | 律速 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| QK | 464 TFLOP | 59.55 s | 7.79 TFLOPS | 66% | 74.5 MB (K の 24 複製読み 24.8 + S 書き 49.7) | 183 GB/s | 43 | 演算 (平衡 34 の 1.3 倍) |
| PV | 464 TFLOP | 80.50 s | 5.77 TFLOPS | 49% | 77.6 MB (P 読み 49.7 + V の複製読み 24.8 + O 書き 3.1) | 145 GB/s | 41 | 演算 (効率が低い) |
| softmax | - | 55.68 s | 29.7e9 elem/s | - | 124.6 MB (S 読み 49.7 + P 書き 49.7 + rel_table 25.2) | 298 GB/s | - | DRAM (346 の 86%) |
| pack_kv | - | 29.73 s | - | - | 58.0 MB (KV 読み 8.3 + 24 複製書き 49.7) | 236 GB/s | - | DRAM (68%) |

- FLOP/byte の機械平衡は P40 で 34 (11.76 TFLOPS / 346 GB/s)。QK は 43 で演算寄り、softmax と pack_kv は純粋な帯域律速
- PV の GEMM は M=128 (head_dim) x N=256 x K=2,020、batch 24。QK (M=2,020 x N=256 x K=128) と FLOP は同じだが 26% 遅い。M=128 が小さいことが効率差の主因と推測する (未確認)
- GQA の 3 複製 (8 kv head -> 24 q head) が pack_kv の書きと QK/PV の読みを 3 倍にしている。ただし QK/PV は演算律速なので効くのは pack_kv 側
- S と P は FP32 で往復 99.3 MB/呼び出し。sm_61 の FP16 が 1/64 なのでどちらも素直には落とせない

## 計測オーバーヘッド

| アーム | ATTN_PROF | prefill ppts | 秒 | attn= ms/chunk | total ms/chunk |
|---|---|---:|---:|---:|---:|
| p128-off | 無し | **195.6** | 669.96 | 917.39 | 2,610.67 |
| p128-prof | 有り | **195.1** | 671.88 | 921.28 | 2,615.90 |
| 差 | - | **-0.26%** | +0.29% | **+0.42%** | +0.20% |
| 参考: occ-final pf128-o1 (別バイナリ・gen 64) | 無し | 197.2 | 664.75 | 897.41 | 2,590.19 |

- 計測有無の差は prefill -0.26%、attn= +0.42% (chunk あたり 5,592 event で +3.9 ms = event 1 個 0.7 us)。合格条件 ±3% に対して十分小さい
- 別バイナリの基準 197.2 ppts に対しては prof 有り -1.07%、無し -0.81%。attn= は 897.41 -> 917.39 と +2.2% 動いており、バイナリ間 (または実行時のホスト負荷) の差が prefill の -0.8% の主因。**同一バイナリ内の prof 有無の差 (+0.42%) が計測コードのコスト**

## 512K (k8v4、参考)

**完走** (中断なし)。prefill **5,220.52 s / 100.4 ppts**、decode 7.0 tps。attention は **3,430.15 s** で prefill の **65.9%** (attn= 3,349.76 ms/chunk)。和と attn= の差は **-0.00%** (chunk ごとの最大 0.24%)

| ステージ | 合計 s | 占率 | 128K との比 |
|---|---:|---:|---:|
| PV SGEMM | 1,221.6 | 35.62% | 15.2x |
| QK SGEMM | 887.3 | 25.87% | 14.9x |
| softmax | 808.3 | 23.57% | 14.5x |
| pack_kv | 418.3 | 12.20% | 14.1x |
| accum/unpack/gate | 74.2 | 2.16% | 14.3x |
| rel_table | 17.7 | 0.52% | 4.0x |
| pack_q | 2.3 | 0.07% | 4.0x |
| online_init | 0.2 | 0.01% | 4.0x |
| **合計** | **3,430.0** | **100%** | **14.5x** |

| 位置 | chunk | global の tile 数/chunk | pack_kv | QK | softmax | PV | accum | rel+pack_q | 合計 ms/chunk |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 序盤 (p0 0-174,592) | 342 | 692 | 141.01 | 304.16 | 281.65 | 424.42 | 26.35 | 19.51 | 1,197.32 |
| 中盤 (175,104-349,184) | 341 | 2,057 | 408.86 | 867.33 | 790.09 | 1,194.14 | 72.50 | 19.51 | 3,352.65 |
| 末尾 (349,696-523,776) | 341 | 3,421 | 676.47 | 1,429.61 | 1,297.94 | 1,962.70 | 118.76 | 19.52 | 5,505.22 |
| **全体** | 1,024 | 2,056 | **408.52** | **866.49** | **789.40** | **1,193.00** | **72.49** | **19.51** | **3,349.63** |

- global 経路 3,317.61 s (96.7%) / SWA 経路 112.41 s (3.3%)。SWA は 128K と同じ **109.77 ms/chunk 一定**
- 最終 chunk (p0=523,776) は attn= **6,579.28 ms** (4,096 tile)。128K の最終 chunk (p0=130,560、1,024 tile) の 1,721.25 ms に対し tile 数 4.0x で 3.82x。**1 chunk あたりの attention は p0 にほぼ比例**し、合計が p0² で伸びるのは chunk 数が 4 倍になるため
- 効率 (実測 tile 数と K_t から計算): QK **7.61 TFLOPS** (128K 7.79)、PV **5.49 TFLOPS** (5.77)。softmax は 338 GB/s で **DRAM ピークの 98%**、pack_kv は 272 GB/s (79%)
- k8v4 KV のため pack_kv の読みは FP16 の半分だが、書き (GQA 3 複製で FP32) が支配的なので 128K の FP16 版とほぼ同じ 12.2% を占める
- 参考: 過去の 512K 記録 (97.6 ppts / 5,373 s、別バイナリ) に対し prefill +2.9%。ただし `STRATA_INK_PROF=2` の eager 実行を含む条件差があるため、単純比較はできない

## 短縮案への所見 (推定)

attention は prefill の 35.1% で、残りは MoE (gu+dn で約 1,520 ms/chunk) が大半。attention を X% 削っても prefill は 0.35X% しか縮まない。それを踏まえた候補 (削減量は実測レートからの外挿で、実装前の推定):

| 案 | 対象区間 | 根拠 (実測) | 推定削減 | 難度 / リスク |
|---|---|---|---:|---|
| PV の GEMM 形状を変えて M を 256 に上げる (P^T x V 形式、または 3 q-head を N 方向にまとめる) | PV 80.50 s (34%) | 同じ FLOP の QK が 7.79 TFLOPS、PV は 5.77 TFLOPS | -21 s (attention -8.9%、prefill -3.1%) | 中 / bit 一致はしない |
| pack_kv の GQA 3 複製をやめる (8 kv head のまま QK/PV を batch 8・N=768 で回す) | pack_kv 29.73 s (12.6%) | 帯域律速 236 GB/s、書き 49.7 -> 16.6 MB | -17 s (attention -7.2%、prefill -2.5%) | 中 / S・P のレイアウト変更が要る |
| S を FP16 で書く (QK の C を CUDA_R_16F に、softmax は FP16 読み) | softmax 55.68 s (23.6%) | softmax は 298 GB/s = DRAM 86%、S 読みが半分になる (124.6 -> 99.7 MB) | -11 s (attention -4.7%、prefill -1.6%) | 中 / scale 後スコアの FP16 精度を品質ゲートで要確認 |
| softmax を row 永続化して rel_table を shared に載せる (現状は KV tile ごとに 1025 float の行を再読) | softmax の rel_table 読み 25.2 MB/呼び出し (全体の 20%) | S/P は減らせないが rel の再読を消せる | -9 s 程度 (attention -3.8%) | 大 / online softmax をタイルループごとカーネル内に取り込む改造が要る |
| rel_table / pack_q の削減 | 5.00 s (2.1%) | 既に小さい | -2 s 程度 | 低 / 効果は限定的 |

- 上の 4 案が全部効いた場合の attention は約 195 s (現在 235.85 s、-17%)、prefill は約 631 s (現在 671.88 s、-6.1%) 相当。ただし案は独立ではなく、PV の形状変更と pack_kv の複製削除は同じレイアウト変更を共有する
- 「PV を速くする」が最短で最大。次点は pack_kv で、こちらは演算を変えずにバイト数だけ減らせる
- softmax は既に DRAM の 86% で、これ以上速くするにはバイトを減らすしかない。QK と融合して S を書かない (SIMT で QK を書き直す) 手もあるが、QK の 7.79 TFLOPS を捨てることになり、3-4 TFLOPS の自作カーネルでは差し引きで負ける可能性が高い

## 未確認

1. softmax が 298 GB/s (DRAM の 86%) で止まる内訳は ncu 無しでは確定できない。S/P の往復 99.4 MB と rel_table の再読 25.2 MB を合算して 86% になるという計算は合うが、実際に rel_table が DRAM まで降りているかは未確認。GPU ロック運用のため ncu は未使用
2. PV が QK より 26% 遅い理由 (M=128 のタイル効率か、出力の書き方か) は未確認。cublas のアルゴリズム選択を変えた比較はしていない
3. 512K は 128K の 14.5x (chunk 数 4x、1 chunk あたり 3.6x) で、理論の 16x より小さい。最終 chunk だけ見ると tile 数 4.0x に対し時間 3.82x で、長い K_t のほうが 1 tile あたりの効率がわずかに良い (未確認要因: K_t が大きいほど cublas の K 方向効率が上がる)
4. `attn=` 位相は eager launch のホスト側ギャップを含む。バイナリ間で ±2% 動くため、桁の小さい差 (rel_table など) は同一アーム内の比較でのみ意味がある
