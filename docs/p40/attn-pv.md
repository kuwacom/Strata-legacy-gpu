# Inkling タイル attention の PV SGEMM 形状実験 (sm_61 / FP32 SGEMM 経路)

- 日時: 2026-11-25
- 対象: `~/work/ink-ap` (branch `perf/ink-attn-pv`、base `perf/ink-attn-prof` `1893427af`)、`engine/src/inkling2/cuda/ink_runner.cuh` のタイル経路
- バイナリ: `build/src/inkling2/strata-inkling` sha256 `b8bde829…`
- ホスト: <host> / Tesla P40 24GB (sm_61, 1531 MHz) / SQ-10 / 本番は停止 -> 復旧済み (healthy、`.env` sha256 `(省略)` 不変)
- 前提: FP32 のまま。K/V のコードとスケール、softmax・マスク・sink の意味は変えていない

## 結論

- 現行の PV は `m=128 (head_dim) x n=256 (q 行) x k=K_t、batch=24`。同じ FLOP の QK (`m=K_t x n=256 x k=128`) より遅い主因は **M の小ささではなく、GQA 3 複製ぶんの V を 3 回読んでいること**だった
- **SHAPE=2 (KV head 8 本を strided batch にまとめ、n=3C=768、batch=8) が有効**: PV 80.5 s -> 71.2 s (**-11.6%**)、attention 235.8/236.1 s -> 226.8 s (-3.9%)、**128K prefill 194.8/195.2 -> 197.8 ppts (+1.4%、run-to-run 0.2% を超える)**
- SHAPE=1 (m/n 入れ替えで O^T を返す形) は **不成立**: PV 84.7 s (+5.1%)、prefill 193.7 ppts (-0.6%)。転置出力を通常レイアウトに戻す 1 パスを含めても、GEMM 単体で既に 2% 遅い
- 数値: PV 出力は同一入力で SHAPE=1/2 とも **bit 一致 (max rel err 0、RMS rel err 0)**。QK / softmax / pack_kv のステージ時間は 4 アームで ±0.4% 以内 (PV 以外は不変)。4K プロンプトの生成 8 token と 32K needle の生成 16 token も 3 形状で同一
- 仮説の切り分け (cublas 直叩き): CTA 数を 48 -> 96 に増やしても効率は変わらず (7.09 -> 7.18 TFLOPS)、K 分割 (beta=1) は悪化 (6.90 / 6.38)、batch を減らすと悪化 (b12 で 6.03)。**V の読みを 1/3 にした b8n768 だけが +10% (7.80)**

## 現行の呼び出し形状 (コードから)

`cublasGemmStridedBatchedEx`、算術型は全て `CUBLAS_COMPUTE_32F` / `CUDA_R_32F`、algo は `CUBLAS_GEMM_DEFAULT` (sm_61 は Tensor Core 無し)。P40 の FP16 は FP32 の 1/64 なので K/V/Q/P の保存型も FP32 (`pfd == 2`)。

| 段 | transA/B | m | n | k | lda / strideA | ldb / strideB | ldc / strideC | batch |
|---|---|---|---|---|---|---|---|---|
| QK | OP_T / OP_N | K_t | C (256) | 128 | 128 / K_t x 128 | 128 / M x 128 | K_t / C x K_t | 24 |
| PV (shape 0) | OP_N / OP_N | 128 | C (256) | K_t | 128 / K_t x 128 | K_t / C x K_t | 128 / C x 128 | 24 |
| PV (shape 1) | OP_T / OP_T | C (256) | 128 | K_t | K_t / C x K_t | 128 / K_t x 128 | C / C x 128 | 24 |
| PV (shape 2) | OP_N / OP_N | 128 | 3C (768) | K_t | 128 / 3 x K_t x 128 | K_t / 3 x C x K_t | 128 / 3 x C x 128 | 8 |

- QK は A=K (`[24, K_t, 128]`)、B=Q (`[24, M, 128]`、q0 のオフセット付き)、C=S (`[24, C, K_t]`)。128K 末尾で 1 呼び出しあたり 16 x 2 x 24 = **768 CTA**
- PV は A=V (`[24, K_t, 128]`)、B=P (`[24, C, K_t]`)、C=O (`[24, C, 128]`)。同じく **48 CTA** しか出ない (128x128 tile 換算)
- P は head ごとに `[C, K_t]` が連続 (softmax の書き方)、V は GQA 3 複製 (`pack_kv` が `3*kh + rep` の 3 本に同じ値を書く)

## 仮説と切り分け (cublas 直叩き、検証専用コードは未コミット)

`~/work/results/attn-pv/exp/pvbench{,2,3,4}.cu`。P40 の実効上限は 4096³ SGEMM で **9.50 TFLOPS**、QK 形状で 7.98 TFLOPS。

| 案 | 形状 | ms/呼び出し | TFLOPS | 判定 |
|---|---|---:|---:|---|
| base (=shape 0) | m128 n256 k2048 b24 (48 CTA) | 0.454 | 7.09 | 基準 |
| q512 (Q ブロック統合) | m128 n512 k2048 b24 (96 CTA、2 倍仕事) | 0.897 | 7.18 | 並列度を上げても伸びない |
| b48 (batch 2 倍) | m128 n256 k2048 b48 (96 CTA、2 倍仕事) | 0.973 | 6.62 | 同上、むしろ微減 |
| b12 (batch 半分) | m128 n256 k2048 b12 (24 CTA、半分仕事) | 0.267 | 6.03 | CTA 不足は確かに痛いが 48 で足りている |
| b1n6144 (全 head を 1 GEMM に) | m128 n6144 k2048 b1 (48 CTA) | 0.448 | 7.20 | 48 CTA でも 7.2 出る = 並列度は律速でない |
| ksplit2 / ksplit4 | k1024 x2 / k512 x4 (beta=1) | 0.467 / 0.505 | 6.90 / 6.38 | K 分割は C の read-modify-write で負ける |
| swap (shape 1) | m256 n128 k2048 b24 (opA=T, opB=T) | 0.462 | 6.97 | 2% 遅い |
| **b8n768 (shape 2)** | m128 n768 k2048 b8 (48 CTA) | **0.412** | **7.81** | **+10%。V の読みが 1/3 (75 -> 58 MB/呼び出し)** |

K_t 依存 (base 形状、pvbench2): k=256 6.11 / 512 6.72 / 1024 6.22 / **1536 3.98** / 2020 6.02 / 2048 7.12 TFLOPS。k が 2048 の倍数でないと cublas のカーネル選択が変わり、1536 は特に悪い。SWA 経路 (K_sub ~ 1278) の PV が 4.3 TFLOPS と低いのはこのため (PV 全体の 12%)。

### 原因の結論

- **M=128 / CTA 48 本は律速ではない** (b1n6144 / q512 / b48 が同効率)。M=128 が小さいという当初仮説は棄却
- **V の 3 重読みが律速**。PV の転送は 75 MB/呼び出し (P 50 + V 25 + O 3) で、V の 2/3 は GQA 複製ぶんの重複。b8n768 で 58 MB に減り +10%
- 転置 (ldb) 仮説も棄却: shape 1 は 2% 遅い

## 変更 (env 1 本、既定 0 = 現行)

| ファイル | 箇所 | 内容 |
|---|---|---|
| `ink_runner.cuh` | `:1834` | `int pf_pv_shape = 0;` |
| 同上 | `:2216` | `STRATA_INK_PV_SHAPE=0/1/2` の読み取りと 1 行ログ (既定 0、未設定なら従来と完全に同じ経路) |
| 同上 | `:3467-3496` | PV の呼び出しを `pf_pv_gemm(C, K_t, vstride, V, P, O)` に集約し、3 形状を分岐 |
| 同上 | `:3522, :3562` | 短い経路 / 長い経路の PV を差し替え。shape 1 のときだけ転置を 1 回はさむ |
| `ink_attn.cuh` | `:1097` | `k_ink_pf_ot_to_o` (shape 1 の O^T `[24,128,C]` -> O `[24,C,128]`、shared 32x129 のタイル転置) |

- 追加カーネルは shape 1 専用の転置 1 本のみ。scratch の追加はゼロ (長い経路は softmax が読み終えた `d_pf_s`、短い経路は未使用の `d_pf_ot` を作業先に使う)
- 既存 env (`STRATA_INK_PF_TILE` / `STRATA_INK_PROF` / `STRATA_INK_ATTN_PROF`) の意味は変えていない

## 数値 (PV 出力の比較)

同一入力・同一プロセスで shape 0 と shape 1/2 の PV を両方実行し D2H 比較 (`pvbench3.cu`、V は GQA 3 複製を同一値、P は exp 値、K_t=2048):

| 比較 | max rel err (max\|O\| 基準) | RMS rel err | 備考 |
|---|---:|---:|---|
| SHAPE=1 vs 0 | **0.000e+00** | **0.000e+00** | bit 一致 |
| SHAPE=2 vs 0 | **0.000e+00** | **0.000e+00** | bit 一致 |
| 同形状 (基準の再現性) | 0.000e+00 | 0.000e+00 | bit 一致 |

- 合格条件 (max rel err <= 1e-5) に対して余裕で合格。累算順が変わらないため差が出ない
- K_t=2020 の比較は検証コード側の不備 (P を stride 2048 で確保したまま ldb=2020 を渡した) で無効。取り直し用の `pvbench4.cu` は用意したが、後述のエンドツーエンド一致で代替した
- PV 以外のステージ: QK / softmax / pack_kv / accum のカーネルと引数は変更なし。アーム間のステージ時間も QK 59.5-60.0 s、softmax 55.6-55.7 s、pack_kv 29.7-29.8 s と一致 (PV だけが動く)
- エンドツーエンド: 4K プロンプト (chunk 512、`--serve` の GEN) の生成 8 token が shape 0/1/2 で同一。32K needle の生成 16 token も shape 0/2 で同一

## 性能 (128K FP16 KV、256 chunk、各 1 本、アーム間 60 s 待機)

`STRATA_INK_PF_TILE=1 STRATA_INK_PROF=2 STRATA_INK_ATTN_PROF=1`、`--max-context 131072 --kv fp16 --prompt-len 131072 --gen 16 --chunk 512 --draft off`。

| アーム | PV_SHAPE | prefill ppts | prefill s | attn s | PV s | PV global | PV SWA | QK s | softmax s | pack_kv s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sh0 | 0 | 194.8 | 673.02 | 236.09 | 80.6 | 71.16 | 9.48 | 59.7 | 55.7 | 29.7 |
| sh1 | 1 | 193.7 | 676.59 | 240.73 | 84.7 | 73.64 | 11.06 | 60.0 | 55.7 | 29.8 |
| **sh2** | **2** | **197.8** | **662.54** | **226.77** | **71.2** | **62.02** | **9.19** | 59.8 | 55.7 | 29.7 |
| sh0b | 0 | 195.2 | 671.53 | 235.59 | 80.4 | 70.96 | 9.47 | 59.5 | 55.6 | 29.7 |

- SHAPE=0 の重複 (sh0 / sh0b): prefill 194.8 / 195.2 (**±0.2%**)、PV 80.6 / 80.4 s (±0.25%)。**SHAPE=2 の prefill +1.4% と PV -11.6% はこのばらつきを超える**
- PV の内訳 (ms/chunk、序盤/中盤/末尾): sh0 131.19 / 316.23 / 499.69 -> sh2 118.18 / 279.30 / 438.93 (-9.9% / -11.7% / -12.1%)。文脈が長いほど効く (V の読みが支配的になるため)
- PV の attention 占率 34.16% -> 31.41%、prefill 占率 12.0% -> 10.7%。PV の実効は 5.9 -> 6.8 TFLOPS (global 経路の呼び出しあたり 0.535 -> 0.466 ms、C=256 / K_t=2020 の平均。277.95 -> 242.27 ms/chunk = -12.8%)
- SHAPE=1 は逆効果 (PV +5.1%、SWA 経路は +16.7%)。転置 1 パス (6.2 MB/呼び出し) に加えて、k=1278 の SWA 形状で cublas が不利なカーネルを選ぶ
- decode は不変 (12.4 / 12.5 tps)。GPU は全アーム 1531 MHz、72-74 C、63-65 W で throttle 無し

### 数値の整合

- 切り分け用マイクロベンチの base は k=2020 で 0.528 ms/呼び出し、エンジン実測は 0.535 ms (C=256、K_t=2020 の平均)。GEMM 単体とエンジン内でほぼ同じ
- 同じく b8n768 は 0.493 ms (単体) に対しエンジン 0.466 ms。エンジン側が単体より速いのは、直前の softmax が書いた P の一部が L2 に残るためと推測 (未確認)

## needle (32K)

`~/work/logs/needle-32k.ids` (深度 49.9%) を `--serve` の `GEN 16` に流し、生成 id を `needle.py` でデコード。

| アーム | 生成 16 token のデコード | 判定 |
|---|---|---|
| sh0 | `7741` を含む (ORCHID-7741) | 通過 |
| sh2 | sh0 と同一 id | 通過 |

## 未確認・注意

1. K_t が 2048 でないときの cublas のカーネル選択 (k=1536 で 3.98 TFLOPS) は未解明。SWA 経路の PV (K_sub ~ 1278、PV の 12%) は 4.3 TFLOPS で、ここには手を付けていない
2. QK に同じ束ね (K を 1 batch で読む) を適用する案は、`n=3C` の B に 3 head ぶんの Q を並べる必要があり、Q の head ストライドが M (chunk 全体) なので C == M (Q_CH=512) にしないと表現できない。未測定
3. `pack_kv` の GQA 3 複製 (29.7 s、12.6%) は shape 2 でも残る。V の複製書きをやめれば同じ理屈でさらに減るが、S/P のレイアウト変更を伴う
4. SHAPE=2 の既定化 (env 既定 0 -> 2) はしていない。sm_70+ の BF16/FP16 経路でも同じ形状が効くかは未測定 (このブランチの分岐は pfd に依らず shape 2 を選ぶので、そのまま試せる)
5. 計測は `STRATA_INK_PROF=2` (eager) と `STRATA_INK_ATTN_PROF=1` の上で取った。4 アームすべて同じ条件なので相対比較は有効
