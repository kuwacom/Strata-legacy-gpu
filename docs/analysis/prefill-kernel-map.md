# prefill-kernel-map: Qwen3.8-Flash-Next の prefill 1 チャンク

作業ツリー: `~/work/wt/codemap`（branch `docs/codemap`）
対象: `src/prefill/prefill.cpp`（7,779 行）、`src/prefill/gemm.cu`（1,496 行）、`src/kernels/cuda/*`

## 0. 前提

- §0 の訂正は `decode-kernel-map.md` と同じ。**本番モデル（`qwen4exp`）の prefill は `src/prefill/prefill.cpp`**。
  `src/qwen35/q36_engine.cu` は `qwen35moe` 用で、本番では使われない。
- 本番設定: `--prefill auto`、`--pack`、`--expert-cache auto`、`--kv int8`、`--kv-resident 32768`、
  `--max-context 131072`、`--ple-io ram`、`STRATA_DENSE_F32=1`。`--batch` は無い（`docker.json` で確認）。
- チャンク長: `--prefill auto` の決定は `src/program/generate.cpp` の chunk 決定節。
  手元にある本番ログ（2026-11-02 の ctx 32768 の run）では **`chunk auto: 8192 tokens`**（110 回）と
  `16384`（19 回）が出ている。131072 の現行設定でのチャンク長は **未確認**（コンテナ停止中のためログを追えず）。
- 既存の実測内訳（30K トークン、62.3 s）は `~/opt-work/src-build/report.md`:
  GDN 投影 GEMM 16.8% / QSA 13.6% / expert gate-up GEMM 13.4% / hc read 12.9%。
  4.8K では GDN 投影 18.5% / expert gate-up 14.9% / hc read 13.1%。

## 1. 1 チャンクの外枠

`Prefill::run_impl`（`src/prefill/prefill.cpp:3190`）。

1. `KvDrain` / `ple_drain` の RAII で DMA の完了を待つ（`prefill.cpp:3247-3252`）。
2. PLE 行の gather（`ple_gather`, `prefill.cpp:3309`）を **`std::async` で先に発行**し、layer 0 の計算と重ねる
   （`prefill.cpp:3487-3492`）。
3. `gr_broadcast`（embedding を 4 本の residual stream へ、`prefill.cpp:3483`）。
4. 層ループ `for (int64_t l = LB; l < LE; ++l)`（`prefill.cpp:3926`）。
5. 層ごとに `for (int half = 0; half < 2; ++half)`（`prefill.cpp:4005`）: hc_attn 半分 -> mixer（GDN/QSA）、
   hc_ffn 半分 -> MoE。
6. チャンク終了時に expert ring / staging の後始末、`pf_step` の計測点。

## 2. 層 1 回ぶんの列（両 half 共通の骨格）

`prefill.cpp:4005-4120` の half ループ。

| # | 呼び出し | file:line | 起動数（Pascal 経路） |
|---|---|---|---|
| 1 | `gr_norm_rs`（または `gr_norm`）: R の RMS norm + xn16 | prefill.cpp:4037-4040 -> native_gr_norm.cu:101 | 1 |
| 2 | hc down（BF16 GEMM）: `bf16_proj` -> `Gemm::bf16` | prefill.cpp:4055 -> gemm.cu:1043 | 3（bf16_to_f32(W) + bf16_to_f32(X) + cublasSgemm） |
| 3 | `gr_silu`（lo の SiLU） | prefill.cpp:4057 -> native_gr_postops.cu | 1 |
| 4 | hc up（BF16 GEMM）または `gr_upmix`（融合） | prefill.cpp:4070-4095 | 3（upmix 時 1） |
| 5 | hc inject（BF16 GEMM） | prefill.cpp:4097 | 3 |
| 6 | `gr_mix_r`（mixed = R への注入） | prefill.cpp:4103-4108 | 1 |

- 注意: **Pascal（sm_61）では BF16 GEMM が `bf16_path()==2`**（`gemm.cu:837-848`）で、
  重みを FP32 に変換（`bf16_to_f32`）-> 活性を FP32 に変換 -> `cublasSgemm` の 3 発になる。
  BF16 のままの `cublasGemmEx` は cc>=80 のみ。P40 では毎回この変換が走る（重みキャッシュなし）。
- `STRATA_HC_UPMIX` / `STRATA_PF_HCDOWN` / `STRATA_HCD_EXACT` は既存の融合スイッチ
  （`prefill.cpp:539-575`）。既定は upmix off、hcdown off（`pf_hcdown()` は env）。

## 3. GDN half（36 層）

`prefill.cpp:4109-4160`。

| # | 呼び出し | file:line | 備考 |
|---|---|---|---|
| 1 | `native_proj(attn_qkv)`: 2560 -> 10240 | prefill.cpp:4115 -> gemm.cu:1353 | dequant_f16 1 + f16_to_f32(W) 1 + f16_to_f32(X) n_sl + sgemm n_sl |
| 2 | `native_proj(attn_gate)`: 2560 -> 6144 | prefill.cpp:4123 | 同上 |
| 3 | `gdn_gates`（dt / ssm_a のゲート） | prefill.cpp:4127 | 1 |
| 4 | `gdn_conv`（conv のみ、または `gdn_conv_recurrence_fused`） | prefill.cpp:4133/4136 | 1（融合時 1） |
| 5 | `gdn_recurrence`（chunk 内の delta rule + out norm） | prefill.cpp:4142 | 1（chunk カーネル） |
| 6 | `native_proj(ssm_out)`: 6144 -> 2560 | prefill.cpp:4146 | 同上 |

- 3 つの投影がそれぞれ `dequant_f16` + FP32 変換 + cuBLAS。`STRATA_DENSE_F32=1` の分岐は
  `gemm.cu:1195-1215`。`n_sl = ceil(T*K / 16M)` スライス（`gemm.cu:1200`）。
  8192 トークン・K=2560 で T*K = 21M -> `n_sl = 2`。
- 変換バッファが足りない初回は `cudaStreamSynchronize` が入る（`gemm.cu:1204`）。

## 4. QSA half（12 層）

`prefill.cpp:4155-4440`。

| # | 呼び出し | file:line |
|---|---|---|
| 1 | `native_proj(attn_k)`: 2560 -> 512 | prefill.cpp:4167 |
| 2 | `native_proj(attn_v)`: 2560 -> 512 | prefill.cpp:4168 |
| 3 | `native_proj(attn_q)`: 2560 -> 12288 | prefill.cpp:4170 |
| 4 | `bf16_proj(indexer.k_proj)`: 2560 -> 128 | prefill.cpp:4172 |
| 5 | `bf16_proj(indexer.q_proj)`: 2560 -> 512 | prefill.cpp:4173 |
| 6 | `rms_rope_rows`（norm + rope 融合、`STRATA_QSA_FUSE_NR`）または norm + `rope` | prefill.cpp:4175-4179 |
| 7 | KV の stage / append（`--kv-resident` で host から stage） | prefill.cpp:4183-4213 |
| 8 | indexer の append + pool（`kv_stage_from_host` と並行） | prefill.cpp:4213-4300 |
| 9 | scores + select（`qsa_block_scores` / `qsa_block_topk` 系） | prefill.cpp:4300-4400 |
| 10 | `qsa_prompt_attn_batch_gated` / `qsa_prompt_attn_batch` / `qsa_pattn_batch` | prefill.cpp:4407/4420/4424 |
| 11 | `gate_attn`（gate 融合が効かない場合） | prefill.cpp:4437 |
| 12 | `native_proj(attn_output)`: 6144 -> 2560 | prefill.cpp:4438 |

- プロンプト attention は **T 行をまとめて 1 回**（`qsa_prompt_attn_batch` 系）。decode の
  `qsa_decode_attn_step` とは別カーネル。
- コメント（`prefill.cpp:4401`）: 融合 gate 版は「accuracy は同等だが bitwise ではない」。
  既定は `qsa_fuse_gate()`（`prefill.cpp:576`）次第。

## 5. MoE half（全 48 層）

`prefill.cpp:4443-5940`。

| # | 呼び出し | file:line |
|---|---|---|
| 1 | `bf16_proj(ffn_gate_inp)`: 2560 -> 512（router、T 行） | prefill.cpp:4450 |
| 2 | `route`（GPU top-k、ids/weights を host へ） | prefill.cpp:4451 |
| 3 | host で (token, k) を expert ごとに grouping | prefill.cpp:4671-4700 |
| 4 | shared expert: gate/up/down の 3 GEMM + scalar gate（bf16 1x1 GEMM） | prefill.cpp:4513-4520 |
| 5 | 専門家の gate/up: `mmq::` 系（`mctx->run` 相当）または `ws::gate_up` / `fused::experts_native` | prefill.cpp:5300-5828 |
| 6 | SwiGLU（`mmq::swiglu` / `swiglu_quant` / `swiglu_interleaved`） | prefill.cpp:5321,5707,5826 |
| 7 | 専門家の down | prefill.cpp:5676-5828 |
| 8 | `moe_combine`（parts × weights + shared + sg） | prefill.cpp:5936 |
| 9 | CPU share（`STRATA_PREFILL_CPU_SHARE`）: `cpu_fut` の行を Dm の末尾へ | prefill.cpp:5900 |
| 10 | peer（layer split）: `moe_combine_peer` / `peer16` | prefill.cpp:5932-5934 |

### 専門家の 3 経路（既存スイッチで選択）
- **MMQ**（既定、`use_mmq`）: 重みを int8 活性で読む。`src/prefill/moe_mmq.cu:472 Context::run` が 1 発。
  gate/up を 1 回、SwiGLU、量子化、down を 1 回 = 層あたり 2 run + 2-3 kernel。
- **STRATA_PF_FUSED=1**（`moe_fused.hpp` / `moe_fused_iq.hpp`）: GPU 側で grouping、`fused::experts_native`
  1 発（`prefill.cpp:4652-4659`）。host sync なし。
- **CPU share**（`STRATA_PREFILL_CPU_SHARE`）: 非 resident の一部を CPU pool で計算（`prefill.cpp:4739-4795`）。

### expert の streaming（`--expert-cache auto` + host arena）
- `Stager`（`prefill.cpp:656`）、ring（`prefill.cpp:249-441`）、`cudaMemcpyAsync` を `prefill.cpp:3609` で発行。
- `stream_all_min()`（既定 1024、`prefill.cpp:255`）以上のチャンクは **全 expert を stream** する。
  8192 トークンのチャンクはこれに該当する（`stream_all`）。
- 1 層あたり最大 512 expert × 48 層の H2D コピーが発生しうる（全 expert を stream する場合）。
  実測では expert copy 待ち 0.1%（report）。

## 6. GEMM の形状と経路

| 投影 | 形状 (N x K) | 重みの型 | 経路（Pascal） | 1 回の起動数 |
|---|---|---|---|---|
| attn_qkv | 10240 x 2560 | IQ4_XS/Q3_K/Q2_0/Q4_K（層で混在） | `Gemm::native` -> dequant_f16 + f16_to_f32(W) + f16_to_f32(X)×n_sl + cublasSgemm×n_sl | 2 + 2·n_sl |
| attn_gate | 6144 x 2560 | Q3_K/IQ4_XS/Q4_K/Q2_0 | 同上 | 2 + 2·n_sl |
| ssm_out | 2560 x 6144 | 同上 | 同上 | 2 + 2·n_sl |
| attn_q | 12288 x 2560 | Q3_K/IQ4_XS/Q2_0 | 同上（31.5M 要素、GEMM_SCRATCH 32M に収まる） | 2 + 2·n_sl |
| attn_k / attn_v | 512 x 2560 | K-quant | 同上 | 2 + 2·n_sl |
| attn_output | 2560 x 6144 | K-quant | 同上 | 2 + 2·n_sl |
| indexer.q_proj | 512 x 2560 | BF16 | `Gemm::bf16` -> bf16_to_f32×2 + sgemm | 3 |
| indexer.k_proj | 128 x 2560 | BF16 | 同上 | 3 |
| ffn_gate_inp | 512 x 2560 | BF16 | 同上 | 3 |
| hc down | 320 x 2560 | BF16 | 同上 | 3 |
| hc up | 2560 x 320 | BF16 | 同上（upmix 時 1） | 3 |
| hc inject | 4 x 2560 | BF16 | 同上 | 3 |
| shexp gate/up | 640 x 2560 | Q2_0 / K-quant | `Gemm::native` | 2 + 2·n_sl |
| shexp down | 2560 x 640 | IQ4_NL/Q4_0/Q5_0/Q8_0/Q2_0 | 同上（640 は Q8_K 不可 -> Q8_0 契約） | 2 + 2·n_sl |
| expert gate/up | 1280 x 2560 | IQ3_XXS 等 | MMQ / fused | 1 run |
| expert down | 2560 x 640 | IQ3_S 等 | MMQ / fused | 1 run |
| output (head) | 248320 x 2560 | Q5_K | decode 側（prefill は最後の 1 行だけ） | - |

- `Gemm::native` の本体: `gemm.cu:1353-1496`。スクラッチは 32M 要素（`GEMM_SCRATCH`）。
  重みが収まらないときは行スライス（`gemm.cu:1465-1476`）。
- `STRATA_PF_W16_CACHE_MIB`（`gemm.cu:1427-1490`）: dequant 済み FP16 重みのキャッシュ。
  auto default は `--batch >= 2` のときだけ on（`generate.cpp:8180`）。**現行プリセットは `--batch` が無いので off**
  （`gemm.cu:1429` で `w16_budget_ = 0`）。有効化した場合は `STRATA_DENSE_F32` の分岐が毎回
  `f16_to_f32(W)` をやり直す点に注意（`gemm.cu:1206`）。
- `STRATA_DEQUANT_SIDE=1`（`gemm.cu:1477-1495`）: dequant を side stream に逃がす。既定 off。
- cuBLAS と自前カーネルの使い分け: **CUDA では dense 投影はすべて cuBLAS**（SGEMM / HGEMM）。
  自前なのは dequant / 変換 / norm / rope / attention / expert（MMQ）/ combine。
  HIP では `wmma_gemm` / hipBLASLt / rocBLAS の分岐があるが本番は CUDA。

## 7. 起動回数の見積もり（**推定**、1 チャンク = 8192 トークン、n_sl=2）

| 区分 | 1 層あたり | 48 層 |
|---|---|---|
| hc（2 half × 6 GEMM 経路） | 2 × (norm 1 + down 3 + silu 1 + up 3 + inject 3 + mix 1) = 24 | 1,152 |
| GDN 投影 3 本（各 2 + 2·n_sl = 6） | 18 | 648（36 層） |
| GDN の conv / gates / recurrence | 3 | 108（36 層） |
| QSA 投影 4 本（各 6） | 24 | 288（12 層） |
| QSA の indexer / scores / select / attn / gate | 約 8 | 96（12 層） |
| router（bf16 3） | 3 | 144 |
| shared expert（3 × 6 + gate 3） | 21 | 1,008 |
| 専門家（MMQ: 2 run + swiglu + quant + down 補助） | 約 5 | 240 |
| combine | 1 | 48 |
| **合計（推定）** | | **約 3,700 kernel 起動 / チャンク** |
| expert streaming の H2D（stream_all 時） | 最大 512 | 最大 24,576 copy |

- この数字は §2-§5 の表から機械的に足した **推定**。実測のカーネル数は取っていない。
- 30K トークンを 8192 チャンクで割ると 4 チャンク弱 -> 1 リクエストで約 15,000 起動（推定）。
- 参考: 62.3 s / 30K トークン = 2.08 ms/token。起動オーバーヘッドが 5-10 µs/発なら
  3,700 × 4 = 14,800 発で 74-148 ms/リクエスト（全体の 0.1-0.2%）。**prefill の律速は起動数ではなく GEMM と帯域**。

## 8. ホットスポット（既存レポートの実測値）

| 順位 | 30K（62.3 s） | 4.8K（9.9 s） |
|---|---|---|
| 1 | GDN 投影 GEMM 16.8% | GDN 投影 GEMM 18.5% |
| 2 | QSA attention 13.6% | expert gate/up GEMM 14.9% |
| 3 | expert gate/up GEMM 13.4% | hc read 13.1% |
| 4 | hc read 12.9% | - |

- プロンプトの GPU 時間は実時間とほぼ同じ（62,285 / 62,425 ms）。expert copy 待ちは 0.1%。
- したがって prefill の改善は (a) GDN/QSA の投影 GEMM の効率、(b) expert gate/up の GEMM、
  (c) hc read の 3 点に集中する。起動削減の効果は限定的（§7）。

## 9. ストリーム・コピー・同期（prefill）

| 項目 | file:line | 内容 |
|---|---|---|
| PLE の非同期 gather + H2D | prefill.cpp:3487-3520 | `std::async` + `cudaMemcpyAsync`。`ple_copy` ストリームと `ple_copied` イベントで compute stream と同期 |
| PLE の `cudaEventSynchronize` | prefill.cpp:3531 | 次のバッファの再利用待ち（2 つ先の gather を待つ） |
| expert staging の ring | prefill.cpp:656-855, 3609 | `cudaMemcpyAsync` を copy stream へ。`ev_copy` / `ev_used` で順序付け |
| KV の host -> VRAM stage | prefill.cpp:4192 | `kv_stage_from_host` + `kv_ready` イベント |
| KV の DMA（`STRATA_KV_HOST_DMA`） | prefill.cpp:4204-4213 | 既定 off（`constexpr bool kv_host_dma = false`） |
| GEMM 変換バッファの grow 時 sync | gemm.cu:1204 | 初回のみ |
| `pf_step` の計測点 | prefill.cpp:3928 ほか | `STRATA_PREFILL_TIMING` |
| チャンク末尾の同期 | prefill.cpp:6059-6100 | `STRATA_HANDOFF_PIPE` 時は host が chunk の完了だけ待つ |

## 10. 未確認
- 現行 `--max-context 131072` での実チャンク長（ログ未取得。旧 run は 8192/16384）。
- `STRATA_PF_W16_CACHE_MIB` は現行プリセット（`--batch` なし）では off（`generate.cpp:8180` の条件）。
  有効化した場合の効果は未測定。
- `qsa_fuse_gate()` / `qsa_fuse_nr()` の既定値（`prefill.cpp:576-582` は env 依存。本番 env 未確認）。
- 専門家経路が MMQ か PF_FUSED か（`--pack` の native layout と `use_mmq` の組合せ。ログの
  "prompt experts on ..." 行は旧 run に無い）。
- 起動回数の実測（`nsys` / `nvprof` はエンジンに対して未実行。report にも「カーネル単位の時間は取っていない」とある）。
