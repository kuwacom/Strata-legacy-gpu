# GDN 投影 GEMM の prefill 高速化（P40 / sm_61）

作業ツリー: `~/work/wt/pgdn`（branch `perf/prefill-gdn`、fork HEAD `9f7d977d`）
対象: `src/prefill/gemm.cu`、`src/prefill/prefill.cpp`、`src/prefill/moe_mmq.cu`、`src/prefill/mmq_kq_lib.cu`、`CMakeLists.txt`
計測: `~/work/results/pgdn/`（生ログ）。数値は実測、または「推定」と明記した見積り

## 0. 結論

- 本番経路（`STRATA_DENSE_F32=1`）の GDN 投影は **FP16 重みを FP32 に widen して `cublasSgemm`**。実測は 8.6-8.8 TFLOPS で、P40 の FP32 ピーク 11.8 TFLOPS の 73%。cuBLAS 自身の最良カーネルも 9.0-9.6 TFLOPS なので、**cuBLAS の調整で取れるのは数 % だけ**（実測: 全 24 アルゴリズムを試して qkv/gate は既定が最速、`ssm_out` だけ ALGO6 が -17.7%）。
- 桁で効くのは **int8 DP4A の MMQ**（P40 の DP4A ピーク 47 TOPS = FP32 の 4 倍）。llama.cpp も Pascal では MMQ を既定で使う（`ggml_cuda_should_use_mmq`: `!fp16_mma_hardware_available` で true）。
- 実装: HIP 専用だった dense MMQ（`Gemm::native_mmq`）を CUDA でも opt-in（`STRATA_DENSE_MMQ=1`）で使えるようにし、Q3_K の MMQ インスタンスを `strata_kq` に足した。
- 結果: **prefill 24K +13.5% / +12.9%、128K +11.7% / +12.1%（A-B-A 中央値）**。GDN 位相は -38.5%、`ssm_out` 位相は -36.9〜-38.8%。TTFT は 24K で -6.6 s、128K で -35〜-36 s。
- 品質: 4K プロンプト最終位置の KL 0.0015 / 0.0076（top-1 一致）、回答は 4 問中 2 問がバイト一致・残り 2 問は同内容の言い換え。bitwise ではない。
- 判定: **採用推奨（opt-in）**。既定は変えない（`STRATA_DENSE_MMQ=1` + `STRATA_MMQ_KQ=1` のときだけ）。

## 1. 形状と呼び出し回数（実コード）

GDN 層は 36 層。1 層あたり 4 本の投影（`prefill.cpp:4109-4146` の half 0 の分岐）。

| 投影 | 形状 N x K | 呼び出し | 1 層の回数 |
|---|---|---|---|
| `attn_qkv.weight` | 10240 x 2560 | `native_proj` -> `Gemm::native`（`prefill.cpp:4115`、`prefill.cpp:2948-2955`） | 1 |
| `attn_gate.weight` | 6144 x 2560 | `native_proj`（`prefill.cpp:4123`） | 1（KVA 時のみ skip、既定 off） |
| `ssm_alpha.weight` | 48 x 2560（`ldy` 96） | `bf16_proj`（`prefill.cpp:4124`） | 2（BF16X2 の hi + lo） |
| `ssm_beta.weight` | 48 x 2560（`ldy` 96） | `bf16_proj`（`prefill.cpp:4125`） | 2 |
| `ssm_out.weight` | 2560 x 6144 | `native_proj`（`prefill.cpp:4146`） | 1 |

- `ssm_v_heads = 48`（`src/core/layer.cpp:316` のコメント）。alpha/beta は N=48 の極端に細い GEMM で、`bf16()` の Pascal 分岐（`gemm.cu:1043` の path 2）が毎回 W と X を FP32 に widen して `cublasSgemm` する。
- 「GDN 投影 GEMM」の位相（`kPfGdn`、`prefill.cpp:4112` -> `kPfGdnConv`、`prefill.cpp:4126`）には **qkv + gate + alpha/beta 4 発 + gdn_gates** が入る。`ssm_out` は別位相（`kPfGdnOut`、`prefill.cpp:4144`）。
- 1 チャンク（8192 トークン）あたりの GDN 投影 GEMM は 36 層 × 4 本 = 144 発（qkv/gate/out の 3 本 + alpha/beta 4 発のうち MMQ が効くのは 3 本）。

## 2. 現在の経路（Pascal, `STRATA_DENSE_F32=1`）

`Gemm::native`（`gemm.cu:1353`）:

1. スクラッチ（32M 要素 = 64 MiB, `GEMM_SCRATCH`）に `dequant_f16` で重みを FP16 へ（`gemm.cu:1496` 付近）。
2. `Gemm::f16`（`gemm.cu:1171`）へ。`dense_f32_on()`（`gemm.cu:885`）と `dense_f32_min_t(N,K)`（`gemm.cu:900`）が真なら:
   - `f16_to_f32(W)` で**重み全体**を FP32 の `tc_w_` に widen（`gemm.cu:1207`）。
   - 活性を `kXSliceElems = 16M` 要素（64 MiB as fp32）ずつ FP32 の `tc_x_` に widen し、`cublasSgemm`（`gemm.cu:1211-1219`）。
3. `dense_f32_min_t` は qkv 10240x2560 / qsa q 12288x2560 が T>=256、gate 6144x2560 が T>=512、out 2560x6144/4096 が T>=4096。それ以外の形（alpha/beta の N=48 を含む）は FP16 の `cublasGemmEx` 経路（`gemm.cu:1226-1250`）。
4. `bf16_path()`（`gemm.cu:837`）は sm_61 で 2（FP32 経由）。BF16 のままの `cublasGemmEx` は cc>=80 のみ。

- 重みの widen は**毎チャンク・毎層・毎投影**。FP32 像のキャッシュはない（FP16 キャッシュ `STRATA_PF_W16_CACHE_MIB` は既定 off、`gemm.cu:1416-1475`）。
- 実測（`bench/results/2026-10-12-p40-sm61-kernels/`）: FP32 経路 8.4-9.8 TFLOPS、FP16 経路は同じ形で 4.3-7.6 TFLOPS と不安定。

## 3. 実測

機材: Tesla P40 24GB（sm_61）、prod と同じデータボリューム、`CONTEXT=512k`、`STRATA_SET=ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2`、`--spec 4` は Terminal の既定。
`BENCH_SEED=42`、`max_tokens=1`（TTFT = prefill）、各セル 1 本目除外・中央値。画像は `strata:fork-9f7d977-base`（base/base2）と `strata:fork-prefill-gdn`（mmq、`STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1`）。

### 3.1 A-B-A（prefill tok/s と TTFT）

`~/work/results/pgdn/summary.txt`（`summarize_prefill.py`）:

| サイズ | kind | base | base2 | **mmq** | mmq vs (base/base2 中央値) |
|---|---|---:|---:|---:|---:|
| 24K | code | 487.7 (57.50 s) | 482.2 (58.17 s) | **550.7 (50.92 s)** | **+13.5%** |
| 24K | prose | 486.4 (52.31 s) | 485.9 (52.38 s) | **549.0 (46.36 s)** | **+12.9%** |
| 128K | code | 424.0 (331.36 s) | 424.5 (331.00 s) | **473.9 (296.46 s)** | **+11.7%** |
| 128K | prose | 423.9 (337.74 s) | 423.9 (337.75 s) | **475.2 (301.30 s)** | **+12.1%** |

- A と A2 の差は -1.1% / -0.1% / +0.1% / 0.0% で、5% 閾値の内側。mmq の差は閾値の外。
- TTFT の短縮: 24K code -6.58 s / prose -5.95 s、128K code -34.9 s / prose -36.4 s。

### 3.2 GDN 位相の内訳（`STRATA_PREFILL_TIMING=1`）

同じプロンプト（24K は 30,240、128K は 135,065 トークン）の 1 回目:

| prompt | 位相 | base | mmq | 差 |
|---|---|---:|---:|---:|
| 30,240 | GPU timeline | 60,296 ms | 53,199 ms | -11.8% |
| 30,240 | gdn（qkv+gate+ab+gates） | 10,431 ms (17.3%) | 6,420 ms (12.1%) | **-38.5%** |
| 30,240 | gdn out proj | 4,111 ms (6.8%) | 2,595 ms (4.9%) | **-36.9%** |
| 30,240 | qsa proj | 4,490 ms (7.4%) | 2,937 ms (5.5%) | -34.6% |
| 135,065 | GPU timeline | 311,612 ms | 279,166 ms | -10.4% |
| 135,065 | gdn | 46,491 ms (14.9%) | 28,711 ms (10.3%) | **-38.3%** |
| 135,065 | gdn out proj | 19,003 ms (6.1%) | 11,626 ms (4.2%) | **-38.8%** |
| 135,065 | qsa proj | 20,338 ms (6.5%) | 13,157 ms (4.7%) | -35.3% |

- GDN 位相の実効: base 8.64 TFLOPS（24K）/ 8.77 TFLOPS（128K）-> mmq 14.0 / 14.2 TFLOPS（推定: FLOPs は同じ）。DP4A ピーク 47 TOPS の約 30%。
- `qsa attn`・`qsa select`・expert GEMM・`hc read` は mmq でもほぼ同じ（MMQ は dense 投影だけ）。

### 3.3 cuBLAS FP32 アルゴリズムの全数スイープ（`algo-sweep-T8192.txt`）

`cublasGemmEx`（CUDA_R_32F / CUBLAS_COMPUTE_32F）の ALGO0..23 を T=8192 で。`cublasSgemm` 既定との比:

| 形状 | 既定 | 最良 |
|---|---:|---|
| qkv 10240x2560 | 47.68 ms / 9.01 TFLOPS | ALGO12 48.65 ms（-2.0%、悪化） |
| gate 6144x2560 | 28.23 ms / 9.13 TFLOPS | ALGO12 27.74 ms（-1.8%） |
| out 2560x6144 | 32.67 ms / 7.89 TFLOPS | **ALGO6 26.88 ms / 9.59 TFLOPS（-17.7%）** |

- 既定の `cublasSgemm` が qkv/gate では最良。`ssm_out` だけ ALGO6 が明確に速い（が、MMQ は同じ形を -38.8% にするので優先度は低い）。
- qkv+gate を 1 本の N=16384 GEMM に連結すると 77.6 -> 81.1 ms（+4.5%、悪化）。
- ALGO は加算順が変わるので、既存の `STRATA_DENSE_F32=1` の bitwise 性（`gemm_f32_parity`）は失われる。既定経路には入れない。

### 3.4 パリティ

`~/work/results/pgdn/parity.log`（`strata:fork-prefill-gdn` のビルドで GPU 実行）:

| テスト | 結果 |
|---|---|
| `gemm_f32_parity --quick` | 134 sizes、bitwise OK（MMQ を入れても既定経路は不変） |
| `gdn_parity --selftest` | 0 failures |
| `gdn_rec_parity` | 0 failures（tc は sm_80+ のため skip） |
| `gdn_chunk_parity --selftest` | sm_61 の shared memory 48 KiB では kkt stage に足りず skip |
| `prefill_mmq_kquant_test` | Q4_K/Q5_K/Q6_K/Q5_1/Q8_0 + **Q3_K** が screen 通過。Q3_K の rel_l2 0.0010-0.0014（1280x2560 と qkv 実形状 10240x2560） |

### 3.5 品質

- `~/work/results/pgdn/kl_at.txt`: 4K プロンプトの最終位置の logits。`pos=3848 T=6` KL **0.00145**、`pos=3854 T=1` KL **0.00758**、top-1 トークンは両方一致。
- `~/work/results/pgdn/kl.txt`（全レコードの KL）: 中央値 0.00019 だが、verify window と bench の記録が混ざるため最大値は意味を持たない（記録の pos で切り分けた上の 2 点が正しい比較）。
- 回答（`answers.json`、greedy 200 tokens × 4 問）: 4 問中 2 問がバイト一致。残り 2 問（空が青い理由 / 100 より大きい素数）は事実関係が同じで言い回しだけが違う。
- 参考: llama.cpp の K-quant MMQ は「first-token KL ~0.03」と自ら書いており、今回の 0.0015-0.0076 はそれより小さい。

## 4. 改善案（期待幅・bitwise・実装コスト・リスク）

期待幅は「GDN 投影 2 位相（24K で 24.1%、128K で 21.0%）に対する割合」と「prefill 全体」の両方。実測でないものは推定。

| # | 案 | 対象 `file:line` | 期待幅 | bitwise | 実装 | リスク |
|---|---|---|---|---|---|---|
| 1 | dense MMQ（int8 DP4A, q8_1 活性） | `gemm.cu:1353`（native のフック 1400）、`gemm.cu:1259`、`moe_mmq.cu:333/520`、`mmq_kq_lib.cu:54`、`CMakeLists.txt:2325` | **実測: GDN -38%、prefill +12〜13%** | いいえ（q8_1 活性の丸め） | 約 40 行 + CMake | 中-高: 精度、P40 dp4a、Q3_K インスタンス追加 |
| 2 | `ssm_out` だけ cuBLAS ALGO6 | `gemm.cu:1199-1219` | 実測: `ssm_out` -17.7% = prefill -1.0〜-1.5%（推定） | いいえ（加算順） | 約 15 行 | 低: 他 T で要再測定 |
| 3 | FP32 重み像のチャンク間キャッシュ | `gemm.cu:1199-1210` | GDN の 1-2%（prefill -0.3%、推定） | はい | 約 40 行 | 低いが VRAM 544 MiB で 2 層分のみ |
| 4 | qkv と gate で FP32 活性像を共有 | `gemm.cu:1200-1219`、`prefill.cpp:4115/4123` | prefill -0.4〜-0.6%（推定） | はい（同じ変換） | 約 40 行 | 低 |
| 5 | alpha+beta を 1 本の N=96 GEMM に融合 | `prefill.cpp:4124-4125` | prefill -0.5〜-1.0%（推定） | いいえ | 約 40 行 | 低-中（ldy の配置） |
| 6 | dequant を FP16 経由せず直接 FP32 へ（同じ丸めで widen） | `gemm.cu:1496` / `kernels/dequant_*` | GDN の 1-2%（prefill -0.3〜-0.5%、推定） | はい（丸めを揃えれば） | 約 30 行 | 低 |
| 7 | dequant を side stream で重ねる（既存 `STRATA_DEQUANT_SIDE=1`） | `gemm.cu:1477-1495` | prefill -1〜-2%（推定） | はい（既存の設計） | 0 行（env のみ） | 低 |
| 8 | チャンク長 8192 -> 16384 | `generate.cpp`（`--prefill`） | 0%（実測で 8.6-8.8 TFLOPS は T 非依存） | はい | 0 行 | 低（設定変更のみ、本番は auto 固定） |

- 3 は VRAM 余裕 544 MiB（分析 doc）では 3 投影 1 層分（約 231 MiB）しか入らず、ヒット率が低い。
- 1 が唯一、桁で効く。2 は MMQ と排他（MMQ が先に取る）なので、MMQ を切ったときの保険。

## 5. 実装した変更

`patches/gdn-mmq-dense.diff`（`git diff`、150 行）:

1. `src/prefill/moe_mmq.cu`
   - `f16_to_f32` / `set_bounds` のカーネルとホスト関数を CUDA でもコンパイル（HIP 専用ガードを外す）。
   - `supported()` に `GGML_TYPE_Q3_K` を追加（CUDA のみ、`STRATA_MMQ_KQUANTS` 時）。
   - `Context::run` の K-quant ライブラリ分岐に `Q3_K` を追加（これが無いと `type 11 is not covered` で exit する。最初のビルドで踏んだ）。
2. `src/prefill/mmq_kq_lib.cu`: `strata_kq_run` に `Q3_K` の `mul_mat_q_case` を追加。
3. `CMakeLists.txt`: `_strata_kq_types` に `q3_k` を追加（`libstrata_kq.so` に Q3_K インスタンスが入る）。
4. `src/prefill/gemm.cu`: `Gemm::native_mmq` と `native()` のフックの `&& defined(__HIPCC__)` を外す。**実行時は opt-in**（`STRATA_DENSE_MMQ=1` のときだけ）。既定の挙動は変わらない。
5. `tests/cuda/prefill_mmq_kquant_test.cpp`: Q3_K を 1280x2560 と 10240x2560（qkv の実形状）で追加。

有効化: `STRATA_DENSE_MMQ=1` と `STRATA_MMQ_KQ=1`（Q3_K は `strata_kq` から読むため）。どちらも既定 off なので、本番 `.env` は変更していない。

### 5.1 実装中に踏んだ点

- 最初のビルド（`mmq`）は `Context::run` の dispatch に Q3_K が無く、テストが `prefill mmq: type 11 is not covered` で exit 1。2 回目のビルド（`mmq2`）で修正し、テストとエンジンの両方が通った。
- `mmq::fits` は P40 の 48 KiB shared memory で Q3_K の tile を持つ（llama.cpp の pascal-dp4a 設定）。`fits` が false なら従来の cuBLAS 経路に落ちる（安全側）。
- MMQ は T<=1024 行ずつ 8 スライスで走る（`Gemm::native_mmq` の `kRows = 1024`）。

## 6. 判定

**採用推奨（opt-in）**。

- prefill: 24K **+13.5% / +12.9%**、128K **+11.7% / +12.1%**（A-B-A 中央値、A-A2 の揺れは 1.1% 以内）。5% 閾値を明確に超える。
- GDN 位相: **-38.5%（24K）/ -38.3%（128K）**、`ssm_out` 位相: -36.9% / -38.8%。実効 8.6-8.8 -> 14.0-14.2 TFLOPS（推定）。
- 品質: KL 0.0015 / 0.0076（top-1 一致）、回答 2/4 バイト一致 + 2/4 同内容。bitwise ではない。
- リスク: (a) 精度が q8_1 活性の丸めに依存する（長文の greedy 回答が言い換わる）、(b) P40 の dp4a はこの doc が初の実測、(c) Q3_K インスタンスで `libstrata_kq.so` が少し大きくなる（VRAM 数 MB）。
- 既定を変えない理由: 既存の `STRATA_DENSE_F32=1` は bitwise 性を売りにしており、MMQ はそれを壊す。owner の判断で `STRATA_SET` に `STRATA_DENSE_MMQ=1` を足す形にする。

### 6.1 適用方法（本番 512K のまま）

```
STRATA_SET="ple-io=ram prefill=auto STRATA_DENSE_F32=1 STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 rope-scaling=yarn rope-scale=2"
```

- イメージは `strata:fork-prefill-gdn`（fork HEAD + このパッチ）。本番 `ghcr.io/niko1221/strata:0.2.7-cuda12` は 0.1.52 で、このパッチは入っていない。
- 戻すときは `STRATA_DENSE_MMQ` と `STRATA_MMQ_KQ` を外すだけ（コードの既定は off）。
