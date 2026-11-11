# P40 (sm_61) のカーネル・GPU・ライブラリのギャップ分析

取得日: 2026-11-10。対象は本番 `strata:fork-v5`（engine 0.1.57、main `80cc9ec7`）。
調査は 4 観点を並列で行い、いずれも読み取りのみ（GPU 実行なし）。実装したものは末尾に記す。

## 0. 前提の実測値

`nvidia-smi` の照会（読み取りのみ）:

| 項目 | 値 |
|---|---|
| ECC | Disabled（帯域ペナルティなし） |
| persistence mode | Enabled |
| PCIe | Gen3 x16 |
| SM clock / mem clock | 1,531 MHz / 3,615 MHz（max と同一、throttle 0x0） |
| MPS | なし |

## 1. cuBLAS / cuBLASLt

現行は `cublas_v2` のみで、cuBLASLt は未使用（`src/prefill/gemm.cu` に集中）。

- `cublasSgemm`（DENSE_F32 と bf16 path 2）、`cublasGemmEx`（f16、ALGO2 フォールバックのみ）
- `cublasSetWorkspace` 32 MiB、`CUBLAS_DEFAULT_MATH`
- **libcublasLt.so.12.9.1.4 には sm_61 の `maxwell_sgemm_{128x128,128x64,128x32,64x64,32x128}` が
  入っている**が、v2 API が実際に選ぶのは `maxwell_sgemm_128x64_tn` と
  `maxwell_sgemm_fp16_{32x128,64x64}_tn` だけ。3 種のタイルが「あるが選ばれない」
- 候補: Lt の heuristic でタイルを選ばせる（`cublasLtMatmulAlgoGetHeuristic`）、
  `CUBLAS_GEMM_ALGO0..23` の掃引、`cublasSgemmStridedBatched`/`Sgemv`（N=4 向け）
- 計測: `CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=stdout`、`CUBLASLT_LOG_LEVEL=2`、`nsys`
- 注意: 本番イメージの cuBLAS は 12.9.1.4、過去の実測は 12.4.5.8。混在に注意
- 出典: CUDA 12.9 cuBLAS / cuBLASLt ドキュメント、`libcublasLt` の `cuobjdump`

## 2. CUDA / Pascal の機能

| 機能 | sm_61 | Strata | 期待 |
|---|---|---|---|
| `__ldg` / `__restrict__` | 可 | 使用済み | - |
| `cudaFuncSetCacheConfig` | 可（L1/shared 独立） | 0 箇所 | 実質 no-op |
| shared > 48 KB/block | 不可 | 該当なし | - |
| `cudaStreamCreateWithPriority` | 可 | verify/dflash2/mtp で使用 | - |
| CUDA graph | 可（O(node) 起動） | 1 window = 1 graph | node 1,972〜2,069（実測） |
| `cudaMallocAsync` / pool | 可 | 0 箇所 | 初回 0.1〜1 ms |
| `cudaLaunchCooperativeKernel` | 可 | 0 箇所 | 0〜0.5% |
| L2 persistence (`accessPolicyWindow`) | **不可**（CC 8.0+） | - | - |
| `-Xptxas -dlcm=cg` | 既定と同じ | - | no-op |
| `-Xptxas -dlcm=ca` | 可 | 未使用 | 0〜2% |
| `-Xptxas -v` | 可 | **未使用**（レジスタ/スピルが見えない） | 計測基盤 |
| `cudaHostAllocWriteCombined` | 可 | 0 箇所 | 0〜1%（write-once staging） |
| `--pcie-mode dma` | 可 | 既定 auto（kernel copy） | 通常 0、miss 時の逃げ道 |

既存 docs との食い違い（更新済み）:

- `docs/references/cuda-graphs-and-launch.md` の node 推定 800〜1000 は実測 1,972〜2,069 と違う
- `docs/references/pascal-gemv.md` の「P40 では今も I=64」は overlay 適用前の記述

## 3. llama.cpp / ik_llama.cpp との差分

- pin `3cf03257` の `mmq-config-pascal-dp4a.cuh` は master と byte 一致（TQ 行を除く）。
  したがって差分は Strata の overlay が入れた **I=128 for J=64（12 型）** だけ
- Pascal DP4A の全型で nthreads 256 / nwarps 8 / occupancy 2 / `MMQ_ITER_K` 256 / stream_k false
- J は 8/16/24/32/40/48/64 のみ。J=56, 72..128 は表に無い
- IQ2_S / IQ2_XS は I=128 だと J=64 で 51,200 B（49,152 B 超過）。**J=48 なら 49,088 B**、
  I=96 なら J=64 で 40,768 B
- Q2_K は I=128 だと J=8 でも 52,256 B で不可、I=96 なら J=64 で 47,104 B
- `fits()` は J=8 しか見ないため、P40 で `supported()` を通った型は false にならない。
  実際に落ちるのは `supported()` の段（K-quant は `STRATA_MMQ_KQ=1` 必須など）
- upstream master には `has_fusion`（gate/up + GLU の MMQ 融合）があるが、
  **DP4A の write-back は `glu_op` を捨てており、Pascal では未実装**。移植は新規実装になる
- upstream 実測: GLU 融合 +4.6〜7.4%（dense）/ +5.7〜17.2%（MoE、DGX Spark）。
  I=128 化は +8.2〜18.0%（1080 Ti, pp512）
- 出典: llama.cpp PR #29948 / #31188 / #31352 / issue #30235、`mmq.cuh` / `mmq.cu`

## 4. SASS（実測 `cuobjdump -res-usage`）

| カーネル | threads | REG | STACK | SHARED | 占有率 |
|---|---:|---:|---:|---:|---|
| `ps_value_d4_kernel` | 64 | **151** | 0 | 8,388 | 6 blk / 18.75% |
| `ps_score_x_kernel<1>` | 256 | 128 | 0 | 20,992 | 2 blk / 25% |
| `bf16_narrow_kernel` | 256 | 64 | 0 | 8,960 | 4 blk / 50% |
| `gu_grouped_t_kernel<0,8>` | 256 | 79 | 0 | 25,600 | 2 blk / 25% |
| MMQ（`libstrata_kq.so`） | 128 | 68-127 | 0 | 動的 | 最大 4 blk |

- 実スピル: `gr_write_norm_rs_kernel` / `gr_write_cvec_norm_rs_kernel`（STACK 40、LDL/STL 9 本）、
  `native_gu_multi_kernel<145,1,1>`（144 B）、`dequant_gu_kernel`（32 B）、
  `block_scores_halve_kernel<16>`（16 B）、`dequant_kernel<11,*>`（16 B）
- `bf16_narrow_kernel` は LDG 128 bit（W）/ 64 bit×2（X）、LDS 128/64 でベクタ化済み、
  行パディング 36/68 でバンク衝突なし（推定）、FFMA:LDS = 9.85:1 で LDS 律速ではない
- `check_cuda_spills.py` の `ALLOWED` に sm_61 が無く、CI のスピルゲートが効いていない

## 5. 優先度付きバックログ（期待効果は推定）

| 優先 | 項目 | 期待 | 状態 |
|---|---|---|---|
| 1 | MMQ gate/up + GLU 融合を DP4A write-back に実装 | +2〜5% | 未着手（新規実装、Pascal では upstream も未対応） |
| 2 | cuBLASLt の heuristic を f16 / DENSE_F32 経路に | 0〜10% | 未着手（タイル 3 種が未使用という物証） |
| 3 | `ps_value_d4` の `__launch_bounds__(64, 8)` | +0.2〜0.7% | **実装・検証中**（151 -> 128 regs、6 -> 8 CTA） |
| 4 | IQ2_S / IQ2_XS の I=128 @ J=48 | 0〜5%（該当時のみ） | **実装・検証中**（表のみ、bitwise） |
| 5 | dense MMQ の `K % 256` 制約を `overread_bytes` で解除（K=640） | +0.3〜1.5% | 未着手（低難度、非 bitwise） |
| 6 | `-Xptxas -v` を sm_61 ビルドに追加し、spill ゲートを sm_61 にも | 計測基盤 | 未着手（低難度） |
| 7 | `-Xptxas -dlcm=ca` | 0〜2% | 未着手（1 フラグ） |
| 8 | `native_gu_multi` などのスピル解消（STACK 144 B） | 0〜1% | 未着手 |
| 9 | `gr_write_norm_rs` の STACK 40 解消 | 0〜0.5% | 未着手 |

## 6. 見送りと判断したもの

- `STRATA_MMQ_TAB` / `STRATA_MMQ_DIRECT`: 期待 0.7〜1.1%、sm_61 は repo 内でも未検証、+37 MiB 常時
- L2 persistence / PDL / `cp.async` / tensor cores: sm_61 に無い
- cooperative kernel: barrier 2 回のみで 0〜0.5%、graph capture 相性が未確認
- `cudaHostAllocWriteCombined`: 通常の prefill では write-once の staging が小さい
- `cublasSgemv` への置換: N=4 は既に `bf16_narrow` で処理済み

## 7. 実装したもの（本ドキュメントと同時）

- `ps_value_d4_kernel` に `__launch_bounds__(64, 8)`（worktree `~/work/wt/mmqtiles`）
- overlay で IQ2_S / IQ2_XS の J=48 行を I=128 に（`cmake/strata_ggml_tq_overlay.cmake`）
- ビルドと A/B の結果は `docs/prefill/mmq-tiles.md` に追記する

