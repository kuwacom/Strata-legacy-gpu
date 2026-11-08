# カーネル融合とメモリ帯域（P40 での意味）

取得日: 2026-11-07。数値は出典の実測、推定は「推定」と明記する。

## 1. 一次情報

### 1.1 NVIDIA: fusion は「メモリ仕事を減らす」

- 出典: NVIDIA Technical Blog, "Kernel Fusion in NVIDIA CUDA: Optimizing Memory Traffic and Launch Overhead", 2026-07-10, https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/ （2026-11-07 取得）
- 例: `sum(abs(x))` を 2 kernel → 1 kernel に融合。中間バッファ（1 GiB）が消え、global memory の移動が **3 GB → 1 GB**。RTX 4090 で **3.51 ms → 1.18 ms（3 倍）**、実効帯域 855 GiB/s（ピーク 1,008 GB/s の約 90%）。
- 決定的な指摘: 「**CUDA Graphs は別レイヤの融合**であり、カーネル本体は融合しない。graph の中のカーネルは別々に走り、中間結果は global memory を経由する。graph で包むとホスト側の µs は削れるが、1 GiB の往復は残る」。
- つまり fusion の利得は「帯域律速のカーネルで、中間の write+read を消すこと」。

### 1.2 Hazy Research: 融合した命令列で帯域利用率 78%

- 出典: "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B", 2025-05-27, https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles （2026-11-07 取得）
- 融合した命令: 「fused RMS norm & QKV & RoPE」「fused RMS norm & up-gate & SiLU」「down-projection + residual」「RMS norm & LM head」。
- vLLM/SGLang は H100 の帯域の 50% しか使えず、megakernel は **78%**。H100 で vLLM の 2.5 倍。
- 示唆: RMSNorm を GEMM の前に融合するのは、**正規化の出力を global に書かずに済む**から。ただし Llama-1B では活性が小さいので、利得の主因は「境界の除去」であり、バイト削減は副次。

### 1.3 Liger-Kernel: 融合カーネルの「厳密性」

- 出典: LinkedIn, *Liger-Kernel*, https://github.com/linkedin/Liger-Kernel （2026-11-07 取得）
- RMSNorm / RoPE / SwiGLU / CrossEntropy を融合し、学習で throughput +20%、メモリ −60%（LLaMA 3-8B, 8×A100, FSDP）。
- 「**Computation is exact—no approximations**」を明示（forward/backward とも単体テストと収束テストで検証）。
- 限界: Triton 実装。Triton の下限は sm_70 なので **Pascal では動かない**（P40 ではこの手法をそのまま使えない）。

### 1.4 llama.cpp: decode の operator fusion と、P40 での形式別の得損

- Strata の `STRATA_FUSE_DECODE=1` は llama.cpp PR #1097 の融合（gate/up GEMV + SwiGLU を 1 kernel、SwiGLU + 中間など）に由来する（`include/strata/kernels/fuse_decode.hpp:2`）。V100 で「1 token あたり 4-16 µs 節約、end-to-end では見えない」（`docs/NVIDIA_V100.md:112`）。
- P40 では**形式によって融合の得損が反転する**（llama.cpp #30334, 2026-10-11, https://github.com/ggml-org/llama.cpp/issues/30334 ）:
  - 融合 OFF: Mixtral Q2_K −5.46%、Q3_K_M **+5.29%**、Llama Q4_0 **+4.72%**、Q4_K_M −1.21%
  - 融合 OFF + small-K: Mixtral Q2_K **+12.04%**（tg128）、pp512 `-ub 1` で **+12.15%**
- 示唆: 「融合は常に速い」は Pascal では成り立たない。形式・形状ごとに A/B が必要。

## 2. Strata が既に持つ融合（再掲）

`strata-existing-optimizations.md` の 3-4 章。要点だけ:

| 融合 | 効果の実測 |
|---|---|
| expert mode 8（gate/up + SwiGLU + q8_1 を 1 kernel） | P40 で 18 expert × 2 token: IQ3_XXS 345 → 252 µs |
| `STRATA_EXPDEC`（窓の活性を shared に、gate/up を 1 kernel） | 2080 Ti で gate/up −11〜−23%、end-to-end +1.1% |
| `STRATA_GU_SWIGLU`（gate/up + SwiGLU + q8_1） | sm_86 で −0.8 / −1.4 ポイント |
| `STRATA_FUSE_DECODE` | V100 で 4-16 µs/token |
| `STRATA_HC_UPMIX`（GR up + mix） | GB10 で decode 1.054x |
| GR read（norm+down+up+inject を 1 呼び出しに） | 既定 |
| `STRATA_LFUSE`（shared expert の gate / scale を combine に畳む） | 既定（条件付き） |
| `STRATA_QFUSE`（GR read が消費側の q8_1 を書く） | 既定（gfx12 系） |

→ **「RMSNorm + GEMM」「GEMV + dequant」「MoE gate/up + down」「attention 融合」は、Strata では概ね実装済み**。残っているのは「層をまたぐ / 同一層内で別々に起動しているカーネルの統合」と「P40 で既定になっていないもの」。

## 3. P40 で融合が効く理由（定量）

P40 の decode window は実測で 49 ms。stage ごとの「バイト数から計算した下限」と実測の差を取ると、**カーネルごとに 7-25 µs の説明できない時間**がある（推定。バイト数はモデル形状から、下限は 227 GB/s で計算）:

| stage（GDN 36 層の合計） | 実測 | 重みバイト（推定） | 227 GB/s の下限 | 差 |
|---|---|---|---|---|
| q8+qkv/q-idx gemv | 4.24 ms（118 µs/層） | qkv Q6_K 10240×2560 = 21.5 MB | 95 µs | **+23 µs** |
| out-proj | 2.79 ms（78 µs/層） | 2560×6144 Q6_K = 12.9 MB | 57 µs | **+21 µs** |
| hc0 down | 1.43 ms（40 µs/層） | 320×10240 bf16 = 6.55 MB | 29 µs | **+11 µs** |
| hc0 up | 1.30 ms（36 µs/層） | 10240×320 bf16 = 6.55 MB | 29 µs | **+7 µs** |
| head（1 window） | 2.87 ms | 248320×2560 Q5_K ≈ 437 MB | 1.93 ms | **+0.94 ms**（末尾 GR read 約 0.4 ms を含む） |

1 層あたり GDN 14 stage + QSA 18 stage だが、重みを読む stage は GDN で 6 程度。48 層 × 6 = 288 stage。1 stage あたり平均 10 µs の空白があると **2.9 ms（window の 6%）**。
これが「融合で削れる上限」で、gfx1030 の GR persistent kernel が 3 launch → 1 launch で **window −6.6%** だった実測と整合する。

## 4. 融合の候補（P40 で未実装のもの）

### 4.1 GR read の norm + down を 1 kernel に（bitwise）

- 現状: `gr_norm_multi_kernel`（`src/kernels/cuda/fused_gr.cu:281`）が xn を scratch に書き、`gr_down_multi_kernel`（`:333`）が読む。1 層 2 回（attn / ffn）× 48 層 = **96 回の境界**。
- 融合すると: 境界 96 個削減 + xn の write/read（40 KB × 2 × 96 = 7.7 MB）削減。
- 期待: **−0.3〜−0.9 ms（0.6-1.8%）**（推定。1 境界 10 µs × 96 = 0.96 ms が上限）。
- リスク: 低。同じ和の順序を保てば bitwise（`gr_parity` がある）。
- sm_61: 使える（既存カーネルの統合のみ）。

### 4.2 GR read を persistent kernel 1 個に（`STRATA_HC_PERSIST`）

- 既に実装済み（`fused_gr.cu:3296`）。gfx1030 で read 1.35-2.10x、window −6.6%（4K）/ −4.4%（32K）。
- 期待: P40 の GR 8.54 ms → 6.0-6.3 ms、**−2.2〜−2.5 ms（4.5-5%）**（推定、1.35-1.46x を当てはめ）。
- リスク: **bitwise でない**（別の和の順序）。起動時レジデンシ検査 + KL 測定が必要。
- sm_61: 使える（CUDA 実装あり、80 block / 256 thread は 30 SM に 2.67 block/SM で常駐可能と推定）。

### 4.3 shared expert を routed expert の grouped launch に合流

- 現状: `shared+quant` が GDN 2.23 ms + QSA 0.73 ms = **2.96 ms（6%）**。shared expert は毎層 1 個で、routed expert と同じ q8_1 行 dot。
- 融合すると: 1 層あたり 1 launch（shared の gate/up/down 3 個）が grouped の 1 個に吸収。48 層で **−96 launch / window**、かつ活性の再読を削減。
- 期待: **−0.5〜−1.2 ms（1-2.4%）**（推定）。
- リスク: 中。grouped カーネルは expert 単位の group なので、shared を 1 group として足すのは構造的に可能。bitwise は group 内の和の順序を保てば可。
- sm_61: 使える（mode 8 の grouped kernel は sm_61 既定）。

### 4.4 router + q8 量子化 + shared gate を 1 launch に

- 現状: `router+plan+bell` 2.04 ms（4%）+ `shared+quant` 2.96 ms。router は 512×2560 = 1.3M 重み（Q? で 0.9 MB）しか読まないのに 42 µs/層かかる = **レイテンシ律速**。
- 融合すると: 層あたり 2-3 launch 削減。
- 期待: **−0.3〜−0.8 ms（0.6-1.6%）**（推定）。
- リスク: 中（bitwise は可）。
- sm_61: 使える。

### 4.5 RMSNorm + GEMV（dense 射影の前段）

- 現状: `k_rms` / `k_add_rms` は独立 kernel（`src/qwen35/q36_engine.cu:113`, `:135`）。production 側は `hc` read の中に norm が入っているので、実は既に融合済み（GR read の norm）。
- したがって「RMSNorm + GEMM」は **Strata では GR read として既に融合されている**。追加の余地は「GR read の norm を down と融合」（4.1）。

### 4.6 attention の融合

- decode attention は QSA で 0.57 ms/window（1.2%）しかない。融合の余地は小さい。
- prefill attention は 1,860 ms / 16,005 ms（11.6%）で大きいが、融合より「int8 K/V の sm_6x カーネル」（`qsa_pattn`）の方が効く（`qsa_pattn.cu`）。

## 5. まとめ

- P40 の融合の価値は「バイト削減」より「**カーネル境界の空白（1 個 10 µs 前後）の除去**」にある。
- 最大の対象は GR read（1 層 6 launch → 2-3 launch）で、persistent 版なら window −4〜−6% が既に他アーキで実測されている。
- RMSNorm+GEMM / GEMV+dequant / MoE gate/up+down は既に実装済みなので再提案しない。
- Triton / Liger / megakernel の手法は P40 では直接使えない（sm_70+ / 巨大 shared）。ただし「境界を消す」という設計思想はそのまま使える。
