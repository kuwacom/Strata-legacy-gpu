# Pascal / P40 向け GEMV・dequant・DP4A の最適化（一次情報と実測）

取得日: 2026-11-07。数値はすべて出典の実測。推定は「推定」と明記する。

## 1. なぜ P40 の decode は帯域で決まるか

decode の dense 射影は GEMV（行列 × ベクトル）で、演算強度は 1 token あたり「重み 1 バイトにつき 1〜2 FLOP」しかない。
P40 の FP32 は 12 TFLOPS、GDDR5 は 346 GB/s なので、**重みを読む時間が計算時間を支配する**。
したがって改善幅は「(a) 読むバイト数を減らす」「(b) 346 GB/s に近づける」「(c) カーネル境界の空白を減らす」の 3 つに分解できる。

Strata の P40 実測（`bench/results/2026-10-12-p40-sm61-kernels/`）:

| 指標 | 実測 |
|---|---|
| 最も速い dequant カーネルの帯域 | Q6_K 10240×2560 を FP16 へ: 0.33 ms = **227 GB/s**（従来 7.25 ms = 10 GB/s の 22.3 倍） |
| decode GEMV の実効帯域 | wide dequant の 227 GB/s の **60-90%**（同 README の表現） |
| decode window の GPU 内訳 | dense GEMV 約 15 ms / VRAM expert 約 12.5 ms / hyper-connection 約 8.5 ms / head 2.9 ms（合計 49 ms） |

→ 「346 GB/s の何 % か」でいうと、dense GEMV は **136-204 GB/s（39-59%）**、最良カーネルでも 227 GB/s（66%）。
残り 34% は「カーネル境界の空白」と「L2 に載らない重みのレイテンシ隠蔽不足」に消えている（後述の CUDA graph / megakernel の項を参照）。

## 2. llama.cpp の Pascal 向け実装（一次情報）

### 2.1 MMQ の tile 幅 `I` を 64 → 128（Q8_0）

- 出典: llama.cpp issue #30235 "CUDA: Pascal DP4A MMQ config uses I=64 for Q8_0; I=128 gives +12% pp512 on sm_61", https://github.com/ggml-org/llama.cpp/issues/30235 （2026-10-09 作成、2026-11-07 取得）
- 環境: Tesla P4 (sm_61), CUDA 12.6, `-DGGML_CUDA_FORCE_MMQ=ON`
- 結果: Q8_0 の 2 モデルで **+12.1% / +11.6%**（pp512、8 ペアすべて勝ち、sign test p=0.0078）。Q4_K_M −0.7%、Q6_K −0.1%（ノイズ）。
- 重要な制約: `I=160` は **45.7% 悪化**、`I>=192` は **コンパイル不能（Pascal の shared memory 上限）**。128 が上限。
- 注意: tg128 は −0.3%。「decode は帯域律速なので tile 変更に反応しないはず」という sanity check を著者自身が使っている。
- 単一カード・pp512 のみ・22 型のうち 3 型のみ、という限界も明記されている。

### 2.2 MMQ の `I=128` を 11 型へ拡張（J=64 の行だけ）

- 出典: llama.cpp PR #31352 "cuda : increase Pascal DP4A MMQ I tile for J=64", https://github.com/ggml-org/llama.cpp/pull/31352 （2026-10-29 作成、2026-11-07 取得）
- 環境: GTX 1080 Ti (sm_61, 28 SM), 6 ペア交互、sign test
- 結果: Qwen3-4B Q8_0 で **pp128 +16.2% / pp512 +17.8% / pp2048 +18.9%**。変更した 11 型すべてが pp512 で **+8.2% 〜 +18.0%**。Qwen3.6-35B-A3B Q2_0（`-ub 128`）で +13.0%、bge-m3 で +6.6%。
- 逆効果: K-quant は `I=128` で Q4_K −18.8%、Q5_K −31.2%、Q6_K −36.7%（したがって K-quant は変更しない）。pp64（入力 49-64 列）は block 数が 40→20 になり 28 SM のうち 8 個が遊ぶため **−3.1%**。
- 正しさ: `MUL_MAT` 2490/2490、`MUL_MAT_ID` 931/931、5 量子化で greedy 64 token が byte 一致。
- 一般性: 変更表は sm_61 / sm_62 のみで選ばれる。

**Strata への適用可否**: Strata の prefill expert は llama.cpp の `mmq.cuh` をそのままリンクしている（`src/prefill/moe_mmq.cu`, `include/strata/prefill/mmq_kq_abi.hpp`）。ggml は commit `3cf03257`（2026-09-20）に pin されており、#30235/#31352 より前なので **P40 では今も `I=64`** のはず。適用手順は `cmake/strata_ggml_tq_overlay.cmake` の overlay 方式で `mmq-config-pascal-dp4a.cuh` を書き換える（もしくは pin を上げる）。sm_61 で使える。**推定 +8〜18%（prefill、IQ3_XXS の gate/up が対象形式に入るかは要確認）**。

### 2.3 MMVQ の warp 数を Pascal 専用に（4 → 2、1 token のみ）

- 出典: llama.cpp PR #25479 "ggml-cuda : optimize mmvq nwarps for Pascal DP4A", https://github.com/ggml-org/llama.cpp/pull/25479 （2026-07-09 作成、2026-11-07 取得）
- 環境: GTX 1060 6GB (sm_61), Llama-3.2-3B, 6 反復のペア比較
- 結果: single-token decode で Q4_0 +4.59%、Q4_K_M +5.44%、Q5_K_M +3.24%、Q8_0 +2.48%、Q6_K −0.32%。**geomean +3.07%**。prefill は全条件で flat（±0.3% 以内）。
- 理屈: "Pascal decode is memory-bandwidth-bound on these small SMs, so the extra warps add scheduling/occupancy pressure without helping throughput."

**Strata への適用可否**: Strata の `kIlRows_61` は「1 warp が何行持つか」を決めるが、block あたりの warp 数は別。`native_mmvq.cu:3667` の表に対して「1 token 列のとき block の warp を減らす」変種を足せる。sm_61 で使える。**推定 +2〜3%（decode、1 token 窓のみ）**。ただし Strata の窓は平均 T 2.14 なので効くのは T=1 の窓だけ。

### 2.4 MMVQ の融合判定が sm_61 を外している（P40 の実測）

- 出典: llama.cpp issue #30334 "CUDA: P40 and the MMVQ fusion check for pascal", https://github.com/ggml-org/llama.cpp/issues/30334 （2026-10-11 作成、2026-11-07 取得）
- 内容: `ggml_cuda_should_fuse_mul_mat_vec_q` の判定が `cc <= GGML_CUDA_CC_PASCAL`（600）で、**P40 の 610 が漏れる**ため融合が ON のまま。
- 実測（Mixtral 8x7B / Llama 3.1 8B、P40、tg128、6 回の中央値）:
  - 融合 OFF: Mixtral Q2_K −5.46%、Q3_K_M **+5.29%**、Llama Q4_0 **+4.72%**、Q4_K_M −1.21%
  - 融合 OFF + small-K 強制: Mixtral Q2_K **+12.04%**（depth 0）、pp512 `-ub 1` で **+12.15%**
- つまり「P40 では量子化形式によって融合の得損が反転する」。一律 ON/OFF ではなく形式別の表が必要。

**Strata への適用可否**: Strata の expert カーネルは独自（mode 8）で ggml の融合判定を使わないが、**「形式別に融合の得損が反転する」という事実**は Strata の `STRATA_GU_SWIGLU` / `STRATA_FUSE_DECODE` の形式別 A/B にそのまま当てはまる。sm_61 で使える（測定手法の示唆）。

### 2.5 MMQ の src1 column stride（歴史的な大当たり）

- 出典: llama.cpp PR #4233 "cuda : tweak mm stride to double perf on P40 + GTX 970", https://github.com/ggml-org/llama.cpp/pull/4233 （2023-11-27 作成、2026-11-07 取得）
- 結果（GTX 970 + P40、`-DLLAMA_CUDA_FORCE_MMQ=ON`）: pp512 が 7B q4_0 で **217.28 → 458.18 tok/s（2.11x）**、13B q4_k_s で **111.45 → 246.14（2.21x）**。
- 原因は `MUL_MAT_SRC1_COL_STRIDE`（活性側の列ストライド）。1024 にすると倍になる。
- 示唆: Pascal の MMQ は「重み tile」より「活性の読み方」で 2 倍動くことがある。tile 幅だけ見ていてはいけない。

**Strata への適用可否**: Strata の MMQ は llama.cpp 由来なので同じ stride 問題を持ちうる。sm_61 で使える。**未測定**（Strata の P40 で `MUL_MAT_SRC1_COL_STRIDE` 相当がどうなっているかは未確認）。

### 2.6 Qwen3.8-Flash-Next を Pascal で動かした外部実測

- 出典: llama.cpp issue #27980 "qwen4exp: slow decode on Pascal (sm_61) — 18 tok/s vs 29.5 for same-class qwen3.5moe", https://github.com/ggml-org/llama.cpp/issues/27980 （2026-08-29 作成、2026-11-07 取得）
- 4× Tesla P40、Qwen3.8-Flash-Next UD-Q3_K_XL (83.8 GiB): decode **18.0-18.1 tok/s**、prefill 338-376 tok/s。比較の Qwen3.5-122B-A10B は 29.5 tok/s。
- 著者の分析: `GGML_CUDA_FORCE_MMQ=1` で変化なし → 量子化 matmul ではない。CPU offload を外しても変化なし。**36/48 層が recurrent なので `ssm_scan` / `ssm_conv` が毎 token 走るのが本命の疑い**（未プロファイル）。
- 意義: 同じモデル・同じ GPU クラスで、llama.cpp は Strata の 45-50 tok/s よりかなり遅い。Strata の GDN 実装（`k_gdn_seq` / fused_gdn）と expert 配置が効いている裏付けになる。同時に「Pascal では recurrent 層の per-token コストが支配的になりうる」という警告でもある。

## 3. dequant の最適化（P40 の実測）

Strata の wide dequant（1 thread / 8 values, 16-bit load, 16-byte store）は sm_61 で既定。`dequant_wide_parity --bench`:

| 形状 | per-group (従来) | wide | 倍率 |
|---|---|---|---|
| Q6_K 10240×2560 | 7.25 ms (10 GB/s) | 0.33 ms (227 GB/s) | 22.3x |
| Q6_K 2560×6144 | 4.45 ms | 0.20 ms | 22.5x |
| Q4_K 10240×2560 | 1.67 ms | 0.32 ms | 5.2x |
| Q5_K 10240×2560 | 1.24 ms | 0.33 ms | 3.8x |
| IQ4_XS 10240×2560 | 1.21 ms | 0.34 ms | 3.6x |

出典: `bench/results/2026-10-12-p40-sm61-kernels/README.md` と `dequant-wide-parity.txt`（360 ケース bitwise）。

限界: 227 GB/s は 346 GB/s の 66%。dequant は「読む（量子化）＋書く（FP16）」なので、読み 1 バイトあたり書き 2-4 バイトになり、実効は「読み+書き」の合計で割られる。Q6_K は 6.56 bpw → FP16 で 16 bpw なので、合計 22.6 bpw。10240×2560 = 26.2M 要素 × 22.6/8 = 74 MB を 0.33 ms → **224 GB/s**。読みだけ見ると 21.5 MB/0.33 ms = 65 GB/s だが、書き込みを含めると 224 GB/s でほぼ上限。つまり **dequant はもう帯域律速で、これ以上は「書かない」以外にない**（→ FP16 キャッシュ = `STRATA_PF_W16_CACHE_MIB`）。

## 4. 実装候補への含意（この文書の結論）

1. **prefill の MMQ tile `I`**: sm_61 で +8〜18% の外部実測があり、Strata は同じ `mmq.cuh` を使っている。最優先。
2. **1 token 窓の warp 数**: +2〜3% の外部実測。Strata の窓は T 2.14 平均なので効果は限定的。
3. **形式別の融合判定**: P40 では形式で得損が反転する。Strata の `GU_SWIGLU` / `FUSE_DECODE` を形式別に A/B する根拠。
4. **活性側の stride**: 過去に 2 倍の事例。Strata では未確認。
5. **dequant は既に上限**: 追加の改善は「FP16 を VRAM に残す」しかない（P40 は expert が VRAM を埋めるので小予算のみ）。
