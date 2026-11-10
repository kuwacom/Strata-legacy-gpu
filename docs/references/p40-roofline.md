# P40 のルーフラインと prefill の位相分解

取得日: 2026-11-11。対象は本番 `strata:fork-v6`（engine 0.1.57）。
数値はすべて実測（`STRATA_PREFILL_TIMING=1` の位相内訳）と、GGUF メタデータから計算した理論値。
計算は「推定」ではなく式を書いて出している。実測と理論の比を「効率」として並べる。

## 1. ハードウェアの上限

| 項目 | 値 | 出典 |
|---|---|---|
| SM 数 | 30（128 core × 30 = 3,840） | `nvidia-smi` / P40 仕様 |
| SM clock | 1,531 MHz（実測、throttle なし） | `nvidia-smi -q -d CLOCK` |
| FP32 | 11.75 TFLOP/s = 3,840 × 2 × 1.531e9 | 計算 |
| INT8 DP4A | 47 TOPS | NVIDIA P40 仕様 |
| VRAM 帯域 | 346 GB/s（GDDR5 3,615 MHz × 384 bit） | P40 仕様。実効は 270〜300 GB/s 程度（推定） |
| PCIe | Gen3 x16（実効 11〜12 GB/s、推定） | `nvidia-smi --query-gpu=pcie.link.gen.current` |
| L2 | 3 MB | GP102 |
| shared/block | 48 KB（opt-in 不可の上限） | Pascal |
| FP16（非テンソル） | FP32 の 1/64 | Pascal |

## 2. モデルの 1 トークンあたりの仕事

GGUF メタデータ（実測）: `block_count=48`、`embedding_length=2560`、`expert_count=512`、
`expert_used_count=10`、`expert_feed_forward_length=640`、`full_attention_interval=4`（GDN 36 層 + 全注意 12 層）、
`hyper_connection.count=4`、`hyper_connection.low_rank=320`、`attention.head_count=24`、
`attention.key_length=256`、`attention.indexer.top_k=2048`。

| 部分 | 1 トークン・1 層の FLOP | 層数 | 合計 (TFLOP, T=137,344) |
|---|---:|---:|---:|
| GDN 射影 in（10240 + 6144 行） | 83.9 M | 36 | 415 |
| 全注意 射影 in（12288 + 512 + 512 + 6144 行） | 99.6 M | 12 | 164 |
| 射影 out（6144 -> 2560） | 31.5 M | 48 | 208 |
| MoE（top-10 × (gate 640 + up 640 + down 640)） | 98.3 M | 48 | 648 |
| 共有 expert（640 × 3） | 9.83 M | 48 | 65 |
| router（512 -> 10） | 2.62 M | 48 | 17 |
| hyper-connection（2 ハーフ × down 320 + up 320） | 26.2 M | 48 | 173 |
| QSA attention（top-2048、24 head × 256） | 50.3 M | 12 | 83 |
| **合計** | **12.3 GFLOP** | | **1,689** |

うち hyper-connection の 173 TFLOP だけは **FP32**（bf16 重みを FP32 に広げて FMA）で、
Pascal に bf16 の積和器は無い。残り 1,516 TFLOP は int8 DP4A で走る。

## 3. 下限と実測（128K prefill、137,344 tokens）

| 位相 | 実測 ms | 理論下限 ms | 効率 | 下限の根拠 |
|---|---:|---:|---:|---|
| gdn | 26,989 | 8.8 | 33% | 415 TFLOP / 47 TOPS |
| gemm gate/up | 27,743 | 9.2 | 33% | 432 TFLOP / 47 TOPS |
| gemm down | 14,460 | 4.6 | 32% | 216 TFLOP / 47 TOPS |
| qsa proj | 12,856 | 3.5 | 27% | 164 TFLOP / 47 TOPS |
| gdn out proj | 11,362 | 3.3 | 29% | 156 TFLOP / 47 TOPS |
| router+shared | 10,179 | 1.7 | 17% | 82 TFLOP / 47 TOPS |
| hc read: down+silu | 12,644 | 7.4 | 58% | 86 TFLOP / 11.75 TFLOP/s |
| hc read: up | 9,387 | 7.4 | 78% | 86 TFLOP / 11.75 TFLOP/s |
| hc write / mix / inject | 15,981 | 2〜3 | 20% | 帯域（後述） |
| qsa attn | 22,534 | 1.8〜 | 8%? | 83 TFLOP / 47 TOPS（疎性の実装依存で上限が読めない） |
| gdn recurrence | 8,100 | 状態の読み書き依存 | - | 786K 状態 × 36 層 |
| qsa select | 7,778 | - | - | top-k 選択 |
| combine | 2,714 | - | - | top-10 の重み付き和 |
| その他 | 8,300 | - | - | ple / dequant / gather / kv stage ほか |
| **合計** | **192,062** | **約 47 + attention** | **25〜30%** | |

## 4. 読み取れること

1. **MoE は 33%**。DP4A の積和そのものは速いが、IQ2_XS/IQ2_XXS の unpack と
   write-back（`scale_gu_rows` + `swiglu` の 2 パス）が乗っている。ここが 42 s。
2. **dense 射影は 27〜33%**。MoE と同じ効率で、N が大きい（10240 / 12288 行）のに
   expert のように「同じ重みを何百トークンも使う」形になっていないため、重みの unpack が
   64 トークンでしか償却されない。**チャンク全体で償却する形（int8 常駐 + 素の DP4A GEMM）が本命**。
3. **router+shared が 17% と最悪**。N=512 / 640 の狭い形で、タイルが 30 SM に満たない。
   llama.cpp の「タイル効率 90% ルール」がそのまま効く可能性がある。
4. **HC は 58〜78%** で、もう限界に近い。ここを縮めるには計算量を減らすしかない（不可）。
5. **attention は 8% 前後**。`ps_value_d4` の占有率を上げただけで 10.9% 速くなったので、
   まだ構造的な余地がある。

## 5. この表の使い方

- 「理論下限」は 1 命令あたりの積和がすべて有効に使えた場合の値で、unpack・スケール・write-back は
  含まない。したがって MoE の 50% は「ほぼ限界」ではなく、まだ 2 倍の余地があるという意味
- 位相の再現性は ±10 ms（27,000 ms の位相で ±0.04%）。1 アーム 6 分の 128K 計測で 0.3% の差は検出できる
- 判定は「対象位相」と「GPU timeline 合計」の両方で行う。位相が縮んで合計が縮まないなら、それは
  別の位相へ仕事が移っただけ
