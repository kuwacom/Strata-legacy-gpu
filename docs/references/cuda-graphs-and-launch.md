# CUDA Graphs・起動オーバーヘッド・persistent kernel・PDL

取得日: 2026-11-07。数値は出典の実測。推定は「推定」と明記する。

## 1. 一次情報

### 1.1 CUDA Programming Guide: CUDA Graphs

- 出典: NVIDIA, *CUDA Programming Guide* 4.2 "CUDA Graphs", https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html （2026-11-07 取得）
- 要点: graph は「定義・instantiate・実行」を分離し、instantiate で launch の準備を済ませる。stream では**カーネルごとに**ホストドライバが同じ準備をするので、短いカーネルではこれが実行時間の大きな割合になる。
- graph の node 種別: kernel / memcpy / memset / host function / event record・wait / conditional / child graph。
- CUDA 12.3 で edge data が入り、`cudaGraphDependencyTypeProgrammatic` が PDL 用に使える。
- **重要**: graph は「カーネル本体を融合しない」。中間結果は global memory 経由のまま（後述の NVIDIA の fusion 記事も同じことを明言）。

### 1.2 pre-Ampere では graph launch が O(node 数)

- 出典: NVIDIA Technical Blog, "Constant Time Launch for Straight-Line CUDA Graphs and Other Performance Enhancements", 2024-09-11, https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/ （2026-11-07 取得）
- 実測（Xeon Silver 4208 + RTX 3060, Ubuntu 22.04）:

| 指標 | CUDA 11.8 | CUDA 12.6 |
|---|---|---|
| straight-line 10 node の repeat launch CPU | 2 µs | 2 µs |
| 100 node | 25 µs | 15 µs |
| **1025 node** | **278 µs** | **175 µs** |
| repeat launch の device runtime（1025 node） | 629 µs | 567 µs |
| 4 並列 chain 100 node の CPU | 73 µs | 67 µs |

- 決定的な一文: **"Launch still takes O(n) time on pre-NVIDIA Ampere architecture hardware."**
  Ampere 以降は「2.5 µs + 約 1 ns/node」のほぼ定数。**Pascal (sm_61) は O(n) 側**。
- また "Optimizing some inter-kernel latency for straight-line graphs reduces device times by up to a noticeable 60 ns/node"（Ampere での話）。

→ **P40 では window graph の node 数が直接コストになる**。1000 node なら 11.8 の式（2 µs + 200 ns/node）で約 202 µs、2000 node で約 402 µs。

### 1.3 llama.cpp への導入（NVIDIA のエンジニアによる実測）

- 出典: llama.cpp issue #6763 "New optimization from NVIDIA to use CUDA Graphs in llama.cpp", 2024-04-19, https://github.com/ggml-org/llama.cpp/issues/6763 （2026-11-07 取得）
- H100-PCIe, llama-2-7B Q4_K_M, `-n 128`: **143.35 → 163.83 tok/s（+14%）**。
- 著者の言葉: 「**graph 自体の実行は約 40% 速い**。全体の 14% に留まるのは graph の生成・起動のオーバーヘッドのため」。
- そして「**カーネル間の gap は主に GPU 側の launch overhead であり、CPU API 呼び出しではない**」。
- 同氏の追加コメント: schedule reset の CPU memset が critical path にあり、先に実行する変更（#6933）で A100 3% 向上。

### 1.4 Pascal (sm_61) で graph を有効にした実測

- 出典: llama.cpp PR #27721 "CUDA graphs on Pascal (sm_61): +40% on MoE, +7% on dense, no regression", 2026-08-25, https://github.com/ggml-org/llama.cpp/pull/27721 （2026-11-07 取得）
- 変更は 1 行: `ggml_cuda_graph_set_enabled` のゲートを `GGML_CUDA_CC_VOLTA` → `GGML_CUDA_CC_DP4A`（610）。
- GTX 1080 (sm_61), driver 581.57, CUDA 12.4, Windows:
  - MoE Ling-3.0-tiny Q6_K: **41.2 → 57.6 tok/s（+39.8%）**、長い生成で 47.7 → 72.2（+51.4%）、prefill 857 → 853（−0.5%）
  - dense Qwen 9B Q5_K_M: **+6.4% / +7.7%**、prefill −0.5%
  - llama-bench tg64: 41.47 → 73.85（**+78%**）、pp512 −1.4%（ノイズ）
- 6 モデルの追試（同一 build、A-B-A-B）:

| モデル | 構成 | tg64 off → on | 最小改善 |
|---|---|---|---|
| granite-3.1-3b-a800m | MoE 40 experts | 81.2 → 124.5 | +27.3% |
| Ling-3.0-tiny | MoE 128 experts | 52.7 → 79.8 | +48.0% |
| **granite-4.0-h-tiny** | **36 mamba + 4 attn** | **57.4 → 79.5** | **+38.5%** |
| OLMoE-1B-7B | MoE 64 | 150.5 → 169.5 | +7.6% |
| LFM2.5-8B-A1B | 18 conv + 6 attn | 116.2 → 128.5 | +10.6% |
| dense Qwen 9B | dense | 28.5 → 30.1 | +5.6% |

- 著者の結論: 「**利得は graph の node 数に従う**。decode が launch bound だから」。文脈が深いほど attention が重くなり相対利得は減るが、mamba hybrid は減りにくい。
- 反例（重要）: P100 (sm_60) では**悪化**した（単一カード・layer split の pp/tg が全量子化で後退、2 カード tensor で約 −3%）。4 カード tensor では graph が効いて 39.58 → 70.62 tok/s に回復。
- 歴史: PR #6766 が 2024 に CC<8 で graph を切った理由は **P40 の回帰（53.84 → 51.67 tok/s）**。当時は dense で約 700 node、MoE は graph 対象外だった。2 年後の再測定では dense は +6〜8%、MoE は +40%。

**Strata への含意**: Strata は既に窓全体を 1 graph に capture しているので、llama.cpp の +40% に相当する利得は**既に取っている**。残るのは (a) node 数の削減（pre-Ampere の O(n)）、(b) 融合によるメモリトラフィック削減、(c) graph 内の待ちノード削減。

### 1.5 megakernel（persistent kernel）の実測 — 「graph でも足りない」

- 出典: Hazy Research, "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B", 2025-05-27, https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles （2026-11-07 取得）
- 問題意識: batch 1 の decode は完全に帯域律速。ところが vLLM / SGLang は H100 の帯域の **最大 50%** しか使えていない。理由は「forward pass を約 100 個のカーネルに分割し、各カーネルの setup/teardown で重みのロードが止まる」から。
- 実測: H100 で dummy kernel の launch cost は **stream で約 2.1 µs、CUDA graph でも約 1.3 µs**。「graph でもまだ多くを残している」。
- 見積り: 「1 層 7 launch × 16 層、1 launch あたり楽観 5 µs の stall で、理論 1350 forward/s が **770 forward/s** に落ちる」。
- PDL への評価: "PDL ... still introduces unnecessary stalls because the PDL synchronization mechanism (cudaGridDependencySynchronize) is very coarse"（Q/K/V 全部を待ってから attention を始める必要がある）。
- 結果: Llama-1B bf16 を **1 forward pass 1 ms 未満**（H100）、B200 で 680 µs。H100 で vLLM の **2.5 倍**、SGLang の 1.5 倍。帯域利用率 **78%**（vLLM/SGLang は 50%）。
- 手法: (1) 約 100 演算を 1 kernel の interpreter で実行（fused RMS norm & QKV & RoPE、fused RMS norm & up-gate & SiLU など）、(2) shared memory を 16 KiB ページに切って命令間で受け渡す（H100 の 213 KB を使用）、(3) global memory の counter 配列で命令間を同期（PDL より細かい粒度）。
- 残コストの内訳（B200, 600 µs 全体）: activation の store/load と整合待ち 250 µs、RMS norm + GEMV 200 µs（うち 95% が GEMV）、重みの load 待ち 30 µs、warp 間同期 40 µs（async barrier 1 回 ~60 ns）、その他 80 µs。

**Strata への含意**: P40 は 30 SM・346 GB/s で、まさにこの「帯域律速 + 小カーネル多数」の条件。ただし Strata の窓は 1 graph なので、megakernel の利得のうち「launch の削減」部分は既に取っている。残るのは「**カーネル境界でメモリパイプラインが止まる**」部分で、これは fusion（次章）と persistent kernel（GR read）で削る。

### 1.6 Strata 自身の実測（graph のホストコスト）

- `include/strata/core/graph.hpp:30-31`: 「**1 呼び出しあたり 10-22 µs**（このマシン）、8 回 submission する層で約 0.1 ms の往復」。これは graph を使わない場合のコスト。
- `include/strata/core/graph.hpp:9-16`: メモリだけを spin してもカーネルは走らない（5,907,703 spin / 500 ms）。`wait_ms`（event を poll）は **0.021 ms**。
- `src/core/verify.cpp`: 窓は capture 済みで、1 window = 1 `cudaGraphLaunch`。

## 2. PDL（Programmatic Dependent Launch）

- 出典: CUDA Programming Guide 4.5, https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html （2026-11-07 取得）
- **sm_90 以上が必須**（"Available starting with devices of compute capability 9.0"）。P40 では**使えない**。
- Strata は実装済み（`include/strata/kernels/pdl.hpp`）。`pdl_supported()` が `cc_major >= 9` と `ptxVersion/binaryVersion >= 90` を検査するので、sm_61 では常に false。Pascal では `kPdlPrefetch=false`、`STRATA_PDL_RESTRICT=__restrict__`（LDG を残すため）。
- 代替: Pascal では「重みを先に読む」prefetch を **手動**でやるしかない（`cp.async` が無いので、register への先行ロードか、次の kernel の先頭で L2 を叩く）。

## 2.5 Dynamic parallelism（CDP）と stream priority

- 出典: CUDA Programming Guide 4.21 "CUDA Dynamic Parallelism", https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html （2026-11-07 取得）
  - CDP2 が CUDA 12 以降の既定。CC 9.0 未満では CDP2 が使われ、旧 CDP1 は `-DCUDA_FORCE_CDP1_IF_SUPPORTED` で opt-in。つまり **P40 (6.1) でも device からの kernel launch は可能**。
  - ただし "System software which is active when controlling dynamic launches may impose an overhead on any kernel which is running at the time, whether or not it invokes kernel launches of its own ... may result in decreased performance. This overhead is, in general, incurred for applications that link against the device runtime library."
  - device 側 launch は非同期だが concurrency は保証されない。子 grid の完了は親 grid の暗黙同期まで見えない（tail launch が必要）。
  - 結論: **P40 で使えるが、device runtime をリンクすると全カーネルにオーバーヘッドが乗る**ため、この用途（帯域律速の GEMV 列）では割に合わない。候補にしない。
- 出典: CUDA Runtime API, *Stream Management* (`cudaStreamCreateWithPriority`, `cudaDeviceGetStreamPriorityRange`), https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__STREAM.html （2026-11-07 取得）
  - "Priorities provide a hint... '0' represents default priority. The range of meaningful numerical priorities can be queried using cudaDeviceGetStreamPriorityRange()."
  - priority は**同時に走るカーネル間の順序にしか効かない**。1 本の stream 内では意味がない。
  - Strata は `STRATA_GRAPH_PRIO=1` で graph の node 優先度を使う実装を持つ（`src/core/verify.cpp:229`）。優先度レンジが 0..0 のカードでは何もしない（`:1179`）。P40 のレンジは**未確認**。

## 3. 起動キュー削減の考え方（P40 向け）
1. **node 数を測る**: `STRATA_VERIFY_NODES=1`（CUDA は個数のみ）と `STRATA_DISPATCH_REPORT=<path>`（窓ごとの呼び出し回数）で実数を取る。**未測定**。
2. **1 window = 1 graph は維持**（既にそう）。
3. **node を減らす**手段は「融合」しかない（graph はカーネルを融合しない）。1 層あたり GDN 約 14 stage / QSA 約 18 stage、48 層で **推定 800-1000 node**。
4. **event / host node を減らす**: `STRATA_FUSE_WAIT=2` は待ちを 1 つに畳む（sm_60 で −1.6）。CPU expert の doorbell 待ちは event node として graph に入るので、待ちの数 = node 数。
5. **優先度**: `STRATA_GRAPH_PRIO` は既存。P40 の優先度レンジが 0..0 なら無効（`verify.cpp:1179`）。

## 4. 期待幅の見積り（推定）

| 施策 | 根拠 | P40 での推定 |
|---|---|---|
| node 数 1000 → 800 | pre-Ampere の 200 ns/node（CUDA 11.8 式） | 起動コスト −40 µs（0.08%）。**ほぼ無視できる** |
| 1 層あたり 3 launch 削減（48 層 = 144 node） | 同上 + device gap 60-200 ns/node | −30〜−60 µs（0.06-0.12%） |
| **カーネル境界のメモリ空白を削る**（融合で重みを 1 回読む） | dense GEMV が 227 GB/s の 60-90% | **−1〜−3 ms（2-6%）** ← 本命 |
| GPU の待ち（waitCPU 2.50 + waitA 1.14 + gap 0.34） | 実測 stage table | **−1〜−2 ms（2-4%）** |

結論: **P40 で「起動キュー削減」自体の直接効果は小さい（0.1% 未満）**。効くのは「融合でメモリトラフィックを減らす」と「GPU の待ちを埋める」の 2 つ。node 数削減はその副作用として付いてくる。
