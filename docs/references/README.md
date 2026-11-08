# Strata × Tesla P40 (sm_61) 高速化リファレンス — 索引と適用候補ランキング

最終更新: 2026-11-07。作業ツリー `~/work/wt/research`（branch `docs/research`）。
GPU は使っていない（ビルド・計測・本番コンテナ操作なし）。数値はすべて出典の実測で、推定は「推定」と明記した。

## 1. このディレクトリの構成

| ファイル | 内容 |
|---|---|
| [strata-existing-optimizations.md](strata-existing-optimizations.md) | Strata が既に実装している最適化の一覧（再提案よけ）。アーキ別既定スイッチ集合と「P40 で測って採らなかったもの」も含む |
| [pascal-arch-limits.md](pascal-arch-limits.md) | P40 の仕様（HBM2 ではなく GDDR5 という訂正を含む）、48 KB shared / L2 3 MB / FP16 1/64 / dp4a、sm_61 で使える・使えない機能の判定 |
| [pascal-gemv.md](pascal-gemv.md) | llama.cpp の Pascal 向け GEMV/MMQ/MMVQ の実測（#30235, #31352, #25479, #30334, #4233）、dequant の帯域上限、Qwen3.8 を P40 で動かした外部実測（#27980） |
| [cuda-graphs-and-launch.md](cuda-graphs-and-launch.md) | CUDA Graphs の一次資料、pre-Ampere の O(node) 起動、llama.cpp の Pascal graph 実測（#6763, #27721）、megakernel、PDL が sm_90+ であること |
| [kernel-fusion.md](kernel-fusion.md) | 融合の一次資料（NVIDIA / Hazy / Liger）、P40 で「境界 1 個 = 約 10 µs」の定量、未実装の融合候補 |
| [moe-runtime.md](moe-runtime.md) | expert caching / 先読み / 投機（HOBBIT, FreeToken #677, DuoServe, FineMoE）、P40 のヒット率 99% と waitCPU 3 ms の意味 |
| [qwen38-flash-next.md](qwen38-flash-next.md) | モデルの公開仕様（論文・NeMo・HF）と Strata の対応、GDN recurrence が既に帯域 floor であること |

## 2. 基準値（P40 の実測、`bench/results/2026-10-12-p40-sm61-kernels/`）

1 window = GPU 49 ms / wall 52-56 ms（engine 0.1.43.1、IQ3_S、8,225 slots、99% hit）。

| 内訳（window あたり） | ms | 割合 |
|---|---|---|
| dense GEMV 系（qkv/q-idx, q+q-idx, out-proj, z, ab, conv, kv-idx） | 13.4 | 27% |
| VRAM expert（mode 8） | 11.16 | 23% |
| hyper-connection read（norm+down+up ×2） | 8.54 | 17% |
| head（+最終 GR read） | 2.87 | 6% |
| router + plan + doorbell | 2.04 | 4% |
| shared expert + 活性量子化 | 2.96 | 6% |
| GDN recurrence（state 3.15 MB × 36 層の read+write） | 0.91 | 1.9% |
| QSA の小ステージ（kv-idx, norm-rope, scores/topk, attention, gate） | 1.55 | 3.2% |
| copy+combine + PCIe grp | 1.12 | 2.3% |
| **GPU の待ち**（waitCPU 3.04 + waitA 1.25 + waitB 0.42 + gap 0.45） | **5.16** | **10.5%** |
| draft layer（別枠、wall 比） | 4.04 | 7.7% |

（合計 49.1 ms。各値は `profile-base.txt` の stage 表から足し上げた。）

prefill（7,039 token、16,005 ms）: GDN 射影 19.6%、expert gate/up 16.0%、QSA attention 11.8%、GR read 12.8%、QSA 射影 8.5%、GDN out 8.3%、gemm down 7.5%、dequant 1.9%。

## 3. 適用候補ランキング（上位 10 件）

「期待幅」は実測の類推か、バイト数からの推定。**太字は推定**。

種別の注記: **#2（GR persist）、#6（head 表の再チューニング）、#7（qsa_pattn）、#9（FP16 キャッシュ）は「既存の実装・opt-in を sm_61 で有効化し、測定・検証する」もの**で、新しいカーネルを書くものではない。新規実装は **#1（ggml overlay）、#3（CPU pool の先読み）、#5（shared expert 合流）、#8（norm+down 融合）**。#4 は設定、#10 は計測。

| # | 候補 | 対象 `file:line` | sm_61 | 期待幅 | 根拠（実測） | 実装コスト | リスク |
|---|---|---|---|---|---|---|---|
| 1 | **Pascal DP4A MMQ の `I` tile を 64→128（J=64 の行）** | ggml `mmq-config-pascal-dp4a.cuh`（pin `3cf03257`、overlay は `cmake/strata_ggml_tq_overlay.cmake`）、engine 側 `src/prefill/moe_mmq.cu` | 使える | **prefill +8〜18%**（pp512-2048、11 型） | llama.cpp #31352: GTX 1080 Ti で Q8_0 pp512 +17.8%、11 型すべて +8.2〜18.0%、greedy 64 token byte 一致、MUL_MAT 2490/2490。K-quant は逆効果（Q4_K −18.8% 等）なので J=64 かつ非 K-quant のみ | 0.5-1 日（overlay パッチ + parity） | 低（bitwise）。pp64 で −3.1% の実測あり。IQ3_XXS が対象形式に入るか要確認 |
| 2 | **GR read を persistent kernel 1 個に**（`STRATA_HC_PERSIST=1` を sm_61 で有効化） | `src/kernels/cuda/fused_gr.cu:3296`（persist_kernel）、`:3013`（entry）、`src/core/verify.cpp:1587`（gr_read_group） | 使える（CUDA 実装あり。80 block × 256 thread は 30 SM に 2.67 block/SM で常駐可能と推定） | **decode −2.2〜−2.5 ms（+4.5〜5%）** | gfx1030: read 1.35-2.10x（T1 51.6→38.1 µs）、window −6.6%（4K）/ −4.4%（32K）、10/10 ペア同方向。P40 の GR は 8.54 ms / 1.26 GB = 148 GB/s（227 GB/s の 65%） | 2-3 日（常駐検査 + KL 測定 + A/B） | 中。**bitwise でない**（別の和の順序）。起動時レジデンシ検査が必須 |
| 3 | **CPU expert pool に 1 層先の予測発行**（RouterLookahead を pool にも使う） | `src/core/expert_source.cpp:3199`（RouterLookahead::run）、`:1127`（warm_scored）、`src/core/verify.cpp:2262`（doorbell_publish） | 使える | **decode −1〜−2 ms（+2〜4%）** | P40: waitCPU 2.50 ms（GDN）+ 0.54（QSA）= 3.04 ms。CPU experts 1.80 ms/window。FreeToken #677 は同型の先読みで 32K decode +22%（43.4→53.0）、miss 206→111、+148 MiB、420 prompt で lossless | 2-3 日 | 中。slot-map の writer を 1 本化しないと壊れる（FreeToken も同じ罠で 1-6/60 が破損）。予測は当たれば bitwise |
| 4 | **sm_61 の bitwise 既定スイッチ集合を作る** | `src/core/nv_defaults.cpp:53`（sm_60 ブロックの直後に分岐を追加） | 使える | **decode +1〜5%** | P100 の 7 個で +17.8%（8/8）、sm_86 の 9 個で +9.6/+14.6%、V100 の FUSE_DECODE で 4-16 µs/token。ただし P40 は hit 99% で adaptive tier がほぼ動かないため ADAPT_* の寄与は小さいと推定 | 1 日 + A/B | 低（すべて bitwise、identity gate で確認） |
| 5 | **shared expert を routed expert の grouped launch に合流** | `src/kernels/cuda/iq_kernels.cu:4230` 付近（native_gu_multi_kernel の launch）、`src/kernels/cuda/shared_expert.cu`、`src/core/verify.cpp:2178` | 使える | **decode −0.5〜−1.2 ms（+1〜2.4%）** | shared+quant が 2.96 ms（6%）、48 層で 96 launch 削減。mode 8 は sm_61 既定 | 3-5 日 | 中。group 内の和の順序を保てば bitwise |
| 6 | **head の GEMV 行クラスを再チューニング**（Q5_K, 2-3 列, ≥12288 行） | `src/kernels/cuda/native_mmvq.cu:3667`（kIlRows_61）、`src/core/native_head.cpp:139`、`src/core/verify.cpp:2608` | 使える | **decode −0.4〜−0.9 ms（+1〜2%）** | head 2.87 ms。microbench では 1 列 1,754 µs（249 GB/s = ほぼ上限）、3 列 2,545 µs（rows 2 で 1,766 µs）。engine の T≈2 は約 150-180 GB/s（推定） | 0.5 日（表のセル + bench） | なし（表は rows/warp だけ変える。bitwise） |
| 7 | **QSA prefill attention を sm_6x カーネルで**（`STRATA_PATTN_SM60=1`） | `src/kernels/cuda/qsa_pattn.cu:35`（pattn_mode）、`:402`（qsa_pattn_batch）、`src/prefill/prefill.cpp:4417` | 使える（`sm6x_device()` は 6.1 を含む） | **prefill +3% 前後（長文）** | P40 の prefill attention は 1,860 ms / 16,005 ms（11.6%）。GP100 で 43.3→30.8 ms/層（26K, 1.4x）。`qsa_pattn_parity` が byte 一致を検査 | 1-2 日（sm_61 で parity + A/B） | 低（同じ演算順。bitwise を謳っている） |
| 8 | **GR read の norm + down を 1 kernel に**（bitwise 版） | `src/kernels/cuda/fused_gr.cu:281`（gr_norm_multi_kernel）、`:333`（gr_down_multi_kernel） | 使える | **decode −0.3〜−0.9 ms（+0.6〜1.8%）** | 96 境界/window 削減 + xn の write/read 7.7 MB 削減。境界 1 個 ≈ 10 µs（推定） | 2 日 | 低（`gr_parity` で bitwise 確認） |
| 9 | **dense 重みの FP16 キャッシュを sm_61 で小予算だけ使う** | `src/prefill/gemm.cu:1404`（`STRATA_PF_W16_CACHE_MIB`）、`:1442`（dequant_f16） | 使える（既定は「expert cache が VRAM を埋めるカードでは 0」） | **prefill +1〜5%（短い prompt）** | P40 の dequant は 559-token prompt で 139 ms / 2,469 ms（5.6%）、7K で 1.9%。engine の scratch では cached/uncached が bitwise（`gemm_w16_parity`） | 0.5 日（env + A/B） | VRAM。実装は 2 GiB の reserve を要求するので、P40 の残り約 8 GiB では expert slot を削るか OOM になりうる |
| 10 | **window graph の node 数を測ってから、まとめられる launch を統合** | `src/core/mtp.cpp:932`（STRATA_VERIFY_NODES）、`include/strata/core/dispatch_stats.hpp:70`（graph_launch） | 使える | **不明（起動コスト自体は 0.1% 未満と推定）** | pre-Ampere の graph launch は O(node) で 200 ns/node（CUDA 11.8）、device gap 60 ns/node。1000 node で約 200 µs。ただし融合によるメモリ削減の方が大きい（kernel-fusion.md） | 0.5 日（計測のみ） | なし |

### ランキングの考え方

- 1 位は「外部で sm_61 実測があり、Strata が同じコード（llama.cpp の `mmq.cuh`）を使っている」ため、確度と費用対効果が最も高い。
- 2 位は「P40 の GR が帯域の 65% しか出ておらず、他アーキで 1.35-2.10x の実測がある」ため、decode の最大単一レバー。
- 3 位は「GPU が 3 ms 待っている」という直接の実測があり、FreeToken の同型手法が lossless で +22% を出しているため。ただし Strata は既に file tier 用の lookahead を持っているので、**新規実装は CPU pool 側だけ**。
- 4 位は費用がほぼゼロで、他アーキの bitwise 集合がそのまま使えるため。ただし P40 のヒット率 99% を考えると控えめに見積もる。
- 5-9 位は個別の fusion / 表 / switch。合計すると decode +4〜8% 程度になる見込み（推定）。
- 10 位は「測ってから決める」項目。P40 では起動コストそのものは小さい。

## 4. 既存 opt-in で sm_61 に未適用のもの（新規実装ではなく A/B 候補）

| switch | 何 | 他アーキの実測 | P40 の期待（推定） | 注意 |
|---|---|---|---|---|
| `STRATA_EXPDEC=1` | 窓の活性を shared に置く expert gate/up | 2080 Ti で gate/up −11〜−23%、end-to-end +1.1% | decode +1〜2% | CUDA は既定 OFF。48 KB shared に収まる（entries 4 個ずつ） |
| `STRATA_HC_Q8=1` | GR 重みを int8 化 | — | decode +1〜2% | **rounding が変わる**（bitwise でない） |
| `STRATA_IQ_STAGE_TABLES=1` | IQ3 の codebook を shared に | P100 で end-to-end 変化なし、3090 Ti で 3-13% 遅い | 0 | 期待しない |
| `STRATA_GR_UP75=1` | GR up の別カーネル | T≥3 で 27.3→24.6 µs | 0〜+1%（T 2.14 平均） | 既に P40 で T1-2 は不利と実測 |
| `STRATA_GU_SWIGLU=1` | gate/up + SwiGLU + q8_1 を 1 kernel | sm_86 で −0.8/−1.4 ポイント | decode +1% | 形式別に A/B（#30334 の教訓） |
| `STRATA_FUSE_DECODE=1` | decode の operator fusion | V100 で 4-16 µs/token | decode +0.5〜1% | bitwise |
| `STRATA_FUSE_WAIT=2` | 待ち 2 つを 1 つに | sm_60 で −1.6 | decode +1% | bitwise |
| `STRATA_SIDE_MIXER=2` | mixer を side stream で | sm_60 で −1.8、sm_86 で −1.2/−2.3 | decode +1〜2% | bitwise |
| `STRATA_ATTN_DAILY=1` / `STRATA_QSA_ROWS=1` | decode attention / block score | sm_86 で小 | decode +0.2% | bitwise |
| `STRATA_B6_MMVQ_ROWS=1` | 複数列で活性断片を再利用 | sm_60 で −1.3 | decode +0.5% | bitwise。il 表と重複する可能性 |
| `STRATA_SPEC_COST=1` | draft 窓をコストで選ぶ | sm_121 で 1.047x | decode +1〜2% | 出力は変わらない（全 draft を verify する） |
| `STRATA_PF_W16_CACHE_MIB=512` | dense FP16 キャッシュ | 上記 #9 | prefill +1〜5% | VRAM |
| `STRATA_MMQ_DIRECT=1` | MMQ が VRAM 常駐 expert を直接読む | — | prefill +? | `-DSTRATA_MMQ_TAB=ON` ビルドが必要、36-38 MiB/GPU |
| `STRATA_PREFILL_ROUTED_STREAM=1` | prefill の copy 先出し | RTX 3090 で 2K +20% | prefill +? | P40 は wait copy 0.2% なので期待薄 |

## 5. 採らない（P40 で無効・不利と分かっている）

| 手法 | 理由 |
|---|---|
| PDL（`STRATA_DF_PDL`） | **sm_90 以上が必須**（CUDA Programming Guide 4.5）。P40 では常に false |
| L2 persistence / access policy window | **CC 8.0 以上**。P40 の L2 は 3 MB でモデルに対して極小 |
| `cp.async` / `memcpy_async` の非同期性 | sm_80+。Pascal では同期コピーに落ちる |
| tensor core 系（`STRATA_GDN_TC`, `STRATA_PF_FUSED` の sm_80+ 経路, Marlin, Triton/Liger） | Pascal に tensor core が無い。Triton は sm_70 下限 |
| `STRATA_GDN_CHUNKED=1` | 93.7 KB shared が必要（48 KB の壁）。128 SM 以上前提 |
| `STRATA_HC_UPMIX=1` | sm_80+ 専用 |
| `STRATA_EXPERT_NIBBLE=1` | sm_60 専用（GP100）。sm_61 では起動しない |
| GDN recurrence の最適化 | 既に 248 GB/s（36 層 × 3.15 MB の state を read+write）で帯域 floor |
| `STRATA_GR_FAST=1` | P40 で T≤4 は遅い（82.0→88.2 µs at T1）と実測済み |
| `STRATA_PREFILL_CPU_SHARE=auto`, `STRATA_ADAPT_OVERLAP=1` | P40 で利得なしと実測済み |
| expert slot を増やす | P100 実測で限界価値 0.15-0.4 s/GiB。P40 は残り VRAM が約 8 GiB で余地小 |
| 混合精度 expert offload（HOBBIT 型） | 出力が変わる。bit 一致方針の外 |
| 2 枚 P40 の layer split | 1 枚より遅い（コミュニティ実測 35.0 tok/s） |

## 6. 情報の確度

| 区分 | 内容 |
|---|---|
| **一次情報で確認** | P40 の仕様（GDDR5 346 GB/s, 30 SM, 48 KB shared, FP16 1/64, dp4a）、PDL が sm_90+、L2 persistence が sm_80+、cooperative launch が CC 6.0+、llama.cpp の各 PR の数値、Strata の P40 stage 実測、Strata の GR persist gfx1030 実測、Qwen3.8 の仕様（論文・NeMo・HF） |
| **推定（本文で明記）** | 各カーネルの「バイト数からの下限」と実測の差、境界 1 個 ≈ 10 µs、候補 2/3/5/6/8 の P40 での期待幅、node 数 800-1000、GR の 148 GB/s |
| **未確認** | GP102 の L2 = 3 MB（一次資料で未取得）、Strata の window graph の実 node 数、`STRATA_PF_W16_CACHE_MIB` を P40 で有効にしたときの VRAM 収支、IQ3_XXS の gate/up が llama.cpp の `I=128` 対象 11 型に入るか |

## 7. 主要出典（URL と取得日 2026-11-07）

NVIDIA / CUDA:
- CUDA Programming Guide 4.2 CUDA Graphs — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- CUDA Programming Guide 4.4 Cooperative Groups — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html
- CUDA Programming Guide 4.5 Programmatic Dependent Launch（sm_90+）— https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
- CUDA Programming Guide 4.15 L2 Cache Control（CC 8.0+）— https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html
- CUDA Programming Guide 4.21 CUDA Dynamic Parallelism — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html
- CUDA Runtime API, Stream Management — https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__STREAM.html
- Pascal Tuning Guide — https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html
- "Constant Time Launch for Straight-Line CUDA Graphs", 2024-09-11 — https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/
- "Kernel Fusion in NVIDIA CUDA", 2026-07-10 — https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/
- Qwen3.8-Flash-Next（NeMo AutoModel）— https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/qwen/Qwen3.8-Flash-Next

ハードウェア:
- Wikipedia, Pascal (microarchitecture) — https://en.wikipedia.org/wiki/Pascal_(microarchitecture)
- Microway, Tesla P40 (GP102) Up Close — https://www.microway.com/hpc-tech-tips/nvidia-tesla-p40-gpu-accelerator-pascal-gp102-up-close/

llama.cpp:
- #30235 Pascal DP4A MMQ I=64→128（2026-10-09）— https://github.com/ggml-org/llama.cpp/issues/30235
- #31352 I=128 for J=64, 11 types（2026-10-29）— https://github.com/ggml-org/llama.cpp/pull/31352
- #25479 mmvq nwarps for Pascal DP4A（2026-07-09）— https://github.com/ggml-org/llama.cpp/pull/25479
- #30334 P40 MMVQ fusion check misses 6.1（2026-10-11）— https://github.com/ggml-org/llama.cpp/issues/30334
- #4233 mm stride doubles P40 prompt（2023-11-27）— https://github.com/ggml-org/llama.cpp/pull/4233
- #6763 CUDA Graphs intro（2024-04-19）— https://github.com/ggml-org/llama.cpp/issues/6763
- #27721 CUDA graphs on Pascal sm_61（2026-08-25）— https://github.com/ggml-org/llama.cpp/pull/27721
- #27980 qwen4exp slow decode on Pascal（2026-08-29）— https://github.com/ggml-org/llama.cpp/issues/27980

モデル・MoE:
- arXiv:2608.30320（2026-08-31）— https://arxiv.org/abs/2608.30320
- Qwen blog — https://qwen.ai/blog?id=qwen3.8-flash-next
- HuggingFace — https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- HOBBIT, arXiv:2411.01433（2024-11-03）— https://arxiv.org/abs/2411.01433
- DuoServe-MoE, arXiv:2509.07379 — https://arxiv.org/html/2509.07379v2
- FineMoE, EuroSys'26 — https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf
- FreeToken #677 router lookahead（2026-10-13）— https://github.com/FlashML-org/FreeToken/issues/677

実装ブログ:
- Hazy Research, "Look Ma, No Bubbles!"（2025-05-27）— https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Liger-Kernel — https://github.com/linkedin/Liger-Kernel

Strata 内の実測（リポジトリ内）:
- `bench/results/2026-10-12-p40-sm61-kernels/`（profile-base.txt, dequant-wide-parity.txt, mmvq-il-bench-p40-run*.txt, gemm-sweep.txt, engine-runs.txt）
- `bench/results/2026-10-12-hc-persist-gfx1030/`（README.md）
- `docs/OLDER_GPUS.md`, `docs/DETAILS.md`, `docs/KVA.md`, `docs/RESEARCH_RUNS.md`, `docs/EXPERTDEC.md`, `docs/GDN_CONTRACT.md`, `docs/MMVQ_IL_TABLE.md`, `docs/NVIDIA_V100.md`, `docs/VRAM_ELASTIC.md`

## 8. 次の提案（最大 3 件）

1. **#1 の MMQ tile を先に試す**: `mmq-config-pascal-dp4a.cuh` の J=64 行だけ `I=128` にし、P40 で `MUL_MAT` parity → pp512/2048/64 の A/B（BENCH_SEED=42、A-B-A 交互 3 回以上）。1 日で白黒つく。
2. **#2 の GR persist を sm_61 で常駐検査から**: `STRATA_HC_PERSIST=1` で起動し、`fused_gr_persist_init` のレジデンシ判定ログと `STRATA_VERIFY_PROFILE=1` の hc0/hc1 の down/up 時間を見る。通れば KL（`STRATA_LOGITS_DUMP`）で品質を確認。
3. **#3 の前に CPU pool の実測を取る**: `STRATA_DECODE_TIMING=1` の waitCPU と CPU experts の内訳を、expert cache を意図的に縮めた状態（`--expert-cache` を下げる）でも取り、waitCPU が「予測で消せる待ち」か「pool の処理能力不足」かを切り分ける。
