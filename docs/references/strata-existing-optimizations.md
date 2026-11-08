# Strata が既に実装している最適化（再提案よけの一覧）

対象: Qwen3.8-Flash-Next を Tesla P40 (sm_61) で動かす Strata エンジン。
この文書の目的は「既にあるもの」を列挙し、後の候補ランキングで同じものを再提案しないこと。
記載は 2026-11-07 時点の `~/work/wt/research`（upstream main = 9f7d977d 系）のコードと `docs/` に基づく。

凡例: **既定** = そのアーキで既定ON / **opt-in** = 環境変数でON / **sm_61未適用** = 実装はあるが P40 では既定で使われない。

---

## 1. デコード窓（verify window）そのもの

| 仕組み | 状態 | 根拠 |
|---|---|---|
| 窓全体を 1 つの CUDA graph として capture / replay | 既定（全 CUDA アーキ） | `src/core/verify.cpp:2748` `Verifier::capture`、`src/core/graph.cpp:56` `CapturedGraph::begin` |
| graph の node 優先度（`cudaGraphInstantiateFlagUseNodePriority`） | opt-in `STRATA_GRAPH_PRIO=1` | `src/core/verify.cpp:229` `graph_prio_on()`、`:2879` |
| Programmatic Dependent Launch (PDL) | opt-in `STRATA_DF_PDL=1`、**sm_90+ のみ**（コード側で `ptxVersion>=90 && binaryVersion>=90` を検査） | `include/strata/kernels/pdl.hpp:1-70`、`src/kernels/cuda/verify_kernels.cu:1759` `pdl_supported()` |
| サイドストリーム分岐（mixer の入力だけ読む仕事を別 stream へ） | `STRATA_SIDE_MIXER=2`（sm_60/86/120 で既定、sm_61 は未設定） | `src/core/verify.cpp:259`、`:1398` fork/join |
| `STRATA_DF_BRANCH`（同上の別実装） | opt-in | `src/core/verify.cpp:1391` |
| pipelined windows（2 窓を別 stream で重ねる） | `--pipeline-windows` | `src/core/verify.cpp:5121` |
| 窓を 2 グループに割り、A の expert を CPU が計算する間に B の mixer を GPU で回す | `groups_ == 2`（既定の分割） | `src/core/verify.cpp:1326-1340` |
| 1 window の GPU 内訳を stamp で取る | `STRATA_VERIFY_PROFILE=1` | `src/core/verify.cpp:2626` 付近、`profile_by_t_report` |
| graph の node 数を列挙 | `STRATA_VERIFY_NODES=1` | `src/core/mtp.cpp:932` `mtp_list_nodes`（CUDA は個数のみ） |
| 起動〜graph 呼び出しのホスト側待ちを event poll にする | 既定 | `include/strata/core/graph.hpp:9-16`（memory spin はカーネルを走らせない、`wait_ms` は 0.021 ms）、`:30-31`（submission 1 回 10-22 µs） |

## 2. dense 側（decode GEMV / prefill GEMM）

| 仕組み | 状態 | 根拠 |
|---|---|---|
| interleaved small-batch matvec + アーキ別 rows/warp 表 | sm_61 既定（`kIlRows_61` / `kIl1Rows_61`） | `src/kernels/cuda/native_mmvq.cu:3667`、`:3809` `kIlArch` |
| 1-4 列 GEMV の interleave 量子化（q8_1 IL） | 既定 | `native_mmvq.cu:3600` 付近 `native_quantize_q8_1_il_kernel` |
| 5-8 列の第2表 | sm_120 のみ | `MMVQ_IL_TABLE.md` |
| hoisted-load 変種 | opt-in `STRATA_MMVQ_IL_PF`、sm_80 のみ | `MMVQ_IL_TABLE.md:40` |
| 複数列で活性断片を使い回す | `STRATA_B6_MMVQ_ROWS=1`（sm_60/86 既定、sm_61 未設定） | `native_mmvq.cu:1230`、`:1550` |
| prefill の dense 重みを FP32 に広げて `cublasSgemm` | opt-in `STRATA_DENSE_F32=1`（sm_6x のみ、cuBLAS 依存の bit 一致） | `src/prefill/gemm.cu:861-912`、`:1199` |
| prefill の広い dequant（1 thread / 8 values, 16-bit load, 16-byte store） | sm_61 既定（`STRATA_DEQUANT_WIDE=0` で戻す） | `docs/OLDER_GPUS.md:259-264`、bench `2026-10-12-p40-sm61-kernels` |
| dequant を side stream へ | opt-in `STRATA_DEQUANT_SIDE` | `src/prefill/gemm.cu:1477` |
| FP16 重みキャッシュ（1 回 dequant して VRAM に保持） | opt-in `STRATA_PF_W16_CACHE_MIB`（auto はバッチ時のみ既定） | `src/prefill/gemm.cu:1404-1450` |
| prefill の MMQ（llama.cpp の mmq.cuh をリンク） | 既定（K-quant / i-quant 共通） | `src/prefill/moe_mmq.cu`、`include/strata/prefill/mmq_kq_abi.hpp` |
| MMQ が VRAM 常駐 expert をコピーせず直接読む | opt-in `STRATA_MMQ_DIRECT`（`-DSTRATA_MMQ_TAB=ON` ビルド） | `docs/DETAILS.md:1079` |

## 3. expert（MoE）

| 仕組み | 状態 | 根拠 |
|---|---|---|
| expert mode 8（codebook grid + 活性を shared に、SwiGLU/q8_1 融合） | sm_61 既定（`STRATA_SM61_TABLE=0` で CUDA layout） | `src/kernels/cuda/iq_kernels.cu:6167-6215` `exp_mode()` |
| expert cache（VRAM slot, LRU/adaptive tier） | 既定 `--expert-cache auto` | `src/core/expert_cache.cpp` |
| adaptive tier（RAM 常駐 arena から swap） | 既定（カード依存） | `docs/VRAM_ELASTIC.md`、`src/core/expert_cache.cpp` |
| foresight swap slots | opt-in `STRATA_FS_SLOTS=N` | `docs/DETAILS.md:1078` |
| 次レイヤの expert を router lookahead で先読み（専用 thread, confidence/budget/deadline 付き） | 実装済み `RouterLookahead`（file tier 用） | `src/core/expert_source.cpp:3199` `run()`、`:1127` `warm_scored`、`:4880` `submit()` |
| io prefetch の予算・締切・confidence 足切り | 実装済み（149-prefetch） | `src/core/expert_source.cpp:1091-1160` |
| expert の重複 slot 回避（helper GPU） | opt-in `STRATA_DISJOINT_ADAPT=1` | `docs/DISJOINT_EXPERT_CACHE.md` |
| PCIe share の自動調整 | 既定（カード依存） | `docs/DETAILS.md:1050` |
| route prior（常駐 expert を優先） | opt-in `STRATA_ROUTE_PRIOR=1.5` | `src/core/verify.cpp:282`、`src/kernels/cuda/route_prior.cu` |
| verify 窓の活性を shared に置く expert カーネル | opt-in `STRATA_EXPDEC=1`（CUDA は既定OFF、gfx12 は既定ON） | `docs/EXPERTDEC.md`、`iq_kernels.cu:2447` |
| gate/up + SwiGLU + q8_1 を 1 kernel | `STRATA_GU_SWIGLU=1`（sm_86/120 既定、sm_61 未設定） | `iq_kernels.cu:4249-4254` |
| expert の 4-bit コード化（IQ2/IQ3 の 16 値グリッド） | opt-in `STRATA_EXPERT_NIBBLE=1`、**sm_60 のみ** | `docs/OLDER_GPUS.md:210-253` |
| codebook を shared に stage | opt-in `STRATA_IQ_STAGE_TABLES=1`（Pascal 向け） | `docs/OLDER_GPUS.md:136` |
| prefill の expert を group 単位でまとめる / weight-stationary | `STRATA_MMQ_WS` は sm_60 専用、`STRATA_PF_FUSED` は gfx12 と sm_120(80SM+) | `docs/DETAILS.md:1052`、`OLDER_GPUS.md:191` |

## 4. hyper-connection（GR）と attention

| 仕組み | 状態 | 根拠 |
|---|---|---|
| GR read（norm + down + up + inject）を 3 kernel / 1 呼び出しに融合 | 既定 | `src/kernels/cuda/fused_gr.cu:2806` `fused_gr_read_multi_slice` |
| 複数 token を 1 起動で読む（T 1-8） | 既定 | 同上 `GrMulti` |
| latency-hidden norm/up（`gr_norm_fast` / `gr_up_fast`） | opt-in `STRATA_GR_FAST=1`、**P40 では T≤4 で遅いと実測**（82.0→88.2 us at T1） | `fused_gr.cu:1755`、bench `2026-10-12-p40-sm61-kernels/README.md` |
| GR up の別カーネル（T≥3 で速い） | `STRATA_GR_UP75=1`（sm_60/86 既定、sm_61 未設定） | `fused_gr.cu:557-559`、`:1547` |
| GR read を 1 つの persistent kernel に（grid barrier 2 回） | opt-in `STRATA_HC_PERSIST=1`、**既定OFF**、gfx1030 で 1.35-2.10x | `fused_gr.cu:3296-3640`、bench `2026-10-12-hc-persist-gfx1030` |
| GR 重みを int8 化（scale/32） | opt-in `STRATA_HC_REQ8=1`（rounding が変わる） | `docs/DETAILS.md:1115` |
| GR up + mix の融合 | `STRATA_HC_UPMIX=1`、**sm_80+ のみ** | `docs/DETAILS.md:1112` |
| GR read を 2 kernel に分割する版 | `STRATA_GR_V3`（sm_120 で既定、sm_86 では遅い） | `docs/DETAILS.md:1114` |
| decode attention 2 cell/step | `STRATA_ATTN_DAILY=1`（sm_86/120 既定） | `src/kernels/cuda/qsa_decode_attn.cu:832` |
| QSA block score を query レジスタで | `STRATA_QSA_ROWS=1`（sm_86 既定） | `src/kernels/cuda/qsa_select.cu:2098` |
| prompt の QSA attention を sm_6x カーネルで（int8 K/V） | opt-in `STRATA_PATTN_SM60=1`（cc 6.x で使える、**sm_61 未測定**） | `src/kernels/cuda/qsa_pattn.cu:35`、`:402` |
| 長文 select の multi-CTA top-k | `STRATA_SELECT_LONG=1`（cc 6.x で既定） | `docs/OLDER_GPUS.md:183-190` |
| decode GEMV / MoE の operator fusion（#1097） | `STRATA_FUSE_DECODE=1`（sm_60 既定） | `include/strata/kernels/fuse_decode.hpp` |
| 待ち合わせ 2 つを 1 つに畳む | `STRATA_FUSE_WAIT=2`（sm_60 既定） | `src/core/verify.cpp:436-452` |

## 5. 先読み・投機・長文

| 仕組み | 状態 | 根拠 |
|---|---|---|
| MTP draft layer（round / step graph を capture） | 既定 `--mtp` | `src/core/mtp.cpp:145-152`（`round_exec_` / `step_exec_`） |
| DFlash2 drafter | opt-in | `src/core/dflash2.cpp` |
| prompt lookup（n-gram 窓, 独自 graph） | 既定 12（8-row CUDA build, 1 GPU） | `docs/DETAILS.md:1088` |
| KVA（late layer を state だけ作る近似） | opt-in `STRATA_KVA=1` | `docs/KVA.md` |
| KV prefetch（層の KV upload を別 stream で重ねる） | opt-in `STRATA_KV_PREFETCH=1` | `docs/KV_PREFETCH.md` |
| prefill head alias（head の VRAM を prompt buffer に貸す） | opt-in `--prefill-head-alias` | `docs/PREFILL_HEAD_ALIAS.md` |
| prefill の routed stream（PCIe copy を先出し） | opt-in `STRATA_PREFILL_ROUTED_STREAM=1` | `docs/DETAILS.md:1121-1133` |
| chunk 自動（`--prefill auto`） | 既定 | `docs/DETAILS.md:1310` |
| CPU expert pool（NUMA / affinity / aux cpus） | 既定 | `docs/DETAILS.md:1169-1246` |
| prefill の CPU share | `STRATA_PREFILL_CPU_SHARE=auto`（小 chunk のみ） | `docs/DETAILS.md:1048` |

## 6. アーキ別の既定スイッチ集合

`src/core/nv_defaults.cpp` の `nv_default_env()` は sm_86 / sm_60 / sm_121 の 3 つだけを定義している。

| アーキ | 集合 | 実測 |
|---|---|---|
| sm_86 | 9 個（ADAPT_NOWAIT/LAG, CPU_IQ_ALL_EXACT1, SIDE_MIXER=2, ATTN_DAILY, GR_UP75, GU_SWIGLU, QSA_ROWS, B6_MMVQ_ROWS） | 2 カードで +9.6% / +14.6%（8/8）、identity gate 一致 |
| sm_60 | 7 個（ADAPT_NOWAIT/LAG, SIDE_MIXER=2, FUSE_WAIT=2, B6_MMVQ_ROWS, FUSE_DECODE, GR_UP75） | P100 で +17.8%（8/8）、identity gate 一致 |
| sm_121 | PF_FUSED, PREFILL_NVFP4=w4a4, HC_UPMIX, SIDE_MIXER=2, SPEC_COST | GB10 で個別実測 |
| **sm_61** | **なし** | — |

→ P40 は「wide dequant / mode 8 / kIlRows_61 / DENSE_F32(opt-in)」の 4 つだけが sm_61 固有で、上の bitwise 集合に相当するものが無い。
これが候補ランキングの 4 番目（`nv_defaults.cpp:53`）の根拠。

## 7. P40 で測って「採らなかった」もの（再提案しない）

bench `2026-10-12-p40-sm61-kernels/README.md` より:

- `STRATA_GR_FAST=1`（latency-hidden GR read）: T1 82.0→88.2 us、T2 92.4→102.6 us。T5 以上でのみ速い（192→181 us）。sm_61 は plain のまま。
- `STRATA_PREFILL_CPU_SHARE=auto`: P40 では利得なし。
- `STRATA_ADAPT_OVERLAP=1`: P40 では利得なし。
- `STRATA_GR_FAST` 以外の GR 変種（`STRATA_HC_SPLIT` の split 版）も T≤4 では不利。

## 8. 未実装であることの確認（候補にする前に）

以下はコードを grep して「無い」ことを確認した:

- `cudaLaunchCooperativeKernel` / `cooperative_groups` の使用: 無し（GR persist は独自の atomic barrier）。
- CUDA dynamic parallelism（`<<<>>>` を device から）: 無し。
- `cudaStreamCreateWithPriority` を decode の本流に使う箇所: `STRATA_GRAPH_PRIO` の graph node 優先度のみ。
- L2 persistence（`cudaLimitPersistingL2CacheSize` / `accessPolicyWindow`）: 無し（sm_80+ 専用なので妥当）。
- sm_61 用の MMQ tile 設定の上書き: 無し（ggml の `mmq-config-pascal-dp4a.cuh` をそのまま使う）。
- CPU expert pool 側の「次レイヤ先出し」: `RouterLookahead` は file tier 用。CPU pool の job 発行は層内で閉じている（`src/core/verify.cpp` の per-layer host 8.48 ms の内訳 plan/actq/jobs/CPU）。

## 9. 出典（リポジトリ内）

この文書の記述は次のファイルの実コードと実測に基づく（すべて 2026-11-07 時点の `~/work/wt/research`）:

| 出典 | 用途 |
|---|---|
| `src/core/verify.cpp`, `src/core/graph.cpp`, `include/strata/core/graph.hpp` | 窓の graph capture、待ち、stage 名 |
| `src/core/nv_defaults.cpp` | アーキ別の既定スイッチ集合 |
| `src/kernels/cuda/fused_gr.cu`, `include/strata/kernels/fused_gr.hpp` | GR read と persistent 版 |
| `src/kernels/cuda/native_mmvq.cu`, `include/strata/kernels/pdl.hpp` | GEMV 表、PDL |
| `src/kernels/cuda/iq_kernels.cu` | expert mode 8、EXPDEC、GU_SWIGLU |
| `src/core/expert_source.cpp`, `include/strata/core/expert_source.hpp` | RouterLookahead、io prefetch |
| `src/prefill/gemm.cu`, `src/prefill/moe_mmq.cu` | DENSE_F32、FP16 キャッシュ、MMQ |
| `src/core/mtp.cpp` | draft graph と node 列挙 |
| `docs/OLDER_GPUS.md`, `docs/DETAILS.md`, `docs/EXPERTDEC.md`, `docs/MMVQ_IL_TABLE.md`, `docs/NVIDIA_V100.md`, `docs/KVA.md`, `docs/KV_PREFETCH.md`, `docs/RESEARCH_RUNS.md`, `docs/GDN_CONTRACT.md`, `docs/VRAM_ELASTIC.md`, `docs/PREFILL_HEAD_ALIAS.md`, `docs/DISJOINT_EXPERT_CACHE.md`, `docs/BATCHING.md`, `docs/COMMUNITY_BENCHMARKS.md` | 仕様・実測・スイッチの意味 |
| `bench/results/2026-10-12-p40-sm61-kernels/` | P40 の profile とカーネル実測 |
| `bench/results/2026-10-12-hc-persist-gfx1030/` | GR persistent の実測 |
| `bench/results/2026-10-10-community-2x-p40-iq3_s/`, `bench/results/2026-10-05-community-2x-p40/` | 2 枚 P40 のコミュニティ実測 |

外部の一次情報は各テーマ別ファイル（`pascal-gemv.md` など）に URL と取得日を記載している。
