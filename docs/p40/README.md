# Tesla P40 (sm_61) notes: Qwen3.8-Flash-Next at 512K context

This branch records work done on a single Tesla P40 (24 GB, compute capability 6.1,
no tensor cores) running Strata's `qwen4exp` engine with Qwen3.8-Flash-Next IQ3_XXS.
The machine also has 98 GiB of RAM and a SATA-class disk.

These notes are machine-specific. Numbers below were measured on that machine with
`BENCH_SEED=42`, A-B-A runs, the first run of each arm discarded, and medians reported.
The 5% threshold is used for adoption unless a change is bitwise and reproducible.

## Result

Compared with the previous production image (`0.2.7`, engine 0.1.52), the final
configuration (`fork-v9`, engine 0.1.57) gives:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| prefill 24K (tok/s) | 485 | 734 | **+51%** |
| prefill 128K (tok/s) | 434 | 758 | **+75%** |
| TTFT 24K (s) | 57.9 | 33.6 | **-42%** |
| TTFT 128K (s) | 278.0 | 181.2 | **-35%** |
| TTFT 480K (s) | ~1,532 | 837.4 (fork-v3) | **-45%** |
| prefill 480K (tok/s) | ~303 | 561.6 (fork-v3) | **+85%** |
| decode 4K code / prose (tok/s) | 52.3 / 43.3 | 44.1-51.9 / 41.3-43.4 | no change |
| decode 24K code / prose (tok/s) | 50.3 / 43.6 | 49.3-52.4 / 41.6-43.4 | no change |

The 480K rows were measured on `fork-v3` and have not been re-run since; `fork-v7` is
about 5% faster than `fork-v6` at 128K, which should carry to 480K.

Quality checks at 512K: needle retrieval at 24K/32K/128K (50%) and at 32K/128K/480K
(50% and 90%, `fork-v3`) was correct; an 8-question 128K extraction set scored 8/8;
`gemm_f32_parity` (5,201 sizes) is bitwise, `gemm_bf16_parity` 0 failures,
`qsa_pattn_parity` 468 cases 0 differ and `prefill_swiglu_quant_test` ok on `fork-v7`.
KL between base and final was within base-vs-base noise, with identical top-1 tokens.

## Production configuration

```
CONTEXT=512k
STRATA_SET="ple-io=ram prefill=32768 kv-resident=32768 draft-conf=0.7 \
  STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 \
  STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 spec=4 \
  STRATA_PATTN_SM60=1 STRATA_SELECT_SIMT=1 STRATA_HC_DOWN_BF16=1"
```

Engine flags observed at startup:

```
--max-context 524288 --kv int8 --kv-resident 32768
--ple-io ram --prefill 32768 --rope-scaling yarn --rope-scale 2
--spec 4 --draft dflash2 --draft-conf 0.7
```

At 512K the INT8 KV cache is 1,056 bytes per cell and 12,672 bytes per token
(12 QSA layers), i.e. 6.19 GiB of pinned RAM for the full 524,288 cells; 32,768
cells stay in VRAM. RoPE scaling is YaRN with factor 2.

## Changes that were kept

| Change | Effect | Notes |
|---|---|---|
| Dense MMQ for GDN projections (`STRATA_DENSE_MMQ=1`, `STRATA_MMQ_KQ=1`) | prefill +11 to +13% | Opt-in. Not bitwise; KL within noise, needle/QA unchanged |
| QSA select thread scorer on sm_61 | 480K prefill -31.6%, 128K -11.5% | Bitwise (scores and selections identical) |
| DP4A MMQ I=128 tiles for J=64 rows | prefill +3.1% | Bitwise; limited to types that fit 48 KB shared memory |
| sm_61 bitwise default switch set | decode +2.5% | 10 existing switches, enabled by architecture |
| Router/publish and wait/copy fusion | 97 fewer graph nodes per token | Bitwise; no measurable speed change |
| `STRATA_PATTN_SM60=1` | prefill +6.6 to +7.1% | Bitwise against the prompt attention path; needle/QA unchanged |
| prompt chunk 32768 (was 8192) | prefill +6.6%, TTFT 128K -9% | Env only; weights are re-read and re-dequantized once per chunk |
| `STRATA_SELECT_SIMT=1` | prefill +1.7% at 128K | Different summation order in the block selector; needle/QA unchanged |
| hc down+inject share one bf16->fp32 conversion | ~1% (inside the combined result) | Bitwise; `gemm_bf16_parity` 0 failures |
| `STRATA_HC_DOWN_BF16=1` (narrow BF16 kernel for hc down/inject) | 128K prefill +1.4%; the hc pair -18% | Not bitwise (K summation order, rel RMS ~3.5e-6); needle/QA passed |
| `__launch_bounds__(64, 8)` on the QSA value kernel | 128K prefill +1.45%; `qsa attn` -10.9% | Bitwise; 151->128 registers, 6->8 CTAs an SM (`qsa_pattn_parity` 468 cases) |
| Counted loop for the hyper-connection write-back (`v[]` was in local memory) | `hc write` 8,709 -> 5,099 ms (-41%); 128K prefill +1.8% | Bitwise; 81,920 B of local traffic per token and half removed (SASS STACK 40 -> 0) |
| Dense MMQ activations quantized straight from FP16 | 128K prefill +1.3%; `gdn out proj` -7.6%, `router+shared` -10.3% | Bitwise; removes an f16->f32->q8_1 round trip of T x K per launch |
| Shared expert down projection (K=640) through dense MMQ | `router+shared` -15.1%; 128K prefill +1.2% | Not bitwise (activations rounded to int8); needle 24K/32K/128K and QA 8/8 passed |
| `STRATA_MMQ_SQ=1` (SwiGLU and the down product's q8_1 in one pass) | `gemm gate/up` -4.2%; 128K prefill +0.7% | `prefill_swiglu_quant_test` ok |
| `STRATA_HC_INJ8=1`: an 8-token kernel for the N=4 inject projection (one warp, 8 tokens, weight reused 8 times) | `hc read: inject` 2,756 -> 897 ms (-67%); 128K prefill +1.0% | `gemm_bf16_parity` 0 failures; 98 -> 301 GB/s |
| `STRATA_HC_MIX16=1`: the hyper-connection mix reads the BF16 image instead of `R` | `hc read: mix` 4,525 -> 3,630 ms (-20%); 128K prefill +0.5% | Not bitwise (BF16 rounding, rel RMS 9.3e-4); needle 3/3, QA 8/8 |
| IQ2_S / IQ2_XS through the I=128 MMQ tile at J=48 (the J=64 row is dropped so it is chosen) | `gemm gate/up` -5.4%; 128K prefill +0.7% | Same per-output sums; parity screen and needle/QA passed |
| `STRATA_HC_MIXINJ=1`: the N=4 inject dot rides on the mix pass that already reads the same `xn16` | inject+mix 4,527 -> 4,040 ms (-11%) | `gr_mixinj_parity` 0 failures (`mixed` bit-identical); needle/QA passed |

## Changes that were rejected

- `--draft-conf 0.5`: the apparent +7.7% did not reproduce; the difference was inside
  A-vs-A drift, effective gain was about +2%, and output is not invariant.
- `STRATA_KV_PREFETCH=1`: +0.8% (KV upload is only ~0.35% of a 128K prefill on this GPU).
- `STRATA_KVA=1`: not applicable with the DFlash2 drafter.
- CPU expert pool look-ahead: waitCPU scales with CPU expert count (memory-bandwidth
  bound), so prefetching did not help (+0.8% at best).
- `STRATA_HC_PERSIST=1`, `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_QFUSE=1`: within noise.
- chunk 4096 (-6%), PF_W16 cache (structurally no-op at 512K because VRAM is full),
  KV `q4_0`, `spec` 3/5, `mtp-draft-vocab`, `STRATA_PTC` options that are unavailable
  on sm_61 (PDL, L2 persistence, `cp.async`, tensor cores).
- Raising the prompt chunk from 8192 to 32768 was kept (prefill +6.6%, TTFT 128K -9%),
  but it is not bitwise and it changes the launcher's `--draft-conf`, so both
  `prefill=32768` and `draft-conf=0.7` are set explicitly.
- QSA value kernel with 128 threads (12 heads split in two groups): bitwise
  (`qsa_pattn_parity` 468 cases, 0 differ) but -5.2% at 24K and -3.7% at 128K.
  It is issue-bound, not latency-bound; see `docs/prefill/qsa-value-reject.md`.
- QK on DP4A (`STRATA_QSA_DP4A=1`, opt-in): the `qsa attn` phase is 4.3% slower and
  128K prefill is -0.6%. DP4A shares the FP32 pipeline on Pascal, so int8xint8 does
  not raise throughput; see `docs/prefill/qsa-dp4a.md`.

## Build

The images were built from this tree with CUDA 12.9.1 for `61-real` only:

```
nvidia/cuda:12.9.1-devel-ubuntu24.04
--cpuset-cpus 36-47 -j 12
-DCMAKE_CUDA_ARCHITECTURES=61-real
-DSTRATA_PORTABLE=OFF -march=native
-DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_NATIVE_EXPERTS=ON
-DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON
```

Scripts used on the test machine are under `scripts/` in the per-worktree branches;
the consolidated build recipe is in `docs/build/`.

## Evidence layout

- `docs/references/` - external recipes and architecture limits
- `docs/analysis/` - kernel maps and optimization candidates
- `docs/context/` - 512K feasibility, rollout, and quality
- `docs/prefill/` - GDN dense MMQ, QSA select, chunk/SIMT experiments
- `docs/decode/` - decode flag sweeps
- `docs/optimization/` - sm_61 defaults, fusion, env sweeps
- `docs/integration/` - v2/v3 integrated builds
- `docs/patches/` - the patches applied in each experiment

## Document index

Start with `docs/p40/README.md` (this file) and `docs/references/p40-roofline.md`.
The per-phase budget and the theoretical floor of each phase on this card are in the
latter; every later change is judged against it with `STRATA_PREFILL_TIMING=1`.

Results of this campaign, newest last:

- `docs/prefill/v7-round.md` - hyper-connection write-back (an array in local memory),
  dense MMQ activations quantized straight from FP16, K=640 through dense MMQ, MoE SQ
- `docs/prefill/v8-round.md` - the N=4 inject kernel, the mix reading the BF16 image,
  IQ2_S/IQ2_XS through the I=128 tile at J=48, and the candidates that lost
- `docs/prefill/hc-fuse2.md` - the inject fused into the mix pass
- `docs/prefill/qsa-value-occupancy.md` - the value kernel's `__launch_bounds__(64, 8)`
- `docs/prefill/qsa-value-reject.md`, `qsa-dp4a.md`, `qsa-simt.md`, `qsa-select.md` -
  the QSA experiments that lost, with numbers
- `docs/prefill/chunk-profile.md` - the chunk size sweep and the per-phase profile
- `docs/prefill/hc-down-bf16.md`, `hc-pair.md`, `gdn-gemm.md`, `dense-mmq-quality.md` -
  the earlier prefill adoptions and their quality gates
- `docs/prefill/prefill-knobs.md` - every prefill env switch and what it needs

External recipes and the machine's limits:

- `docs/references/p40-roofline.md` - per-phase FLOPs, bytes, measured efficiency
- `docs/references/gap-analysis.md` - cuBLAS/cuBLASLt, CUDA/Pascal, llama.cpp and SASS
  gap analysis with a prioritised backlog
- `docs/references/pascal-arch-limits.md`, `pascal-gemv.md`,
  `cuda-graphs-and-launch.md`, `qwen38-flash-next.md` - architecture and model notes
- `docs/analysis/prefill-kernel-map.md`, `decode-kernel-map.md`,
  `optimization-candidates.md` - which kernel runs where
- `docs/optimization/` - sm_61 defaults, fusion, env sweeps

## Caveats

- One P40, 98 GiB RAM, SATA-class disk. Results do not transfer unchanged to other
  cards, and the decode path is dominated by GPU-reach wait rather than kernel count.
- The final image was built locally and is not published here.
- Greedy answers are not reproducible run to run on this CPU (no AVX-512); quality
  gates use parity tests, KL, needle retrieval, and fixed-question accuracy rather
  than exact string equality.

## Measurement harness

`tools/p40/` holds the scripts this campaign measured with: `build-fork.sh`,
`runPhase.sh`, `runArms.sh`, `verify.sh` and `phaseDiff.py`, with a README that explains
the one-arm budget, the GPU/build lock protocol and the A/B discipline. They target one
P40 and the test machine's paths.

## P40 (sm_61) での Inkling-2-Mini 最適化と測定

Tesla P40 1 枚 (24 GB, sm_61) で Inkling-2-Mini SQ-10 を動かし、prefill を中心に
最適化した記録。数値はすべてこのホストでの実測で、A/B はアーム間に 60 s を置き、
初回アームは捨て値。品質は parity テストと bit 一致で確認した。

### 到達点

| 文脈 | 起点 (tok/s) | 最終 (tok/s) | 差 |
|---|---:|---:|---:|
| 32K | 178.6 | 274.2 | +95.6 |
| 128K | 94.6 | 212.4 | +117.8 |
| 512K | 97.6 | 114.2 | +16.6 |

- 32K は 119.50 s、512K は 4,592.52 s (前回 32K 126.96 s、512K 4,691.47 s)。128K は FP16 KV、512K は `--kv k8v4 --kv-offload off`
- 512K の decode は 5.2 tok/s (draft dspark) で、draft off の過去値 7.5 より遅い。prefill の改善とは別扱い
- 512K の最新は draft off で 4,593 s、decode 7.7 tok/s、host スピルなし、VRAM ピーク 17.5 GB (17,499 MiB)
- 起点は upstream 0.2.21 の移植直後 (32K と 128K は素の fork、512K は同じ k8v4 構成)

### decode の到達点

| 文脈 | 起点 (tok/s) | 最終 (tok/s) | 差 |
|---|---:|---:|---:|
| 8K | 20.3 | 33.0 | +62.6% |
| 32K | 18.0 | 27.0 | +50.0% |
| 128K | 13.8 | 21.6 | +56.5% |
| 256K | 10.4 | 16.3 | +56.7% |

- 起点は移植直後の draft off。8K/32K は unembed の FP32 常駐、128K/256K はそれに global attention の 1 ウェーブ化と res_ctrl 共有が乗る
- 256K の VRAM ピークは 21,223 MiB / 24,576 MiB で host スピルなし。needle 32K は 7741
- 8K/128K/256K は同一セッションで連続測定 (env は `STRATA_INK_PF_TILE=1 STRATA_INK_PROF=2` のみ)。32K は dspark (gen 64) の 1 本

### 施策

- タイル型 FP32 prefill attention (sm_61)。`STRATA_INK_PF_TILE=1`
- PV SGEMM の形状バリアント。既定 `STRATA_INK_PV_SHAPE=2` (0 で従来形)
- GQA pack の複製削減。既定 `STRATA_INK_GQA_PACK=1` (pack_kv を 3 -> 1 コピー、QK/PV を batch=8)
- S_K MoE の occupancy 版。既定 `STRATA_INK_SK_OCC=1`
- S_K dn の活性 shared 化。既定 `STRATA_INK_SK_DN_SMEM=1` (dn -36%、bit 一致)
- SF4 MoE の起動を実在 expert に限定。既定 `STRATA_INK_SF4_COMPACT=1`
- MoE MMQ の復活 (`-DSTRATA_MMQ=ON`、opt-in)。移植 5 段には QSA value 8 CTA/SM と
  hyper-connection の BF16 down/inject も含む
- 計測用: `STRATA_INK_PROF`、`STRATA_INK_ATTN_PROF`、`STRATA_PREFILL_TIMING`
- unembed の FP32 常駐。既定 `STRATA_INK_HEAD_F32=1` (sm_61 単機 + full logits 経路、bit 一致)
- decode global attention の 1 ウェーブ化。既定 `STRATA_INK_DEC_ATTN=2` (sm_61、M=1 / global / fp16 KV、bit 一致)
- res_ctrl の depth mix 共有。既定 `STRATA_INK_DEC_RC=1` (M=1 decode、bit 一致)
- decode の区間別 prof。`STRATA_INK_DEC_PROF=1` (既定 off、ゼロオーバーヘッド)

### ブランチの積み順

`d46f5f268` (upstream/main = v0.2.21) から線形に積む。

1. `port/upstream-0.2.21` (5 コミット)
2. `perf/mmq-revival` (1 コミット)
3. `perf/ink-final` (9 コミット)
4. `perf/ink-dn-shared` (1 コミット)
5. `perf/ink-final2` (1 コミット)

この 3 つを `inkling` に取り込むと 15 コミット。`perf/ink-final` とのツリー差は
移植分と MMQ 分だけ。`perf/ink-dn-shared` と `perf/ink-final2` を足すと 17 コミットになり、
`perf/ink-final2` とのツリー差は移植分と MMQ 分と docs だけになる。

decode の最適化はその上に次の順で積む。

6. `perf/ink-head-f32` (1 コミット) -> `perf/ink-head-default` (1 コミット)
7. `perf/ink-decode-prof` (1 コミット)
8. `perf/ink-dec-attn` (decode prof を除く 1 コミット。decode prof が先に入っていることが前提)
9. `perf/ink-dec-rc` (decode prof を除く 1 コミット)

decode prof は `perf/ink-decode-prof` / `perf/ink-dec-attn` / `perf/ink-dec-rc` に同一パッチとして
存在するので 1 回だけ入れる。この 5 コミットと既定の整理 1 コミットを足すと 23 コミットになる。

### 文書一覧

計画と移植:

- `port-plan.md` - フォーク 32 コミットを upstream 0.2.21 へ移す計画
- `gdn-regression.md` - GDN prefill が upstream で遅い原因
- `gemm-regression.md` - gemm gate/up の後退と回復
- `inkling-attn-plan.md` - prefill attention の改善計画
- `ink-attn-verify.md` - タイル attention の検証手順

結果:

- `ink-final.md` - 最終構成の 32K / 512K
- `ink-final2.md` - dn shared 既定化後の 512K (draft off) と 32K / 128K
- `dn-smem.md` - S_K dn の活性 shared 化
- `attn-stages.md` - attention のステージ別内訳 (128K / 512K)
- `attn-pv.md` - PV SGEMM の形状バリアント
- `attn-gqa.md` - GQA pack 削減
- `sk-occ.md`, `sk-occ-final.md` - S_K MoE の occupancy
- `sf4-launch.md` - SF4 MoE の起動削減
- `moe-prof.md` - MoE S_K 区間の prof
- `occ-final.md` - occupancy 版の最終確認
- `decode-optimization.md` - decode の区間別内訳と 4 施策 (unembed 常駐 / global attention / res_ctrl 共有 / prof) の統合結果

ランタイムと測定:

- `inkling-runbook.md` - ビルド、起動、bench の手順
- `inkling-gguf-runbook.md` - GGUF の扱い
- `inkling-512k-plan.md` - 512K の計画
- `inkling-baseline.md`, `inkling-128k.md`, `inkling-128k-512k.md` - 文脈長ごとの測定
- `inkling-gguf-uncensored.md` - GGUF (uncensored) の測定
- `inkling-moe-survey.md`, `inkling-moe-prereq.md` - MoE カーネルの調査
- `qwen-baseline.md`, `qwen-mmq-ab.md` - Qwen 側の基準値と MMQ の A/B

リファレンス:

- `ci-migration-notes.md` - フォークのイメージ CI を 0.2.21 のツリーへ移すときの差分
