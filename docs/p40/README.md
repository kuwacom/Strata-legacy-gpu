# Tesla P40 (sm_61) notes: Qwen3.8-Flash-Next at 512K context

This branch records work done on a single Tesla P40 (24 GB, compute capability 6.1,
no tensor cores) running Strata's `qwen4exp` engine with Qwen3.8-Flash-Next IQ3_XXS.
The machine also has 98 GiB of RAM and a SATA-class disk.

These notes are machine-specific. Numbers below were measured on that machine with
`BENCH_SEED=42`, A-B-A runs, the first run of each arm discarded, and medians reported.
The 5% threshold is used for adoption unless a change is bitwise and reproducible.

## Result

Compared with the previous production image (`0.2.7`, engine 0.1.52), the final
configuration (`fork-v7`, engine 0.1.57) gives:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| prefill 24K (tok/s) | 485 | 732 | **+51%** |
| prefill 128K (tok/s) | 434 | 740 | **+71%** |
| TTFT 24K (s) | 57.9 | 33.7 | **-42%** |
| TTFT 128K (s) | 278.0 | 185.5 | **-33%** |
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

## Caveats

- One P40, 98 GiB RAM, SATA-class disk. Results do not transfer unchanged to other
  cards, and the decode path is dominated by GPU-reach wait rather than kernel count.
- The final image was built locally and is not published here.
- Greedy answers are not reproducible run to run on this CPU (no AVX-512); quality
  gates use parity tests, KL, needle retrieval, and fixed-question accuracy rather
  than exact string equality.

