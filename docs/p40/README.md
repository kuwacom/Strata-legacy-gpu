# Tesla P40 (sm_61) notes: Qwen3.8-Flash-Next at 512K context

This branch records work done on a single Tesla P40 (24 GB, compute capability 6.1,
no tensor cores) running Strata's `qwen4exp` engine with Qwen3.8-Flash-Next IQ3_XXS.
The machine also has 98 GiB of RAM and a SATA-class disk.

These notes are machine-specific. Numbers below were measured on that machine with
`BENCH_SEED=42`, A-B-A runs, the first run of each arm discarded, and medians reported.
The 5% threshold is used for adoption unless a change is bitwise and reproducible.

## Result

Compared with the previous production image (`0.2.7`, engine 0.1.52), the final
configuration (`fork-v3`, engine 0.1.57) gives:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| prefill 24K (tok/s) | 485 | 595 | **+22.6%** |
| prefill 128K (tok/s) | 434 | 586 | **+35.1%** |
| TTFT 24K (s) | 57.9 | 47.2 | -18.5% |
| TTFT 128K (s) | 278.0 | 205.7 | -26.0% |
| TTFT 480K (s) | ~1,532 | 889.5 | **-42%** |
| prefill 480K (tok/s) | ~303 | 528.7 | **+74%** |
| decode 4K code / prose (tok/s) | 52.3 / 43.3 | 52.5 / 43.3 | no change |
| decode 24K code / prose (tok/s) | 50.3 / 43.6 | 51.9 / 44.5 | noise |

Quality checks at 512K: needle retrieval at 32K/128K/480K (50% and 90%) was correct;
an 8-question 128K extraction set scored 8/8 with byte-identical answers; KL between
base and final was within base-vs-base noise, with identical top-1 tokens.

## Production configuration

```
CONTEXT=512k
STRATA_SET="ple-io=ram prefill=auto STRATA_DENSE_F32=1 \
  rope-scaling=yarn rope-scale=2 \
  STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 spec=4"
```

Engine flags observed at startup:

```
--max-context 524288 --kv int8 --kv-resident 32768
--ple-io ram --prefill auto --rope-scaling yarn --rope-scale 2
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

## Changes that were rejected

- `STRATA_PATTN_SM60=1`: prefill +6.5% but changed real answers in 2 of 4 prompts;
  held back pending a larger quality evaluation.
- `STRATA_KV_PREFETCH=1`: +0.8% (KV upload is only ~0.35% of a 128K prefill on this GPU).
- `STRATA_KVA=1`: not applicable with the DFlash2 drafter.
- CPU expert pool look-ahead: waitCPU scales with CPU expert count (memory-bandwidth
  bound), so prefetching did not help (+0.8% at best).
- `STRATA_HC_PERSIST=1`, `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_QFUSE=1`: within noise.
- chunk 4096 (-6%), PF_W16 cache (structurally no-op at 512K because VRAM is full),
  KV `q4_0`, `spec` 3/5, `mtp-draft-vocab`, `STRATA_PTC` options that are unavailable
  on sm_61 (PDL, L2 persistence, `cp.async`, tensor cores).
- `chunk 16384` (+3%) and `STRATA_SELECT_SIMT=1` (+1.5%) remain conditional: the first
  is not bitwise and changes the launcher's `--draft-conf`; the second is below 5%.

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

