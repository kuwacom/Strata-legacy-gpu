# P40 prefill measurement harness

The scripts this campaign used to measure one change on a single Tesla P40 without
disturbing the serving container for longer than a benchmark takes. They were written
for the test machine (`llm-02`), so the paths and the GPU lock are that machine's.

## Why one arm is enough

`STRATA_PREFILL_TIMING=1` prints, after every prompt, a line like

```
strata prefill timing: 137344 tokens, GPU timeline 178582 ms, wall 178614 ms, host staging 512 ms:
  embed+steps 537 (0.3%) ... gdn 26283 (14.7%) ... gemm gate/up 25923 (14.5%) ...
```

`phaseDiff.py A.log B.log 137344 137344` prints that budget side by side. The phases
reproduce to **±10 ms on a 27,000 ms phase (±0.04%)** and the total to ±0.25%, so a
change worth 0.3% is visible from a single 128K prompt. That is what made the loop
affordable: build (about 12 min) + one 128K arm (about 6 min) per candidate.

The rule that came out of the campaign: judge on the phase the change targets **and**
on the GPU timeline. A phase that shrinks while the timeline does not is a phase whose
work moved somewhere else. Host-side phases (`ple`, `host grouping`, `embed+steps`)
move with the machine's CPU load and must not be read as engine changes.

## Files

| file | what it does |
|---|---|
| `build-fork.sh` | `WT=<worktree> TAG=<image> LABEL=<name>` builds that worktree in the CUDA container and tags a docker image. Takes `/tmp/strata-build.lock`; exits 2 if another build holds it. |
| `runPhase.sh` | `IMG=<image> TAG=<name>` stops the serving container, snapshots `/data`, runs one fresh arm with `STRATA_PREFILL_TIMING=1`, runs one prompt, and restores production (with a check that `.env` matches the canonical `prod.env`). Takes `/tmp/strata-gpu.lock`. |
| `runArms.sh` | the same for several arms in one lock: `MANIFEST=arms.txt IMG=... TAG=...`, where each line is `<label>|<extra STRATA_SET>|<tokens>`. One production stop and one restore for the whole batch. |
| `verify.sh` | the full gate for an adopted build: the CUDA parity tests on the GPU, then 24K x2 / 128K, decode, needle at 24K/32K/128K and the 8-question 128K QA, then restore. |
| `phaseDiff.py` | the phase-by-phase diff of two engine logs. |

## The lock protocol

Both locks are directories: `mkdir` is the atomic acquire, `rm -rf` the release. The
GPU lock is held for the whole stop/measure/restore cycle, so nothing else touches the
card while the serving container is down. Production is restored in an `EXIT` trap, and
the trap verifies the container reports `healthy` **and** that
`~/Strata/docker/compose/.env` still equals the canonical `~/opt-work/prod.env` before
releasing, so a failed restore leaves the lock in place for a human to see.

One operational lesson: restarting or smoke-testing production outside the lock while
another benchmark runs puts two models in VRAM and drags the card to 77 tok/s. Everything
that touches the GPU goes through the lock.

## A/B discipline

- one change per build unless the changes touch disjoint phases (the phase budget still
  attributes them)
- 128K first (6 min), then 24K x2 and the quality gates only for a change that won
- quality gates: `gemm_f32_parity` (bitwise), `gemm_bf16_parity`, `qsa_pattn_parity`
  (468 cases), `prefill_swiglu_quant_test`, `gr_mixinj_parity`, then needle at
  24K/32K/128K and the 8-question 128K QA
- a change is adopted only if the targeted phase and the timeline both improve, and the
  quality gates pass; otherwise it is committed on its own branch with the measured
  number so nobody repeats it
