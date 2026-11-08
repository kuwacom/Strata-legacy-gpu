#!/bin/bash
# MMQ I=128 parity on the P40: the prompt path's screens, in the CUDA devel image with the build tree mounted.
# The GPU lock must be held (a GPU is needed); every test skips (77) when it finds no card.
#   parity.sh <outdir>
set -u
out=$1
wt=/home/manager/work/wt/mmq
img=nvidia/cuda:12.9.1-devel-ubuntu24.04
mkdir -p "$out"
for t in prefill_mmq_v31_test prefill_mmq_scatter_test prefill_swiglu_quant_test native_grouped_parity; do
  echo "=== $t"
  sudo docker run --rm --gpus all -v "$wt/build-native":/b -w /b "$img" "./$t" > "$out/$t.log" 2>&1
  echo "exit=$?"
  tail -25 "$out/$t.log"
done
