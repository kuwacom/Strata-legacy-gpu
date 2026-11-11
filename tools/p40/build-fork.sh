#!/bin/bash
# 任意の worktree から fork イメージをビルドする共通スクリプト
#   WT=~/work/wt/foo TAG=strata:fork-x LABEL=x bash build-fork.sh
# ビルドロック (/tmp/strata-build.lock) を取る。GPU は使わない
set -u
wt=${WT:?set WT to the worktree}
label=${LABEL:-$(basename "$wt")}
tag=${TAG:-strata:fork-$label}
img=nvidia/cuda:12.9.1-devel-ubuntu24.04
cpus=${BUILD_CPUS:-0-11}
res=${RES:-$HOME/work/results/build-$label}
tests=${EXTRA_TARGETS:-"gemm_f32_parity qsa_pattn_parity"}
mkdir -p "$res" "$wt/engines"
if ! mkdir /tmp/strata-build.lock 2>/dev/null; then
  echo "build lock held by $(cat /tmp/strata-build.lock/owner 2>/dev/null)" >&2; exit 2
fi
echo "build-fork $label $(date -Is) $wt" > /tmp/strata-build.lock/owner
trap 'rm -rf /tmp/strata-build.lock; echo "build lock released $(date -Is)"' EXIT

t0=$(date +%s)
sudo docker run --rm --cpuset-cpus "$cpus" -v "$wt":/work/strata -v "$wt/engines":/work/engines -w /work/strata \
  -e BUILD_DIR=build-native -e MODE=native -e EXTRA_TARGETS="$tests" "$img" sh /work/strata/scripts/build-in-container.sh \
  > "$res/build-$label.log" 2>&1
rc=$?; t1=$(date +%s)
echo "build exit $rc in $((t1-t0))s (log: $res/build-$label.log)"
[ $rc -eq 0 ] || { grep -iE "error|Error [0-9]" "$res/build-$label.log" | head -20; tail -n 40 "$res/build-$label.log"; exit $rc; }

sudo docker run --rm --cpuset-cpus "$cpus" -v "$wt":/work/strata -v "$wt/engines":/work/engines -w /work/strata \
  -e BUILD_DIR=build-native -e LABEL="$label" -e OUT=/work/engines/$label "$img" sh /work/strata/scripts/package-in-container.sh \
  > "$res/package-$label.log" 2>&1
rc=$?; t2=$(date +%s)
echo "package exit $rc in $((t2-t1))s (log: $res/package-$label.log)"
[ $rc -eq 0 ] || { tail -n 40 "$res/package-$label.log"; exit $rc; }

eng=$(ls -d "$wt"/engines/$label/*/ 2>/dev/null | head -1); eng=${eng%/}
if [ -z "$eng" ]; then echo "no engine package under $wt/engines/$label" >&2; exit 3; fi
sudo env STRATA_BUILD_COMMIT=$(git -C "$wt" rev-parse HEAD) "$wt/docker/build.sh" --flavor cuda --engine "$eng" --tag "$tag" \
  > "$res/image-$label.log" 2>&1
rc=$?; t3=$(date +%s)
echo "image exit $rc in $((t3-t2))s tag=$tag total=$((t3-t0))s"
[ $rc -eq 0 ] || { tail -n 40 "$res/image-$label.log"; exit $rc; }
echo BUILD_ALL_OK
