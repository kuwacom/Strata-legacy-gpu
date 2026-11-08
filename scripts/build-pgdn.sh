#!/bin/bash
# pgdn worktree のビルド: ビルドロック -> コンテナで configure/build -> パッケージ -> docker/build.sh
# 使い方: scripts/build-pgdn.sh <label> [tag]
# CPU 36-47、-j 12。ロックはこのスクリプトが取り、終了時に必ず解放する
set -u
wt=~/work/wt/pgdn
scripts=~/work/wt/build/scripts
res=~/work/results/pgdn
label=${1:-pgdn}
tag=${2:-strata:fork-prefill-gdn}
img=nvidia/cuda:12.9.1-devel-ubuntu24.04
mkdir -p "$res" "$wt/engines"
if ! mkdir /tmp/strata-build.lock 2>/dev/null; then
  echo "build lock held by $(cat /tmp/strata-build.lock/owner 2>/dev/null)" >&2; exit 2
fi
echo "build $label $(date -Is)" > /tmp/strata-build.lock/owner
trap 'rm -rf /tmp/strata-build.lock; echo "build lock released $(date -Is)"' EXIT

t0=$(date +%s)
sudo docker run --rm --cpuset-cpus 36-47 -v "$wt":/work/strata -v "$wt/engines":/work/engines \
  -v "$scripts":/work/build-scripts -w /work/strata \
  -e BUILD_DIR=build-native -e MODE=native -e SRC=/work/strata \
  -e EXTRA_TARGETS="gemm_f32_parity gdn_parity gdn_chunk_parity gdn_rec_parity prefill_mmq_kquant_test" \
  "$img" sh /work/build-scripts/build-in-container.sh > "$res/build-$label.log" 2>&1
rc=$?; t1=$(date +%s)
echo "build exit $rc in $((t1-t0))s (log: $res/build-$label.log)"
[ $rc -eq 0 ] || { tail -n 60 "$res/build-$label.log"; exit $rc; }

sudo docker run --rm --cpuset-cpus 36-47 -v "$wt":/work/strata -v "$wt/engines":/work/engines \
  -v "$scripts":/work/build-scripts -w /work/strata \
  -e BUILD_DIR=build-native -e LABEL="$label" -e OUT=/work/engines/$label -e SRC=/work/strata \
  "$img" sh /work/build-scripts/package-in-container.sh > "$res/package-$label.log" 2>&1
rc=$?; t2=$(date +%s)
echo "package exit $rc in $((t2-t1))s (log: $res/package-$label.log)"
[ $rc -eq 0 ] || { tail -n 60 "$res/package-$label.log"; exit $rc; }

eng=$(ls -d "$wt"/engines/$label/*/ | head -1); eng=${eng%/}
sudo env STRATA_BUILD_COMMIT=$(git -C "$wt" rev-parse HEAD) "$wt/docker/build.sh" --flavor cuda --engine "$eng" --tag "$tag" \
  > "$res/image-$label.log" 2>&1
rc=$?; t3=$(date +%s)
echo "image exit $rc in $((t3-t2))s (log: $res/image-$label.log) tag=$tag total=$((t3-t0))s"
[ $rc -eq 0 ] || { tail -n 60 "$res/image-$label.log"; exit $rc; }
sudo docker image inspect "$tag" --format 'image {{.RepoTags}} {{.Size}}'
echo BUILD_ALL_OK
