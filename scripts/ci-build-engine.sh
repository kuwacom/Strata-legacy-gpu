#!/bin/sh
# Build the CUDA engine from source and package it, for the fork's image workflow. Runs inside the CUDA devel
# container (nvcc compiles for the listed architectures; no GPU is needed) and leaves an engine component package
# that docker/build.sh can turn into an image.
#
#   docker run --rm -v "$PWD":/work/strata -v "$PWD/engine":/work/engine -w /work/strata \
#     -e ARCHS="61-real;75-real;86-real;89-real;120" nvidia/cuda:12.9.1-devel-ubuntu24.04 \
#     sh scripts/ci-build-engine.sh
#
# Environment:
#   ARCHS        CMAKE_CUDA_ARCHITECTURES for the engine (default 61-real;75-real;86-real;89-real;120)
#   VARIANT      package variant tag, goes in the folder name (default cuda12)
#   MIN_DRIVER   the minimum NVIDIA driver recorded in the package (default 575.51, CUDA 12.9's)
#   BUILD_DIR    build directory (default build-ci)
#   OUT          engine package output directory (default /work/engine)
#   LABEL        build id suffix (default ci)
#   SRC          source tree (default /work/strata, else the current directory)
#   JOBS         parallel compile jobs (default: nproc)
#
# Unlike scripts/build-in-container.sh (the P40 recipe: sm_61 only, -march=native) this is the portable build: every
# host source is built with STRATA_PORTABLE=ON so the package runs on any x86_64 machine, and the architecture list is
# what the caller asks for. The runtime feature switches (STRATA_EXPERIMENTAL_SM60, the K-quant MMQ library, ...) are
# the ones the fork's P40 work needs and are harmless on newer cards.
set -eu
archs=${ARCHS:-61-real;75-real;86-real;89-real;120}
variant=${VARIANT:-cuda12}
min_driver=${MIN_DRIVER:-575.51}
build_dir=${BUILD_DIR:-build-ci}
out=${OUT:-/work/engine}
label=${LABEL:-ci}
jobs=${JOBS:-$(nproc)}
src=${SRC:-/work/strata}
[ -d "$src" ] || src=$(pwd)
export PATH=/venv/bin:$PATH
cd "$src"

# the pinned cmake/ninja live in requirements-build.txt so the build matches the release's
if [ ! -x /venv/bin/cmake ]; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends build-essential ca-certificates git libgomp1 libatomic1 \
    python3 python3-pip python3-venv unzip >/dev/null
  python3 -m venv /venv
  /venv/bin/pip install -q -r requirements-build.txt
fi

# third_party/llama.cpp is not vendored; setup.get_llama_cpp() fetches the pinned commit
[ -f third_party/llama.cpp/ggml/CMakeLists.txt ] || python3 -c "import setup; print(setup.get_llama_cpp())"

echo "ci-build-engine: archs=$archs variant=$variant jobs=$jobs src=$src"
/venv/bin/cmake -G Ninja -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$archs" -DSTRATA_PORTABLE=ON \
  -DSTRATA_GGML_DIR="$src/third_party/llama.cpp"
/venv/bin/cmake --build "$build_dir" -j "$jobs" --target strata strata_kq

# ---- package (the same steps as scripts/package-in-container.sh, with the architecture list parameterised)
stage="$src/stage-$label"
rm -rf "$stage"
mkdir -p "$stage" "$out"
# $out is usually a bind mount, so clean its contents instead of the directory itself
find "$out" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
cp "$build_dir/strata" "$stage/"
if [ -f "$build_dir/libstrata_kq.so" ]; then cp "$build_dir/libstrata_kq.so" "$stage/"; fi
ver=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -1)
ver=${ver:-dev}
cudafull=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
# "61-real;75;120" -> "61;75;120" for BUILD.json, and one --arch sm_NN per entry for the packager
nums=$(printf '%s' "$archs" | tr ';' '\n' | sed -n 's/^\([0-9][0-9]*\).*/\1/p' | paste -sd';' -)
arch_args=""
for n in $(printf '%s' "$nums" | tr ';' ' '); do arch_args="$arch_args --arch sm_$n"; done
printf '{"version": "%s", "source": "fork %s", "archs": "%s", "cuda": "%s", "portable": true}\n' \
  "$ver" "$label" "$nums" "$cudafull" > "$stage/BUILD.json"
python3 packaging/collect_libs.py --binary "$build_dir/strata" --search /usr/local/cuda/lib64 --out "$stage" \
  --also 'libcublasLt.so*'
# shellcheck disable=SC2086
python3 packaging/package_engine.py --src "$stage" --backend cuda --version "$ver" --out "$out" \
  $arch_args --min-driver "nvidia=$min_driver" --variant "$variant" --platform linux-x86_64 \
  --build-id "ci-$label-$ver-cuda$cudafull" --include SYSTEM-LIBS.txt --include 'libstrata_kq*'
for d in "$out"/*/; do python3 packaging/package_engine.py --verify "$d"; python3 packaging/binary_deps.py "$d"; done
ls -la "$out"
echo CI_BUILD_ENGINE_OK
