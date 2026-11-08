#!/bin/sh
# コンテナ内で configure とビルドを行う。P40 (sm_61) 専用で、ホスト側のコードは -march=native
# 元は ~/opt-work/src-build/build-in-container.sh（0.1.52 で成功したもの）。フォーク HEAD 用に
# ターゲットを strata / strata_kq に絞り、ggml はピン留めの llama.cpp から取る
set -eu
build_dir=${BUILD_DIR:-build-native}
mode=${MODE:-native}
src=${SRC:-/work/strata}
export PATH=/venv/bin:$PATH
cd "$src"
# cmake 4.4.3 / ninja 1.13.2 は requirements-build.txt から venv に入れる（配布版と同じ版）
if ! command -v ninja >/dev/null 2>&1 || [ ! -x /venv/bin/cmake ]; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends build-essential ca-certificates git libgomp1 libatomic1 python3 python3-pip python3-venv unzip >/dev/null
  python3 -m venv /venv
  /venv/bin/pip install -q -r requirements-build.txt
fi
# フォーク HEAD は third_party/llama.cpp を同梱しない（.gitignore 対象）。setup.get_llama_cpp() が
# ピン留めコミット 3cf03257 を取ってくる。ホスト側で用意済みならそれをそのまま使う
[ -f third_party/llama.cpp/ggml/CMakeLists.txt ] || python3 -c "import setup; print(setup.get_llama_cpp())"
if [ "$mode" = native ]; then
  # 配布版は STRATA_PORTABLE=ON。ここでは全ホストコードを Broadwell 向けにする
  host="-DSTRATA_PORTABLE=OFF -DCMAKE_C_FLAGS=-march=native -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_CUDA_FLAGS=-Xcompiler=-march=native"
else
  host="-DSTRATA_PORTABLE=ON"
fi
# shellcheck disable=SC2086
/venv/bin/cmake -G Ninja -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61-real -DSTRATA_GGML_DIR="$src/third_party/llama.cpp" $host
# テストは必要になったときだけ EXTRA_TARGETS で足す（既定はエンジンと K-quant MMQ ライブラリのみ）
# shellcheck disable=SC2086
/venv/bin/cmake --build "$build_dir" -j 12 --target strata strata_kq ${EXTRA_TARGETS:-}
grep -E "CMAKE_(C|CXX|CUDA)_FLAGS:|GGML_NATIVE|STRATA_PORTABLE|CMAKE_CUDA_ARCHITECTURES" "$build_dir/CMakeCache.txt"
echo BUILD_OK
