#!/bin/sh
# Dockerfile の engine-build 段の後半と同じ手順でエンジン部品パッケージを作る（vision なし）
# 元は ~/opt-work/src-build/package-in-container.sh。出力先だけ OUT で差し替えられるようにした
set -eu
build_dir=${BUILD_DIR:-build-native}
label=${LABEL:-native}
src=${SRC:-/work/strata}
stage=${STAGE:-/work/stage-$label}
out=${OUT:-/work/engine-$label}
export PATH=/venv/bin:$PATH
cd "$src"
# ビルドとは別のコンテナで動かすため、python3 はここで入れる（ビルド側の venv は引き継がれない）
if ! command -v python3 >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends python3 >/dev/null
fi
rm -rf "$stage" "$out"; mkdir -p "$stage" "$out"
cp "$build_dir/strata" "$stage/"
cp "$build_dir/libstrata_kq.so" "$stage/"
ver=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -1)
cudafull=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
echo "{\"version\": \"$ver\", \"source\": \"fork $label\", \"archs\": \"61\", \"cuda\": \"$cudafull\", \"portable\": false}" > "$stage/BUILD.json"
python3 packaging/collect_libs.py --binary "$build_dir/strata" --search /usr/local/cuda/lib64 --out "$stage" --also 'libcublasLt.so*'
python3 packaging/package_engine.py --src "$stage" --backend cuda --version "$ver" --out "$out" --arch sm_61 \
  --min-driver nvidia=575.51 --variant cuda12 --platform linux-x86_64 --build-id "fork-$label-$ver-cuda$cudafull" \
  --include SYSTEM-LIBS.txt --include 'libstrata_kq*'
for d in "$out"/*/; do python3 packaging/package_engine.py --verify "$d"; python3 packaging/binary_deps.py "$d"; done
ls "$out"
echo PACKAGE_OK
