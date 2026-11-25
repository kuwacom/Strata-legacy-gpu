# フォークのイメージ CI を upstream 0.2.21 へ移すときの差分

## 前提

- 対象は fork `main` の `.github/workflows/fork-images.yml` と `scripts/ci-build-engine.sh`
- `inkling` の基点 `d46f5f268` (v0.2.21) にはこの 2 ファイルが無い。0.2.21 のツリーでは CI そのものが存在しないため、このブランチでは追加も変更もしていない
- 以下は 0.2.21 のツリーを読んで分かった差分。ビルドは実行していないので、実機確認していない項目は「未確認」と書く

## パスの対応

| fork `main` | 0.2.21 |
|---|---|
| `CMakeLists.txt` | `engine/CMakeLists.txt` |
| `src/**` | `engine/src/**` |
| `include/**` | `engine/include/**` |
| `cmake/**` | `engine/cmake/**` |
| `packaging/` | `tools/packaging/` |
| `docker/build.sh` | `tools/docker/build.sh` |
| `requirements-build.txt` | `terminal/requirements-build.txt` |
| `scripts/ci-build-engine.sh` | 無い (CI 用に新規追加が要る) |

## CI ビルドスクリプト

- CMake のソースはリポジトリ直下ではなく `engine/`。`cmake -S .` は `-S engine` にする。ビルドディレクトリと成果物の位置はそのままでよい
- `--target strata strata_kq` は 0.2.21 の既定では成立しない。`strata_kq` は `-DSTRATA_MMQ=ON` と `-DSTRATA_MMQ_KQUANTS=ON` の両方が立つときだけ `engine/cmake/backends/cuda.cmake` で定義される (どちらも既定 OFF)。修正は次のどちらか
  - 両フラグを足して `strata_kq` も積む
  - `strata_kq` と `--include 'libstrata_kq*'` を外す。0.2.21 の既定ターゲットは `strata` と `strata-inkling` (`STRATA_BUILD_INKLING` / `STRATA_BUILD_INKLING2` が既定 ON)
- `python3 -c "import setup; print(setup.get_llama_cpp())"` は使えない。0.2.21 に root の `setup.py` は無く (`terminal/setup.py` のみ)、ggml は `engine/third_party/strata-ggml` と `strata-llama` に同梱されている。この取得ステップは不要のはず (未確認)
- `packaging/collect_libs.py`、`packaging/package_engine.py`、`packaging/binary_deps.py` は `tools/packaging/` にある
- バージョン抽出元が変わる。root の `CMakeLists.txt` は無く、`engine/CMakeLists.txt` の `project(... VERSION 0.1.67)` はエンジン版。製品版は `terminal/strata_cli/__init__.py:15` の `PRODUCT_VERSION = "0.2.21"` で、`app/host/package.json` と `app/desktop/src-tauri/Cargo.toml` の 0.2.21 と一致する。workflow の version はここから取る
- `tools/docker/build.sh --version` の既定も `engine/CMakeLists.txt` を読むため 0.1.67 になる。製品版を明示して渡す

## ワークフロー

- `paths` は `src/**` などを `engine/src/**`、`engine/include/**`、`engine/cmake/**`、`engine/CMakeLists.txt`、`engine/src/inkling2/**`、`tools/packaging/**`、`tools/docker/**`、`terminal/**` に置き換える
- イメージ作成は `docker/build.sh` ではなく `tools/docker/build.sh`
- 0.2.21 の `tools/docker/build.sh` は `--engine` のほかに `--shape-engine` を受ける。`strata-inkling` をイメージに含めるなら、shape engine のパッケージも作って渡す
- sm_61 を含む arch リストには `-DSTRATA_EXPERIMENTAL_SM60=ON` が必須 (`engine/cmake/backends/cuda.cmake` の guard)。既存スクリプトは ON にしている
- arch の `120` が CUDA 12.9.1 で通るかは未確認
- checkout、ghcr ログイン、push の各ステップはそのまま使える

## 未確認

- `tools/packaging/package_engine.py` が 0.2.21 の `strata-inkling` を engine component として扱えるか
- `-DSTRATA_MMQ=ON` で `strata_kq` を積んだとき、同梱の `strata-llama` だけで MMQ の K-quant インスタンスまでビルドできるか
- CUDA 12.9.1 での sm_120 コンパイル
