# Pascal DP4A MMQ の I tile 64 -> 128（J=64）を P40 (sm_61) で A/B

作業ツリー `~/work/wt/mmq`（branch `perf/mmq-tile`、HEAD 9f7d977d）。2026-11-06 に実施。
数値はすべて実測。推定は「推定」と明記する。

**結論（v1 §1-10 / v2 §11 / v3 §12）**: 対象形式を IQ3_S まで広げても prefill は +2.75〜+3.44% で 5% に届かず、**採用推奨ではない**。v3 で J=48/40 も I=128 にしたが +2.31〜+3.26% で変わらず（本番は expert を id 順にグループ化するため J=48/40 が選ばれない）。残る IQ2_XS / IQ2_S の 19 層は I=96 が shared に収まるものの fallback=false で out-of-bounds write になるため使えず、この tile 変更の上限は約 3.5〜4%（推定）。パリティは v2/v3 で完全 bitwise。

## 1. 出典と狙い

llama.cpp PR #31352 "cuda : increase Pascal DP4A MMQ I tile for J=64"（2026-10-29 作成、2026-11-07 取得、
https://github.com/ggml-org/llama.cpp/pull/31352）。GTX 1080 Ti (sm_61) で Qwen3-4B Q8_0 の pp512 +17.8%、
変更した 11 型すべてが +8.2〜+18.0%、`MUL_MAT` 2490/2490、greedy 64 token byte 一致。
K-quant は逆効果（Q4_K −18.8% 等）なので変更しない。`I>=192` は shared memory でコンパイル不能。

Strata は llama.cpp の `mmq.cuh` をそのままリンクしている（`src/prefill/moe_mmq.cu`）ので、
同じ変更が原理的に効く。ggml のピンは `3cf03257`（PR より前）なので、P40 では今も `I=64`。

## 2. 変更（パッチ `patches/mmq-i128-sm61.diff`）

1. `cmake/strata_ggml_tq_overlay.cmake`
   - overlay が書き出す `mmq-config-pascal-dp4a.cuh` の `J=64` の行だけ `I=64 -> 128` にする。
     対象は PR と同じ 11 型（Q1_0 Q2_0 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 IQ2_XXS IQ3_XXS IQ4_XS IQ4_NL）。
     K-quant・IQ2_XS・IQ2_S・IQ3_S・MXFP4・NVFP4 は `I=64` のまま。
   - overlay が template instance も書くようにし、pinned の 9 型の instance も overlay 側に置く。
2. `CMakeLists.txt`
   - `strata_mmq` の instance は overlay のものがあればそれを使う。
   - **これが無いと変更はカーネルに届かない**。pinned の `template-instances/mmq-instance-*.cu` は
     `#include "../mmq.cuh"` を自分のディレクトリで解決するので、pinned の `mmq.cuh` / config を読み込む。
     overlay の include path を前に置いても、quote include は自ファイルの隣が先に探される。
     その状態ではホスト側の `fits()` / `overread_bytes()` だけが `I=128` を見て、
     実際の `mul_mat_q` は `I=64` のままになる（速度は変わらず、パリティも通ってしまうので気づきにくい）。
3. `tests/cuda/prefill_mmq_v31_test.cpp`（パリティ用、`patches/parity-j64-test.diff`）
   - 既存の screen は expert あたり 1〜17 行しか作らず `J=64` を踏まない。`{64}` と `{80}` の batch を足した。

### カーネルに入った証拠

IQ3_XXS の J=64 fallback=false カーネルの SASS（`cuobjdump -sass`）:
変更前 1636 行 / REG 96 → 変更後 2796 行 / REG 122。J=48・J=56 のカーネルは不変。
（`~/work/results/mmq/sass-pinned-iq3_xxs.txt` と `sass-overlay-iq3_xxs.txt`）

## 3. このモデルで変わる範囲

`native_experts.txt` の型（48 層）: gate/up = IQ2_XXS 9 層、IQ2_XS 9、IQ3_XXS 7、IQ3_S 13、IQ2_S 10。
down = Q2_0 35、IQ4_NL 13。

| 対象 | 型 | 層数 | 変更 |
|---|---|---|---|
| gate/up | IQ2_XXS, IQ3_XXS | 16 / 48 | I=128 |
| gate/up | IQ2_XS, IQ2_S, IQ3_S | 32 / 48 | 変わらず |
| down | Q2_0, IQ4_NL | 48 / 48 | I=128 |

PR の 11 型のうち、このモデルで使われるのは 4 型だけ。gate/up の 2/3 は対象外。

## 4. ビルド条件

| 項目 | 値 |
|---|---|
| ソース | 9f7d977d + 上記パッチ |
| ggml / llama.cpp | ピン `3cf03257`（`setup.get_llama_cpp()` で取得、ホストに展開） |
| ビルドコンテナ | `nvidia/cuda:12.9.1-devel-ubuntu24.04`、`--cpuset-cpus 36-47`、`-j 12` |
| CUDA アーキ | `-DCMAKE_CUDA_ARCHITECTURES=61-real` のみ |
| flags | `STRATA_PORTABLE=OFF`、`-march=native`（GGML_NATIVE=ON）、`STRATA_ENABLE_CUDA=ON`、`STRATA_EXPERIMENTAL_SM60=ON`、`STRATA_NATIVE_EXPERTS=ON`、`STRATA_Q6K_EXPERTS=ON`、`STRATA_MMQ_KQUANTS=ON`、`STRATA_BUILD_TESTS=ON` |
| 追加ターゲット | `prefill_mmq_v31_test prefill_mmq_scatter_test native_grouped_parity prefill_swiglu_quant_test` |
| パッケージ | `package-in-container.sh`（vision なし、arch sm_61、variant cuda12、min-driver 575.51） |
| イメージ | `strata:fork-mmq-i128`（747,356,675 bytes） |
| ログ | `~/work/results/mmq/build-i128-instances.log`、`package-i128.log`、`image-i128.log` |

ビルドは 2 回（1 回目は instance 差し替え前。イメージも 2 回作った）。最終イメージは 2 回目のもの。

## 5. パリティ

GPU ロック下、`nvidia/cuda:12.9.1-devel-ubuntu24.04 --gpus all` でビルドツリーのテストを実行
（`~/work/results/mmq/ab1/parity/`）。

| テスト | 結果 |
|---|---|
| `prefill_mmq_v31_test` | 6 batch × 9 型 = 54 product すべて合格。J=64 を踏む rows 64 / 80 でも `rel_l2` 0.0011 前後、`max/rms` 0.0045 前後（小さい batch と同じ水準）。FP64 参照に対する screen で、bit-exact ではない |
| `prefill_mmq_scatter_test` | 0 failures（5 型 × T=1,2,7,600,2099、活性 q8_1 のバイト一致） |
| `prefill_swiglu_quant_test` | 18 ケースすべて 0 bytes differ（byte 一致） |
| `native_grouped_parity` | 24 ケース `bitwise equal to v1`、0 failures（変更と無関係の巻き添え確認） |

### logits の bitwise 比較（`STRATA_LOGITS_DUMP`）

4K code プロンプト 1 本、max_tokens 128、temperature 0。

| 比較 | 結果 |
|---|---|
| A vs A（同じイメージ 2 回、決定性の確認） | 72/72 レコード byte 一致 |
| A vs B | rec 0..62 は byte 一致。これは 4K プロンプトの prefill 全体 + 生成 97 token を含む。初めて差が出るのは rec 63（pos 4894 = 生成 101 token 目）で max \|d\| 0.286。以降 argmax は一致（最初のリクエストで位置が揃う窓すべて、生成 126 token 目まで）。リクエスト 2 以降は投機の窓構造（受理数）が変わるため位置がずれ、比較できない |

- **prefill の MMQ 変更は bitwise**: プロンプト全体の logits が byte 一致している
- **decode は 101 token 目から logits がわずかにずれる**。ただし argmax は一致（生成 126 token 目まで確認）し、出力テキストも同じ。
  原因は未確認（推定: drafter の Q8_0 行列も同じ MMQ を通るため draft が変わる、またはタイミング依存の経路。
  decode の expert 積は expert あたり 1 token 前後で `J=8` なので、変更した `J=64` は踏まないはず）
- 回答テキスト（"Paris / 391"）は A・B・A2 で同一

## 6. 速度（A-B-A、`~/bench.py`、BENCH_SEED=42、spec=4、1 本目除外の中央値）

A = `strata:fork-9f7d977-base`（I=64）、B = `strata:fork-mmq-i128`（I=128）。
4K/24K/30K × code のみ × 3 回、max_tokens 128。A1・B1・A2 の順（別ブロックにまたがる）。

| サイズ | A1 | B1 | A2 | A 中央値 | B/A |
|---|---:|---:|---:|---:|---:|
| prefill 4K | 438.3 | 457.1 | 441.1 | 439.7 | **+4.0%** |
| prefill 24K | 481.8 | 494.0 | 478.8 | 480.3 | **+2.9%** |
| prefill 30K | 482.6 | 492.4 | 479.5 | 481.1 | **+2.4%** |
| decode 4K | 53.46 | 51.84 | 51.45 | 52.46 | −1.2% |
| decode 24K | 49.90 | 50.66 | 49.51 | 49.71 | +1.9% |
| decode 30K | 52.66 | 49.17 | 51.54 | 52.10 | −5.6%（参考値） |

- A1 と A2 の差は 0.6% 以内で、B の +2.4〜+4.0% はそのばらつきより大きい
- プロンプト実トークン数は rep ごとに違う（4K = 3,184 / 4,793 / 5,097、24K = 24,944〜27,642、
  30K = 29,593〜33,053）が、A と B は同じ並びの同じプロンプト
- decode は 4K/24K で ±2% 以内（ノイズ）。30K decode の −5.6% は rep のばらつきが大きく（48.87〜52.25）、
  prefill の変更が decode に入る経路が無いことから **未確認**（熱の影響も無い: 全 arm で 1531 MHz、
  温度は A1 51 C / B1 60 C / A2 61 C）

### 内訳（`STRATA_PREFILL_TIMING=1`、31,020 token のチャンク、1 回）

| 位相 | A1 | A2 | B1 | B/A |
|---|---:|---:|---:|---:|
| gemm gate/up | 8,505 ms | 8,504 ms | 7,983 ms | **−6.1%** |
| gemm down | 4,939 ms | 4,938 ms | 3,904 ms | **−20.9%** |
| GPU timeline 合計 | 62,341 ms | 62,743 ms | 60,857 ms | −2.7% |

- down は 48/48 層が対象で、640 行の出力が `I=128` で 10 -> 5 tile になるため伸びが大きい
- gate/up は 16/48 層だけなので −6.1% にとどまる。全体では MMQ の節約は約 1.5 s / 62 s = 2.5%

## 7. 判定: **採用推奨ではない**（ただし方向は正しく、機構も確認できた）

- prefill の改善は 4K +4.0%、24K +2.9%、30K +2.4% で、**5% 閾値に届かない**
- パリティは prefill bitwise・出力テキスト一致で問題なし
- 理由は対象形式の不足: PR の 11 型のうちこのモデルが使うのは 4 型だけで、gate/up の 2/3（IQ2_XS /
  IQ2_S / IQ3_S、32/48 層）は `I=64` のまま
- decode への影響は 4K/24K でノイズ範囲

## 8. 再現

```sh
# 1. ggml（ピン）を用意
cd ~/work/wt/mmq && python3 -c "import setup; print(setup.get_llama_cpp())"

# 2. パッチを当てる（patches/mmq-i128-sm61.diff、patches/parity-j64-test.diff）
git apply patches/mmq-i128-sm61.diff patches/parity-j64-test.diff

# 3. ビルド（ビルドロック、CPU 36-47、-j 12）
mkdir /tmp/strata-build.lock
sudo docker run --rm --cpuset-cpus 36-47 \
  -v ~/work/wt/mmq:/work/strata -v ~/work/wt/mmq/engines:/work/engines \
  -v ~/work/wt/build/scripts:/work/scripts:ro -w /work/strata \
  -e BUILD_DIR=build-native -e MODE=native \
  -e EXTRA_TARGETS="prefill_mmq_v31_test prefill_mmq_scatter_test native_grouped_parity prefill_swiglu_quant_test" \
  nvidia/cuda:12.9.1-devel-ubuntu24.04 sh /work/scripts/build-in-container.sh
sudo docker run --rm --cpuset-cpus 36-47 -v ~/work/wt/mmq:/work/strata \
  -v ~/work/wt/mmq/engines:/work/engines -v ~/work/wt/build/scripts:/work/scripts:ro \
  -w /work/strata -e BUILD_DIR=build-native -e LABEL=i128 -e OUT=/work/engines/i128 \
  nvidia/cuda:12.9.1-devel-ubuntu24.04 sh /work/scripts/package-in-container.sh
eng=$(ls -d ~/work/wt/mmq/engines/i128/*/ | head -1); eng=${eng%/}
sudo env STRATA_BUILD_COMMIT=$(git -C ~/work/wt/mmq rev-parse HEAD) \
  ~/work/wt/mmq/docker/build.sh --flavor cuda --engine "$eng" --tag strata:fork-mmq-i128
rm -rf /tmp/strata-build.lock

# 4. パリティ（GPU ロック下）
mkdir /tmp/strata-gpu.lock && echo "mmq-i128 $(date -Is)" > /tmp/strata-gpu.lock/owner
bash ~/work/wt/mmq/scripts/parity.sh ~/work/results/mmq/parity

# 5. A-B-A（GPU ロック、本番停止、復元まで含む）
SKIP_PARITY=1 SKIP_DUMP=1 DUMP_ARM=dump-A2 DEADLINE_S=3300 \
  bash ~/work/wt/mmq/scripts/block.sh ~/work/results/mmq/ab2 \
  A1:strata:fork-9f7d977-base B1:strata:fork-mmq-i128 A2:strata:fork-9f7d977-base

# 6. 集計
python3 ~/work/wt/mmq/scripts/analyze2.py ~/work/results/mmq/ab2/A1 ~/work/results/mmq/ab2/B1 ~/work/results/mmq/ab2/A2
python3 ~/work/wt/mmq/scripts/cmp_reqs.py ~/work/results/mmq/ab1/dump-A/logits.bin ~/work/results/mmq/ab1/dump-B/logits.bin
```

生ログ: `~/work/results/mmq/`（`ab1/`、`ab2/`、`parity/`、`logits-cmp-AB.txt`、`logits-cmp-AA.txt`、
`logits-reqs-AB.txt`、`logits-argmax-AB.txt`、`sass-*.txt`、`block1.out`、`block2.out`）。

## 9. 逸脱・注意

- ベンチは code のみ（prose は測っていない）。時間予算のため
- 30K の decode は指示外の参考値
- A-B-A は 2 ブロックにまたがった（B1 と A2 が同じブロック、A1 は前のブロック）。A1/A2 の差は 0.6% 以内
- 1 回目のブロックで `block.sh` を実行中に上書きしてしまい、A1 の手前でスクリプトが壊れて中断した。
  本番は復元されず（別エージェントが復元・ロック取得）、その時点で本番 healthy・`.env` 一致・残留なしを確認した
- logits dump は 1 本 1.7 GB（A）/ 1.2 GB（B）/ 0.27 GB（A2）。`STRATA_LOGITS_DUMP` は窓ごとに全 logits を書く
- decode 101 token 目以降の logits 差の原因は未確認。prefill は byte 一致なので、変更した prefill 経路自体は bitwise
- `ab1/dump-*/engine-environ.txt` に `STRATA_LAUNCH_KEY` が入る（起動ごとのローカルキー）。docs には書かない

## 10. 次の提案（最大 3 件）

1. **IQ2_XS / IQ2_S / IQ3_S にも `I=128` を適用**する。この 3 型は gate/up の 32/48 層を占める。
   shared memory は IQ2_XS・IQ2_S が 44,864 B、IQ3_S が 43,808 B（推定、`MMQ_DP4A_TXS_Q8_0_16` /
   `Q8_0` と `J=64` から計算）で sm_61 の 48 KB に収まる。gate/up の対象が 16 -> 48 層になり、
   内訳から +1.5〜2% 上乗せが見込める（推定）。1 ビルド + 1 ブロックで確認できる
2. **down の `I=160` を試す**（640 行の積は tile 数が減るほど良い可能性）。ただし #30235 では Q8_0 の
   `I=160` が 45.7% 悪化しており、期待は低い。まず shared memory の収支を確認してから
3. **decode 101 token 目の差の切り分け**: drafter を FP16 GEMM に固定（`STRATA_MTP_BATCH_F16=1`）して
   同じ logits 比較をする。差が消えれば drafter の Q8_0 MMQ が原因と確定する。1 ブロック
## 11. v2: IQ3_S を追加（2026-11-06/07）

v1 の +2.4〜4.0% では 5% に届かない。gate/up の対象を 16/48 層から増やすため、
IQ2_XS / IQ2_S / IQ3_S の J=64 行にも I=128 を当てられるか確認した。

### 11.1 適用した形式と、除外した形式

shared memory を実測した（`mmq_get_nbytes_shared(config, 610)`、`smpbo = 49,152 B`）。
`~/work/results/mmq/v2/smem-pinned-i64.txt` / `smem-overlay-i128.txt`、`scripts/smem_check.cu`。

| 形式 | J=64 / I=64 | J=64 / I=128 | 判定 |
|---|---:|---:|---|
| IQ3_S | 28,224 B | **46,976 B** | 適用 |
| IQ2_XS | 30,336 B | 51,200 B | **除外**（49,152 B 超） |
| IQ2_S | 30,336 B | 51,200 B | **除外**（同上） |
| IQ3_XXS / IQ2_XXS / Q2_0 / IQ4_NL / Q8_0 | 28,224 B | 46,976 B | 適用（v1 から） |

- IQ2_XS / IQ2_S は Q8_0_16 レイアウト（16-bit スケール）で I に対する傾きが大きく、I=128 では
  51,200 B になる。48 KB を超えると `mul_mat_q_switch_J` が J=64 を飛ばして J=48（2 tile）を選ぶため、
  **変更するとかえって遅くなる**。よって I=64 のまま
- この 2 形式は gate/up の 19/48 層を占める。v2 の対象は IQ2_XXS(9) + IQ3_XXS(7) + IQ3_S(13) = **29/48 層**

### 11.2 ビルド・イメージ

v1 と同じ条件（CPU 36-47、`61-real`、`-j 12`、CUDA 12.9.1）。テストは `prefill_mmq_v31_test` に
IQ3_S のケースを足した（`patches/parity-j64-test.diff`）。イメージ `strata:fork-mmq-i128-v2`
（747,364,775 bytes）。ログ `~/work/results/mmq/v2/build.log`、`package.log`、`image.log`。

### 11.3 パリティ

| テスト | 結果 |
|---|---|
| `prefill_mmq_v31_test` | 6 batch × 9 ビルド済み形式（IQ3_S を含む）= 54 product すべて合格。IQ3_S は rows 17 で rel_l2 0.00111、rows 64 で 0.00104、rows 80 で 0.00113、max/rms 0.0042〜0.0046（他形式と同じ水準） |
| `prefill_mmq_scatter_test` | 0 failures |
| `prefill_swiglu_quant_test` | ok（byte 一致） |
| `native_grouped_parity` | 24 ケース bitwise、0 failures |

logits dump（4K code、temp 0、max_tokens 128）:

| 比較 | 結果 |
|---|---|
| A vs A2（同一イメージ） | 72/72 レコード **byte 一致**（max \|d\| 0） |
| A vs B（v2） | 72/72 レコード **byte 一致**（max \|d\| 0） |

- v2 では **prefill も decode も完全に bitwise**。v1 で見えた「生成 101 token 目からの差」は再現しなかった
  （v1 の 1 回だけの現象。原因は未確認のまま。v2 で差が無いため drafter FP16 の切り分けは不要と判断し未実施）
- 回答テキスト（"Paris / 391"）は A・B・A2 で同一

### 11.4 速度（A-B-A、`~/bench.py`、BENCH_SEED=42、spec=4、1 本目除外の中央値、code と prose）

A = `strata:fork-9f7d977-base`（I=64）、B = `strata:fork-mmq-i128-v2`。

prefill (tok/s):

| サイズ/種類 | A1 | B1 | A2 | A 中央値 | B/A |
|---|---:|---:|---:|---:|---:|
| 4K/code | 447.5 | 461.0 | 449.8 | 448.7 | **+2.75%** |
| 4K/prose | 440.7 | 457.5 | 443.8 | 442.3 | **+3.44%** |
| 24K/code | 487.6 | 503.1 | 488.3 | 488.0 | **+3.12%** |
| 24K/prose | 487.6 | 502.2 | 486.6 | 487.1 | **+3.11%** |
| 30K/code | 478.1 | 493.1 | 477.1 | 477.6 | **+3.25%** |
| 30K/prose | 477.4 | 492.7 | 476.9 | 477.2 | **+3.26%** |

- A1 と A2 の差は 0.7% 以内。B は全 6 条件で +2.75〜+3.44%
- v1（code のみ、+2.4〜+4.0%）と比べると、24K/30K が +2.9/+2.4% → +3.1/+3.3% に上がった。
  IQ3_S の 13 層ぶんの改善

decode (tok/s、参考):

| サイズ/種類 | A1 | B1 | A2 | A 中央値 | B/A |
|---|---:|---:|---:|---:|---:|
| 4K/code | 47.36 | 51.98 | 52.55 | 49.96 | +4.04% |
| 4K/prose | 42.95 | 42.56 | 43.30 | 43.13 | −1.31% |
| 24K/code | 51.82 | 51.15 | 53.21 | 52.52 | −2.60% |
| 24K/prose | 42.41 | 42.63 | 46.41 | 44.41 | −4.01% |
| 30K/code | 50.13 | 49.69 | 51.89 | 51.01 | −2.59% |
| 30K/prose | 43.00 | 43.78 | 42.80 | 42.90 | +2.06% |

- decode はノイズが支配的（同じイメージの A1 と A2 でも 24K/prose が 42.41 vs 46.41 = +9.4%）。
  B の差は方向も一定せず、**decode への影響は検出できない**

内訳（`STRATA_PREFILL_TIMING=1`、34,404 token のチャンク、1 回）:

| 位相 | A1 | A2 | B1 | B/A1 |
|---|---:|---:|---:|---:|
| gemm gate/up | 9,748 ms | 9,582 ms | 8,552 ms | **−12.3%** |
| gemm down | 5,618 ms | 5,537 ms | 4,457 ms | **−20.7%** |
| GPU timeline 合計 | 70,972 ms | 70,836 ms | 68,569 ms | −3.4% |

- gate/up は v1 の −6.1%（16 層）から −12.3%（29 層）へ。層数比（29/16 = 1.8）とほぼ比例し、
  「I=128 が効いている層が増えた」ことと整合する
- 節約は 2,357 ms / 70,972 ms = 3.3%。wall の prefill 改善 +3.1〜3.3% と一致する

### 11.5 判定: **採用推奨ではない**（v1 と同じ。ただし v2 が現状の上限）

- prefill は +2.75〜+3.44% で 5% に届かない
- パリティは完全 bitwise（A/A2/B の 72/72 レコード一致）、出力も同一
- **到達可能な上限の推定**: gate/up の残り 19 層（IQ2_XS 9 + IQ2_S 10）は shared memory の制約で
  I=128 にできない。仮にできたとして、gate/up 全体 9,748 ms のうち 19/48 ≈ 3,858 ms を −12.3% すると
  475 ms の追加節約で、合計 2,832 ms / 70,972 ms = **約 4.0%**（推定）。この tile 変更だけでは
  5% に届かない
- down は既に −20.7% で、これ以上の tile 変更の余地は小さい（I=160 は #30235 で Q8_0 が 45.7% 悪化）

### 11.6 再現（v2）

```sh
# パッチ（v2 は v1 のパッチ + IQ3_S の 1 語 + テストの IQ3_S ケース）
cd ~/work/wt/mmq && git apply patches/mmq-i128-v2.diff patches/parity-j64-test.diff

# shared memory の確認（GPU 不要）
sudo docker run --rm -v ~/work/wt/mmq:/w nvidia/cuda:12.9.1-devel-ubuntu24.04 sh -c \
  "cd /w/scripts && nvcc -arch=sm_61 -std=c++17 -I/w/build-native/ggml-tq-overlay \
   -I/w/build-native/ggml-tq-overlay/include -I/w/third_party/llama.cpp/ggml/src/ggml-cuda \
   -I/w/third_party/llama.cpp/ggml/src -I/w/third_party/llama.cpp/ggml/include \
   smem_check.cu -o smem_check2 /w/build-native/ggml/src/libggml-base.a -lcudart -lcublas && ./smem_check2"

# ビルド・パッケージ・イメージは §8 と同じ（LABEL=i128v2、tag strata:fork-mmq-i128-v2）

# パリティ + logits（A, B=v2, A2）
DUMP_A=strata:fork-9f7d977-base DUMP_B=strata:fork-mmq-i128-v2 DUMP_ARM=dump-A2 \
  DUMP_IMAGE=strata:fork-9f7d977-base bash ~/work/wt/mmq/scripts/block.sh ~/work/results/mmq/v2/ab

# A-B-A（code + prose）
SKIP_PARITY=1 SKIP_DUMP=1 SIZES=4000,24000,30000 REP=3 MAXT=128 KINDS=code,prose \
  bash ~/work/wt/mmq/scripts/block.sh ~/work/results/mmq/v2/ab \
  A1:strata:fork-9f7d977-base B1:strata:fork-mmq-i128-v2
# 2 本目の A は別ブロックで
SKIP_PARITY=1 SKIP_DUMP=1 SIZES=4000,24000,30000 REP=3 MAXT=128 KINDS=code,prose \
  bash ~/work/wt/mmq/scripts/block.sh ~/work/results/mmq/v2/ab A2:strata:fork-9f7d977-base

python3 ~/work/wt/mmq/scripts/analyze3.py ~/work/results/mmq/v2/ab/A1 ~/work/results/mmq/v2/ab/B1 ~/work/results/mmq/v2/ab/A2
python3 ~/work/wt/mmq/scripts/cmp_logits.py ~/work/results/mmq/v2/ab/dump-A/logits.bin ~/work/results/mmq/v2/ab/dump-A2/logits.bin
python3 ~/work/wt/mmq/scripts/cmp_logits.py ~/work/results/mmq/v2/ab/dump-A/logits.bin ~/work/results/mmq/v2/ab/dump-B/logits.bin
```

### 11.7 v2 の逸脱・注意

- ベンチは 3 ブロックに分かれた（A1+B1、A2）。A1/A2 の差は 0.7% 以内で、分けても比較は成立する
- logits dump は 1 本 268 MB（v1 の 1.7 GB は dump の引数渡しのバグで 12 リクエストを回していたため。
  v2 は 1 リクエスト 128 token のみ）
- drafter FP16（`STRATA_MTP_BATCH_F16=1`）の切り分けは未実施。v2 で A/B が bitwise 一致し、
  切り分ける差が存在しないため
- decode の比較はノイズが大きく、3 回の中央値では ±4% 程度の差を判定できない

## 12. v3: J=48/40 の I=128 と IQ2_XS/IQ2_S の I=96 検証（2026-11-07）

v2 は +2.75〜+3.44% で 5% に届かなかった。残る gate/up の 19 層（IQ2_XS 9 + IQ2_S 10）を I=128 に
できないか、また 4K の小さい expert group が選ぶ J=48/40 にも I=128 を広げられるかを確認した。

### 12.1 shared memory と switch_J（実測）

`mmq_get_nbytes_shared(config, 610)`（smpbo = 49,152 B）と `mul_mat_q_switch_J` と同じ選択ループで
測った（`scripts/smem_check_v3.cu`、結果 `~/work/results/mmqv3/smem-final.txt`）。

| 形式 | J | I=64 | I=96 | I=128 |
|---|---:|---:|---:|---:|
| IQ2_XS / IQ2_S | 64 | 30,336 B | **40,768 B（収まる）** | 51,200 B（超過） |
| IQ2_XS / IQ2_S | 48 | 28,224 B | 38,656 B | 49,088 B（64 B 余裕） |
| IQ3_S / IQ3_XXS / IQ2_XXS / Q2_0 / IQ4_NL / Q8_0 | 64 | 28,224 B | 37,600 B | 46,976 B |
| 同上 | 48 | 26,112 B | 35,488 B | 44,864 B |

- J=56 は config が無い（switch_J も選ばない）
- switch_J の選択（ncols_opt = expert group の max_rows）: IQ2_XS は n=64 → J=64、n=80 → J=40、
  n=96 → J=48、n≥100 → J=64。I=96 を入れても n=64/100/200/469/600/1200 は J=64 を選ぶ
  （IQ3_XXS は n=80 → J=40 I=128、n=96 → J=48 I=128、n≥100 → J=64 I=128）

### 12.2 I=96 は除外（out-of-bounds write）

IQ2_XS / IQ2_S の J=64 行（fallback=true と false の両方）を I=96 にしてビルドし、v3.1 screen を回した。

- 数値 screen は合格した（rows 64/80 の rel_l2 0.00113、他形式と同じ水準）
- しかし `compute-sanitizer --tool memcheck` は **32 件の Invalid __global__ write** を出す。
  `mul_mat_q<GGML_TYPE_IQ2_XS, J=64, fallback=false>` の block 13（最後の tile）で、dst の
  327,680 B（64 rows × 1280 floats）の割当の 1〜257 byte 後を書いている（`~/work/results/mmqv3/ab/memcheck.log`）
- 原因: 1280 % 96 = 32。nrows_x=1280 は 128 で割り切れるので fallback=false が選ばれ、
  `ggml_cuda_mmq_write_back_dp4a` は fallback=false のとき i_max を見ない。最後の tile が
  nrows_x を越えて隣の expert 行を壊す。数値 screen が通るのは、壊れた行を後から正しい書き込みが
  上書きする競合が今回は勝ったためで、再現性は保証されない
- したがって IQ2_XS / IQ2_S は J=64・J=48 とも I=64 のまま。J=48 の I=128 は 49,088 B で
  64 B しか余裕がなく、対象も 81..96 行の group だけなので適用しない
- 安全に I=96 を使うには fallback=true を強制する（`mul_mat_q_case` の routing 変更）必要がある。
  境界チェックのコストがあり、v3 では採らない（§12.8 の提案 1）

### 12.3 適用した変更

v2（gate/up 29/48 層と down 48/48 層の I=128）に、J=48/40 の I=128 を 12 型へ足した
（Q1_0 Q2_0 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 IQ2_XXS IQ3_XXS IQ3_S IQ4_XS IQ4_NL）。パッチ
`patches/mmq-i128-v3.diff`、テストは `patches/parity-j64-test.diff`（batch {48} と {96} を追加、
`ids_dst` に 256 B のパディング）。

- 512 experts × top-16 なので、4K の expert group の max_rows はおよそ 100〜160
- 本番は expert を id 順にグループ化する（row-count sort は `STRATA_MMQ_RESIDENT_SORT_NE=1` などの
  opt-in で、本番の `STRATA_SET` には入っていない）。そのため group の max_rows は全体の最大に近く、
  4K でも 96 を超える。J=48/40 は選ばれず、内訳も v2 と同じ（§12.6）
- 実測でも gate/up の短縮は v2 と同じ −11.7〜−12.2%（J=48/40 を足しても変わらない）

### 12.4 ビルド・イメージ

| 項目 | 値 |
|---|---|
| ソース | 9f7d977d + `patches/mmq-i128-v3.diff` + `patches/parity-j64-test.diff` |
| ビルド | `nvidia/cuda:12.9.1-devel-ubuntu24.04`、CPU 36-47、`-j 12`、`61-real` |
| イメージ | `strata:fork-mmq-i128-v3`（747,421,238 bytes、engine 0.1.57、strata の sha256 `5bb452df2df63dd74413519560238f274948ab0b5adc785fb27700b5b849790d`） |
| ログ | `~/work/results/mmqv3/final-build.log`、`final-package.log`、`final-image.log` |

### 12.5 パリティ

| テスト | 結果 |
|---|---|
| `prefill_mmq_v31_test` | 8 batch × 9 型 = 72 product すべて合格。J=48 を踏む rows 48/96 も rel_l2 0.0010〜0.0012、max/rms 0.0043〜0.0053 |
| `prefill_mmq_scatter_test` | 0 failures |
| `prefill_swiglu_quant_test` | 18 ケースすべて 0 bytes differ |
| `native_grouped_parity` | 24 ケース bitwise、0 failures |

logits dump（4K code、temp 0、max_tokens 128）:

| 比較 | 結果 |
|---|---|
| A vs A2（同一イメージ base） | rec 63（pos 4894、生成 101 token 目）から 9/72 レコードが差、max \|d\| 1.017 |
| A vs B（v3） | A vs A2 と同一の 9/72 レコード・同一の max \|d\| |
| A2 vs B | **72/72 レコード byte 一致（max \|d\| 0）** |

- v3 の prefill は bitwise（A2 と B が完全一致）。A と A2 の差は同一イメージ同士でも出る
  実行ごとの揺らぎで、v1 で見えた rec 63 / pos 4894 と同じ（原因は未確認のまま）
- argmax は全レコード一致（`cmp_reqs.py`: req 0/1 とも argmax differs 0）、回答テキストも 3 アーム同一

### 12.6 速度（A-B-A、`~/bench.py`、BENCH_SEED=42、spec=4、1 本目除外の中央値）

A = `strata:fork-9f7d977-base`（I=64）、B = `strata:fork-mmq-i128-v3`。4K/24K/30K × code/prose × 3 回。

| サイズ/種類 | A1 | B1 | A2 | A 中央値 | B/Amed |
|---|---:|---:|---:|---:|---:|
| 4K/code | 446.7 | 456.8 | 446.1 | 446.4 | **+2.31%** |
| 4K/prose | 441.7 | 455.1 | 442.0 | 441.9 | **+2.99%** |
| 24K/code | 488.5 | 502.5 | 487.9 | 488.2 | **+2.93%** |
| 24K/prose | 489.4 | 503.9 | 486.8 | 488.1 | **+3.26%** |
| 30K/code | 480.2 | 493.8 | 477.2 | 478.7 | **+3.15%** |
| 30K/prose | 478.1 | 493.0 | 477.4 | 477.8 | **+3.21%** |

- A1 と A2 の差は 0.7% 以内。B は全 6 条件で +2.31〜+3.26%
- v2（+2.75〜+3.44%）とほぼ同じで、J=48/40 の追加は検出できない
  （4K も +2.3〜+3.0% で、J=48/40 が効いていれば 4K だけ上がるはずだが上がっていない）
- decode (tok/s、参考): 4K/code A 中央値 52.70 → B 51.78（−1.73%）、24K/code 52.29 → 49.00（−6.28%）。
  A1/A2 の差が 4K/code で 3.6%、24K/code で 1.6% あり、ノイズが支配的で方向は一定しない

内訳（`STRATA_PREFILL_TIMING=1`、同一プロンプトの 24,493 / 29,333 / 30,302 token チャンク）:

| 位相 | 24,493 A1 → B1 | 29,333 A1 → B1 | 30,302 A1 → B1 |
|---|---|---|---|
| gemm gate/up | 6,623 → 5,840 ms（−11.8%） | 8,107 → 7,155 ms（−11.7%） | 8,336 → 7,344 ms（−11.9%） |
| gemm down | 3,857 → 3,049 ms（−20.9%） | 4,713 → 3,735 ms（−20.7%） | 4,862 → 3,840 ms（−21.0%） |

- v2 の内訳（gate/up −12.3%、down −20.7%）と同じ。J=48/40 の追加分は乗っていない

### 12.7 判定: **採用推奨ではない**（v3 は v2 と同じ上限）

- prefill は +2.31〜+3.26% で 4.5% に届かない。v2 と同等で、v3 で足した J=48/40 は効果なし
- パリティは完全 bitwise（A2 vs B が 72/72 一致）、出力も同一
- 上限: 残る gate/up 19 層（IQ2_XS 9 + IQ2_S 10）は I=96 が shared に収まる（40,768 B）が、
  fallback=false の 1280 行で out-of-bounds write になるため使えない。fallback=true を強制して
  I=96 にしても、14 tile（I=128 なら 10 tile）で境界チェックが乗るため、追加節約は 200 ms 前後、
  全体で **+3.5〜3.7%（推定）**。この tile 変更だけでは 4.5% に届かない
- 4.5% を狙うなら別の軸が必要（down の I=160 は #30235 で Q8_0 が 45.7% 悪化、K-quant は逆効果）

### 12.8 次の提案（最大 3 件）

1. **IQ2_XS / IQ2_S を fallback=true 固定で I=96 にする**。`mul_mat_q_case` の routing を型で分岐し、
   境界チェック付きのカーネルで 14 tile（vs 20 tile）にする。compute-sanitizer で out-of-bounds が
   消えることと速度を 1 ブロックで確認できる。見込みは gate/up 19 層で −5〜−10%（推定）
2. **4K の expert group の max_rows を実測する**。`STRATA_MMQ_RESIDENT_SORT_NE=1` を入れた場合と
   入れない場合で J=48/40 が選ばれるかを engine ログに出す。入っていないなら J=48/40 の変更は
   v3 の差分から落とせる
3. **decode の logits 揺らぎ（rec 63 / pos 4894）の切り分け**。A と A2 が同一イメージでも差が出るため、
   drafter を FP16（`STRATA_MTP_BATCH_F16=1`）にして同じ比較をする。差が消えれば spec の draft 経路

### 12.9 v3 の逸脱・注意

- ベンチは 2 ブロック（A1+B1、A2）。A1/A2 の差は 0.7% 以内
- compute-sanitizer は I=96 の probe バイナリ（`~/work/results/mmqv3/i96-probe/`）で実行した。
  最初は J=8 の既存 overread（ids_dst 7 ints に対して J=8 を読む）で止まったため、テスト側で
  `ids_dst` に 256 B のパディングを足して IQ2_XS の kernel まで到達させた
- 4K のプロンプト実トークン数は 2,931〜5,097、24K は 22,274〜30,309、30K は 34,010〜36,677
  （文字数ベースの生成のため要求サイズとずれる）。A/B は同じ並びの同じプロンプト
- decode はノイズが大きく（A1/A2 で 3.6% 差）、v3 の decode への影響は検出できない
- `~/work/results/mmqv3/ab/prod-environ.txt` に `STRATA_LAUNCH_KEY` が入る（起動ごとのローカルキー）。
  docs には書かない
- ビルドロックと GPU ロックは終了時に解放した。本番は `~/opt-work/prod.env` で復元し、healthy・
  `--max-context 524288`・残留なしを確認した
