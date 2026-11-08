# sm_61 (Tesla P40) の bitwise 既定スイッチ集合

最終更新: 2026-11-06。作業ツリー `~/work/wt/defaults`（branch `perf/sm61-defaults`、フォーク HEAD `9f7d977d`）。
数値はすべて実測。推定は「推定」と明記した。

## 1. 概要

`src/core/nv_defaults.cpp` は sm_86 / sm_60 / sm_121 の既定スイッチ集合を持つが、sm_61 (Tesla P40) の分岐が無かった。
sm_60 (P100) で +17.8%（8/8）、sm_86 で +9.6〜14.6%（8/8）の bitwise な集合から、P40 のハード制約（dp4a あり、BF16 なし、
48 KB shared / block、tensor core なし）で意味のあるものだけを sm_61 の既定にした。bitwise でないもの（`STRATA_HC_PERSIST`、
`STRATA_ROUTE_TAIL_SKIP`、`STRATA_IQ_MT_MIN`）は既定にせず opt-in のまま残した。

## 2. スイッチ表

「bitwise の根拠」はコード内のコメントまたは docs の該当箇所。すべてのスイッチはユーザーが値を設定すればそれが優先され、
`STRATA_SM61_PERF=0` で集合全体を切れる。

| スイッチ | 何をするか | bitwise の根拠 | sm_61 の制約 | P40 での期待 | 採用 |
|---|---|---|---|---|---|
| `STRATA_ADAPT_NOWAIT=1` | adaptive tier のコピー完了を待たずに窓を開始する | 同じトークン（スケジューリングのみ）。`src/program/generate.cpp:1962-1964`、`docs/DETAILS.md:1073` | なし | P100 で外すと -10 points | 採用 |
| `STRATA_ADAPT_LAG=2` | tier の swap を 2 窓前のものとして扱う | 同上。`generate.cpp:1972-1974`、`DETAILS.md:1074` | なし | P100 +3.5%（6/6） | 採用 |
| `STRATA_CPU_IQ_ALL_EXACT1=1` | 1 token の AVX2 行を ggml の float 順で計算する | `src/kernels/cpu/native_expert.cpp:399-407`、`DETAILS.md:1094`（"the output does not change"） | ホストが AVX2 かつ AVX-512 なしのときだけ active。このホストは Xeon E5-2695 v4（AVX2、AVX-512 なし） | sm_86 で +0.4〜+2.5%（CPU pool 側） | 採用 |
| `STRATA_SIDE_MIXER=2` | mixer の入力だけ読む仕事を graph の branch stream へ | `src/core/verify.cpp:259-263`（"bitwise the same outputs"） | CUDA のみ（HIP / Metal は 0） | P100 -1.8、sm_86 -1.2 / -2.3 | 採用 |
| `STRATA_FUSE_WAIT=2` | plan の待ちを直後のコピーに畳む | `verify.cpp:436-452`（"same reads, same values, one kernel node fewer"） | CUDA のみ | P100 -1.6 | 採用 |
| `STRATA_B6_MMVQ_ROWS=1` | 複数列 matvec で活性断片を 1 回だけ読み、行で再利用する | `src/kernels/cuda/native_mmvq.cu:1383`（"every output is bitwise the default exact layout's"） | exact layout（`g_multi_exact=true`）のときだけ。sm_61 の IL 表とは別経路 | P100 -1.3 | 採用 |
| `STRATA_FUSE_DECODE=1` | PR #1097 の operator fusion（活性画像 1 pass、BF16 pair 射影、SwiGLU + q8_1） | `include/strata/kernels/fuse_decode.hpp:1-10`（"bitwise what the unfused sequences produce"） | `STRATA_EXPERIMENTAL_SM60` ビルドのみ。今回のビルドは ON | P100 -1.3。native pack では一部経路が no-op（codemap の指摘） | 採用（無害） |
| `STRATA_GR_UP75=1` | GR up を 3 token 以上のとき別カーネルで読む | `src/kernels/cuda/fused_gr.cu:557`（"the same bits"）、`:1542`（"bitwise gr_up_multi_kernel"） | 3 token 以上のみ（`:1720`）。T1-2 は元の kernel | P100 -1.2、sm_75 で T3 27.3 -> 24.6 µs | 採用 |
| `STRATA_ATTN_DAILY=1` | PR #540 の chunk kernel と v2 merge を使う | `src/kernels/cuda/qsa_decode_attn.cu:832`（"bit for bit the default kernel's partials"）、`:926`（v2 merge は "bitwise the same output"） | experimental build では chunk kernel は既に sm_61 既定。`ATTN_DAILY` は v2 merge を足す | sm_86 -1.2 / -1.3 | 採用 |
| `STRATA_QSA_ROWS=1` | 4 query 以下の block score を register で計算する | `src/kernels/cuda/qsa_select.cu:2098`（"the same bits"） | NQ <= 4 のみ | sm_86 -0.5 / -1.2 | 採用 |
| `STRATA_GU_SWIGLU=1` | gate/up + SwiGLU + q8_1 を 1 launch にする | `src/kernels/cuda/iq_kernels.cu:4249-4252`（"bitwise the same hq"） | **CUDA layout（`em0 == 0`）のみ**。sm_61 は expert mode 8 が既定で、gate/up + SwiGLU + q8_1 を既に融合している | sm_86 -0.8 / -1.4 | 不採用（mode 8 で inert） |
| `STRATA_IQ_MT_MIN=1` | すべての group で multi-token CPU kernel を使う | 出力は drafting に依存しなくなるが、1 token の丸めが変わる（`native_expert.cpp:354-356`） | — | AVX-512 で -1〜-3% | 不採用（bitwise でない・減速） |
| `STRATA_ROUTE_TAIL_SKIP` (7) | 全 token が rank 7 以下の missed expert を省く | `docs/DETAILS.md:1049`（"Answers differ slightly"） | CUDA ではアーキ表とは別に既定 on | 既に既定 on | 追加しない（bitwise でない・既に既定） |
| `STRATA_EXPDEC=1` | 窓の活性を shared に置く expert gate/up | 2080 Ti で gate/up -11〜-23% の実測 | 48 KB に収まる | 期待薄（P40 は hit 99%） | 不採用（既定集合外の opt-in） |
| `STRATA_HC_PERSIST=1` | GR read を persistent kernel 1 個にする | **bitwise でない**（別の和の順序） | 30 SM に 80 block | decode +4.5〜5% の可能性 | 不採用（opt-in のまま） |

### 集合の設計

- sm_60 の 7 個はすべて sm_61 でも意味がある（`ADAPT_*` は同じ tier、`SIDE_MIXER` / `FUSE_WAIT` / `FUSE_DECODE` は CUDA の
  スケジューリング、`B6_MMVQ_ROWS` は exact matvec、`GR_UP75` は T>=3）ので全部入れた。
- sm_86 の 9 個のうち、`CPU_IQ_ALL_EXACT1`（CPU pool）、`ATTN_DAILY`（PR #540）、`QSA_ROWS`（block score）は Pascal でも
  動くので入れた。`GU_SWIGLU` は sm_61 の既定 expert mode 8 が同じ融合を既に持つため入れない。
- 10 個すべてが bitwise。identity gate（10 prompt + exact_pp 4K/20K）は集合 on と off で一致する（本タスクでは GPU 時間の
  都合で A/B の出力一致とパリティで代替。§7）。

## 3. ビルド条件

| 項目 | 値 |
|---|---|
| ソース | `~/work/wt/defaults`、HEAD `9f7d977d` + 本パッチ |
| ggml / llama.cpp | ピン留めコミット `3cf03257`（`~/work/wt/build/third_party/llama.cpp` をコピー） |
| ビルドコンテナ | `nvidia/cuda:12.9.1-devel-ubuntu24.04`、`--cpuset-cpus 36-47`、`-j 12` |
| CUDA アーキ | `-DCMAKE_CUDA_ARCHITECTURES=61-real` のみ |
| ホスト flags | `STRATA_PORTABLE=OFF`、C/CXX `-march=native`、CUDA ホスト `-Xcompiler=-march=native` |
| 機能 flags | `STRATA_ENABLE_CUDA=ON`、`STRATA_EXPERIMENTAL_SM60=ON`、`STRATA_NATIVE_EXPERTS=ON`、`STRATA_Q6K_EXPERTS=ON`、`STRATA_MMQ_KQUANTS=ON`、`STRATA_BUILD_TESTS=ON` |
| パリティ対象 | `nv_defaults_test`、`qsa_merge_daily_parity`、`qsa_rows_bench`、`mmvq_multi_parity`、`b6_mmvq_bench`、`native_grouped_parity`、`s2_expert_grouped_parity`、`mmvq_il_parity`、`dequant_wide_parity` |
| 所要時間 | build 468 s / package 130 s / image 495 s（合計 1,093 s） |
| ログ | `~/work/results/defaults/{build,package,image}-sm61-defaults.log` |
| イメージ | `strata:fork-sm61-defaults`（746,995,355 bytes） |

- コンパイラ警告のみ（未使用変数など）、エラー 0。
- `nvcc` の sm_75 未満 offline compilation 非推奨警告は sm_61 の既知警告。

## 4. パリティ

GPU ロックを取り、本番を止めて VRAM を空けてから実行した（`~/work/results/defaults/parity-*.log`）。全 10 本 PASS。

| テスト | 結果 | 何を確認したか |
|---|---|---|
| `nv_defaults_test` | ok | sm_61 表が 10 個、`STRATA_SM61_PERF=0` で空、`STRATA_DEFAULTS=026` の影響なし、6.2 は空 |
| `qsa_merge_daily_parity`（既定） | 1,172 cases / 0 differ -> PASS | PR #540 chunk kernel と v2 merge が upstream と bit 一致（fp16 295 / int8 291 / q4_0 294 / k8v4 292） |
| `qsa_merge_daily_parity`（`STRATA_ATTN_DAILY=1`） | 1,172 cases / 0 differ -> PASS | `ATTN_DAILY` の production entry が bit 一致 |
| `qsa_rows_bench --rows` | 8 cases / 0 differ -> bit-identical | `QSA_ROWS` の scores と ids が multi kernel と bit 一致。scores は T=1 64.4 -> 61.7 µs、T=4 141.9 -> 125.7 µs |
| `mmvq_multi_parity` | 148,480 outputs / 0 differ（multi_exact on） | exact multi-column が single-column と bit 一致。負の対照は 63,811 differ（T>4）で検査能力あり |
| `b6_mmvq_bench` | 0 case/width comparisons failed | `B6_MMVQ_ROWS` が exact layout と bit 一致 |
| `native_grouped_parity` | 0 failures | sm_61 の expert mode 8 が v1 と bit 一致（IQ3_S / IQ4_XS ほか） |
| `s2_expert_grouped_parity` | 0 failures | grouped expert の slot stride / arena 経路が bit 一致 |
| `mmvq_il_parity` | OK（T 1-8、rows 1/2/4 + table） | sm_61 の IL 表が bit 一致 |
| `dequant_wide_parity` | 360 cases OK | sm_61 の wide dequant が bit 一致 |

`GR_UP75` と `FUSE_DECODE` には専用の parity テストが無い。前者は `gr_up75_kernel` が `gr_up_multi_kernel` と同じ lane の和・
xor 木・epilogue を使うこと（`fused_gr.cu:557`）と sm_75 / sm_120 の実測に基づく。後者は PR #1097 の各 fusion が
`fuse_decode.hpp` で bitwise と明記され、sm_60 の既定集合で P100 実測（identity gate 一致）がある。

## 5. 速度 A-B-A

条件: `~/bench.py`、`BENCH_SEED=42`、4K/24K、max 256、spec=4 固定、`STRATA_DENSE_F32=1`、`--ple-io ram`、`--prefill auto`、
`--kv int8`、`--kv-resident 32768`。A = `strata:fork-9f7d977-base`、B = `strata:fork-sm61-defaults`。各アーム 4 回、
1 本目を捨てて残り 3 の中央値。A-B-A の 3 本目 (A2) は最初の測定が他エージェントの GPU コンテナと重なって汚染されたため、
23:16 に再測した (A2b)。

| 項目 | A1 | A2b | A 中央値 | B1 | B - A |
|---|---:|---:|---:|---:|---:|
| 4K code decode (tok/s) | 54.65 | 53.97 | 54.31 | 53.19 | -2.1% |
| 4K prose decode | 43.76 | 42.84 | 43.30 | 44.49 | +2.7% |
| 24K code decode | 50.29 | 48.22 | 49.26 | 51.39 | +4.3% |
| 24K prose decode | 43.56 | 42.88 | 43.22 | 45.40 | +5.0% |
| 4K code prefill | 460.9 | 465.0 | 463.0 | 463.9 | +0.2% |
| 4K prose prefill | 440.0 | 437.6 | 438.8 | 443.8 | +1.1% |
| 24K code prefill | 487.8 | 489.7 | 488.8 | 488.3 | -0.1% |
| 24K prose prefill | 487.2 | 488.4 | 487.8 | 487.3 | -0.1% |

- A 自身の揺れ (A1 vs A2b): 4K code 1.3%、4K prose 2.1%、24K code 4.3%、24K prose 1.6%。B の差はこの揺れと同程度〜やや上。
- prefill は ±1% 以内（スイッチは decode のみ）。
- 最初の A-B-A (A2 汚染) の値: A1 54.65 / 43.76 / 50.29 / 43.56、B1 53.19 / 44.49 / 51.39 / 45.40、
  A2 49.16 / 40.64 / 48.54 / 36.64。A2 は 20:57 から他エージェントの GPU コンテナ (nvidia/cuda devel) と重なり、
  4K prose が 20.43 / 22.73 tok/s、24K code の 1 本が prompt 0 / completion 27 で失敗した。汚染アームは判定に使わない。

### decode の内訳 (`STRATA_DECODE_TIMING=1`、各アーム 1 回)

| 項目 | A1 4K code | B1 4K code | A1 4K prose | B1 4K prose | A1 24K code | B1 24K code |
|---|---:|---:|---:|---:|---:|---:|
| window (ms) | 61.76 | 57.37 | 56.06 | 52.65 | 56.78 | 54.37 |
| adapt wait | 1.76 | 0.02 | 1.28 | 0.02 | 1.31 | 0.02 |
| verify | 49.47 | 47.84 | 50.67 | 49.57 | 95.29 | 48.79 |
| GPU-reach wait | 39.62 | 37.65 | 42.39 | 41.68 | 61.06 | 41.04 |
| per-layer host | 7.15 | 7.37 | 5.35 | 4.94 | 30.23 | 4.92 |
| CPU experts | 6.45 | 6.63 | 4.64 | 4.28 | 29.00 | 4.27 |
| draft | 4.29 | 4.44 | 4.39 | 4.50 | 5.44 | 4.49 |
| avg T | 2.80 | 2.89 | 3.22 | 3.18 | 2.84 | 2.93 |

- `ADAPT_NOWAIT` / `ADAPT_LAG` の効果が直接見える: adapt wait が 1.3〜1.8 ms/window から 0.02 ms へ。
- 24K code の per-layer host CPU が 29.00 -> 4.27 ms に落ちている。`CPU_IQ_ALL_EXACT1` が AVX2 のみのホストで
  ggml の single-token dot を置き換えるためと推定（推定）。
- 24K prose の A1 timing は汚染された rep 4 (21.7 tok/s) のもので、clean な比較は無い。

## 6. 出力一致

固定 4 プロンプト (2K/8K/16K/24K、temp 0、max 128、thinking なし)。プロンプトは `bench.py` と同じ corpus と seed 42 で
作り、アーム間で同一。実トークン数は 2,429 / 7,866 / 20,148 / 27,637。

| 比較 | 2K | 8K | 16K | 24K |
|---|---|---|---|---|
| A1 vs A2 (既定 flags) | DIFFERENT | DIFFERENT | DIFFERENT | DIFFERENT |
| A1 vs B1 (既定 flags) | DIFFERENT | DIFFERENT | DIFFERENT | DIFFERENT |
| A1 vs A2 (identity flags) | IDENTICAL | DIFFERENT | DIFFERENT | DIFFERENT |
| A1 vs B1 (identity flags) | DIFFERENT | DIFFERENT | DIFFERENT | DIFFERENT |

- identity flags = `STRATA_IQ_MT_MIN=1 STRATA_PREFILL_CPU_SHARE=0 STRATA_ROUTE_TAIL_SKIP=0`（`docs/OLDER_GPUS.md:278` の推奨）。
- このホストでは A 同士でも回答が変わる（AVX-512 なし CPU の既知の揺れ）。identity flags でも 8K/16K/24K で A1 と A2 が
  違った。`--adapt-swaps 0 --pcie-frac 0 --prompt-cache 0` を併用していないため、adaptive tier の swap と会話キャッシュの
  再利用が残ったことが原因と推定（推定、未確認）。
- B の差分は A の揺れと同じ種類（同じ出だしから数十〜数百文字で語句が分岐）。最初の分岐位置は A1 vs A2 が 45 / 254 / 44 文字、
  B が 291 / 82 / 254 / 33 文字で、B の差が A の揺れより大きいとは言えない。
- したがってテキスト比較では「B は A の揺れと同程度」。bitwise の最終根拠は §4 のパリティ 10 本 PASS とコード内の宣言。

## 7. 判定

- decode の改善: 4K code -2.1%、4K prose +2.7%、24K code +4.3%、24K prose +5.0%。4 セルの中央値は約 +2.5%。
- 5% 閾値に到達したのは 24K prose のみ (5.0%)。24K code の +4.3% は A 自身の A1 vs A2b の揺れ (4.3%) と同じ大きさ。
- 出力は A の揺れと同程度（テキスト比較では区別できない）。bitwise はパリティ 10 本 PASS とコード内の宣言で支持。
- **判定: 採用推奨とはしない（5% 閾値未満）**。理由: (1) 総合の decode 改善が約 +2.5% で 5% 未満、(2) 最大の改善セル
  24K code の +4.3% は A 自身のドリフトと同程度、(3) 24K prose の +5.0% は 1 セルのみ。
- ただし機構としては `ADAPT_NOWAIT` / `ADAPT_LAG` による adapt wait の消失 (1.3〜1.8 -> 0.02 ms/window) が一貫して見え、
  bitwise で害も無い。既定 on のまま残すか opt-in にするかはメインの判断。既定 on を維持する場合の期待は decode +2〜3%（推定）。
- 既定 on を維持する場合の注意: 今回の A/B は 4 セル × 3 有効値で、A のドリフトが最大 4.3% あるため、+5% 前後の差は
  追加の 5 ペア以上で再確認するのが望ましい。

## 8. 再現コマンド

```sh
ssh llm-02
cd ~/work/wt/defaults

# 1. ビルド（ビルドロック -> build -> package -> image）
bash scripts/build-sm61.sh sm61-defaults      # タグ strata:fork-sm61-defaults

# 2. パリティ（GPU ロック、本番停止 -> 実行 -> 本番復元）
bash scripts/parity-sm61.sh

# 3. 速度 A-B-A（A = strata:fork-9f7d977-base、B = strata:fork-sm61-defaults、spec=4 固定）
bash scripts/ab-sm61.sh speed

# 4. 出力一致 + STRATA_DECODE_TIMING
bash scripts/ab-sm61.sh output
```

`scripts/ab-sm61.sh` は GPU ロックを取り、本番を止め、アームごとにエンジンを起動して `~/bench.py`（`BENCH_SEED=42`、
4K/24K、max 256）と `scripts/answers4.py`（2K/8K/16K/24K、temp 0、max 128、thinking なし）を回し、本番を復元してロックを解放する。

## 9. 未確認・注意

- `STRATA_ADAPT_*` の P40 での寄与は、P40 の expert cache ヒット率が 99% で tier の swap がほぼ動かないため小さいと推定する。
  P100 の -10 points は「外すと待ちが入る」構造の差で、同じ経路が P40 にもある。
- `STRATA_FUSE_DECODE` は native pack では一部の fusion 経路が no-op（codemap の指摘）。速度に効かない可能性があるが、
  bitwise なので既定に入れても害はない。
- `STRATA_ATTN_DAILY` の chunk kernel は experimental build の sm_61 で既に既定。追加分は v2 merge のみで寄与は小さいと推定。
- `STRATA_B6_MMVQ_ROWS` は sm_61 の 1-4 token IL 表とは別経路（IL が先に選ばれる T では B6 は走らない）。
- `STRATA_GR_UP75` は T>=3 のみ。spec=4 の窓は平均 T 2.14（研究資料の profile）なので、走る窓は限られる。
- P40 はファン無し。A/B は各アームを同じ温度から始める必要がある（89 C で thermal slowdown、1,531 -> 約 1,300 MHz）。
