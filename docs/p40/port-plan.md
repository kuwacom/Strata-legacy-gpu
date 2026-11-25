# 移植計画: フォーク 32 コミット -> upstream 0.2.21

- 日付: 2026-11-24
- fork: `origin/main` 0a050270f (32 コミット, 分岐点 9f7d977d2)
- upstream: `upstream/main` d46f5f268 (product 0.2.21 / engine 0.1.67)
- リポジトリ: `<host>:~/work/Strata-legacy-gpu`
- 本計画は静的調査のみ。ビルド・GPU 測定・3-way apply の実試行はしていない (「未確認」に列挙)

## 1. 前提

### パス対応
| fork (旧) | upstream 0.2.21 (新) |
|---|---|
| `src/prefill/*` | `engine/src/prefill/*` |
| `include/strata/*` | `engine/include/strata/*` |
| `src/kernels/cuda/*` | `engine/src/kernels/cuda/*` |
| `src/core/*` | `engine/src/core/*` |
| `cmake/strata_ggml_tq_overlay.cmake` | 無し (overlay 機構ごと廃止。`engine/cmake/*` に分割) |
| ルート `CMakeLists.txt` | `engine/CMakeLists.txt` + `engine/cmake/backends/cuda.cmake` |
| `src/prefill/moe_mmq.cu` | 無し (`moe_helpers.cu` = Strata 自前分 / `mmq_absent.cpp` = MMQ のスタブ) |

### 最重要の結論: upstream は MMQ を撤去した
- `engine/cmake/backends/cuda.cmake:94`: 「llama.cpp の MMQ は 0.2.16 でツリーを出た」(原文: llama.cpp MMQ left the tree in 0.2.16)
- `engine/src/prefill/mmq_absent.cpp:24-36` が `mmq::quantize` / `quantize_scatter` / `Context::run` / `f16_to_f32` を「absent」スタブにする
- `STRATA_PREFILL_MMQ` という CMake オプション自体が upstream に存在しない (grep 0 件)。`strata_mmq` ターゲットも無い
- `engine/third_party/strata-llama/ggml/src/ggml-cuda/` (283 ファイル) は残るが、`engine/cmake` / `engine/CMakeLists.txt` から参照されない (vision テストの oracle 用)。つまり `mmq-config-pascal-dp4a.cuh` を書き換えても engine には 1 バイトも届かない
- `Gemm::native_mmq` は upstream にも残る (`engine/src/prefill/gemm.cu:1391`) が、`mmq::built()` が false なので常に false を返す
- 代替: MoE prompt は `native_expert_i8` (`engine/include/strata/kernels/i8_expert_rule.hpp`, MMQ 無しビルドでは全カードで既定 = `engine/src/prefill/prefill.cpp:234` の `native_expert_i8_set_mmq_free()`) と fused int8 (`moe_fused_iq.cu`, cc>=75 のみ) に置き換わった

この 1 点で、fork の perf 12 件のうち 4 件 (ca93f701e, bb9c9c140, a2cb22715, 06f08e13b) と 50c284eca の MMQ 部分は移植先を失う。

## 2. 32 コミットの分類

| # | commit | 日付 | 種別 | 判断 |
|---|---|---|---|---|
| 1 | 50c284eca perf: QSA thread scorer と 512K production build | 11-08 | コード (混在) | 分割: qsa_select=破棄 / nv_defaults=移植 / verify+router=移植(要改変) / MMQ+overlay=破棄 |
| 2 | 221e3e3c8 docs: P40 512K ノート統合 | 11-08 | docs (28 ファイル) | 保留 (パスが旧ツリー。数値は資産) |
| 3 | 5d6928c75 docs: P40 fork 入口 | 11-08 | docs | 保留 (書き直し前提) |
| 4 | b9a45bb0d docs: PATTN / draft-conf 検証 | 11-08 | docs | 保留 (PATTN は upstream に同名スイッチで存在) |
| 5 | 5b54c2f06 docs: PATTN を本番構成に採用 | 11-09 | docs | 保留 (docs/p40/ は upstream に無い) |
| 6 | 02e8c6637 perf(prefill): hc down+inject の bf16->fp32 変換共有 | 11-09 | コード | **移植** |
| 7 | b337633aa docs: prefill profile, chunk 32768, hc-pair | 11-09 | docs | 保留 |
| 8 | f0945a661 docs: DP4A score path 不採用 | 11-09 | docs | 保留 (判断記録として有効) |
| 9 | 463ab392a perf(prefill): hc down を BF16 で読む | 11-09 | コード | **移植** |
| 10 | 6d064727d perf(prefill): hc inject も BF16 | 11-09 | コード | **移植** |
| 11 | e797f4e0f perf(prefill): BF16 narrow 採用 (fork-v5) | 11-09 | docs のみ | 保留 (9/10 と一緒に書き直す) |
| 12 | 80cc9ec7f docs: 融合 down+inject は不採用 | 11-09 | docs | 保留 (upstream は HIP で実装済み = 参考価値あり) |
| 13 | c9eec7a9a perf(qsa): value kernel を 8 CTA/SM に | 11-09 | コード | **移植** |
| 14 | 54d4171f7 perf(qsa): value kernel 占有修正 (fork-v6) | 11-09 | docs のみ | 保留 (13 と一緒に) |
| 15 | e0c4000a3 perf(hc): write-back の k をカウント | 11-10 | コード | **移植** (優先度低) |
| 16 | ca93f701e perf(prefill): dense MMQ 活性を FP16 から量子化 | 11-10 | コード | **破棄** (MMQ 撤去) |
| 17 | bb9c9c140 perf(prefill): prompt MoE の MMQ tile probe と gate/up 融合 | 11-10 | コード | **破棄** (MMQ 撤去) |
| 18 | a2cb22715 perf(prefill): shared expert down (K=640) を dense MMQ へ | 11-10 | コード | **破棄** (MMQ 撤去) |
| 19 | 6732e0325 Merge perf/dense | 11-10 | merge | 破棄 (内容は #16-#18 が個別に持つ) |
| 20 | 2e7d57eb5 Merge perf/mmq-glu | 11-10 | merge | 破棄 (同上) |
| 21 | e1d8b20f8 docs: v7 round | 11-10 | docs | 保留 (MMQ 前提の数値は無効) |
| 22 | 52e908424 perf(hc): inject の 8-token kernel と mix の BF16 読み | 11-11 | コード | 分割して **移植** (INJ8 / MIX16 の 2 つ) |
| 23 | 06f08e13b perf(prefill): IQ2_S/IQ2_XS を I=128 J=48 で | 11-11 | コード (cmake) | **破棄** (MMQ 撤去。overlay も無い) |
| 24 | a1970db19 Merge perf/mmq-i128 | 11-11 | merge | 破棄 (#22/#23 が個別に持つ) |
| 25 | a9433270a docs: v8 round | 11-11 | docs | 保留 |
| 26 | e7c5543cb perf(prefill): inject を mix に融合 (STRATA_HC_MIXINJ) | 11-11 | コード | **移植** (MIX16 の後) |
| 27 | d1de43b45 docs: v9 | 11-11 | docs | 保留 |
| 28 | 1a2c8a686 docs: 4 方向ギャップ分析 | 11-11 | docs | 保留 (参照価値) |
| 29 | f28f12051 docs: P40 文書の索引 | 11-11 | docs | 保留 (上の docs 群のあと) |
| 30 | a9767bb11 tools(p40): 測定ハーネスを docs と同居 | 11-11 | tools | **移植** (パス改変) |
| 31 | aef2ba5df ci: フォークイメージをソースからビルド | 11-11 | CI | **移植** (要書き換え) |
| 32 | 0a050270f docs(images): 初回測定と起動方法 | 11-11 | docs | 移植 (#31 と一緒) |

内訳: コード 12 / merge 3 / docs 15 / tools 1 / CI 1 = 32。
コード 12 件の判断: 移植 7 (02e8c6637, 463ab392a, 6d064727d, c9eec7a9a, e0c4000a3, 52e908424, e7c5543cb) + 50c284eca の 2 部分 / 破棄 4 (ca93f701e, bb9c9c140, a2cb22715, 06f08e13b) + 50c284eca の 2 部分。

## 3. コードコミットの個別調査

### 3.1 50c284eca (分割して扱う)

変更ファイル (fork): `src/kernels/cuda/qsa_select.cu` 6 行, `src/core/nv_defaults.cpp` 25, `src/core/nv_defaults_test.cpp` 23, `src/core/verify.cpp` 49, `src/kernels/cuda/native_router.cu` 92, `include/strata/kernels/native_router.hpp` 6, `include/strata/core/layer.hpp` 5, `src/core/layer.cpp` 20, `src/prefill/gemm.cu` 8, `src/prefill/mmq_kq_lib.cu` 3, `src/prefill/moe_mmq.cu` 16, `CMakeLists.txt` 10, `cmake/strata_ggml_tq_overlay.cmake` 22, `tests/cuda/prefill_mmq_kquant_test.cpp` 4

| 部分 | 新パス | シンボルの存否 | upstream の同等物 | 衝突リスク | 判断 |
|---|---|---|---|---|---|
| a: qsa_select の sm_61 既定 | `engine/src/kernels/cuda/qsa_select.cu:2164` | `scores_thread_on()` 存続 | あり: `STRATA_SCORES_THREAD=1` で同じ thread-per-head カーネル (upstream の主張は同じビット)。既定は sm_75/sm_70 のみ | 低 | 破棄 (env で代替) |
| b: nv_defaults の sm_61 表 | `engine/src/core/nv_defaults.cpp` (6.0 分岐が :164、6.1 分岐なし), `engine/src/core/nv_defaults_test.cpp` | `STRATA_SM61_PERF` は upstream に無い | なし | 低 (分岐とログ文字列の追加だけ) | 移植 (要再測定) |
| c: verify の fuse_wait / route_publish | `engine/src/core/verify.cpp:491` (fuse_wait_on), `:2636-2660` (record_window の router ループ), `engine/src/kernels/cuda/native_router.cu:219/228` の隣, `engine/include/strata/kernels/native_router.hpp:16-18` | `copy_from_mapped_multi` は既存 (`engine/include/strata/kernels/elementwise.hpp:156`)。`native_router_top10_publish` / `moe_route_publish` は無い | なし (fuse_wait の既定は sm_75 のみ) | 高 (record_window の分岐が再編: fuse_pub 条件, hits_.d_res, route_prior, always_publish) | 移植 (要改変) |
| d: MMQ / Q3_K / overlay / CMake | 新パス無し | `strata_mmq` / `strata_kq` ターゲット無し | なし | - | 破棄 |

(b) の根拠: upstream の sm_60 表は 7 個 (`nv_defaults.cpp:164-176`) に減り、fork の 10 個のうち CPU_IQ_ALL_EXACT1 / ATTN_DAILY / QSA_ROWS は sm_70 表 (`:136-162`) で実績がある。0.2.21 でも名前は全て存続 (各 2-6 ファイル)。よって表を作り直して 1 個ずつ A/B するのが正しい。
(c) の根拠: upstream の `record_window` は 1 行 router を `fuse_pub` 条件で分岐する形に変わっており、fork の `route_published` フラグ方式をそのまま当てると二重 publish になる。手で入れ直す。

### 3.2 02e8c6637 hc down/inject の変換共有 (bf16_pair)

- 変更: `include/strata/prefill/gemm.hpp` 12, `src/prefill/gemm.cu` 45, `src/prefill/prefill.cpp` 19
- 新パス: `engine/include/strata/prefill/gemm.hpp` (bf16_pair 宣言), `engine/src/prefill/gemm.cu` (`Gemm::bf16_pair` + `tc_w2_`), `engine/src/prefill/prefill.cpp:5124` の直前に hcpair
- upstream の存否: `bf16_pair` は無い (`Gemm` に無し。`engine/src/core/layer.cpp:115` の `project_bf16_pair` は別物)。`Gemm::bf16` の Pascal path 2 は健在 (`gemm.cu:1199-1255`)、`kXSliceElems` / `grow` / `bf16_to_f32` も同じ
- upstream の同等物: なし。`hcd` 経路 (`strata_pf_hcdown_bf16`, `wmma_gemm.cu:675`) は HIP gfx11 専用で CUDA では false (`wmma_gemm.cu:766`)
- 衝突リスク: 低 (新規関数 + デストラクタ 1 行 + 呼び出し 3 行。upstream の `if (!hcd && !hdown && !bf16_proj(...))` は :5124 に同一文がある)
- 判断: 移植。bitwise (2 回呼びと同じ cublasSgemm を同じ順)
- 測定: 128K の hc down/inject 位相

### 3.3 463ab392a hc down を BF16 のまま読む (bf16_narrow)

- 変更: `include/strata/prefill/gemm.hpp` 10, `src/prefill/gemm.cu` 約 240, `src/prefill/prefill.cpp` 19
- 新パス: `engine/include/strata/prefill/gemm.hpp` (`bf16_narrow`), `engine/src/prefill/gemm.cu` (`bf16_narrow_kernel` + `Gemm::bf16_narrow`), `engine/src/prefill/prefill.cpp` (スイッチ `hc_down_bf16` + down 呼び出し)
- upstream の存否: `bf16_narrow` は無い
- upstream の同等物: `strata_pf_hcdown_bf16` (`engine/src/prefill/wmma_gemm.cu:502-700`、呼び出し `prefill.cpp:5117`) と `Gemm::bf16_hcd_exact` (`gemm.cu:1025`) は `strata_wmma_gfx11_device()` ゲートで CUDA では false。P40 では無効なので fork のカーネルは依然必要
- 衝突リスク: 低 (新規コード。呼び出しは 3.2 の後)
- 判断: 移植
- 注意: 適用条件が `ldx == D` なので upstream の `hc_pad()` (= `STRATA_PF_HCDOWN` / `STRATA_HCD_EXACT` で `ldx = D + 64`, `prefill.cpp:1007`) とは排他。両方 =1 にすると hcd が勝ち、CUDA では hyper-connection の射影を実行できず停止する
- 測定: hc read 位相 + `gemm_bf16_parity` (narrow は累積順が違い非ビット、相対 1e-4 で判定)

### 3.4 6d064727d hc inject も BF16

- 変更: `src/prefill/prefill.cpp` のみ
- 新パス: `engine/src/prefill/prefill.cpp` (upstream の :5157 の行)
- upstream の存否: 同等の分岐は無い (upstream は inject を `strata_pf_hcdown_bf16` に含めるか `bf16_proj` か)
- 衝突リスク: 低 (3.2 / 3.3 の後に当てる前提)
- 判断: 移植。3.10 (a) INJ8 の routing 前提でもある

### 3.5 c9eec7a9a qsa value kernel を 8 CTA/SM に

- 変更: `src/kernels/cuda/qsa_pattn.cu` 1 行
- 新パス: `engine/src/kernels/cuda/qsa_pattn.cu:256` (`__launch_bounds__(64)` を `(64, 8)` へ)
- upstream の存否: カーネルは存続、`__launch_bounds__(64)` のまま (upstream :256)
- upstream の同等物: なし
- 衝突リスク: 低 (1 行)
- 判断: 移植。レジスタ 151 -> 128 の副作用 (スピル) は P40 で要再確認
- 測定: `STRATA_PATTN_SM60=1` の QSA attention 位相 (128K)

### 3.6 e0c4000a3 write-back の k カウント

- 変更: `src/prefill/kernels.cu` (2 カーネル)
- 新パス: `engine/src/prefill/kernels.cu` (スカラ経路 :250-262、cvec カーネル :286-320)
- upstream の存否: カーネルは存続。修正は入っていない (スカラ経路は `int k = 0; for (int d = threadIdx.x; ...)` のまま = upstream :250-265, :290-300)
- upstream の同等物: なし。ただし upstream は `gr_write_norm_rs_kernel` に float4 経路を追加済み (:208-250) で、N=2560 かつ整列時はそちらが走るため効果は小さい。`gr_write_cvec_norm_rs_kernel` (:276) は scalar のみなのでこちらには効く
- 衝突リスク: 低-中 (upstream でベクタ経路が挿入され行番号がずれるが、ハンクの文脈は一致)
- 判断: 移植 (優先度低)。bitwise

### 3.7 ca93f701e dense MMQ 活性を FP16 から量子化

- 変更: `include/strata/prefill/moe_mmq.hpp` 4, `src/prefill/moe_mmq.cu` 119, `src/prefill/gemm.cu` 7
- 新パス: 無し。`mmq::quantize` の実体は upstream に無く `engine/src/prefill/mmq_absent.cpp:24` がスタブ。`engine/src/prefill/moe_helpers.cu` (242 行) にも quantize は無い
- upstream の同等物: 別物として `quantize_f16_to_q8_1_kernel` が `engine/src/kernels/cuda/dense_i8_tc.cu:159` にある (dense i8 tensor core 用で MMQ とは別経路)
- 衝突リスク: 適用先なし
- 判断: 破棄。MMQ を戻す場合にだけ復活させる

### 3.8 bb9c9c140 MMQ tile probe / j_opt / gate-up 融合 / STRATA_MMQ_SQ

- 変更: `cmake/strata_ggml_tq_overlay.cmake`, `include/strata/prefill/moe_mmq.hpp` (j_opt, gu, gu_ok), `src/prefill/moe_mmq.cu` 約 300, `src/prefill/prefill.cpp`
- 新パス: 無し (`Product::j_opt` / `mmq::gu` / `mmq::swiglu_quant` を載せる `Context::run` がスタブ)
- upstream の同等物: 機能的にはあり。gate/up を 1 タイルで回して SwiGLU まで書くのは `moe_fused_iq.cu` の fused int8 experts (cc>=75) と `native_expert_i8` (`i8_expert.cuh`, P40 可, `i8_expert_rule.hpp` で MMQ 無しビルドの既定)。`STRATA_MMQ_SQ` 相当は `mmq::swiglu_quant` 前提で移植先なし
- 衝突リスク: 適用先なし
- 判断: 破棄 (upstream の fused int8 が上位互換。ただし P40 での実測は未取得)

### 3.9 a2cb22715 shared expert down (K=640) を dense MMQ へ

- 変更: `include/strata/prefill/gemm.hpp` 3, `include/strata/prefill/moe_mmq.hpp` 5 (`k_in_padding`), `src/prefill/gemm.cu` 約 35
- 新パス: 無し。`Gemm::native_mmq` は upstream にもあるが `mmq::` がスタブ。`k_in_padding` は upstream に無い
- upstream の同等物: なし (dense の K=640 は FP16/cuBLAS 経路のまま)
- 衝突リスク: 適用先なし
- 判断: 破棄

### 3.10 52e908424 inject 8-token kernel と mix の BF16 読み (2 分割)

(a) INJ8 (`bf16_narrow8_kernel`, `STRATA_HC_INJ8`) と `gemm_bf16_parity` の narrow 節
- 新パス: `engine/src/prefill/gemm.cu` (`bf16_narrow` 内の分岐 + 新カーネル), `engine/src/prefill/gemm_bf16_parity.cu` (narrow 節)
- upstream の存否: 無し。upstream の inject は `strata_pf_hcdown_bf16` (HIP) か `bf16_proj`
- 衝突リスク: 低 (3.3 の新規コードの内側)
- 前提: N == 4 (HC)、K % 8 == 0、`ldx == D`
- 判断: 移植。非ビット (butterfly 和) なので parity の narrow 節で確認

(b) MIX16 (`gr_mix_r16`, `STRATA_HC_MIX16`, `STRATA_HC_MIX16_CHECK`)
- 新パス: `engine/include/strata/prefill/kernels.hpp`, `engine/src/prefill/kernels.cu` (upstream の `gr_mix_r_kernel` :160 の隣), `engine/src/prefill/prefill.cpp:5163`
- upstream の存否: `gr_mix_r16` は無い。`STRATA_HC_MIX*` は 1 つも無い (upstream の HC 系は HC_PERSIST / HC_Q8* / HC_REQ8 / HC_SPLIT / HC_UPMIX のみ)
- upstream の同等物: `gr_upmix` (`kernels.cu:2985`) は mix を up に融合するが CUDA は cc>=8 のみ (:3008-3012) で P40 では使えない
- 衝突リスク: 低-中 (mix の分岐が upstream で upmixed / gr_unfused / gr_mix_r の 3 分岐に増えている)
- 前提: `ldx == D`、`m.xn16_lo == nullptr`、`!xf16` (upstream の既定で成立)
- 判断: 移植。数値は 1 回丸めが増える (非ビット) ので CHECK で差を出し needle/QA でゲート

### 3.11 06f08e13b IQ2_S/IQ2_XS を I=128 J=48 で

- 変更: `cmake/strata_ggml_tq_overlay.cmake` のみ
- 新パス: 無し。該当設定は `engine/third_party/strata-llama/ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh:174-195` に I=64 のまま残るが、この木はエンジンにコンパイルされない (`strata_mmq` ターゲット無し、`STRATA_PREFILL_MMQ` 無し)
- upstream の同等物: なし (0.2.21 の MMQ config は I=64 のみ。I=128 は既定行 :272 の 1 件だけ)
- 衝突リスク: 適用先なし
- 判断: 破棄

### 3.12 e7c5543cb inject を mix に融合 (STRATA_HC_MIXINJ)

- 変更: `CMakeLists.txt` 6 (パリティターゲット), `include/strata/prefill/kernels.hpp` 5, `src/prefill/gr_mixinj_parity.cu` 新規 252, `src/prefill/kernels.cu` 79, `src/prefill/prefill.cpp` 28
- 新パス: `engine/include/strata/prefill/kernels.hpp`, `engine/src/prefill/kernels.cu`, `engine/src/prefill/prefill.cpp` (:5157-5164 の置換), 新規 `engine/src/prefill/gr_mixinj_parity.cu` + テスト登録 (`engine/cmake/backends/cuda.cmake` のテスト節)
- upstream の存否: `gr_mix_r16_inj` は無い
- upstream の同等物: なし (HC の融合は UPMIX = sm_80+ と HCDOWN = HIP のみ)
- 衝突リスク: 中 (inject から mix の順序を 1 カーネルにし、`pt.mark(kPfHcInj)` / `pt.mark(kPfHcMix)` の位置が変わる。位相別内訳の意味が変わる)
- 判断: 移植 (MIX16 の後)。`m.inj` を mix の後に書くため、次ハーフの `gr_write_norm_rs` が読む順序は保たれる

## 4. フォーク独自スイッチ 5 種の依存関数の upstream での所在

前提: 5 種とも upstream に無い (grep 0 件)。依存する関数がどう変わったかをファイルと行で示す。

| fork スイッチ | 依存する関数 (fork) | upstream 0.2.21 の所在と変化 |
|---|---|---|
| `STRATA_HC_DOWN_BF16` | `Gemm::bf16_narrow` (fork `src/prefill/gemm.cu:1218-1520`、定義は :1393) | `engine/src/prefill/gemm.cu:1154` の `Gemm::bf16` が down を処理 (Pascal は path 2 = `bf16_to_f32` + `cublasSgemm`, :1199-1255)。narrow 相当は `engine/src/prefill/wmma_gemm.cu:502-700` の `kernel_hcdown` / `:675` の `strata_pf_hcdown_bf16` だが HIP gfx11 専用 (`strata_wmma_gfx11_device()`)、CUDA は `wmma_gemm.cu:766` の false スタブ。呼び出し条件は `prefill.cpp:5110-5124` |
| `STRATA_HC_INJ8` | 同上 (N=4 の分岐, fork `gemm.cu:1333-1390` の `bf16_narrow8_kernel`) | upstream は inject を down と同一カーネルで処理 (`wmma_gemm.cu:502-700` が `nd + ni` を 1 回の GEMM に。HIP のみ)。CUDA では `prefill.cpp:5157` の `bf16_proj` のまま |
| `STRATA_HC_MIX16` | `gr_mix_r16` (fork `src/prefill/kernels.cu:167-200, 2863-2870`) | `engine/src/prefill/kernels.cu:160` の `gr_mix_r_kernel` が R (FP32) を読み、`:2979` の `gr_mix_r` が呼び出し。upstream に 16-bit 版は無い。呼び出しは `prefill.cpp:5163` |
| `STRATA_HC_MIXINJ` | `gr_mix_r16_inj` (fork `kernels.cu:195-265, 2938-2948`) | upstream の融合は `gr_upmix` (`kernels.cu:2985-3016`) で up+mix のみ、CUDA は cc>=8 (:3008-3012)。inject と mix の融合は無い |
| `STRATA_MMQ_SQ` | `mmq::swiglu_quant` (fork `src/prefill/moe_mmq.cu`) | `engine/src/prefill/mmq_absent.cpp` が `mmq::` の全体をスタブ (`:24` quantize, `:25` quantize_scatter, `:33` Context::run, `:36` f16_to_f32)。upstream の MoE prompt は `native_expert_i8` (`engine/include/strata/kernels/i8_expert_rule.hpp:37` `i8_switch_effective`, `engine/src/kernels/cuda/iq_kernels.cu:6915` `native_expert_i8_set_mmq_free`) と fused int8 (`moe_fused_iq.cu`, cc>=75) |
| (参考) IQ2_S/IQ2_XS の MMQ tile | `cmake/strata_ggml_tq_overlay.cmake` | `engine/third_party/strata-llama/ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh:174-195` は I=64 のまま。ただし engine はこの木をコンパイルしない |

## 5. upstream で新しく入った P40 / sm_61 向けの変更

### 5.1 文書の所在 (依頼の 3 点)
- `docs/OLDER_GPUS.md:244-262` が Tesla P40 節。既定で効くのは 3 つ: 広い dequant (`STRATA_DEQUANT_WIDE=0` で off, `engine/src/kernels/cuda/dequant_bf16.cu:288`)、expert mode 8 (`STRATA_SM61_TABLE=0`, `iq_kernels.cu:6790` / `native_mmvq.cu:4201`)、interleaved matvec の P40 表 (`STRATA_MMVQ_IL=0`, `verify.cpp:194`)。opt-in が `STRATA_DENSE_F32=1` (`gemm.cu:998`, 既定 off, 8.4-9.8 TFLOPS)
- `docs/OLDER_GPUS.md:150-215` が GP100 (sm_60) 向け opt-in: `STRATA_DOWN_WS`, `STRATA_PATTN_SM60`, `STRATA_SELECT_LONG` (+`STRATA_SELECT_LONG_AT`, `STRATA_TOPK_MULTI`, `STRATA_SCORES_DX`), `STRATA_Q8_SM60`, `STRATA_IQ_STAGE_TABLES`, `STRATA_EXPERT_NIBBLE`。この節は「cc 6.1 では誰も測っていない」と明記している (PATTN / SELECT_LONG)
- `docs/bench-results/2026-10-05-community-2x-p40/README.md`: v0.1.39 (commit 6f32ec0) の基準線。1 枚 P40 で decode 19.6 tok/s、prompt 348 / 388 tok/s (4K / 16K)、2 枚で 34.8 tok/s / 353-501 tok/s。sm_61 向けの新スイッチの記述は無い
- `docs/RELEASE_GRID.md` はリリース検証グリッドで、セルは sm_86 / 89 / 120 / RDNA / SYCL / Metal のみ。P40 セルは無い。0.2.16 から 0.2.21 の P40 向け変更履歴を示す文書は見つからない (未確認)

### 5.2 0.2.16 以降の新スイッチ (P40 に関係するもの)

| スイッチ | 既定 | 実装 | P40 での意味 |
|---|---|---|---|
| MMQ 撤去 (スイッチではない) | - | `engine/cmake/backends/cuda.cmake:94`, `mmq_absent.cpp` | prompt MoE は `native_expert_i8` へ。0.2.16 以降のリリースビルドに MMQ は無い |
| `STRATA_EXPERT_I8` | MMQ 無しビルドでは全カードで実効 1 (`i8_expert_rule.hpp:36-41`, `prefill.cpp:234`) | `i8_expert.cuh`, `iq_kernels.cu:6840-6920` | P40 の prompt expert の新しい既定。`=0` で FP16 経路 |
| `STRATA_I8_N8` | 対象範囲で既定 on (`i8_expert_rule.hpp:63`) | 同上 | 8 エントリのスライス。P100 の実測は 5.7-5.8 行/expert が交差点 |
| `STRATA_I8_FEWROWS` | P100 (6.0) 6 / 他 7 (`i8_expert_rule.hpp:65-68`) | 同上 | 少行の層を grouped カーネルへ。P40 は「他」= 7 |
| `STRATA_SCORES_THREAD` | sm_75 / sm_70 のみ (`qsa_select.cu:2164-2178`) | thread-per-head scorer | P40 では `=1` が必要。我々の qsa_select パッチの代替 |
| `STRATA_SELECT_LONG` (+`_AT`, 既定 65536) | opt-in (`verify.cpp:200-224`) | `qsa_select.cu:1618-1620` | sm_6x の長文 decode select。P40 未測定 |
| `STRATA_PATTN_SM60` | off (`qsa_pattn.cu:37`) | `qsa_pattn.cu` | 既存。我々は本番で =1 (P40 実測は我々のみ) |
| `STRATA_PF_HCDOWN` / `STRATA_HCD_EXACT` | off | `wmma_gemm.cu:502-766`, `gemm.cu:1025` | HIP gfx11 専用。P40 では無効 (CUDA では false、=1 にすると起動時エラー) |
| `STRATA_HC_UPMIX` | sm_120 / sm_121 の表で on | `kernels.cu:2985-3016` | CUDA は cc>=8 のみ。P40 では無効 |
| `STRATA_HC_Q8` / `_REQ8` / `_SPLIT` / `_PERSIST` | パック依存 | `prefill.cpp:5085-5095` ほか | HC 射影を int8/Q8_0 で保持 (VRAM 節約 -> expert cache 増)。P40 で試す価値あり (未測定) |
| `STRATA_CVEC_FUSE` | off | `kernels.cu:272`, `prefill.cpp:1031` | control vector と次ハーフの norm を 1 パスに。3.6 の k カウントが効く経路 |
| `STRATA_DENSE_F32` | off | `gemm.cu:998` | 既存 (P40 の唯一の opt-in 大物) |

### 5.3 P40 で使えない / 使うべきでないもの
- `STRATA_PF_FUSED` 系 (fused int8 prompt experts): `moe_fused.cu:708` の `available()` が `cc >= 75`。P40 は対象外
- `STRATA_HC_UPMIX` / `STRATA_PF_HCDOWN` / `STRATA_HCD_EXACT`: 上記のとおり CUDA sm_61 では false
- `STRATA_MMQ_KQ` / `STRATA_DENSE_MMQ`: 名前は残るが実効なし (5.2 の 1 行目)
- 本番の Qwen3.8 IQ3_XXS は gate/up IQ3_XXS と down IQ4_NL で、`i8_expert.cuh` のコメントにある `native_expert_i8` の対象形式に入る (P40 で動く見込み)
- Inkling SQ-10 の gate/up の型番と `native_expert_i8` の対象可否は未確認

## 6. 本番 STRATA_SET と upstream スイッチの重複表

正: `~/Strata/docker/compose/.env` の `STRATA_SET` (17 項目)

| # | STRATA_SET 項目 | upstream 同名 | upstream の既定 | 重複 / 競合 | 0.2.21 での扱い |
|---|---|---|---|---|---|
| 1 | `ple-io=ram` | CLI `--ple-io` | 自動 (RAM 量で決まる) | 重複 (同じ意味) | そのまま |
| 2 | `prefill=32768` | CLI `--prefill` | `auto` は sm_80+ で 32768 上限 | 重複 | そのまま |
| 3 | `kv-resident=32768` | CLI `--kv-resident` | - | 重複 | そのまま |
| 4 | `draft-conf=0.7` | CLI `--draft-conf` | - | 重複 | そのまま |
| 5 | `spec=4` | CLI `--spec` | 4 (docs の既定) | 重複 | そのまま |
| 6 | `rope-scaling=yarn` | CLI | - | 重複 | そのまま |
| 7 | `rope-scale=2` | CLI | - | 重複 | そのまま |
| 8 | `STRATA_DENSE_F32=1` | あり (`gemm.cu:998`) | off | 重複 (同じ意味) | そのまま (P40 の主力) |
| 9 | `STRATA_PATTN_SM60=1` | あり (`qsa_pattn.cu:37`) | off | 重複 (同じ意味) | そのまま |
| 10 | `STRATA_SELECT_SIMT=1` | あり (`qsa_select.cu:2327/2396`) | off | 重複。**競合の可能性**: 新しい `STRATA_SELECT_LONG` と同時に使ったときの組み合わせは未確認 | そのまま (要 A/B) |
| 11 | `STRATA_DENSE_MMQ=1` | 名前のみ (`gemm.cu:1391`) | - | 実効なし (mmq スタブ) | 削除候補 |
| 12 | `STRATA_MMQ_KQ=1` | 名前のみ (`nv_defaults.cpp`, gfx12 の K-quant 用) | - | 実効なし (CUDA sm_61 では無関係) | 削除候補 |
| 13 | `STRATA_MMQ_SQ=1` | 無し | - | 移植先なし | 削除 |
| 14 | `STRATA_HC_DOWN_BF16=1` | 無し | - | 我々の移植で追加 | 移植後に再評価 |
| 15 | `STRATA_HC_INJ8=1` | 無し | - | 同上 | 移植後に再評価 |
| 16 | `STRATA_HC_MIX16=1` | 無し | - | 同上 | 移植後に再評価 |
| 17 | `STRATA_HC_MIXINJ=1` | 無し | - | 同上 | 移植後に再評価 |

追加を推奨する upstream スイッチ (STRATA_SET に無い)
| スイッチ | 理由 | P40 実測 |
|---|---|---|
| `STRATA_SCORES_THREAD=1` | fork の qsa_select パッチと等価 (同じビット) | 未 (upstream の主張のみ) |
| `STRATA_SELECT_LONG=1` | 128K 超の decode select。128K 本番では閾値 65536 cells を超える | 未 (4x GP100 で +2.9% @115K) |
| `STRATA_EXPERT_I8` | 既定で有効。明示不要。`=0` は FP16 経路 (比較用) | 未 |
| `STRATA_I8_FEWROWS` | 既定 7。P40 の交差点は P100 と違う可能性 | 未 |

## 7. 推奨適用順序 (1 コミット = 1 変更 = 1 測定)

前提: 各段の後に 128K の同一プロンプトで `STRATA_PREFILL_TIMING=1` の位相別と GPU timeline を取る。判定は対象位相と timeline の両方。数値が動かない段は即座に切り分けるため、段ごとに 1 コミットにする。

| 段 | 内容 | 対象ファイル (新パス) | ゲート / 測定 |
|---|---|---|---|
| 0 | upstream 0.2.21 のベースライン (STRATA_SET から #11-#13 を外したもの) | - | 128K の位相別 + timeline + needle 24K/32K/128K + QA 8 問。fork-v9 (178,582 ms / 758 tok/s) との差をここで把握する |
| 1 | c9eec7a9a (qsa value kernel 8 CTA) | `engine/src/kernels/cuda/qsa_pattn.cu:256` | PATTN の QSA 位相。スピルが出れば戻す |
| 2 | e0c4000a3 (write-back の k カウント) | `engine/src/prefill/kernels.cu` | hc write 位相 (差は小)。bitwise 確認 |
| 3 | 02e8c6637 (bf16_pair) | `engine/include/strata/prefill/gemm.hpp`, `engine/src/prefill/gemm.cu`, `engine/src/prefill/prefill.cpp:5124` | hc down+inject 位相。bitwise (gemm_bf16_parity) |
| 4 | 463ab392a (down を bf16_narrow) | 同上 + `prefill.cpp` (スイッチ) | hc read 位相。`gemm_bf16_parity` の narrow 節 |
| 5 | 6d064727d (inject も bf16_narrow) | `engine/src/prefill/prefill.cpp:5157` | hc inj 位相 |
| 6 | 52e908424 (a) INJ8 | `engine/src/prefill/gemm.cu`, `engine/src/prefill/gemm_bf16_parity.cu` | hc inj 位相。parity (非ビット、相対 1e-4) |
| 7 | 52e908424 (b) MIX16 | `engine/include/strata/prefill/kernels.hpp`, `engine/src/prefill/kernels.cu`, `engine/src/prefill/prefill.cpp:5163` | hc mix 位相。`STRATA_HC_MIX16_CHECK=2` の差 + needle/QA |
| 8 | e7c5543cb (MIXINJ) | `engine/include/strata/prefill/kernels.hpp`, `engine/src/prefill/kernels.cu`, `engine/src/prefill/prefill.cpp:5157-5164`, 新規 `engine/src/prefill/gr_mixinj_parity.cu` | hc inj+mix 位相。`gr_mixinj_parity`。位相名の意味が変わる点を記録 |
| 9 | 50c284eca (b) sm_61 の nv_defaults 表 | `engine/src/core/nv_defaults.cpp`, `engine/src/core/nv_defaults_test.cpp` | decode (4K / 128K)。1 スイッチずつ A/B してから表に入れる |
| 10 | 50c284eca (c) route_publish + fuse_wait | `engine/src/core/verify.cpp`, `engine/src/kernels/cuda/native_router.cu`, `engine/include/strata/kernels/native_router.hpp`, `engine/include/strata/core/layer.hpp`, `engine/src/core/layer.cpp` | decode の verify window。手で入れ直す (衝突 高) |
| 11 | a9767bb11 (tools/p40) | `tools/p40/*` | 測定なし。パスを現行ツリーへ |
| 12 | aef2ba5df + 0a050270f (CI + images) | `.github/workflows/fork-images.yml`, `scripts/ci-build-engine.sh`, `docs/IMAGES.md` | イメージビルドのみ。`cmake -S engine` へ書き換え |
| 13 | docs 群 (保留とした 14 件) | `docs/` (旧 `docs/p40/` は新設) | 数値を段 1-10 の結果で置き換えてから |

順序の根拠:
- 1-2 は 1 行 / 数行で bitwise。先に入れて土台を固める
- 3-8 は fork の元の順序 (02e8 -> 463a -> 6d06 -> 52e9 -> e7c5) を保つ。3 と 5 は 4 の前提、7 は 8 の前提
- 9-10 は decode 側で prefill と独立。nv_defaults を先にすると route_publish の効果が表に埋もれるので、decode は 9 -> 10 の順で個別に測る
- 11-13 は性能に影響しない

## 8. 破棄・保留の一覧と、失うもの

### 破棄 (8 コミット)
- コード 4: ca93f701e, bb9c9c140, a2cb22715, 06f08e13b (すべて MMQ 前提)
- 50c284eca の a (qsa_select の sm_61 既定) と d (MMQ / Q3_K / overlay / CMake)
- merge 3: 6732e0325, 2e7d57eb5, a1970db19 (内容は個別コミットが持つ)
- docs のうち、破棄したコードの測定記録 (v7 の MMQ 数値など) は 保留 扱いにして数値だけ残す

### 保留 (docs 14)
221e3e3c8, 5d6928c75, b9a45bb0d, 5b54c2f06, b337633aa, f0945a661, e797f4e0f, 80cc9ec7f, 54d4171f7, e1d8b20f8, a9433270a, d1de43b45, 1a2c8a686, f28f12051。
いずれも旧 `src/` パスと旧ビルドの数値なので、段 1-10 の測定後に書き直す。

### 失うもの (重要)
- prompt MoE の MMQ 経路 (fork の 758 tok/s @128K の主要因)。0.2.21 では `native_expert_i8` に置き換わる。upstream 自身が 7K-20K で遅いと記録している (#6251, `docs/DETAILS.md:1507`) ため、**128K の prefill は fork-v9 より遅くなる可能性が高い**。段 0 で必ず数値を取る
- dense MMQ による K=640 (shared expert down) の高速化
- IQ2_S / IQ2_XS の I=128 tile
- これらを戻すには MMQ の再統合 (llama.cpp の mmq.cu / quantize.cu を engine に戻す) という別プロジェクトになる。本移植の範囲外とし、必要なら別計画にする

## 9. 未確認事項
1. 3-way apply の実試行はしていない (worktree を作れない制約)。衝突リスクはハンクの文脈比較による静的評価
2. 0.2.21 のリリース feed の `strata-engine-0.1.67-linux-x86_64-cuda-cuda12` に sm_61 が含まれるか。CI 定義 (`.github/workflows/backends.yml:68`) は `60-real;61-real;70-real;86-real` + `sm60: ON` だが、リリース実体との一致は未確認
3. `STRATA_SELECT_SIMT=1` と `STRATA_SELECT_LONG=1` の同時使用 (両方 decode の scorer を変える)
4. `STRATA_SCORES_THREAD=1` が P40 で既定カーネルと同一ビットか (upstream の主張のみ、P40 実測なし)
5. `native_expert_i8` の P40 実測 (docs にあるのは P100 / A4000 / RDNA2 のみ)
6. MIX16 / MIXINJ / INJ8 の数値影響。fork の CHECK 値は fork ビルド上のもので、0.2.21 の層構成では再取得が必要
7. `STRATA_HC_Q8` 系 (新) と我々の HC カーネルの併用 (upstream は `wd->q8_scale != nullptr` のとき別経路。fork の条件は `q8_scale == nullptr` を要求するので共存はする見込み)
8. 0.2.16-0.2.21 の P40 向け変更履歴 (リリースノート相当の文書が無い)

## 10. まとめ
- コード 12 件: 移植 7 (うち 52e908424 は 2 分割) + 50c284eca の 2 部分、破棄 4 + 50c284eca の 2 部分
- 破棄候補: ca93f701e, bb9c9c140, a2cb22715, 06f08e13b と 50c284eca の qsa_select / MMQ 部分
- 50c284eca は 4 分割のため、破棄と移植の両方に 1 件ずつ数えている (合計は 8 + 14 + 11 = 33 で、32 コミットより 1 多い)
- 適用順序: 1 c9eec7a9a -> 2 e0c4000a3 -> 3 02e8c6637 -> 4 463ab392a -> 5 6d064727d -> 6 52e908424(INJ8) -> 7 52e908424(MIX16) -> 8 e7c5543cb -> 9 50c284eca(nv_defaults) -> 10 50c284eca(route_publish) -> 11 tools -> 12 CI -> 13 docs
- 最初に必ず段 0 (upstream 0.2.21 のベースライン) を取る。MMQ 撤去の影響で fork-v9 の 758 tok/s は再現しない見込み
