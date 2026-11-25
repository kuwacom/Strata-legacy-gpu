# gemm gate/up の後退 (+38,501 ms) の原因と回復手段

- 日付: 2026-11-24
- 対象: Qwen3.8-Flash-Next IQ3_XXS / 128K prefill (107,460 tokens / 1 チャンク) on Tesla P40 (sm_61)
- 比較: fork-v9 (engine 0.1.57, `strata:fork-v9`) vs upstream 0.2.21 (engine 0.1.67, `strata:up-0.1.67`)
- 入力: `~/work/results/qwen-baseline.md`, `~/work/port-plan.md`, `~/work/CONTEXT.md`
- 制約: 静的調査のみ。ビルド・GPU 測定はしていない。ソースの定数から出した値は「計算値」、推測は「推定」と書く

## 1. 結論

1. **+38,501 ms の内訳: +10,719 ms は位相の付け替え、実際の後退は +27,782 ms (1.90 倍)。**
   0.1.67 の専門家 GEMM は gate/up と down を 1 つの呼び出しで回し、その経路に `kPfGemmD` のマークが無い。
   fork で `gemm down` に出ていた 10,719 ms が `gemm gate/up` に吸われた。位相名の比較で「+190%」に見えるが、
   同じ仕事どうしの比較は 30,978 -> 58,760 ms (+89.7%)。
2. **走っているカーネルが違う。** fork = llama.cpp MMQ の `mul_mat_q<GGML_TYPE_IQ3_XXS>` (int8 dp4a、
   Pascal 用パッチで 128 エントリ/タイル)。0.1.67 = 自前の `native_expert_i8` (int8 dp4a、64 エントリ/タイル)。
3. **1.90 倍の主因 (推定): P40 の shared 48 KB/ブロックの制約で 128 エントリのタイルが使えないこと。**
   `native_expert_i8` は 128 エントリの tile 8 (計算値 54,800 B) を選ぼうとするが、P40 の上限 49,152 B を超えるので
   tile 4 (64 エントリ、計算値 30,720 B) に落ちる。重みタイルのデコードを半分のエントリでしか償却できない。
   fork 側は同じ 48 KB に収まる範囲で I を 64 -> 128 に上げるパッチを当てていた (`docs/patches/mmq-mmq-i128-v3.diff`)。
4. **dense 側も同じ理由で MMQ を失っている。** GDN / QSA 射影 / shared expert は `Gemm::native` -> 量子化解除 +
   cuBLAS FP16 になった (起動ログの `STRATA_DENSE_MMQ has no effect in this engine`)。`gdn` `qsa proj`
   `gdn out proj` `router+shared` の +29.8 s はこれで説明できる (推定。fork 自身の記録では dense MMQ は prefill +11-13%)。
5. **回復手段は実質 1 つ: MMQ を戻す。** upstream の 0.2.15 が MMQ を積んだ最後のリリースで、
   `-DSTRATA_MMQ=ON` 構成がそのまま移植先に載る (移植先の `moe_mmq.hpp` は v0.2.15 とバイト一致、
   `mmq.cuh` も sha256 一致を確認)。工数 中 (3-5 日)。`native_expert_i8` のチューニングでは届かない。

## 2. 位相の分解 (107,460 tokens、同一チャンク)

engine log の 1 行 (107,460 tokens) から。fork は `engine-ref.log`、upstream は `engine-up-core.log`。

| phase | fork-v9 ms | 0.1.67 ms | 差 |
| --- | ---: | ---: | ---: |
| gemm gate/up | 20,259 | 58,760 | +38,501 (+190.0%) |
| gemm down | 10,719 | (行が無い) | -10,719 |
| dequant | 1,591 | 163 | -1,428 |
| gather | 934 | 277 | -657 |
| **合計 (上 4 つ)** | **33,503** | **59,200** | **+25,697** |

分解:

- **+10,719 ms = 位相の吸収。** 0.1.67 の i8 経路は `pt.mark(kPfGemmGU)` の直後に
  `strata::prefill::i8grp::run(...)` を 1 回だけ呼び、その中で gate/up・SwiGLU・H の q8_1 化・down まで走る
  (`engine/src/prefill/prefill.cpp:7180-7181`)。`kPfGemmD` のマークが無いので down の分が gate/up に乗る。
  up 3 アーム (up-core / up-core-min / up-sw) すべてで `gemm gate/up` は 58,760-58,780、`gemm down` 行なしで一致。
- **+27,782 ms = 実体の後退。** fork の gate/up + down = 30,978 ms に対し 58,760 ms = 1.897 倍。
  形の比で検算: gate/up は [1280 x 2560]、down は [2560 x 640] で FLOP 比 2.0。fork の位相比 20,259 : 10,719 = 1.89 で
  ほぼ一致する。つまり 0.1.67 の 58,760 は「GU + down の全部」と読めて、ゲート/アップだけが 2.9 倍になったのではない。
- 参考: `dequant` が -1,428 ms、`gather` が -657 ms なのは、i8 が blob を滞在場所のまま読むため (gather を作らない)。

## 3. カーネルの特定

### 3.1 旧 (fork-v9, engine 0.1.57) — `gemm gate/up`

| 項目 | 実体 | 場所 |
| --- | --- | --- |
| 呼び出し | `mmq::Context::run(gu, m.cs)` | `src/prefill/prefill.cpp:5888` (Product gu) / `:5907` (run) |
| 実カーネル | `mul_mat_q_case<GGML_TYPE_IQ3_XXS>` -> llama.cpp `mul_mat_q<IQ3_XXS, I, J, ...>` (DP4A) | `src/prefill/moe_mmq.cu:677` (ディスパッチ), `:722` (`ck(..., "mul_mat_q")`) |
| タイル | Pascal DP4A の config を I=64 -> 128 に差し替え (J=64/48/40) | `cmake/strata_ggml_tq_overlay.cmake` の `mmq-config-pascal-dp4a.cuh` 節 / `docs/patches/mmq-mmq-i128-v3.diff` |
| 融合 | `mmq::swiglu_quant` (SwiGLU + down 用 q8_1 を 1 パス、`STRATA_MMQ_SQ=1`) | `src/prefill/prefill.cpp:5916` |
| down | `mmq::Context::run(dn, m.cs)` = `mul_mat_q_case<GGML_TYPE_IQ4_NL>` | `src/prefill/prefill.cpp:5936` |
| dense (GDN/QSA 射影/shared expert) | `Gemm::native_mmq` -> `mul_mat_q_case<...>` / `strata_kq` (Q3_K 等) | `src/prefill/gemm.cu:1529` (定義), `:1703` (呼び出し), `src/prefill/mmq_kq_lib.cu:57` |

実行確認 (fork の engine log): `prefill gemm: dense MMQ on (STRATA_DENSE_MMQ=1)` と
`strata: K-quant prompt kernels from .../libstrata_kq.so (STRATA_MMQ_KQ=1)` (どちらも測定アームの起動時に出力)。

fork の P40 記録 (`docs/p40/README.md`) では、この gate/up は次の積み上げ:
`STRATA_MMQ_SQ=1` で -4.2%、IQ2_S/IQ2_XS の I=128/J=48 で -5.4%、DP4A I=128 (J=64) で prefill +3.1%。
`STRATA_MMQ_GU=1` (write-back への GLU 融合) は +76.9% 悪化で不採用 (`docs/prefill/v7-round.md:80`) — つまり fork も
gate/up の融合はしておらず、素の MMQ + SQ で走っていた。

### 3.2 新 (0.2.21, engine 0.1.67) — `gemm gate/up`

| 項目 | 実体 | 場所 |
| --- | --- | --- |
| 呼び出し | `strata::prefill::i8grp::run(...)` | `engine/src/prefill/prefill.cpp:7181` |
| 呼び出し (下位) | `strata::kernels::native_expert_i8(...)` | `engine/include/strata/prefill/i8_group.hpp:88` |
| 実カーネル (gate/up) | `i8_kernel<18 /*IQ3_XXS*/, GU=true, TM=4>` | `engine/src/kernels/cuda/iq_kernels.cu:6985` (関数), `:7030` (分岐); 実体 `engine/src/kernels/cuda/i8_expert.cuh:543` |
| 実カーネル (down) | `i8_kernel<20 /*IQ4_NL*/, GU=false, TM=4>` | `engine/src/kernels/cuda/iq_kernels.cu:7045`; 実体 `i8_expert.cuh:543` |
| 前処理 | `i8_tiles_kernel` (1 ブロック、タイル表) + `i8_put_record_kernel` (1 ブロック、専門家表) | `i8_expert.cuh:489`, `iq_kernels.cu:6980` |
| 位相マーク | `pt.mark(kPfGemmGU)` のみ。down のマーク無し | `engine/src/prefill/prefill.cpp:7180` |

実行確認 (0.1.67 の engine log、測定アーム):
`strata: STRATA_DENSE_MMQ has no effect in this engine (it has no MMQ kernels): the dense projections of a prompt run on the FP16 path.`
と `strata: prompt kernels: every chunk runs 48 of 48 layers on the int8 kernels.` (48/48 層 = 全層 i8)。

`moe_f16_kq.cu` と `moe_fused_iq.cu` は P40 では**走らない**: 前者は gfx11 専用 (`engine/src/prefill/moe_f16_kq.hpp:1`)、
後者の fused int8 は `available() = cc >= 75` (`engine/src/prefill/moe_fused.cu:708`)。P40 に残るのは i8 だけ。

### 3.3 タイル選択 (なぜ 64 エントリなのか)

`native_expert_i8` の選択 (`engine/src/kernels/cuda/iq_kernels.cu`):

- `:7008` `tm = (avg >= 96 && b_ok) ? 8 : max(2, min(4, (avg+8)/16))`。1 専門家あたりの平均行数は
  (107,460 tokens x top-10) / 512 experts = 約 2,099 行/層 (計算値) なので `tm = 8` が選ばれる。
- `:7025` `while (tm > 2 && i8_smem_bytes(...) > smem_limit) tm = tm == 8 ? 4 : tm - 1;`
  `smem_limit` は `cudaDevAttrMaxSharedMemoryPerBlockOptin` (`i8_expert.cuh:1214` の `i8_smem_limit`) = P40 では 49,152 B。
- tile 8 の shared は `I8BSmem<18>` の定数から 13,700 dword = **54,800 B (計算値)** で超過 -> `tm = 4`。
  tile 4 (`I8Smem<18,4>`) は 7,680 dword = **30,720 B (計算値)** で収まる。

fork 側は同じ 48 KB の枠で I=128 (128 エントリ) が収まっていた (overlay の注記: IQ3_S の I=128 タイルは 46,976 B、
IQ2_S/IQ2_XS だけが 51,200 B で入らないため J=48 に限定)。**エントリ数 128 対 64 = 重みデコードの償却が 2 倍違う。**
実測 1.90 倍はこの差と整合する (推定)。

## 4. 質問への回答

### 4.1 `STRATA_DENSE_MMQ=1` は fork で何をしていたか

- 専門家ではなく **dense 射影** (`Gemm::native`) を MMQ に載せるスイッチ。`Gemm::native_mmq` は
  `#if defined(STRATA_PREFILL_MMQ)` の中 (`src/prefill/gemm.cu:1529`)、呼び出しは `:1702`。
  K が 256 の倍数でない形 (shared expert の down、K=640) も `mmq::k_in_padding` で載せていた。
- 0.1.67 では同じ場所が `#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)` になり
  (`engine/src/prefill/gemm.cu:1570`, 呼び出し `:1716`)、CUDA では関数ごと存在しない。
  `engine/src/prefill/prefill.cpp:2132` が「効かない」と 1 回だけ言う (`dense_mmq_notice.hpp`)。
- したがって **`gate/up` の dense 部分 (shared expert) には 0.1.67 では効かない**。shared expert は
  `native_proj(m.gemm, ...)` (`engine/src/prefill/prefill.cpp:5697-5700`) 経由で FP16 + cuBLAS に落ちる。

### 4.2 `native_expert_i8` は gate/up の dense 部分に効くか

効かない。`native_expert_i8` は routed expert のグループ専用 (`i8_group.hpp:62` の `run` が blob ポインタ列を受ける)。
dense 行列 (shared expert、GDN/QSA 射影) は `Gemm::native` の担当で、0.1.67 では
`dequant -> cublasGemmEx` (`engine/src/prefill/gemm.cu:1666` 以降の `Gemm::native`) か、`STRATA_DENSE_F32=1` の
`cublasSgemm`。`dense_i8_tc.cu` (int8 テンソルコア) は `current_cc() >= 75` (`:1683-1702`) で P40 対象外。

### 4.3 `STRATA_DENSE_F32` の意味は変わっていない

- 旧 `src/prefill/gemm.cu:863-935` と新 `engine/src/prefill/gemm.cu:973-1045` は **diff が空 (バイト一致)**。
- 意味: Pascal (cc 6.x) 限定の opt-in。指定形 (K=2560 の 10240/12288/6144 行、N=2560 の 6144/4096 列) かつ
  トークン数がしきい値以上のときだけ、FP16 重みの FP32 コピー + `cublasSgemm` にする (8.4-9.8 TFLOPS)。
  ビット単位で FP16 呼び出しと一致する範囲だけ。0.1.57 と 0.1.67 で差は無い。

### 4.4 sm_61 で無効化されている新しい分岐 (cc ゲート)

| 分岐 | ゲート | P40 で |
| --- | --- | --- |
| fused int8 prompt experts (`moe_fused.cu`) | `cc >= 75` (`:708`) | 無効 |
| fused int8 / WMMA (`moe_fused_iq.cu`, `moe_f16_kq.cu`) | gfx11/gfx12 (HIP) | 無効 |
| dense int8 テンソルコア (`dense_i8_tc.cu`) | `current_cc() >= 75` (`:1702`) | 無効 |
| Volta の tensor-core タイル (`i8h_volta.cuh`, tile 5) | cc 7.0 (`i8_expert_rule.hpp` の `i8_volta`, `iq_kernels.cu` の `i8_volta()`) | 無効 |
| **i8 の tile 8 (128 エントリ)** | **cc ではなく shared 48 KB/ブロック** | **無効 (これが主因、推定)** |
| MXFP8 dense (`STRATA_CUBLASLT_MXFP8`) | cuBLASLt の MXFP8 (sm_89 以降、推定) | 無効 |
| `STRATA_HC_UPMIX` (gr_upmix) | CUDA は cc >= 8 (`kernels.cu:3008-3012`) | 無効 -> `hc read: mix` が 1 -> 3,547 ms |

## 5. MMQ を戻す場合の作業項目

前提: upstream は 0.2.16 で MMQ のビルド経路を削除した (`3e26b4e15` 「remove llama.cpp's MMQ build path」)。
最後に MMQ を積んだのは **v0.2.15** で、削除理由は「どのリリースビルドも MMQ をコンパイルしていなかった」。
つまり 0.2.21 は「`STRATA_MMQ=OFF` の構成」そのもので、復元は既知の構成に戻す作業になる。

確認済みの事実:
- `git show v0.2.15:include/strata/prefill/moe_mmq.hpp` は移植先 `engine/include/strata/prefill/moe_mmq.hpp` と
  **バイト一致** (diff 0)。`launch_splits_k` / `prep_group` も移植先のヘッダにあり、v0.2.15 の `moe_mmq.cu` が実装済み。
- v0.2.15 の vendored `ggml-cuda/mmq.cuh` と移植先 `engine/third_party/strata-llama/ggml/src/ggml-cuda/mmq.cuh` は
  **sha256 一致**。
- 移植先の `third_party/strata-llama/ggml/src/ggml-cuda/` に MMQ のソースは全量残っている (161 ファイル):
  `mmq.cu`, `mmq.cuh`, `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh`, `quantize.cu`,
  `mmq-config-{pascal-dp4a,pascal-older,ampere,blackwell,cdna,gcn,rdna2,rdna3,rdna3-5,rdna4}.cuh`,
  `template-instances/mmq-instance-{q1_0,q2_0,q4_0,q4_1,q5_0,q5_1,q8_0,iq1_s,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_nl,iq4_xs,q2_k,q3_k,q4_k,q5_k,q6_k,mxfp4,nvfp4}.cu`。
  コンパイルされていないだけ (vision テストの oracle 用)。

### 5.1 ソース (ファイル単位)

1. `engine/src/prefill/moe_mmq.cu` — `git show v0.2.15:src/prefill/moe_mmq.cu` (615 行) をそのまま置く。
   移植先ヘッダと API 一致なので改変不要。fork 版 (1,166 行) は `j_opt` / `gu` / `k_in_padding` / `quantize_f16` があり
   **移植先ヘッダに無い**ので使わない (使うとヘッダも fork 版に戻す必要が出る)。
2. `engine/src/prefill/ggml_cuda_host.cu` — v0.2.15 から復元 (`ggml_backend_cuda_context` の最小ホスト実装)。
3. `engine/src/prefill/mmq_kq_lib.cu`, `mmq_kq_loader.cpp`, `engine/include/strata/prefill/mmq_kq_abi.hpp` —
   K-quant dense (Q3_K/Q4_K/Q5_K/Q5_1/Q6_K) を戻すなら必要。IQ3_XXS パックでも起動時に `libstrata_kq.so` を
   読んでいたので、dense 回復には要る。専門家だけ戻す段では不要。
4. `engine/src/prefill/gemm.cu` — `:1570` と `:1716` の `#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)` を
   `#if defined(STRATA_PREFILL_MMQ)` に戻す (fork と同じ)。
5. `engine/src/prefill/mmq_absent.cpp` — ターゲットから外す (ファイルは残す。MMQ off ビルド用)。
6. `engine/src/prefill/prefill.cpp` — **変更不要**。`#define STRATA_PF_OWN_ONLY 1` は
   `#if defined(STRATA_PREFILL_FUSED) && !defined(STRATA_PREFILL_MMQ)` (`:222`) の中なので、
   `STRATA_PREFILL_MMQ=1` にすれば自動的に外れる (i8 既定・起動行・own_only 系が全部 MMQ 前提に戻る)。
7. 任意 (最小構成では不要): `moe_mmq_tab.cuh/.hpp`, `mmq_tab_instance.cu.in`, `moe_iq4_w4a8.cu`, `moe_ws*.cu`,
   `mmq_nvfp4_{w4a4,w4a8}.cu`, `mmq_expert_bench.cpp`, `mmq_nvfp4_parity.cpp`。

### 5.2 ビルド

8. `engine/cmake/backends/cuda.cmake` — v0.2.15 の `macro(strata_cuda_mmq)` (185 行) と
   `strata_ggml_mmq_types()` を復元。`_strata_mmq_types = q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0`
   (+ `STRATA_MMQ_KQUANTS` で q3_k/q4_k/q5_k/q5_1/q6_k を `strata_kq` へ)。
9. `engine/CMakeLists.txt` — `option(STRATA_MMQ ...)`、`strata_cuda_mmq()` の呼び出し、
   `target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1)`、`strata_pf_own` から
   `mmq_absent.cpp` を外す。
10. `engine/cmake/strata_ggml.cmake` — ggml-base は `third_party/strata-ggml` から、MMQ は
    `third_party/strata-llama/ggml/src/ggml-cuda` から取ることになる。`block_q8_1` などの型が両ツリーで
    一致しているか要確認 (`mmq.cuh` の sha256 一致から同一スナップショットの見込み)。
11. **Pascal の I=128 タイル** — fork の `cmake/strata_ggml_tq_overlay.cmake` のうち
    「`mmq-config-pascal-dp4a.cuh` の I を 64 -> 128 (J=64/48/40)」の節だけを持つ最小 overlay を作る
    (trellis 型の節は移植先の ggml に無いので落とす)。overlay は build ツリーに `mmq.cuh` / `mmq-load-tiles.cuh` /
    `mmq-config-*.cuh` / `quantize.cu` のコピーとインスタンスを書き、`strata_mmq` の include 先頭に置く。
    インスタンスは `"../mmq.cuh"` を自分の隣から解決するので、**パッチした config は同じフォルダのインスタンス経由でしか届かない**。
    これを省くと I=64 のまま (fork 実測で prefill -3.1%)。
12. テスト — `tools/check_no_mmq.py` / ctest `no_mmq_symbols` は「MMQ off のときだけ」に条件付け。
    `prefill_mmq_kquant_test` / `mmq_expert_bench` / `mmq_nvfp4_parity` を戻すなら再登録。
13. `tools/check_vendored.py` の manifest (vendored を増やす場合)。

### 5.3 旧 `cmake/strata_ggml_tq_overlay.cmake` が何をしていたか

configure 時に、ピン留めした llama.cpp の `ggml-cuda` ヘッダを build ツリーへ**オーバーレイ**していた:
`ggml.h` に trellis 型 (144-149) を足す / `mmq.cuh` にそのルーティングを IQ2_XS と同じ形で足す /
`mmq-config-*.cuh` の IQ2_XS 行を trellis 型ぶん複製する / `tq.cuh`・`tq-mmq-tiles.cuh` を足す /
インスタンスを書く。加えて fork 独自に **`mmq-config-pascal-dp4a.cuh` の I を 128 に上げる**節
(`docs/patches/mmq-mmq-i128-v3.diff`)。overlay は `strata_mmq` ターゲットにだけ先頭 include で効かせ、
他のターゲットは無改変のピン留めファイルを見る。アンカーが 1 つでも見つからなければ
`STRATA_TQ_OVERLAY_OK=OFF` にして MMQ の trellis だけ諦める (ビルドは通る)。

-> 移植先で必要なのは trellis 部分ではなく **I=128 の節だけ**。

### 5.4 工数・リスク

- 工数: **中 (3-5 日)**。大半はビルドの揉み (1-2 日) と品質ゲート。
- リスク:
  - `third_party/strata-llama` は vision テストの oracle。ここから `.cu` を engine に取り込むと
    同じ木を 2 用途で使うことになる (include 先頭に overlay を置くので衝突はしにくいが、`ggml.h` の版差は要確認)。
  - `no_mmq_symbols` テストと `mmq_absent.cpp` の二重管理が戻る。
  - 22 ファイル前後の追加で upstream 追従コストが戻る (削除コミット 1 つ分)。
  - 品質: MMQ はビット単位で FP16 経路と違う。`needle 24K/32K/128K` `QA 8 問` `prefill_swiglu_quant_test` は必須。

## 6. 代替案の評価

前提: P40 の DP4A 実測上限は 35.5 TFLOPS (`~/work/CONTEXT.md`)。fork の MMQ はこの仕事で
30,978 ms / 405 TFLOP (計算値: 幾何 512 experts x top-10 x n_ff 640 x 48 層) = **約 16 TFLOPS**、
i8 は 58,760 ms で **約 8.6 TFLOPS (推定)**。どちらも上限の 46% / 24%。

| 案 | 期待回復 | 工数 | リスク | 判定 |
| --- | --- | --- | --- | --- |
| **MMQ 復活 (v0.2.15 の `-DSTRATA_MMQ=ON`)** | gate/up -27.8 s (1.90 -> 1.0 倍)。dense も戻せば `gdn`/`qsa proj`/`gdn out proj`/`router+shared` の +29.8 s の大半 | 中 (3-5 日) | 中 | **推奨** |
| i8 の 128 エントリタイルを P40 で成立させる (重みステージのダブルバッファを 1 本化して 54,800 -> 44,560 B、計算値) | 不明。最大で 2 倍の一部 | 中 | 中-高 (上流ファイルへの fork パッチ、数値はビット同一を維持) | 次点。MMQ を戻さない判断なら検討 |
| `STRATA_I8_N8=1` (8 エントリスライス) + `STRATA_I8_TM=3` の掃引 | 0-6% (推定。R9700 では tile 4 で N8 が 1.6-2.9% 遅い) | 小 (env のみ、1 アーム 15 分) | 低 | 先に潰す |
| `STRATA_EXPERT_I8=0` (量子化解除 + FP16 GEMM) | 負の見込み (P40 の cuBLAS FP16 は 4.3-7.6 TFLOPS、`docs/OLDER_GPUS.md` の実測)。ただし dequant 込みで測る価値はある | 小 | 低 | 同時に 1 アーム |
| cublasLt / MXFP8 で dense を速く | P40 対象外 (sm_89 以降、推定) | - | - | 不可 |
| `Gemm::native` を int8 化 (dense_i8_tc の sm_61 対応) | dense の 10-15 s | 大 | 高 | 対象外 |
| 量子化を軽くする (IQ4_XS 等) | 帯域は減るが品質と引き換え | 大 | 高 | 対象外 |

## 7. 推奨する 1 手

**MMQ を戻す (v0.2.15 の `-DSTRATA_MMQ=ON` 構成の復元)。** ただし 2 段に切る:

1. **段 1 (専門家だけ)**: `moe_mmq.cu` + `ggml_cuda_host.cu` + `strata_mmq`
   (q2_0/iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/iq4_nl/iq4_xs/q8_0 のインスタンス) + `STRATA_PREFILL_MMQ=1` +
   `native_mmq` の HIPCC ガード解除 + I=128 の最小 overlay。
   これで `gemm gate/up` の +27.8 s が戻る見込み (fork 実測 16 TFLOPS に対し i8 8.6 TFLOPS 推定)。
   ゲート: 128K 1 アームの位相別 + needle 24K/32K/128K + QA 8 問 + `prefill_swiglu_quant_test`。
2. **段 2 (dense)**: `strata_kq` (Q3_K/Q4_K/Q5_K/Q5_1/Q6_K) と `STRATA_DENSE_MMQ=1`。
   これで `gdn` `qsa proj` `gdn out proj` `router+shared` の後退のうち GEMM 分が戻る。

段 1 の前に、30 分で終わる env 2 アーム (`STRATA_I8_TM=3`, `STRATA_I8_N8=1`) を回して
「i8 のチューニングで数 % 以上動くか」を潰しておく。動かなければ段 1 に進む。

### 戻す価値があるか

**ある。** 理由:
- +27.8 s は全体の +66.4 s の 42% で、1 つの原因としては最大。
- 128K prefill を fork-v9 並 (767 tok/s) に戻す手段が他に無い。i8 のタイル調整で 2 倍は戻らない見込み (推定)。
- 実装は既知の構成の復元で、新規開発ではない (ヘッダ・`mmq.cuh` の一致を確認済み)。
- 一方で、**上流に合わせる必要が本当にあるのか**は別途判断。P40 の本番を fork-v9 のまま維持し、
  上流は必要な修正だけ cherry-pick する方が安い可能性がある (移植計画の前提を揺さぶる事実)。

## 8. 未確認事項

1. `STRATA_I8_TM` / `STRATA_I8_N8` の P40 実測 (未取得)。
2. `native_expert_i8` の tile 8 が P40 で使えないことの実機確認 (計算値のみ。`STRATA_I8_SMEM_LIMIT` で
   小さい値を偽装すれば `native_expert_i8_last_launch()` で観測できる)。
3. fork の overlay が実際に I=128 でビルドされたか (configure ログ未確認。`STRATA_TQ_OVERLAY_OK` の値)。
   ただし fork の P40 記録に「DP4A MMQ I=128 tiles ... prefill +3.1%」があるので、このモデルで効いた実績はある。
4. `gdn` / `qsa proj` / `gdn out proj` の +29.8 s のうち、dense MMQ の損失が何割か (候補は他に
   0.1.57 -> 0.1.67 のコード差。位相だけでは切れない)。
5. 移植先 `third_party/strata-llama` の ggml と `third_party/strata-ggml` の `ggml-common.h` が同一か
   (`mmq.cuh` は一致。`block_q8_1` のレイアウトが違えば MMQ は壊れる)。
6. v0.2.15 の `moe_mmq.cu` を移植先の CUDA 12.9 / sm_61 でコンパイルできるか (未試行)。
7. 1.90 倍のうち、タイル以外 (命令列・占有・L2 の当たり) の寄与 (未分離)。
