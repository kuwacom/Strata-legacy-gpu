# GDN prefill: upstream 0.1.67 が fork-v9 (0.1.57) より +21.6 s (+59%) 遅い原因

- 日時: 2026-11-24
- 対象: `~/work/Strata-legacy-gpu` (`origin/main`, p40-512k-v1-29 = engine 0.1.57) と `~/work/port` (`v0.2.21-5`, engine 0.1.67)
- 入力: `~/work/results/qwen-baseline.md`, `~/work/results/baseline/engine-*.log`
- 108K = 107,460 tokens、36 GDN 層

## 1. 結論

**GDN のカーネルは 1 バイトも変わっていない。+21.6 s は全部 GDN の dense 射影 (attn_qkv / attn_gate / ssm_out) の GEMM 経路差で、0.1.67 が CUDA の dense int8 dp4a MMQ (`Gemm::native_mmq`) を捨てたため。**

- conv+gates: 1,157 -> 1,155 ms (-0.2%)
- recurrence: 6,372 -> 6,254 ms (-1.9%)
- 射影 (in): 20,751 -> 37,189 ms (**+16,438, +79.2%**)
- 射影 (out): 8,262 -> 13,560 ms (**+5,298, +64.1%**)
- 合計: 36,542 -> 58,158 ms (+21,616, +59.1%)

GDN の再帰・conv を実行するカーネルは同一で、実測も同一。動いたのは行列積だけ。

## 2. 対比表

### 2.1 位相の意味 (最初に確認すべき点)

`gdn` は親ブラケットではなく **射影そのもの**。`prefill.cpp` のコメントが明示している。

- 旧 `src/prefill/prefill.cpp:4248` `pt.mark(kPfGdn, cs)` -> `:4262` `pt.mark(kPfGdnConv, cs);   // "gdn" is the projections in; the rest on their own lines`
- 新 `engine/src/prefill/prefill.cpp:5226` / `:5240` (同一文言)

タイマは「連続する 2 マークの間を前の位相に課金」するので、`gdn` = `native_proj(wqkv)` + `native_proj(wg)` + `bf16_proj(wa)` + `bf16_proj(wb)` の 4 本だけ。`gdn conv+gates` / `gdn recurrence` / `gdn out proj` は別枠で、合計に二重計上は無い。

### 2.2 カーネル・起動条件の対比

| 項目 | 旧 0.1.57 | 新 0.1.67 | 同一? |
| --- | --- | --- | --- |
| GDN 再帰 (sm_61 の実体) | `src/prefill/kernels.cu:3233` `gdn_rec_cols_pipe_kernel<<<HV * NCB, dim3(CB, RG)>>>` | `engine/src/prefill/kernels.cu:3277` 同一 | 同一 |
| GDN conv (tiled) | `kernels.cu:3038` `gdn_conv_tiled_kernel<<<dim3(C/128, (T+CONV_TILE-1)/CONV_TILE), 128>>>` | `kernels.cu:3079` 同一 | 同一 |
| GDN conv (history) | `kernels.cu:3040` `gdn_conv_hist_kernel<<<C/128, 128>>>` | `kernels.cu:3081` 同一 | 同一 |
| GDN gates | `kernels.cu:3014` `gdn_gates_kernel<<<blocks_for(T*HV), 256>>>` | `kernels.cu:3055` 同一 | 同一 |
| chunk 幅 (再帰) | `src/prefill/gdn_chunk.cu:29` `constexpr int CH = 64;` | `engine/src/prefill/gdn_chunk.cu:31` `CH = 64` | 同一 |
| 幾何 (S/HK/HV/C) | `gdn_chunk.cu:28` 直値 `128, 16, 48, 10240` | `gdn_chunk.cu:30` `kCompiledGeometry.ssm_*` | 値は同一 (`engine/include/strata/core/geometry.hpp:33-36` が 128/16/48/10240) |
| NSE (head あたり CTA) | `gdn_chunk.cu:31` `NSE = S/NS = 4` | 同一 | 同一 |
| `__CUDA_ARCH__` ゲート (wmma) | `gdn_chunk.cu:232` `>= 700` | `gdn_chunk.cu:234` `>= 700` | 同一 |
| `__CUDA_ARCH__` ゲート (TC) | `gdn_tc.cuh:32` `>= 800` | `gdn_tc.cuh:32` `>= 800` | 同一 |
| wmma カーネル本体 | `gdn_wmma12.inc` 16,873 B | 同一 16,873 B | バイト一致 |
| GDN 呼び出し側 | `prefill.cpp:4248-4283` | `prefill.cpp:5226-5261` | **diff なし** |
| 射影の実体 | `Gemm::native` -> `Gemm::native_mmq` (CUDA でも有効) | `Gemm::native` -> MMQ 無し、dequant + cuBLAS | **差あり** |
| プロンプト chunk | `--prefill 32768` -> 16,384 (ログ) | 同じ | 同一 |

sm_61 では `gdn_chunk.cu` (`>= 700`) も `gdn_tc.cuh` (`>= 800`) もコンパイル対象外なので、GDN の再帰は旧新とも `kernels.cu` の `gdn_rec_cols_pipe_kernel` 1 本。`STRATA_GDN_CHUNKED` / `STRATA_GDN_TC` は両方 opt-in かつ未設定。

### 2.3 GDN 関連ファイルの `git diff` (sm_61 / 128K 経路に効くもの)

| ファイル | 差分 | sm_61 の 128K に効くか |
| --- | --- | --- |
| `gdn_wmma12.inc` | 差分なし | 効かない (sm_70+ 専用) |
| `gdn_tc.cuh` | 2 行: `static_assert` の総バイト数を形状依存にしただけ (`:32` のゲートは不変) | 効かない (sm_80+ 専用) |
| `gdn_chunk.cu` | 2 ハンクのみ: `geometry.hpp` の include と S/HK/HV/C の定数化 (`:29-30`)、ゲート行が 2 行ずれただけ | 効かない (sm_70+ 専用) |
| `prefill.cpp` GDN ブロック | なし (`diff` で一致) | — |
| `kernels.cu` GDN 領域 | `__gfx900__` 分岐の追加と、gfx12 の `gdn_chunked_auto()` / `STRATA_GDN_CHUNKED` 既定の変更のみ | **効かない** (HIP 専用の `#if defined(__HIPCC__)` 内) |
| `gemm.cu` (GDN 射影が通る) | **4 箇所のガードに `&& defined(__HIPCC__)` が付いた** = dense MMQ が CUDA から消えた | **これが原因** |

`kernels.cu` の gfx12 変更 (`gdn_chunked_auto`, `STRATA_GDN_CHUNKED_MIN=30000`) は魅力的に見えるが、`gdn_rec_chunked()` の入口が `if (!gfx11 && !gfx12) return false;` で、CUDA では到達しない。無関係。

### 2.4 射影で何が走っているか

旧 (0.1.57): `Gemm::native` -> `native_mmq` -> llama.cpp の `mul_mat_q` (int8 dp4a)

- `src/prefill/gemm.cu:4` `#if defined(STRATA_PREFILL_MMQ)` (include)
- `:653` (Context の破棄), `:1529` (`native_mmq` 本体), `:1702-1703` `if (beta == 0.0f && ... && native_mmq(...)) return;`
- CUDA でもビルドされ、`STRATA_DENSE_MMQ=1` で選ばれる
- `src/prefill/moe_mmq.cu:341` `case GGML_TYPE_Q3_K:   // the dense GDN projections' Q3_K weights (STRATA_DENSE_MMQ on CUDA)`
- K の条件 `kMaxK = 8192`、`K % 256 == 0` (`gemm.cu:1533`, `:1548`)。GDN の K は 2560 / 6144 でどちらも満たす

新 (0.1.67): `Gemm::native` -> dequant + cuBLAS

- `engine/src/prefill/gemm.cu:6`, `:753`, `:1570`, `:1716` すべて `#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)`
- CUDA では `native_mmq` が存在せず、`:1717` の呼び出しもコンパイルされない
- 残るのは `STRATA_DENSE_F32=1` の cublasSgemm (FP32 コピー)。`gemm.cu:971-983` のコメントが P40 実測 8.4-9.8 TFLOPS と明記

実効 TFLOPS (107,460 tokens × 36 層 の FLOPs から):

| 射影 | FLOPs | 旧 0.1.57 | 新 0.1.67 |
| --- | ---: | ---: | ---: |
| in (qkv 10240x2560 + gate 6144x2560) | 3.245e14 | 15.6 TFLOPS | 8.73 TFLOPS |
| out (ssm_out 2560x6144) | 1.217e14 | 14.7 TFLOPS | 8.97 TFLOPS |

新の 8.7-9.0 は `gemm.cu:974-983` の「P40 の cublasSgemm FP32 コピーは 8.4-9.8 TFLOPS」と一致。旧の 15-16 は dp4a (上限 35.5) の範囲。

**対照群**: BF16 重みの射影は両者で不変 (`hc read: up` 7,320 -> 7,319 ms)。BF16 は元から MMQ 対象外なので、MMQ を失った Q3_K の射影だけが遅くなった、という形になっている。

## 3. sm_61 で無効化される新規の cc ゲート

**無し。**

- GDN カーネルの `__CUDA_ARCH__` ゲートは旧新で同一 (`>= 700` / `>= 800`)。新規追加なし
- `Gemm::native` に増えた CUDA 分岐は 2 つだが、どちらも sm_61 を除外しない
  - `dense_fp4_tc` (ggml type 40 = MXFP4) — 今回の Q3_K は対象外
  - `STRATA_CUBLASLT_MXFP8` — コンパイル時 opt-in、sm_61 では未定義
- `dense_f32_on()` の条件は `cc >= 60 && cc < 70` (`gemm.cu`) で sm_61 を含む

ただし **cc ゲートではない別の「既定値の消失」** がある (GDN の原因ではない):

- 旧 `src/core/nv_defaults.cpp:53-70` に sm_61 表 (10 スイッチ) がある
- 新 `engine/src/core/nv_defaults.cpp` の分岐は `(12,0)` `:104` / `(7,0)` `:136` / `(6,0)` `:164` / `(12,1)` `:176` のみで、**(6,1) が無い** -> P40 は空の表
- 実測ログでも一致: 旧は `strata: sm_61 (Pascal, P40): 10 exact speed switches on by default ...` を出す。0.1.67 の起動 (banner 30069 以降) ではこの行が 1 度も出ない
- sm_61 表の中身は decode/スケジューリング寄り (`ADAPT_*`, `SIDE_MIXER`, `QSA_ROWS` など)。prefill 側で関係するのは `STRATA_GR_UP75` (hc read up) だけで、その位相は実測で不変。**GDN の +21.6 s には寄与していない**

## 4. スイッチで戻せるか

**戻せない。**

- `STRATA_DENSE_MMQ` は 0.1.67 で無効。`engine/include/strata/prefill/dense_mmq_notice.hpp:14` が「この engine には MMQ カーネルが無いので効果なし」と明言し、`engine/src/prefill/prefill.cpp:2132-2134` が起動時に 1 回その行を出す
- `STRATA_MMQ_KQ` / `STRATA_MMQ_SQ` も CUDA では無効 (カーネル自体がツリーに無い)
- 実測ログの裏付け
  - 旧 ref: 自身の起動 (banner 29871) の後に `prefill gemm: dense MMQ on (STRATA_DENSE_MMQ=1)` (29939) と `strata: K-quant prompt kernels from .../libstrata_kq.so (STRATA_MMQ_KQ=1)` (29940) が出る
  - 新 up-core-min: 0.1.67 の起動 (banner 30069) 以降、この 2 行は出ない。最後に出るのは 29588/29589 = 直前の 0.1.57 の起動。代わりに `prefill gemm: the dense projections' FP16 GEMMs run as cublasSgemm on FP32 copies (STRATA_DENSE_F32=1 ...)`
  - 新 up-core (本番 env で `STRATA_DENSE_MMQ=1`): `strata: STRATA_DENSE_MMQ has no effect in this engine (it has no MMQ kernels)` (29967)
- `STRATA_DENSE_F32=1` は既に P40 で最良の残存経路。0 にすると cuBLAS の FP16 呼び出し (同コメントで 4.3-7.6 TFLOPS) に落ちてさらに遅い

### 戻すべき差分

上流が意図的に外している (0.2.16 の P4-SPLIT) ので、ガードを 1 行戻すだけでは終わらない。

1. **ガード (これだけでは足りない)**: `engine/src/prefill/gemm.cu` の 4 箇所 `:6` `:753` `:1570` `:1716` から `&& defined(__HIPCC__)` を外す。fork の `50c284eca` がまさにこれ (旧 `src/prefill/gemm.cu` の同 4 箇所)
2. **MMQ 本体の再導入**: 0.1.67 は llama.cpp の MMQ をツリーから外している
   - `engine/cmake/backends/cuda.cmake:94` 「llama.cpp's MMQ left the tree in 0.2.16」
   - 同 `:136` は `src/prefill/mmq_absent.cpp` (スタブ) をリンク
   - 同 `:160-167` は `tools/check_no_mmq.py` で MMQ シンボルが 1 つでもあるとテストを落とす
   - `strata_mmq` / `strata_kq` ターゲット、`moe_mmq.cu`、`mmq_kq_lib.cu`、`mmq_kq_loader.cpp`、q3_k のテンプレートインスタンスが**すべて無い** (旧 `CMakeLists.txt:2325-2340` が CUDA 用 q3_k を追加していた)
3. **純 CUDA の選択肢 (推奨)**: 上流が P4-SPLIT で MMQ を切った以上、`STRATA_MMQ_*` を復活させるより、Q3_K 用の int8 dp4a dense GEMM を Strata 自身のカーネルとして書くほうが筋が通る。sm_61 の dp4a 表は既にある (`engine/src/kernels/cuda/native_mmvq.cu:4199` `kIlRows_61`)。対象は 3 形状だけ (K=2560 で N=10240 / 6144、K=6144 で N=2560) なので、汎用 MMQ を戻すより小さい

なお fork 側の追加最適化 2 件 (`a2cb22715` shared expert down の MMQ、`ca93f701e` 活性を FP16 のまま量子化) は GDN には不要。

## 5. 次の実測で確認すべき項目

1. **切り分けの直接確認**: 1 層だけ GDN 射影を cuBLAS と dp4a で撃ち分けるマイクロベンチ (`bench/` の gemm プローブ)。107,460 tokens ではなく T=16,384 × 1 層で、10240x2560 / 6144x2560 / 2560x6144 の 3 形状。期待値は旧 15-16 / 新 8.7-9.0 TFLOPS
2. **重み型の確認**: `attn_qkv.weight` / `attn_gate.weight` / `ssm_out.weight` の ggml type が Q3_K であることを GGUF メタデータで確認 (本レポートは fork のソースコメント `moe_mmq.cu:341` と実測ログからの推定)
3. **sm_61 既定値の寄与**: `STRATA_SM61_PERF=0` を付けた 0.1.57 アームを 1 本。これで「10 スイッチの消失」が prefill に寄与していないこと (本レポートの推定) を実測で潰せる。差が ±0.5% 以内なら推定どおり
4. **内訳の推定検証**: 本レポートは +16.4 s のうち qkv 63% / gate 37% と FLOPs 比例で推定した。N が違うので実測でずれる可能性がある。射影ごとに位相マークを分けた計測版で 1 アーム
5. **chunk 幅**: 両アームの `strata serve: prompt chunk` 行が起動時に出ていない (ログが共有・追記式のため、同一引数の過去起動の行しか無い)。recurrence 時間が一致しているので同じと判断したが、chunk 数を直接出すログが望ましい

## 6. 未確認・注意

- **未確認**: GDN 4 重みの ggml type (上記 5-2)
- **未確認**: 0.1.67 の `Gemm::native` に Q3_K 専用の新カーネルが無いこと。起動ログの告知と TFLOPS の一致から「dequant + cuBLAS」と判断したが、ソースの全経路は追っていない (推定)
- **未確認**: +16.4 s の qkv / gate 内訳 (推定)
- **注意**: `engine-*.log` は arm ごとの独立ファイルではなく、共有 state の追記ログのコピー。1 ファイルに 0.1.52 / 0.1.57 / 0.1.67 の起動が混在する。本レポートは「0.1.67 の banner (engine-up-core-min.log:30069) 以降」を実行区間として扱った。この区間に他の起動は無い
- **注意**: `qwen-baseline.md` の「GDN (gdn + conv+gates + recurrence + out proj)」の合計は二重計上ではない (`gdn` は射影のみ)。ただし `gdn` を「層全体」と読むと +16.4 s を「再帰が遅くなった」と誤読する
- **注意**: 本レポートは読み取りのみ。ビルド・GPU 測定・本番操作はしていない
