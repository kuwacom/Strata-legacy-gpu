# prefill の QSA select（block score + top-k）を sm_61 で速くする

作業ツリー: `~/work/wt/pqsa`（branch `perf/prefill-qsa`、フォーク HEAD `9f7d977d`）
計測: Tesla P40 24 GB、`strata:fork-9f7d977-base`（sm_61 build、engine 0.1.57）、本番と同じ 512K 構成
（`--max-context 524288 --kv int8 --kv-resident 32768 --spec 4 --rope-scaling yarn --rope-scale 2`,
`STRATA_DENSE_F32=1`）。数値はすべて実測。推定は「推定」と書いた。

## なぜ select が長文で最大のレバーになるか

QSA の attention そのものは選ばれた 2,051 cell に固定される（`include/strata/kernels/qsa.hpp:282`
`kTopkMaxCells = 32768`、`qsa_selection_width` は `idx_top_k + idx_block - 1 = 2051`）ので、480K でも
attention の計算量は 2,051 cell 分。一方 **select（block score と top-k）は query × 全 block で O(N²)**。
この非対称が比率の差になる: 実測で 25.9K の select は 2.5%、480K では **39.3%**（attention は 25.9K で
13.8%、480K で 8.9% と O(N) のまま）。

## 1. 現在の経路（sm_61 = P40 の既定）

`src/prefill/prefill.cpp:4286` で `pt.mark(kPfQsaSel, cs)`（`STRATA_PREFILL_TIMING` の "qsa select" の開始）、
`:4399` の `kPfQsaAttn` までが select。

| # | 呼び出し | file:line | sm_61 で実際に走るカーネル |
|---|---|---|---|
| 1 | `for (t0 = 0; t0 < T; t0 += m.sel_batch)` — `sel_batch = 256` | prefill.cpp:4287 / :1104 | 1 呼び出し = 256 query |
| 2 | `active = 最後の query の n_bid + 1` | prefill.cpp:4289-4291 | このバッチの最大 block 数 |
| 3 | `qsa_block_scores_tc(...)` | prefill.cpp:4296 -> qsa_select.cu:2204 | **false を返す**（sm_80 未満。`STRATA_SELECT_SIMT=1` のときだけ `block_scores_simt_kernel`） |
| 4 | `qsa_block_scores(...)` | prefill.cpp:4299 -> qsa_select.cu:2089 | `block_scores_kernel`（qsa_select.cu:35、grid `(reach/8, nq)`、warp = 1 (query, block)） |
| 5 | `qsa_block_topk(...)` | prefill.cpp:4301 -> qsa_select.cu:2971 | `block_topk_wide_kernel`（qsa_select.cu:858、1024 thread、1 CTA = 1 query、radix 4 パス + count + emit の 6 パス） |

- `block_scores_kernel` は warp が 1 つの (query, block) を担当し、key 128 float = 512 B を **query ごとに
  読み直す**。4 head × 4 積 = 16 MAC のあと 20 シャッフル（qsa_select.cu:35-62）。
- `qsa_block_scores_tc` が false になる理由: `#if !STRATA_SEL_SM80` で trap になり（qsa_select.cu:174-199）、
  sm_61 では `block_scores_simt` を `STRATA_SELECT_SIMT=1` のときだけ使う。
- top-k は `counted = false`（`topk_active_turing_device()` が Turing のみ true、qsa_select.cu:2359）なので
  `reach = max_blocks = 131,072`（512K 容量）になり、`fit = 1024 * 33 = 33,792` を超えるため wide kernel。
  128K プロンプト（active 32,769 block）でも wide kernel に落ちる。

### 形状

| 項目 | 値 | 出典 |
|---|---|---|
| indexer 次元 / head / block | 128 / 4 / 4 cells | qsa_select.cu:22 |
| 1 query の選択幅 cap | 2,051 cells | qsa.hpp:133 |
| select 呼び出しあたりの query | 256 | prefill.cpp:1104 |
| QSA 層 | 12（48 層中） | geometry |
| 480K の prefill chunk | 8,192 tokens（`--prefill auto`） | engine log "prompt chunk auto: 8192 tokens" |

### 480K の仕事量（推定）

480K を 8,192 chunk で 59 chunk。chunk の位置 `p0` で query 8,192 × `p0/4` block。
合計 block-score 数 = 12 層 × Σ(p0/4 + 2048) × 8192 ≈ **3.56 × 10¹¹**（推定）。
`block_scores_kernel` は 1 block-score あたり key 512 B を読むので **182 TB**（推定）。P40 の HBM 346 GB/s
なら 526 s。実測の select 617 s のうち score 部が約 526 s、top-k が約 91 s という按分になる（推定）。


## 2. 実測

### 2.1 本番ログの履歴（同一構成・同一カード。私の計測ではない。参考）

`STRATA_PREFILL_TIMING=1` の行が本番の engine ログに残っていた（`--max-context 524288`、`--spec 4`、
`STRATA_DENSE_F32=1`、`--kv int8`）。私の計測ではないので「履歴」と書く。

| prompt tokens | GPU timeline | qsa select | 割合 | qsa attn | 割合 |
|---|---|---|---|---|---|
| 28,281 | 59,089 ms | 2,131 ms | 3.6% | 7,893 ms | 13.4% |
| 27,540 | 57,657 ms | 2,006 ms | 3.5% | 7,679 ms | 13.3% |
| 79,648 | 176,411 ms | 18,498 ms | 10.5% | 22,805 ms | 12.9% |

- select は 28.3K → 79.6K で 8.7 倍（token は 2.8 倍）。**O(N²)** にほぼ乗っている。
- 一方 qsa attn は 2.9 倍（token 比 2.8 倍）で **O(N)**。選ばれた 2,051 cell に固定されているため。
- 線形部（total − select）は 2.01 ms/token（28.3K）と 1.98 ms/token（79.6K）でほぼ一定。

### 2.2 今回の計測（`strata:fork-9f7d977-base`、sm_61、512K 構成、`BENCH_SEED=42`）

`STRATA_PREFILL_TIMING=1` を付けた run（arm A2）の結果は §2.3。まず timing 無しの run（arm A）の TTFT。

| arm | prompt tokens | TTFT | prefill tok/s | 備考 |
|---|---|---|---|---|
| A | 28,711 | 58.7 s | 489.1 | 24K 相当（暖機を兼ねる） |
| A | 135,072 | 310.2 s | 435.4 | 128K 相当（code） |
| A | 135,064 | 15.2 s | 8,902 | **無効**: 会話キャッシュが 131,072 token を再利用した |
| A | 517,175 | 1,736.5 s | 297.8 | 480K 相当（prose、コーパス 1,728,109 文字） |

- 517K の 1,736 s は §2.1 の fit と一致する: 線形部 517,175 × 2.0 ms = 1,034 s、select 2.7e-6 × 517,175² = 721 s、
  合計 1,755 s（実測 1,736 s）。つまり **517K の select は約 41%**。
- 128K prose が無効になった原因: `--conversation-cache-mib`（既定で有効）が同一プレフィックスのプロンプトを
  再利用する。以後の run は `bench-nc.py` で先頭に nonce を置き、必ずフル prefill にした。

### 2.3 内訳（arm A2、`STRATA_PREFILL_TIMING=1`、base イメージ）

24K（25,856 / 28,723 tokens）と 128K（135,081 / 135,076 tokens）の 2 本ずつ。数値は GPU timeline の ms（%）。

| 段 | 25.8K | 28.7K | 135.1K | 135.1K |
|---|---|---|---|---|
| GPU timeline | 52,050 | 57,678 | 311,448 | 312,589 |
| qsa select | 1,295 (2.5%) | 1,707 (3.0%) | **48,517 (15.6%)** | **48,515 (15.5%)** |
| qsa attn | 7,177 (13.8%) | 8,012 (13.9%) | 38,875 (12.5%) | 38,884 (12.4%) |
| gdn | 8,920 (17.1%) | 9,934 (17.2%) | 46,525 (14.9%) | 47,235 (15.1%) |
| gemm gate/up | 7,240 (13.9%) | 7,982 (13.8%) | 36,894 (11.8%) | 37,020 (11.8%) |
| qsa proj | 3,875 (7.4%) | 4,280 (7.4%) | 20,329 (6.5%) | 20,542 (6.6%) |
| gemm down | 4,203 (8.1%) | 4,650 (8.1%) | 21,416 (6.9%) | 21,425 (6.9%) |
| gdn out proj | 3,591 (6.9%) | 3,944 (6.8%) | 19,002 (6.1%) | 19,114 (6.1%) |
| hc read: down+silu | 3,070 (5.9%) | 3,452 (6.0%) | 16,091 (5.2%) | 16,099 (5.2%) |

- select は 25.8K → 135K で **37.5 倍**（token は 5.2 倍）。指数 p = ln(37.5)/ln(5.22) = **2.19**。
  79.6K → 135K では p = 1.82。**O(N²) より少し緩い**（selection 幅が 2,051 cell で飽和するため、と推定）。
- attn は 25.8K → 135K で 5.4 倍（token 比 5.2 倍）で **O(N)**。select との非対称がはっきり出ている。
- 480K の実測は §2.4。

### 2.4 480K（arm A2、480,888 tokens）

| 段 | ms | % |
|---|---|---|
| **GPU timeline** | **1,571,598** | 100 |
| **qsa select** | **617,173** | **39.3%** |
| qsa attn | 139,660 | 8.9% |
| gdn | 170,374 | 10.8% |
| gemm gate/up | 131,928 | 8.4% |
| gemm down | 76,078 | 4.8% |
| qsa proj | 73,800 | 4.7% |
| gdn out proj | 68,021 | 4.3% |
| hc read: down+silu | 57,232 | 3.6% |
| router+shared | 39,573 | 2.5% |
| hc read: up | 33,284 | 2.1% |
| hc write | 29,887 | 1.9% |
| gdn recurrence | 27,934 | 1.8% |
| dequant | 21,047 | 1.3% |
| kv stage | 15,261 | 1.0% |
| hc read: inject | 18,838 | 1.2% |
| hc read: mix | 15,896 | 1.0% |
| qsa indexer | 24 | 0.0% |

- **select が最大の単一項目（39.3%）**。依頼文の「42%」とほぼ同じ（プロンプト長 480,888 vs 480K、実測値）。
- select の N 依存: 25.8K 1,295 ms → 135K 48,517 ms → 480.9K 617,173 ms。
  135K→480.9K の指数は ln(617173/48517)/ln(480888/135081) = **1.93**（ほぼ O(N²)）。
- attn は 8.9% で、select の 1/4.4。128K では 12.5% だったので、長くなるほど相対的に薄まる。
- `qsa indexer` は 24 ms（0.0%）。**indexer の再計算は無い**（案 #7 は見込みなしの裏付け）。
- `kv stage` は 128K の 1,216 ms から 15,261 ms に増える（KV を RAM から stage するため）。select の次に伸びる項目だが 1.0%。

### 2.5 マイクロベンチ（`qsa_select_bench`、capacity 524288、256 query）

`build-native/qsa_select_bench <ctx> 256 5 524288`。scores は 1 呼び出し（256 query × その ctx の block）の ms。

| ctx | blocks | warp 版（既定） | **thread 版** | 倍率 | SIMT=1 | SIMT=2 |
|---|---|---|---|---|---|---|
| 32,768 | 8,193 | 3.500 ms | **0.746 ms** | 4.7x | 0.347 ms | 0.399 ms |
| 131,072 | 32,769 | 13.982 ms | **2.234 ms** | 6.3x | 1.317 ms | 1.495 ms |
| 491,520 | 122,881 | 52.411 ms | **7.926 ms** | 6.6x | 4.883 ms | 5.496 ms |

- thread 版は **selections identical 256/256、cells differing 0.0000%、score rel diff mean 0 max 0**（3 ctx すべて）。
  ビット一致。
- SIMT=1 はさらに 1.6x 速いが、128K で 255/256（cells 0.0002%）と選択が 1 件だけ変わる。非ビット。
- top-k（同じ 256 query）:

| ctx | `block_topk_ref`（最古） | 既定の wide kernel | counted で register kernel になる場合 |
|---|---|---|---|
| 32,768 | 0.966 ms | **0.283 ms** | 0.757 ms（遅い） |
| 131,072 | 7.075 ms | **0.859 ms** | 0.866 ms（ほぼ同じ） |
| 491,520 | 51.066 ms | **3.017 ms** | - |

→ **top-k は wide kernel が最速**。`counted` を sm_61 に広げると 32K では register kernel（0.757 ms）に
落ちて 2.7 倍遅くなるので、その案は**採らない**（実装したが測定で棄却した）。

### 2.6 A-B-A（prefill、`BENCH_SEED=42`、1 本目除外、中央値）

A = `strata:fork-9f7d977-base`、B = `strata:fork-qsa`（この変更）。プロンプトは `bench-nc.py`（先頭 nonce で
会話キャッシュの再利用を防ぐ）。

128K 相当（code/prose、実測 131K〜143K token）:

| arm | prompt | TTFT | prefill tok/s |
|---|---|---|---|
| A（変更前） | 135,072 | 310.19 s | 435.4 |
| A（変更前） | 135,088 | 313.29 s | 431.2 |
| A（変更前） | 135,083 | 316.08 s | 427.4 |
| **B（変更後）** | 135,091 | 277.44 s | **486.9** |
| **B（変更後）** | 140,526 | 291.42 s | **482.2** |
| **B（変更後）** | 131,107 | 271.59 s | **482.7** |
| **B（変更後）** | 143,185 | 296.09 s | **483.6** |
| A（B の後） | 135,088 | 314.13 s | 430.0 |
| A（B の後） | 140,526 | 331.25 s | 424.2 |
| A（B の後） | 131,107 | 305.49 s | 429.2 |
| A（B の後） | 143,190 | 337.27 s | 424.6 |

- **A-B-A が閉じた**: select は A 48,517 → B 13,120 → A 48,518 ms（135,081 token の同じ長さの対）。
  A は 4 標本で 45,688〜54,555 ms（15.1〜16.3%）、B は 4 標本で 12,552〜14,381 ms（4.7〜4.9%）で再現する。
- A の throughput 中央値 429.2 tok/s（7 標本）、B 483.2 tok/s（4 標本）→ **+12.6%**。
- token 数がほぼ同じ 1 対（A 135,081 / B 135,084）の GPU timeline: **311,448 → 275,606 ms = −11.5%**。
  A の 2 回目（A3）は 312,533 ms なので、A のばらつきは ±0.4%。
- qsa select の実測: **48,517 ms (15.6%) → 13,120 ms (4.8%) = −73%（3.7x）**。
- 他の段はほぼ変わらない（qsa attn 38,875 → 38,883、gdn 46,525 → 46,090、gemm gate/up 36,894 → 約 36,000）。


24K 相当（実測 24,946〜30,260 token）:

| arm | prompt | GPU timeline | qsa select |
|---|---|---|---|
| A | 28,723 | 57,678 ms | 1,707 ms (3.0%) |
| A | 25,856 | 52,050 ms | 1,295 ms (2.5%) |
| A | 28,732 | 57,852 ms | 1,710 ms (3.0%) |
| A | 25,868 | 52,085 ms | 1,297 ms (2.5%) |
| A | 30,257 | 60,599 ms | 1,955 ms (3.2%) |
| A | 27,647 | 55,574 ms | 1,542 ms (2.8%) |
| A | 25,973 | 52,111 ms | 1,310 ms (2.5%) |
| A | 24,949 | 50,207 ms | 1,198 ms (2.4%) |
| B | 28,729 | 56,818 ms | 1,213 ms (2.1%) |
| B | 25,976 | 51,551 ms | 1,055 ms (2.1%) |
| B | 30,260 | 59,540 ms | 1,300 ms (2.2%) |
| B | 27,646 | 54,621 ms | 1,151 ms (2.1%) |
| B | 24,946 | 49,829 ms | 1,001 ms (2.0%) |

- 同程度の長さの対で **−1.0%（25.9K）／−1.8%（28.7K）**。select 自体は 1,710 → 1,213 ms（1.4x）。
  24K では top-k と起動の固定費が残るので、ここでは効果が小さい。**5% 閾値には届かない**。
- throughput の中央値: A 482.7 tok/s（8 標本）、B 490.0 tok/s（6 標本）→ **+1.5%**。
- B の 24K は「top-k の counted 変更」も含むイメージで測っている（この変更は 24K で +0.5 ms/呼び出しの
  小さな退行）。最終パッチはそれを落としているので、24K の改善は実際にはもう少し大きい（推定 +0.5%）。


480K: A は 480,888 token で GPU timeline 1,571,598 ms、select 617,173 ms。B は 480,892 token で
1,075,054 ms、select 127,321 ms。**select −79.4%、prefill −31.6%**。

480K の内訳（A = base / B = fork-qsa、token 数はほぼ同一）:

| 段 | A（480,888） | B（480,892） | 差 |
|---|---|---|---|
| **GPU timeline** | **1,571,598 ms** | **1,075,054 ms** | **−31.6%** |
| **qsa select** | **617,173 ms (39.3%)** | **127,321 ms (11.8%)** | **−79.4%** |
| qsa attn | 139,660 (8.9%) | 139,606 (13.0%) | ±0 |
| gdn | 170,374 (10.8%) | 166,046 (15.4%) | −2.5% |
| gemm gate/up | 131,928 (8.4%) | 130,768 (12.2%) | −0.9% |
| gemm down | 76,078 (4.8%) | 75,929 (7.1%) | ±0 |
| qsa proj | 73,800 (4.7%) | 72,566 (6.7%) | −1.7% |
| gdn out proj | 68,021 (4.3%) | 66,946 (6.2%) | −1.6% |
| hc read: down+silu | 57,232 (3.6%) | 57,194 (5.3%) | ±0 |
| router+shared | 39,573 (2.5%) | 39,527 (3.7%) | ±0 |
| dequant | 21,047 (1.3%) | 21,242 (2.0%) | ±0 |
| kv stage | 15,261 (1.0%) | 15,283 (1.4%) | ±0 |

- 差の −496 s はほぼ全部が select（−490 s）。他の段は動いていない。**変更が狙った所だけを削っている**。
- prefill throughput: 480,888 / 1,571.598 = 306.0 tok/s → 480,892 / 1,075.054 = 447.3 tok/s（**+46%**）。
- 残った select 127 s の内訳は、マイクロベンチの比から score 部 ≈ 83 s、top-k ≈ 44 s（推定）。






## 3. 改善案

期待幅は §2 の実測と roofline からの**推定**。bitwise は「今の warp 版とビット一致するか」。

| # | 案 | 期待幅（推定） | bitwise | 実装コスト | リスク |
|---|---|---|---|---|---|
| 1 | prompt の block score を thread-per-head 版（`block_scores_thread_kernel`）へ。key を 64 query で共有しシャッフルを消す | 480K の select 617 s のうち score 部 **約 526 s**（key 182 TB の推定）→ **約 40 s**（推定）。prefill 1,572 s → 約 1,085 s（**−31%**、推定） | する | 小（既定の分岐 1 か所） | 小（ビット一致、24K では 1 CTA/SM の影響が出うる） |
| 2 | top-k の `counted` を sm_61 に広げる（容量ではなく実際の n_bid で register kernel を選ぶ） | 24K〜90K のプロンプトで wide kernel の 6 パス → register kernel の 1 パス。**90K 超では `turing_wide` の帯（22,528 block 超）が wide を選ぶので変わらない** | する | 小（`topk_active_bound_device` の 1 条件） | 小（ids は同一） |
| 3 | `STRATA_SELECT_SIMT=1`（タイル FP32 scorer、既存 opt-in） | RTX 2080 Ti の実測で 6.8x（256 query, 120K-250K）。P40 でも score 部 5x 級 | **しない**（和の順序が違う） | なし（env のみ） | 中（near-tie で選択が変わりうる。KL と回答一致の確認が要る） |
| 4 | `STRATA_PATTN_SM60=1`（prompt attention を sm_6x カーネルへ） | attention は 28K で 13.4%、480K で約 9%。pattn は partials と merge を消す | する（decode カーネルと一致、と docs が主張。要確認） | なし（env のみ） | 中（sm_61 未測定。このビルドは sm_6x カーネルを持つ） |
| 5 | block score のタイル配置を 30 SM 向けに（TS_KB 32→16 で 2 CTA/SM、または TS_Q 64→128） | #1 の上にさらに 10-30%（推定） | する（タイルは結果に影響しない） | 中（カーネル 1 本のパラメータ） | 小 |
| 6 | chunk/merge の融合・起動削減 | qsa attn の merge を消す（#4 がそれ）。単体では数 % | する | 中 | 小 |
| 7 | indexer の再利用・キャッシュ | **見込みなし**: `qsa indexer` は 28.3K で 1 ms（0.0%）。層・チャンクごとに 1 回しか走っていない | - | - | - |
| 8 | `sel_batch` 256 → 64（thread 版の key 再読を 4x → 1x） | 小: thread 版は 128 FLOP/byte で compute-bound（P40 の ridge は 35）なので帯域削減は効かない | する | 小 | 小 |

## 4. 実装

`patches/0001-qsa-select-sm61-thread-scorer.diff`（`src/kernels/cuda/qsa_select.cu`、+5/−1）。

**prompt の block score を thread-per-head 版へ（sm_61 の既定）** — `scores_thread_on()`（qsa_select.cu:2060）。
既存の `block_scores_thread_kernel`（qsa_select.cu:1257、TS_Q=64 / TS_KB=32 / TS_BR=512）を、Turing
（`sm75_default`）に加えて sm_61 でも既定で選ぶ。カーネル自体は無変更。`STRATA_SCORES_THREAD=0` で
従来の warp 版に戻る。

新しいカーネルは足していない。当初は top-k の `counted` も sm_61 に広げる 2 つ目の変更を入れて
ビルドしたが、§2.5 の測定で 32K の top-k が遅くなると分かったので**パッチから外した**（測定したイメージ
`strata:fork-qsa` には入っている。128K/480K の選択には影響しない）。


ビルド: `nvidia/cuda:12.9.1-devel-ubuntu24.04`、`--cpuset-cpus 36-47`、`-j 12`、
`-DCMAKE_CUDA_ARCHITECTURES=61-real -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_BUILD_TESTS=ON`、`MODE=native`。
コンパイル 523 s、パッケージ 96 s。イメージは base と同じ Terminal を使い、engine ディレクトリだけ差し替えて
commit した（`strata:fork-qsa`、775 MB）。engine のバイナリの sha256 は base の
`f13c3858…` から `3b231433…` に変わっている。


## 5. パリティと品質

変更は**スコアと選択がビット一致**するので、同じ入力に対して同じ計算になる（KL や回答一致の測定は不要）。

### 5.1 `qsa_select_bench` のビット一致（これが本命の検査）

3 context × 256 query で、thread 版のスコアは warp 版と
`score rel diff mean 0 max 0`、`selections identical 256/256`、`cells differing 0.0000%`。
`register top-k identical to the reference 256/256` も 3 context すべてで成立（top-k は無変更）。

### 5.2 テストバイナリ（`STRATA_BUILD_TESTS=ON` でビルド、P40 上）

| テスト | 結果 |
|---|---|
| `qsa_parity --selftest` | rc=0 |
| `qsa_merge_daily_parity`（`STRATA_ATTN_DAILY=0` / `=1`） | rc=0 / rc=0 |
| `qsa_scores_parity` | rc=0 |
| `qsa_topk_active_parity`（`STRATA_TOPK_ACTIVE_ANY=1`） | rc=0 |
| `qsa_pattn_parity` | rc=0 |
| `qsa_rows_bench --rows --iters 20` | rc=0（QSAROWS の rows カーネルが multi とビット一致） |
| `qsa_slots_parity` | rc=77 = SKIP（"qsa_decode_attn_slots declined (the pre-sm_75 …)"。sm_86 以降の経路なので P40 では対象外） |
| `qsa_prompt_attn_parity` | rc=2 "qsa_prompt_attn_batch refused the pools"（この構成の KV pool では走らない。変更とは無関係） |
| `qsa_select_long_parity` | rc=2（`STRATA_SCORES_DX=1` / `STRATA_TOPK_MULTI=1` の明示が必要、と自分で言う） |
| `decode_cluster_parity` | ビルドされない（sm_90 の cluster。61-real ビルドの対象外） |

- **エンドツーエンドの回答文字列の比較は実施していない**（未確認）。ビット一致から従う、という論拠だけ。
- `qsa_prompt_attn_parity` と `qsa_select_long_parity` は変更前でも同じ理由で走らない（変更に依存しない）。

## 6. 判定

| 指標 | 結果 |
|---|---|
| **prefill 480K（同一 token 数の対）** | **−31.6%**（1,571,598 → 1,075,054 ms、480,888 → 480,892 token） |
| **qsa select 480K** | **−79.4%**（617,173 → 127,321 ms、39.3% → 11.8%） |
| prefill 480K throughput | 306.0 → 447.3 tok/s（**+46%**） |
| prefill 128K（同一 token 数の対） | **−11.5%**（311,448 → 275,606 ms） |
| prefill 128K（throughput 中央値、A-B-A） | **+12.6%**（429.2 → 483.2 tok/s） |
| qsa select 128K | **−73%**（48,517 → 13,120 ms、15.6% → 4.8%） |
| prefill 24K | −1.0% / −1.8%（5% 閾値に届かない） |
| qsa select 24K | 1,707 → 1,213 ms（1.4x。top-k と起動の固定費が残る） |
| qsa select マイクロ（491K、256 query） | **6.6x**（52.411 → 7.926 ms）、ビット一致 |
| パリティ | ビット一致（スコア・選択・top-k の register 版） |
| リスク | 小: 既存カーネルの選択だけ、ビット一致、`STRATA_SCORES_THREAD=0` で即戻せる |

**判定: 採用推奨。** 480K で prefill −31.6%（TTFT 1,572 s → 1,075 s、26 分 → 18 分）、128K で −11.5%。
24K では −1% で誤差。select は 480K で 39.3% → 11.8% になり、最大の単一項目ではなくなった
（次は qsa attn 13.0%、gdn 15.4%、gemm gate/up 12.2%）。

推定は 1 つだけ残る: 480K の残り select 127 s のうち top-k が約 44 s（マイクロベンチの比からの按分）。

## 7. 計測の注意

- **会話キャッシュ**: `--conversation-cache-mib` は既定で有効で、同一プレフィックスのプロンプトを
  再利用する。最初の 128K prose が 15.2 s で返ったのはそのため（無効な標本として捨てた）。以後は
  `bench-nc.py` / `longbench.py` でプロンプト先頭に nonce を置き、必ずフル prefill にした。
- **GPU ロックの競合**: 別タスク（`v2-block`）が 60 分ブロックを連続で回しており、この作業の GPU ブロックは
  何度も待たされた。そのため (a) 480K の A は 1 回、(b) B も 1 回しか取れていない。
- **ビルドの重なり**: ビルド（CPU 36-47）を、別タスクの GPU ブロックが動いている時間帯に実行した
  （GPU ロックが 1 時間以上連続で取れなかったため）。そのタスクの計測には CPU 競合の影響がありうる。
- **測ったイメージとパッチの差**: `strata:fork-qsa` には「top-k の counted を sm_61 へ」も入っている。
  これは 24K で +0.5 ms/呼び出しの退行なのでパッチから外した。128K/480K の選択には影響しない
  （同じ wide kernel が選ばれる）。24K の B はその分だけ不利に出ている。
- **A-B-A の 3 本目**: B の後に A をもう一度測った（arm A3）。§2.6 の表に含めてある。



### 棄却した案

- **top-k の `counted` を sm_61 へ**: 実装して測ったが、32K で register kernel（0.757 ms）に落ちて
  wide kernel（0.283 ms）より 2.7 倍遅い。24K〜90K で小さな退行になるので**棄却**（パッチから外した）。
- **`STRATA_SELECT_SIMT=1`**: thread 版より 1.6x 速いが非ビット（128K で 1/256 の query が違う選択）。
  ビット一致を保てないので既定にはしない。さらに速さが要る場合の次の候補。
- **indexer の再利用**: 実測 24 ms（480K で 0.0%）。対象外。


