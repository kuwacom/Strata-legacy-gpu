# kernel fusion: verify window の node 削減（P40 / sm_61）

作業ツリー `~/work/wt/fusion`（branch `perf/fusion`、fork HEAD `9f7d977d`）。計測は 2026-11-07 07:20-08:21 UTC、
`ssh llm-02` の Tesla P40 1 枚。イメージは base `strata:fork-9f7d977-base`（engine 0.1.57）と
融合版 `strata:fork-fusion`（同じ 0.1.57、下記のパッチ込み）。本番と同じ 512K 構成
（`--max-context 524288 --kv int8 --kv-resident 32768 --ple-io ram --rope-scaling yarn --rope-scale 2 --spec 4 --spec-min-p 0.5`、
`STRATA_DENSE_F32=1`）。

## 1. 結論

- T=1 window の node 数は **2069 -> 1972（-97、-4.7%）**。T=2..6 は **-49（-2.2〜-2.3%）**。
- 実装した融合は 3 つ: (a) router top-10 と CPU pool の doorbell publish を 1 launch に、
  (b) plan の wait と copy を 1 launch に、(c) window 入力の staging copy 3 発を 1 発に。
- パリティは **bitwise**。カーネル単体のパリティに加え、bench なしの同一ワークロードで
  logits dump 227 record が **227/227 bitwise 一致**（max abs diff 0、KL 0）、固定 4 プロンプトの回答も一致。
- decode 速度は **差なし（ノイズ内）**。A-B-A の中央値は基準比 **-2.6%**、base 自身の揺れが **+4.8%** で、
  5% 閾値に届かない。24K の decode timing は base 24.52 ms/token、融合 24.58 ms/token（+0.2%）、
  closing base 23.17 ms/token（base の揺れ -5.5%）。ボトルネックは GPU-reach wait（CPU expert pool）で、
  graph の node 起動ではない。

## 2. node 名とソースの対応（T=1、融合後 1972 node）

`STRATA_VERIFY_NODES=1` の出力を `c++filt` で demangle し、count の多い順に並べたもの。
count は融合後（`~/work/results/fusion/arms/fusion/engine.log`）。「base」列は融合前の同じ node の count
（base の log は 60 文字で切られていたため、上位 40 件のみ判明しているものだけ記す）。

| node（demangle） | count | base | 定義 file:line | 呼び出し site |
|---|---:|---:|---|---|
| `native_quantize_q8_1_kernel` | 145 | 145 | `src/kernels/cuda/native_mmvq.cu:159` | `src/core/layer.cpp:276,376`（`gemv_quantized`） |
| `bf16_f32_mmvf_kernel<256>` | 122 | 122 | `src/kernels/cuda/native_bf16.cu:62` | `src/core/verify.cpp:2227`（router）、`:1531,1542`（PLE）、`:1868,1887,1918`（indexer） |
| `gr_down_staged_kernel<1,true>` | 97 | 97 | `src/kernels/cuda/fused_gr.cu:1313` | `src/core/verify.cpp:1628`（`gr_read_group`）、`:2591,2613,2630`（head） |
| `gr_norm_split_kernel` | 97 | 97 | `src/kernels/cuda/fused_gr.cu:1163` | 同上 |
| `gr_up_multi_kernel<1,true,false>` | 97 | 97 | `src/kernels/cuda/fused_gr.cu:394` | 同上 |
| `wait_flag_ge_kernel` | **96** | **144** | `src/kernels/cuda/verify_kernels.cu:1313` | `src/core/verify.cpp:2432,2442,2464`（plan / PCIe / CPU rows） |
| `native_q6_k_mmvq_kernel<false>` | 71 | 71 | `src/kernels/cuda/native_mmvq.cu:737` | `gemv_quantized` |
| `swiglu_q8_1_entries_kernel` | 58 | 58 | `src/kernels/cuda/iq_kernels.cu:2851` | `native_expert_grouped`（`iq_kernels.cu:6271`） |
| `mmvq_multi_kernel<21,1,false,true>` | 55 | 55 | `src/kernels/cuda/iq_kernels.cu:1892` | `native_expert_grouped` |
| `add_hits_kernel` | 48 | 48 | `src/kernels/cuda/s2_expert_grouped.cu:736` | `src/core/verify.cpp:2498`（combine hits） |
| `wait_copy_i32_kernel` | **48** | 0 | `src/kernels/cuda/verify_kernels.cu:1641` | `src/core/verify.cpp:2427`（新既定: sm_61） |
| `rebase_ptrs_kernel` | 48 | 48 | `src/kernels/cuda/verify_kernels.cu:660` | `src/core/verify.cpp:2446`（`pcie_mode==2`） |
| `fetch_blobs_kernel` | 48 | 48 | `src/kernels/cuda/verify_kernels.cu:646` | `src/core/verify.cpp:2445`（`pcie_mode==2`） |
| `route_publish` | **48** | 0 | `src/kernels/cuda/native_router.cu:155` | `src/core/verify.cpp:2251`（n==1 の router） |
| `sigmoid_scale_rows_vec4_kernel` | 48 | 48 | `src/kernels/cuda/shared_expert.cu:182` | `shared_expert_multi`（`shared_expert.cu:216`）、`verify.cpp:2334` |
| `native_swiglu_quantize_q8_1_kernel` | 48 | 48 | `src/kernels/cuda/native_mmvq.cu:176` | `native_mmvq.cu:3109`、shared expert |
| `quantize_q8_1_kernel` | 48 | 48 | `src/kernels/cuda/iq_kernels.cu:2838` | `native_expert_grouped` |
| `combine_k10_vec4` | 48 | 48 | `src/kernels/cuda/native_moe.cu:67` | `src/core/verify.cpp:2493,2498,2504` |
| `copy_rows_from_mapped_kernel` | 48 | 48 | `src/kernels/cuda/elementwise.cu:238` | `src/core/verify.cpp:2473` |
| `native_q4_k_mmvq_kernel<false>` | 45 | 45 | `src/kernels/cuda/native_mmvq.cu:602` | `gemv_quantized` |
| `native_small_mmvq_kernel<IQ4NLBlock,4,true>` | 40 | 40 | `src/kernels/cuda/native_mmvq.cu:849` | `gemv_quantized` |
| `gdn_step_norm_multi_kernel<false,false>` | 36 | 36 | `src/kernels/cuda/verify_kernels.cu:318` | `src/core/verify.cpp:1802` |
| `gdn_conv_l2_multi_kernel` | 36 | 36 | `src/kernels/cuda/verify_kernels.cu:75` | `src/core/verify.cpp:1747` |
| `gdn_ab_multi_kernel<1,true>` | 36 | 36 | `src/kernels/cuda/verify_kernels.cu:177` | `src/core/verify.cpp:1755` |
| `norm_rope_kernel<false>` | 36 | 36 | `src/kernels/cuda/native_rope.cu:71` | `src/core/verify.cpp:1835,1880,1954,1966` |
| `native_iq4_xs_mmvq_kernel<true>` | 31 | 31 | `src/kernels/cuda/native_mmvq.cu:507` | `gemv_quantized` |
| `native_down_multi_kernel<42,true,false>` | 30 | 30 | `src/kernels/cuda/iq_kernels.cu:2239` | `native_expert_grouped` |
| `native_down_lds_kernel<42>` | 30 | 30 | `src/kernels/cuda/iq_kernels.cu:2205` | `native_expert_grouped` |
| `native_mmvq_il_kernel<IlIQ4XS,1,4,1>` | 26 | 26 | `src/kernels/cuda/native_mmvq.cu:2161` | `gemv_quantized`（il 経路） |
| `native_gu_multi_kernel<17,true,true>` | 20 | 20 | `src/kernels/cuda/iq_kernels.cu:1974` | `native_expert_grouped` |
| `copy_from_mapped_kernel` | 14 | 14 | `src/kernels/cuda/elementwise.cu:229` | `src/core/verify.cpp`（PLE / hist） |
| `native_gu_multi_kernel<21,false,true>` | 13 | 13 | `src/kernels/cuda/iq_kernels.cu:1974` | `native_expert_grouped` |
| `native_gu_fused_kernel<21>` | 13 | 13 | `src/kernels/cuda/iq_kernels.cu:6063` | `native_expert_grouped` |
| `append<false>` | 12 | 12 | `src/kernels/cuda/native_qsa_indexer.cu:120` | `src/core/verify.cpp:1243`（indexer append） |
| `attn_chunk_kernel_pre75<1,2,false>` | 12 | 12 | `src/kernels/cuda/qsa_decode_attn.cu:545` | `src/core/verify.cpp:1280` |
| `attn_merge_kernel` | 12 | 12 | `src/kernels/cuda/qsa_decode_attn.cu:699` | 同上 |
| `block_scores_multi_kernel` | 12 | 12 | `src/kernels/cuda/qsa_select.cu:1033` | `src/core/verify.cpp:1270` |
| `block_topk_wide_kernel<false>` | 12 | 12 | `src/kernels/cuda/qsa_select.cu:858` | `src/core/verify.cpp:1271` |
| `gate` | 12 | 12 | `src/kernels/cuda/native_qsa.cu:94` | `src/core/verify.cpp:1312` |
| `resolve_kernel` | 12 | 12 | `src/kernels/cuda/kv_stream.cu:214` | `src/core/verify.cpp:1276` |
| `copy_kernel` | 12 | 12 | `src/kernels/cuda/kv_stream.cu:232` | 同上 |
| `kv_append_q8_kernel<true>` | 12 | 12 | `src/kernels/cuda/kv_q8.cu:187` | `src/core/verify.cpp:1241` |
| `copy_from_mapped_multi_kernel` | **1** | 0 | `src/kernels/cuda/elementwise.cu:403` | `src/core/verify.cpp:1447`（新規） |
| `copy_i32_from_mapped_kernel` | **1** | **51** | `src/kernels/cuda/elementwise.cu:393` | `src/core/verify.cpp:1432`（tok の 1 発だけ残る） |
| `doorbell_publish_res_kernel` | **0** | **48** | `src/kernels/cuda/elementwise.cu:329` | `route_publish` に吸収 |
| `route`（単一 token router） | **0** | **48** | `src/kernels/cuda/native_router.cu:45` | `route_publish` に吸収 |
| `memcpy` | 1 | 1 | driver node | `src/core/layer.cpp:1258`（QSA q/gate split） |

## 3. 融合候補（削減 node 数、bitwise、実装コスト、リスク）

「削減 node」は T=1 window の実測または推定。推定は「推定」と明記した。実装行数はパッチの行数（新規・変更）。

| # | 候補 | 対象 file:line | 削減 node | bitwise | 既存の融合 / opt-in | 実装行数 | リスク |
|---|---|---|---:|---|---|---|---|
| 1 | router top-10 + doorbell publish を 1 launch に（**実装**） | `native_router.cu:155`（`route_publish`）、`layer.cpp:442`、`verify.cpp:2243` | **-48（実測）** | はい（ids/weights/rows/ring のパリティ） | なし | 約 90 行 | 低: publish の順序が router 直後へ移るだけ |
| 2 | plan の wait + copy を 1 launch に（**実装**） | `verify_kernels.cu:1641`（`wait_copy_i32`）、`verify.cpp:455`（sm_61 既定） | **-48（実測）** | はい（同じ読み） | 既存カーネル + `STRATA_FUSE_WAIT`（Turing 既定） | 約 15 行 | 低: 1 block の copy になる |
| 3 | window 入力の staging copy 3 発を 1 発に（**実装**） | `elementwise.cu:403`、`verify.cpp:1441` | **-1（実測）** | はい（`verify_parity` が multi と個別を bitwise 比較） | 既存 `copy_from_mapped_multi` | 約 10 行 | 低 |
| 4 | shared expert を routed expert の grouped launch に合流（研究 #5） | `shared_expert.cu:216`、`iq_kernels.cu:6271`、`verify.cpp:2334` | -48〜-96（推定） | はい（group 内の和の順序を保てば） | なし | 200〜400 行 | 中-高: expert 単位 group の再構成。速度 +1〜2.4%（研究の推定） |
| 5 | q8_1 量子化を norm/combine の epilogue へ（C4） | `verify.cpp:180,292,2313-2331`、`fused_gr.cu:246` | -96（既存資料の推定） | はい（コメント "the same bytes"） | 既存 opt-in `STRATA_QFUSE` / `STRATA_VERIFY_QDEDUP`（既定 off） | 0 行（env） | 中: 丸め順。env-sweep の P40 実測は +0.5%（ノイズ） |
| 6 | GR read の norm + down を 1 kernel に | `fused_gr.cu:281,333`、`verify.cpp:1628` | -97（推定） | はい（同じ和の順序なら） | なし（persistent 版は別 opt-in） | 約 150 行 | 中: tile / shared memory の再設計 |
| 7 | PCIe staging の no-op node を落とす | `verify.cpp:2442,2445,2446,2452`、`verify_kernels.cu:646,660` | -96（`--pcie-mode dma` で fetch/rebase）〜-240（推定、grouped/wait も） | はい（PCIe share が空なら） | 既存 `--pcie-mode`（既定 auto=2） | 0 行（設定） | 中: PCIe share が非空のときの挙動 |
| 8 | combine と CPU rows の copy を 1 launch に | `native_moe.cu:67`、`elementwise.cu:238`、`verify.cpp:2473,2493` | -48（推定） | はい | なし | 約 50 行 | 中: mapped host 読みの帯域 |
| 9 | QSA の norm + rope（C5） | `native_rope.cu:220`、`verify.cpp:1835,1880` | 0（本番は既に融合） / -36（canonical のみ） | 未確認（decode 用関数） | 既存 `native_qsa_rms_norm_rope` | 0 行 | 低（本番は既に有効） |
| 10 | QSA の rope + kv_append | `kv_q8.cu:187`、`native_rope.cu:220`、`verify.cpp:1241` | -12（推定） | 未確認 | なし | 約 50 行 | 中: append の host mirror と順序 |

## 4. 実装した融合

### 4.1 router top-10 + doorbell publish（`patches/router-publish.diff`）

- `route_publish`（`src/kernels/cuda/native_router.cu:155`）: warp 0 が `route` と同じ top-10 を計算し、
  同じ block が x/ids を mapped host memory へ書き、ring を 1 回進める。`doorbell_publish_res` の
  `d_res[id] < 0` による「miss のときだけ x をコピー」も同じ。
- `moe_route_publish`（`src/core/layer.cpp:442`）: `moe_route` と同じ GEMV + top-10 の後、
  publish を同じ launch で行う。
- `verify.cpp:2243` の `fuse_pub` は **n == 1** かつ native 512x10、`route_prior` / `route_resident` が
  無効、`remote_opt_` / `always_publish_` / `ar_on()` でないときだけ。n>1 の batched router は従来どおり。
- 効果: T=1 window で `route` 48 + `doorbell_publish_res` 48 -> `route_publish` 48（-48 node）。
  publish が router 直後へ移るため、CPU pool が resident_plan / shared expert と重なりやすくなる（推定）。

### 4.2 plan の wait + copy（`patches/verify-fusion.diff`）

- `fuse_wait_on()` の CUDA 既定に sm_61 を追加（`verify.cpp:455`）。`STRATA_FUSE_WAIT=2` と同じ経路で、
  `wait_flag_ge` + `copy_i32_from_mapped` を `wait_copy_i32_from_mapped` 1 発にする。
- 効果: 48 層 × 1 node = **-48 node**（`wait_flag_ge` 144 -> 96、plan の `copy_i32` 48 -> 0）。
- rows 側（`STRATA_FUSE_WAIT=1`）は opt-in のまま。P40 では未測定（時間切れ）。

### 4.3 window 入力の staging copy（`patches/verify-fusion.diff`）

- `verify.cpp:1441`: step と pos の 2 発を `copy_from_mapped_multi` 1 発に。tok は stream 分岐のため
  別のまま（`copy_i32` 1 発が残る）。
- 効果: **-1 node**（`copy_i32_from_mapped` 51 -> 1、`copy_from_mapped_multi` 1 発が加わる）。
- tok も `!br` のとき multi に含めればもう -1 node にできる（未実装）。

## 5. パリティ

### 5.1 カーネル単体（`~/work/results/fusion/parity.log`）

- `native_multi_parity`: `router+publish n=1` が random / exact ties / NaN の 3 分布で
  **bitwise（ids, weights, rows, ring）**。multi router も 19 case × 3 分布で bitwise。
- `router_top10_parity`: 0 failures（host reference と ids exact、weights 最悪 1.4e-7）。
- `verify_parity`: PASS。`copy_from_mapped_multi` が個別 copy と bitwise（multi=1 と fallback=0、
  aligned / 1025 words / 4B-aligned）。
- `shared_expert_parity`: 0 failures。`gr_parity`: 0 failures（GR read + q8_1、T=1..8、graph replay 含む）。
- `quantize_act_parity`: 3 failures。**1e-7 の極小 Q8_K** の丸めで、今回の変更とは無関係（base でも同じ）。

### 5.2 エンドツーエンド（`STRATA_LOGITS_DUMP`、4K プロンプト、max_tokens=1）

bench なしの 2 アーム（base3 と fusion2）は同じ 227 record を作り、**227/227 が bitwise 一致**
（max abs diff 0.000000e+00、T=1 の KL 0.000000e+00）。固定 4 プロンプトの回答も 4/4 一致。

参考: bench ありの base/fusion は record 数が 588/606 で、window の切り方が run ごとに違う。
base vs base3（同じ base イメージ）でも 138 record 中 41 しか一致せず、max abs diff 2.9e+01、
KL 1.9e+01 になる。したがって **base vs fusion の logits 差は融合由来ではなく、run 間の
window 切り方・expert cache 状態の揺れ**と判断する。回答も base 自身が prompt 2 で揺れる
（base vs base2 で DIFF、base vs base3 で prompt 2/3 が DIFF）。fusion vs fusion2 も prompt 3 で揺れる。
bench なしの base3 vs fusion2 が一致していることが、融合の bitwise 性の直接証拠。

## 6. node 数の before / after（`STRATA_VERIFY_NODES=1`）

| T | base | fusion | 差 | % |
|---:|---:|---:|---:|---:|
| 1 | 2069 | 1972 | -97 | -4.69% |
| 2 | 2130 | 2081 | -49 | -2.30% |
| 3 | 2177 | 2128 | -49 | -2.25% |
| 4 | 2186 | 2137 | -49 | -2.24% |
| 5 | 2197 | 2148 | -49 | -2.23% |
| 6 | 2205 | 2156 | -49 | -2.22% |

- T=1 だけ -97 なのは router 融合が n==1 限定のため。T>=2 は wait 融合（-48）と staging copy（-1）。
- T=5 は今回 capture された（env-sweep では未 capture）。base 2197 / fusion 2148。
- node 種別はすべて kernel（memcpy 1 件を除く）。graph は 1 window = 1 launch のまま。

## 7. 速度（A-B-A、BENCH_SEED=42、spec=4、1 本目除外）

### 7.1 bench（decode 4K/24K code/prose、各 3 回、rep 1/2 の中央値）

同一 rep は同一プロンプト。基準は前後 2 つの base の平均。

| サイズ | 種類 | base | base2 | 基準 | fusion | B-基準%（rep 1/2 の中央値） | base 揺れ%（rep 1/2 の中央値） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K | code | 41.53 | 44.81 | 43.17 | 41.64 | -3.60 | +7.86 |
| 4K | prose | 38.01 | 40.02 | 39.02 | 38.94 | -0.09 | +5.22 |
| 24K | code | 44.75 | 45.91 | 45.33 | 43.58 | -3.89 | +2.57 |
| 24K | prose | 39.78 | 42.27 | 41.03 | 40.19 | -2.04 | +6.29 |

- 8 つの (サイズ, 種類, rep) セルの B-基準% の中央値は **-2.61%**、base 揺れの中央値は **+4.75%**。
  **5% 閾値に届かない**。
- prefill 24K（max_tokens=1、1 回）: base 486.6、fusion 486.1、base2 486.1 tok/s（-0.1%）。

### 7.2 decode timing（24K、`STRATA_DECODE_TIMING=1`）

| アーム | ms/token（中央値） | verify | GPU-reach wait |
|---|---:|---:|---:|
| base | 24.52 | 22.01 | 18.01 |
| fusion | 24.58 | 22.08 | 17.87 |
| base2 | 23.17 | 20.89 | 18.44 |

- fusion は base 比 **+0.2%**（ms/token）。closing base は -5.5% で、base の揺れの方が大きい。
- GPU-reach wait が約 18 ms/token（verify の約 8 割）を占め、node 起動削減の効果はこの下に埋もれる。
- 推定: 97 node × 200 ns/node = 約 19 µs/window。24K の 1 token あたり 24.5 ms に対して 0.08% 程度で、
  5% 閾値には届かない。node 数は減ったが、律速は CPU expert pool の待ち。

## 8. 判定

| 項目 | 判定 | 根拠 |
|---|---|---|
| node 削減 | **採用可（実測 -97 / T=1）** | 2069 -> 1972。T>=2 も -49。パリティ bitwise |
| decode 速度 | **差なし（ノイズ内）** | A-B-A 中央値 -2.6%、base 揺れ +4.8%、5% 閾値未満。timing は +0.2% |
| パリティ / 品質 | **bitwise** | カーネル単体 bitwise、bench なし 227/227 record bitwise、回答 4/4 一致 |
| リスク | 低 | router 融合は n==1 の publish 順序のみ。wait 融合は既存カーネル。staging copy は既存 multi |

**採用推奨 / 差なし / 悪化** のうち、速度は **差なし**、node 削減とパリティは **採用推奨**。
速度目的では採用理由にならないが、graph の node 数を 4.7% 減らしても bitwise を保てることは確認できた。
次の一手は CPU expert pool 側（GPU-reach wait）か、shared expert 合流（研究 #5）のような
より大きな融合で、node 起動ではなく実処理を削る方向。

## 9. 成果物

- パッチ: `~/work/wt/fusion/patches/`
  - `fusion-all.diff`（全変更）
  - `router-publish.diff`（native_router / layer）
  - `verify-fusion.diff`（verify.cpp: wait / staging copy / node dump）
  - `parity-test.diff`（native_multi_parity の router+publish テスト）
- docs: このファイル（`~/work/wt/fusion/docs/optimization/kernel-fusion.md`）
- 生ログ: `~/work/results/fusion/`
  - `arms/{base,fusion,base2,fusion2,base3}/`（`engine.log`、`bench-*.jsonl`、`answers.json`、`logits.bin`）
  - `parity.log`、`restore.txt`、`block-driver.log`、`block2-driver.log`
  - `scripts/`（`build-fusion.sh`、`fusion-arm.sh`、`fusion-restore.sh`、`fusion-block.sh`、`fusion-block2.sh`、
    `nodes.py`、`kl_all.py`、`summarize.py`、`paired.py`）
- イメージ: `strata:fork-fusion`（engine 0.1.57、sm_61、BUILD.json "fork fusion"）

## 10. 再現

```sh
# ビルド（CPU 36-47、-j 12、ビルドロック）
~/work/results/fusion/scripts/build-fusion.sh fusion strata:fork-fusion

# 1 ブロック（GPU ロック、本番停止、パリティ、base/fusion/base2、復元）
~/work/results/fusion/scripts/fusion-block.sh

# パリティだけ
sudo docker run --rm --gpus all -v ~/work/wt/fusion:/work/strata \
  nvidia/cuda:12.9.1-devel-ubuntu24.04 sh -c 'cd /work/strata/build-native && ./native_multi_parity && ./verify_parity --selftest'

# node 内訳
python3 ~/work/results/fusion/scripts/nodes.py ~/work/results/fusion/arms/fusion/engine.log 1

# A-B-A 集計
python3 ~/work/results/fusion/scripts/paired.py \
  base=~/work/results/fusion/arms/base/bench-*.jsonl \
  base2=~/work/results/fusion/arms/base2/bench-*.jsonl \
  fusion=~/work/results/fusion/arms/fusion/bench-*.jsonl
```

## 11. 注意（推定・未確認）

- 速度の差は 3 回 × rep 1/2 の中央値で、base 自身が 4.8% 動く。5% 閾値の判定は「ノイズ内」で、
  有意差ではない。サンプル数を増やすなら 5 回以上 × 交互配置が要る。
- `STRATA_FUSE_WAIT=1`（rows の wait+copy）は今回測定していない。node はさらに -48 になる見込み（推定）。
- shared expert 合流（#4）と GR norm+down（#6）は実装していない。削減 node と速度は推定。
- PCIe staging の no-op node（#7）は `--pcie-mode dma` で fetch/rebase の 96 node が消える見込みだが、
  PCIe share が非空のときの挙動は未確認。
- 本番復元は 08:21:23 に healthy・512K・`STRATA_DENSE_F32=1` のみ・`.env` 一致・preset に計測用 `--set` なしを
  確認して GPU ロックを解放した。その後 08:32 に別タスク `pknobs-main` が GPU ロックを取り、
  `strata-fork`（dflash2）を起動して本番コンテナを停止している。現在の本番停止はこの別タスクによるもので、
  本タスクの残留ではない。preset ファイルはボリューム上で prod の args と `{"STRATA_DENSE_F32":"1"}` のみを確認。
- `fusion-restore.sh` の preset 表示は `/data/config/strata/launch-presets/docker.json` を見ており、
  実際の場所（`/data/state/strata/launch/docker.json`）ではないため JSON エラーになった。
  preset の内容はボリュームから直接確認した（上記）。
- `quantize_act_parity` の 3 failures は 1e-7 の極小 Q8_K で、今回の変更前からある挙動（base でも同じ）。
- API キーは表示していない（`grep ^API_KEY=` で変数に入れるだけ）。
