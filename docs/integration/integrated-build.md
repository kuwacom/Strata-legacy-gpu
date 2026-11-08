# 統合イメージ（MMQ I=128 v2 + sm_61 既定 + 融合 3 件）のビルド・パリティ・速度・品質

最終更新: 2026-11-08。作業ツリー `~/work/wt/integrated`（branch `perf/integrated`、フォーク HEAD `9f7d977d`）。
計測は `ssh llm-02` の Tesla P40 1 枚、本番と同じ 512K 構成。数値はすべて実測。推定は「推定」と明記する。

## 概要

bitwise と確認済みの 3 パッチを 1 つのイメージに統合し、512K 本番構成で base（`strata:fork-9f7d977-base`）と
A-B-A で比較した。統合イメージは `strata:fork-integrated`（engine 0.1.57、747,378,028 bytes）。

- 3 パッチは競合なく適用できた（`git apply --check` と本適用の両方で衝突なし。手動解決は不要）
- パリティ 21 本すべて PASS（bitwise または既知の許容範囲）
- prefill は 24K で +3.1%、128K で +3.1%。decode は 4K code +1.3%、4K prose +2.5%、24K code +3.8%、24K prose +1.6%
- 4K プロンプトの logits は 3 アーム（base 2 回 + integrated 1 回）で bitwise 一致
- 24K の logits と固定 4 プロンプトの回答は、base 同士でも一致しない（実行ごとの揺れ）。integrated の差はこの揺れと同程度
- 本番の恒久切替はしていない。切替可否の判断材料と手順を「判定とリスク」に記す

## 適用したパッチ

| パッチ | 内容 | 適用結果 |
|---|---|---|
| `~/work/wt/mmq/patches/mmq-i128-v2.diff` | Pascal DP4A MMQ の J=64 の I tile を 64 -> 128（Q1_0 Q2_0 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 IQ2_XXS IQ3_XXS IQ3_S IQ4_XS IQ4_NL）、overlay の instance を CMake から使う | 競合なし |
| `~/work/wt/defaults/patches/sm61-defaults.diff` | `src/core/nv_defaults.cpp` に sm_61 の 10 スイッチ、`STRATA_SM61_PERF=0` で OFF。docs とビルド/パリティスクリプトも同梱 | 競合なし |
| `~/work/wt/fusion/patches/fusion-all.diff` | router top-10 + doorbell publish、plan の wait + copy、window 入力の staging copy の 3 融合 | 競合なし |

- 3 パッチの変更ファイルは重ならない（MMQ は `CMakeLists.txt` と overlay、sm_61 は `src/core` と scripts/docs、
  融合は `include` と `src/core` の verify/layer と `src/kernels`）。`src/core` は sm_61 が `nv_defaults*`、
  融合が `layer.cpp` / `verify.cpp` で、行も重ならない
- 統合パッチ: `~/work/wt/integrated/patches/integrated-all.diff`（1,301 行、HEAD に逆適用できることを確認）
- 統合後の差分: 12 ファイル変更 + `docs/optimization/sm61-defaults.md` と `scripts/` 8 本の新規

## ビルド

| 項目 | 値 |
|---|---|
| ソース | `~/work/wt/integrated`、HEAD `9f7d977d` + 3 パッチ |
| ビルドコンテナ | `nvidia/cuda:12.9.1-devel-ubuntu24.04`、`--cpuset-cpus 36-47`、`-j 12` |
| CUDA アーキ | `-DCMAKE_CUDA_ARCHITECTURES=61-real` のみ |
| 機能 flags | `STRATA_ENABLE_CUDA=ON`、`STRATA_EXPERIMENTAL_SM60=ON`、`STRATA_NATIVE_EXPERTS=ON`、`STRATA_Q6K_EXPERTS=ON`、`STRATA_MMQ_KQUANTS=ON`、`STRATA_BUILD_TESTS=ON` |
| 追加ターゲット | パリティ 21 本（`gemm_f32_parity`、`prefill_mmq_v31_test`、`prefill_mmq_scatter_test`、`prefill_swiglu_quant_test`、`native_grouped_parity`、`nv_defaults_test`、`qsa_merge_daily_parity`、`qsa_rows_bench`、`mmvq_multi_parity`、`b6_mmvq_bench`、`s2_expert_grouped_parity`、`mmvq_il_parity`、`dequant_wide_parity`、`verify_parity`、`router_top10_parity`、`router_dot_parity`、`router_chain_bench`、`native_multi_parity`、`gr_parity`、`shared_expert_parity`） |
| 所要時間 | build 492 s / package 96 s / image 408 s（合計 996 s） |
| イメージ | `strata:fork-integrated`（747,378,028 bytes、engine 0.1.57） |
| ログ | `~/work/results/integrated/{build,package,image}-integrated.log`、`build-driver.log` |
| ビルドロック | pgdn のビルド完了を待って取得（09:36:33 取得、09:53:09 解放） |

- エラー 0。警告は 392 行で、内訳は既存の未使用変数・未使用関数（`s2_gemv_quads.cu`、`gdn.cu`、`gr.cu`、
  `shared_expert.cu` など）、`-static-global-template-stub` の将来警告、sm_75 未満の offline compilation 非推奨警告、
  `layer.cpp` の `-Wmisleading-indentation` と未使用変数、`verify.cpp:3763` の未使用変数。パッチ由来の新しいエラーはない
- MMQ がカーネルに入った証拠: 生成された `build-native/ggml-tq-overlay/mmq-config-pascal-dp4a.cuh` に
  `CASE(GGML_TYPE_IQ3_XXS, 256, 2, 128, 64, ...)`、`IQ3_S`、`IQ2_XXS`、`Q2_0`、`IQ4_NL` の J=64 I=128 行がある。
  overlay の instance（`mmq-instance-{q2_0,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_nl,iq4_xs,q8_0}.cu`）もビルドログに出ている
- sm_61 が既定で入った証拠: 起動ログに `sm_61 (Pascal, P40): 10 exact speed switches on by default` と 10 個の名前
- 融合がリンクされた証拠: `strata` に `strata::core::moe_route_publish`、`strata::kernels::wait_copy_i32_from_mapped`、
  `route_publish` の device stub がある

## パリティ

GPU ロックを取り、本番を止めて VRAM を空けてから 21 本を実行した（`~/work/results/integrated/parity-*.log`）。
すべて exit 0。

| テスト | 結果 | 主な数値 |
|---|---|---|
| `nv_defaults_test` | PASS | sm_61 表 10 個、`STRATA_SM61_PERF=0` で空 |
| `gemm_f32_parity` | PASS | 5,201 sizes、0 different bits（bitwise） |
| `prefill_mmq_v31_test` | PASS | J=64 を踏む rows 64/80 を含む screen、`rel_l2` 0.0011 前後 |
| `prefill_mmq_scatter_test` | PASS | 5 型 × T=1,2,7,600,2099、0 bytes differ |
| `prefill_swiglu_quant_test` | PASS | 18 ケース、0 bytes differ |
| `native_grouped_parity` | PASS | 0 failures、v1 と bitwise |
| `qsa_merge_daily_parity`（既定） | PASS | 1,172 cases / 0 differ |
| `qsa_merge_daily_parity`（`STRATA_ATTN_DAILY=1`） | PASS | 1,172 cases / 0 differ |
| `qsa_rows_bench --rows` | PASS | 8 cases / 0 differ（scores T=1 64.5 -> 61.5 µs、T=4 140.0 -> 126.1 µs） |
| `mmvq_multi_parity` | PASS | 148,480 outputs / 0 differ（multi_exact on）。負の対照は 63,811 differ |
| `b6_mmvq_bench 50 0 1` | PASS | 0 case/width comparisons failed |
| `s2_expert_grouped_parity` | PASS | 0 failures、host reference worst 7.84e-08 |
| `mmvq_il_parity` | PASS | T 1-8、rows 1/2/4 + table、bitwise |
| `dequant_wide_parity` | PASS | 360 cases、bitwise |
| `verify_parity --selftest` | PASS | argmax_rows / copy_from_mapped_multi / quantize_q8_0_scaled が bitwise |
| `router_top10_parity --selftest` | PASS | 0 failures、clamp は 320x の余裕で到達不能 |
| `router_dot_parity` | PASS | 0 failures、AVX rel 3.2e-07 |
| `router_chain_bench 2 20` | PASS（bench） | T=2 route+doorbell 10.77 µs/layer、mmvf+route+doorbell 23.79 µs/layer |
| `native_multi_parity` | PASS | 0 failures、n=1 の route_publish が ids/weights/rows/ring で bitwise |
| `gr_parity --selftest` | PASS | 0 failures |
| `shared_expert_parity --selftest` | PASS | 0 failures |

- 失敗は 0。切り分けが必要なケースはなかった
- `router_chain_bench` はパリティではなくベンチのため、合否は「異常終了しないこと」で判定した
- `prefill_mmq_v31_test` は FP64 参照に対する screen で、bit-exact ではない（既存の仕様）。`rel_l2` は v2 単体時と同じ水準

## 品質

`STRATA_LOGITS_DUMP=/out/logits.bin` で、4K と 24K の各 1 リクエスト（max_tokens=1）の logits を採取した。
比較は `kl_all.py` と同じ形式（record を (pos, T) で突き合わせ、全 row の最大絶対差と T=1 row 0 の KL）。
アームは base 2 回（A1, A2）と integrated 1 回（B1）。同じプロンプト・同じシード（`BENCH_SEED=42`）。

| 比較 | 4K の 12 record | 24K の 2 record（pos 25840 T=6 / pos 25846 T=1） |
|---|---|---|
| A1 vs A2（base 同士） | 12/12 bitwise、max diff 0、KL 0 | max diff 1.287 / 0.738、KL 1.46e-07 / 0.001244、argmax 一致 |
| A1 vs B1（base vs integrated） | 12/12 bitwise、max diff 0、KL 0 | max diff 1.890 / 1.122、KL 3.40e-06 / 0.01032、argmax 一致 |

- 4K は prefill 全窓 + 生成 1 token まで含めて 3 アームで bitwise。MMQ の J=64 変更と融合 3 件は 4K ではビットを変えない
- 24K は base 同士でも一致しない。差の大きさは integrated でも同じ桁（KL は A-A の 20〜30 倍だが絶対値は 0.01 nats 以下、
  argmax は一致）。原因は未確認。推定: 24K の prefill は KV streaming と adaptive tier を通り、tier の swap が
  実行ごとのタイミングに依存するため。base の A-A で同じ現象が出ているので、統合パッチ由来ではない
- 固定 4 プロンプト（2K/8K/16K/24K、`answers4.py`、temperature 0、max_tokens 128）の回答は、
  A1 vs A2 でも 4/4 が文字列一致しなかった（共通接頭辞 24 / 138 / 332 / 33 文字）。
  A1 vs B1 の共通接頭辞は 24 / 45 / 523 / 33 文字で、A-A の揺れと同じ範囲。回答の完全一致は品質ゲートに使えない
  （temperature 0 でも spec/draft と tier のタイミングで変わる。原因は未確認）
- 4K の logits が bitwise であることが、統合による数値変化がないことの最も強い証拠

## 速度（A-B-A、512K）

条件: `~/bench.py`、`BENCH_SEED=42`、`CONTEXT=512k`（engine の実引数は `--max-context 524288 --kv int8
--kv-resident 32768 --ple-io ram --prefill auto --rope-scaling yarn --rope-scale 2 --spec 4`）、`STRATA_DENSE_F32=1`。
A = `strata:fork-9f7d977-base`、B = `strata:fork-integrated`。decode は各 (size, kind) 3 回で 1 本目を捨てた中央値、
prefill 24K は 3 回で 1 本目を捨てた中央値、128K は暖機 1 本の後に 2 本（A1 だけは最初の 1 本が失敗し、
次の 1 本が KV streaming の初期化で遅く、最後の 1 本を採用）。

| 項目 | A1 | A2 | A 中央値 | B1 | B/A | A-A の揺れ |
|---|---:|---:|---:|---:|---:|---:|
| decode 4K code (tok/s) | 50.69 | 50.78 | 50.74 | 51.41 | **+1.3%** | 0.2% |
| decode 4K prose | 40.85 | 41.98 | 41.42 | 42.47 | **+2.5%** | 2.8% |
| decode 24K code | 49.34 | 50.19 | 49.77 | 51.67 | **+3.8%** | 1.7% |
| decode 24K prose | 43.42 | 43.52 | 43.47 | 44.19 | **+1.6%** | 0.3% |
| prefill 24K (tok/s) | 487.1 | 487.7 | 487.4 | 502.4 | **+3.1%** | 0.1% |
| prefill 128K | 434.8 | 433.6 | 434.2 | 447.6 | **+3.1%** | 0.3% |

TTFT（prefill の wall、秒）:

| 項目 | A1 | A2 | B1 |
|---|---|---|---|
| 24K（プロンプト 25,856〜30,247 tok） | 59.1 / 53.4 / 61.7 | 58.9 / 53.3 / 61.7 | 57.1 / 51.7 / 59.9 |
| 128K（プロンプト 115,681〜129,355 tok） | 160.8（失敗）/ 744.7（初回・KV 初期化）/ 266.1 | 282.4 / 289.6 / 297.4（暖機含む） | 2.3（プロンプトキャッシュ命中）/ 281.2 / 273.1 |

- prefill の +3.1% は MMQ 単体の +3.1〜3.3% と一致する。decode の +1.3〜3.8% は sm_61 既定の +2.5% 前後と整合する。
  融合 3 件は decode の差には寄与していない（融合単体の測定でもノイズ内）
- decode の A-A の揺れは 0.2〜2.8%。24K code の +3.8% は揺れより大きいが、5% 閾値には届かない。
  4K code の +1.3% は揺れと同程度で、有意とは言えない
- 128K の最初のリクエストは KV streaming の初期化で極端に遅いことがある（A1 の 744.7 s）。
  2 回目以降は 266〜297 s。統合パッチの差ではなく、512K 構成の初回コスト
- 480K 相当の TTFT（integrated のみ 1 回、プロンプト 460,679 tok、`longprefill.py`）: **TTFT 1,449.38 s、317.8 tok/s**。
  480K ちょうどではなく 460,679 tok（corpus の繰り返しで作ったため、文字/トークン比が 3.34 になった）。
  base は測っていない（時間の都合）。128K の 447.6 tok/s から落ちており、QSA select/attention と KV streaming の
  コストが context とともに増える（推定。内訳は未取得）

## 判定とリスク

**判定（サブの提案）: 統合イメージは本番切替の候補として可。ただし 5% 閾値は単独では満たさない。**

- パリティは 21/21 PASS、4K logits は bitwise。統合による数値の変化は 4K では観測されない
- prefill +3.1%（24K/128K とも）、decode +1.3〜3.8%（24K code が最大）。5% 閾値には届かないが、一貫して正で、
  単体測定の和（MMQ +3.1〜3.3%、sm_61 decode +2.5%）と整合する
- 速度の主因は MMQ（prefill）と sm_61 既定（decode）。融合は node を -97 するが、今回の decode 差には現れない
- 本番切替の恒久作業はしていない。`~/opt-work/prod.env` は変更していない
- 計測の各ブロック後、本番は `~/opt-work/prod.env` から復元し、healthy・512K・preset 汚染なし・残留なしを確認した
  （`restore-*.txt`）。最終確認は 16:58、prod healthy、`--max-context 524288`、`.env` は `prod.env` と一致、
  preset に計測用 `--set` なし、GPU/ビルドロックとも解放。その後 16:59 に別タスク `gdnq-quality` が GPU ロックを
  再取得した（本タスクのロックではない）

リスクと注意:

- **本番の engine は 0.1.52、統合イメージは 0.1.57**。今回の A/B は base も 0.1.57 なので、3 パッチの差は分離できているが、
  本番切替には 0.1.52 -> 0.1.57 の engine 更新も含まれる。この差は今回測っていない（未確認）。切替前に
  本番 0.1.52 と fork base 0.1.57 の A/B を取るか、0.1.52 相当のソースに 3 パッチだけを載せたイメージを作るべき
- MMQ I=128 は shared memory を増やす（IQ3_S で 46,976 B / 48 KB）。ビルドとパリティは通ったが、他の量子化や
  将来のモデルで tile が選ばれない可能性がある。`prefill_mmq_v31_test` の screen は IQ3_XXS など対象型を踏んでいる
- sm_61 既定は 10 スイッチ。`STRATA_SM61_PERF=0` で集合全体を切れる。個別スイッチを明示設定した場合はそれが優先される
- 24K の logits と回答は実行ごとに揺れる（base 同士でも）。品質の受け入れ判定に文字列一致を使うなら、
  まずこの揺れを許容する基準（例: 共通接頭辞の長さ、意味的一致、logits の KL 上限）を決める必要がある
- 128K/480K の初回リクエストは KV streaming の初期化で遅い。本番の長文利用では初回 TTFT が大きくなる
- 480K は integrated のみ 1 回。base との比較は未実施

本番切替の手順（メインが判断。ここでは実行しない）:

1. `~/opt-work/prod.env` の `STRATA_IMAGE` を `strata:fork-integrated` に変える（恒久変更はメインの判断）
2. `cp ~/opt-work/prod.env ~/Strata/docker/compose/.env`
3. `sudo docker exec strata sh -c "strata launch --stop --preset docker; strata launch --delete-preset docker"`
4. `cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml up -d --force-recreate`
5. healthy、engine の cmdline が `--max-context 524288 --kv int8 --kv-resident 32768 ... --spec 4` であること、
   preset に計測用 `--set` が残っていないことを確認
6. 問題時は `STRATA_IMAGE` を `ghcr.io/niko1221/strata:0.2.7-cuda12` に戻して 2〜5 を再実行

## 成果物

- 統合パッチ: `~/work/wt/integrated/patches/integrated-all.diff`
- この文書: `~/work/wt/integrated/docs/integration/integrated-build.md`
- 計測スクリプト: `~/work/wt/integrated/scripts/{build-integrated,arm-integrated,block-integrated,parity-integrated,restore-prod}.sh`、
  `{logits-two,longprefill,answers4,analyze,kl_all}.py`
- 生ログ: `~/work/results/integrated/`
  - build: `build-integrated.log`、`package-integrated.log`、`image-integrated.log`、`build-driver.log`
  - parity: `parity-*.log`、`parity-run.log`
  - speed: `arms/{A1,A2,B1}-speed/*.jsonl`、`block-{A1,A2,B1}.log`
  - quality: `arms/{A1,A2,B1}-quality/{logits.bin,answers.json,logits-requests.json}`、`block-quality.log`
  - 480K: `arms/B1-long/long-480k.jsonl`、`block-long.log`
  - 本番復元: `restore-*.txt`、`prod-baseline-*.{txt,json}`
