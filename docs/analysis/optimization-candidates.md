# optimization-candidates: 起動削減・融合・帯域活用の候補

作業ツリー: `~/work/wt/codemap`（branch `docs/codemap`）
前提: `decode-kernel-map.md` / `prefill-kernel-map.md` の地図。本番モデルは `qwen4exp` で、
`src/qwen35/q36_engine.cu` は本番経路ではない（§0 の訂正を参照）。

**この表の「期待幅」はすべて推定**。実測はしていない（GPU 不使用の制約）。数値の根拠は各行の「根拠」に書いた。
パリティは「ビット一致」と「数値一致（許容誤差）」を区別して書く。

> **2026-11-07 更新（`c1-native-graph.md` の調査より）**: C1 は「不能/不要」と判明（本番は既に 1 window = 1 graph）。
> C2/C4/C9 は本番では既存の opt-in スイッチ（既定 off）、C3/C5 は本番で既に有効。canonical 経路にのみ未実装が残る。
> 表と各節をこの事実に合わせて訂正した。

## 1. 優先度順の候補表

| # | 優先 | 候補 | 対象 file:line | 期待幅（推定） | リスク | 検証方法 |
|---|---|---|---|---|---|---|
| C1 | **不能/不要** | native pack で単一トークン graph を使う（前回案: 96 graph launch -> 1） | `src/program/generate.cpp:9000, 19830, 9518-9522`、`src/core/verify.cpp:2748, 3369` | 本番は既に 1 window = 1 graph launch。`session_loop` を有効化すると 1 -> 96 の退行 | 高: native expert の実装が verifier 側にしかない | 詳細は `c1-native-graph.md`。判断は「不能（現行設計では不要）」 |
| C1b | P1 | 代替: verifier の host 項（wait/pool/host/commit）を削る | `src/core/verify.cpp:2388-2400`、`generate.cpp:20810` | `--stats` の "verify window" 行で測る。効果は未測定 | 中: graph の node 構成変更 | `verify_parity.cpp`, `batch_rows_parity.cpp`, `resident_plan_parity.cpp` |
| C2 | P1 | expert hit 判定をホストループから device（`moe_hit_select`）へ | `src/core/expert_source.cpp:5317-5360`、`src/core/session.cpp:868-873`、本番は `verify.cpp:1292-1320`（`STRATA_VERIFY_DEVICE_PLAN`） | 本番は既存 opt-in（既定 off、コード内で「neutral」と測定済み）。まず env A/B | 中: residency 表の更新タイミング | `resident_plan_parity.cpp`, `route_resident_parity.cu`, `s2_expert_grouped_parity.cpp` |
| C3 | P2 | shared expert の gate/up を pair に（本番は既に `STRATA_LFUSE_PAIR` 既定 on） | `src/kernels/cuda/shared_expert.cu:225-231`（multi = 本番）、単一経路 `:436-445`（canonical のみ） | canonical の単一トークン経路だけ 48 起動/token 削減 | 低: コメントが「bitwise the two calls」 | `shared_expert_parity.cpp` |
| C4 | P1 | q8_1 量子化を norm / combine の epilogue に融合（本番は既存 opt-in `STRATA_QFUSE` / `STRATA_VERIFY_QDEDUP`） | `src/core/layer.cpp:276, 376`（canonical）、本番は `src/core/verify.cpp:180, 292, 1601, 2176, 2313-2331` | env で A/B。canonical 経路の 96 起動/token は未実装のまま | 中: 量子化の丸め順が変わるとビット不一致 | `quantize_act_parity.cpp`, `elementwise_parity.cpp`, `verify_parity.cpp` |
| C5 | P3 | QSA の norm + rope を融合（本番は既に `native_qsa_rms_norm_rope`） | `src/core/layer.cpp:1147-1149`（canonical）、本番は `src/core/verify.cpp:1815, 1860, 1934, 1946` | canonical の単一トークン経路だけ 36 起動/token 削減 | 中: ビット一致は未確認 | `qsa_nr_parity.cu`（prompt 経路の融合は bit-for-bit）, `rope_parity.cpp`, `qsa_parity.cpp` |
| C6 | P2 | q/gate split の `cudaMemcpy2DAsync` を stride 読みに置換 | `src/core/layer.cpp:1258` | 12 memcpy node/token 削減 + 12 層ぶんの 12 KB コピー削減 | 中: 下流カーネルが連続レイアウト前提 | `--dump-layers`（qcur のビット一致） |
| C7 | P2 | `kv_stream_resolve` の copy を間引く / `--kv-resident` を増やす | `src/core/layer.cpp:1276`、`src/kernels/cuda/kv_stream.cu:214-233` | 12 層 × (resolve 1 + copy 1) の起動と PCIe トラフィック削減。QSA 5.3% の一部（推定） | 高: page table の整合性。誤ると attention が静かに壊れる | `tests/core/kva_test.cpp`、`tests/prefill/routed_stream_test.cpp`、`qsa_nr_parity` |
| C8 | P2 | `f32_to_bf16_bulk` を norm の epilogue から供給 | `src/core/layer.cpp:493`（shared expert 入力）、`layer.cpp:425`（router 入力） | 48-96 起動/token 削減 | 低-中: bf16 丸めは同じ式なら一致 | `--dump-layers` のビット比較 |
| C9 | P3 | `STRATA_FUSE_DECODE=1`（本番では no-op） | `include/strata/kernels/fuse_decode.hpp:9`、`src/core/layer.cpp:265,327,375,484,1166,1198`、`src/kernels/cuda/s2_expert_grouped.cu:613,1309` | 本番（verifier）に呼び出しが無いため効果ゼロ。canonical と s2 hit 経路のみ | 低: ヘッダが bitwise と明記 | `--dump-layers` のビット比較（canonical 構成で） |
| C10 | P2 | prefill: BF16 重みの FP32 変換をキャッシュ | `src/prefill/gemm.cu:1090-1145`（bf16_path==2）、`:1450-1490`（w16 cache の前例） | hc read 12.9% のうち変換分。158 MB/chunk の再変換を削減（推定 1-3%） | 中: P40 の VRAM 余裕（expert cache と競合） | `gemm_bf16_parity`（`src/prefill/gemm_bf16_parity.cu`） |
| C11 | P2 | prefill: 活性の FP32 像を層で 1 回だけ作る | `src/prefill/gemm.cu:1195-1215`（`f16_to_f32(X)` を投影ごとに実行） | 302 投影 × n_sl の変換を 48 層ぶんに削減。約 550 起動/chunk + 帯域（推定） | 低: 変換は exact（FP16 -> FP32） | `gemm_f32_parity`（5,201 サイズのビット一致実績） |
| C12 | P2 | prefill: expert gate/up の経路選択（PF_FUSED / MMQ タイル） | `src/prefill/prefill.cpp:4527-4659`、`src/prefill/moe_mmq.cu:472` | expert gate/up 13.4% の改善余地。数 %（推定） | 中-高: 経路ごとに丸めが違う | `tests/cuda/prefill_mmq_*_test.cpp`、`prefill_fused_moe_test.cpp` |
| C13 | P3 | prefill: QSA prompt attention の gate 融合を既定化 | `src/prefill/prefill.cpp:4407`、`qsa_fuse_gate()`（`prefill.cpp:576`） | QSA 13.6% の一部 | 中: コメントが「bitwise ではない」 | KL / logits 比較（`--dump-logits`） |
| C14 | P3 | PLE のホスト gather を非同期化（prefill は既に async） | `src/core/layer.cpp:1593-1620`（decode） | 1 トークン 1 回のホスト処理 + H2D 160 float。`--stats` の PLE 項で測る | 中: issue/collect の順序 | `--stats` の PLE 項、`--no-ple` との A/B |
| C15 | P3 | spec window の T と `--spec-min-p` の再調整 | `src/program/generate.cpp`（`--spec`）、`src/core/verify.cpp:3220` | expert 同期待ち 8.2% の一部（report）。T を下げると起動数も下がる | 低: 既存スイッチのみ | A-B-A ベンチ（`~/bench.py`、BENCH_SEED=42） |

## 2. 各候補の根拠

### C1: 単一トークン graph（96 launch -> 1）— 訂正: 不能/不要
- 前回の記述は誤りだった。詳細と根拠は `c1-native-graph.md`。要点:
  - 本番（native pack）の decode は `Verifier` のウィンドウ経路で、**1 window = 1 graph launch**
    （capture は `src/core/verify.cpp:2748`、launch は `:3369`、commit は `:3887`）。
  - `!native_pack`（`src/program/generate.cpp:9000`, `19830`, `9518-9522`）は plan v0.3 P6
    「native pack は verify window だけで decode する」設計。root commit `01529aaa` から存在し、
    単独の導入コミットは無い（`git log -S "!native_pack"` の 4 件は他用途の変更）。
  - 理由（コードから確認）: native pack の GPU expert は `native_expert_grouped` + device plan
    （`verify.cpp:1296-1320, 2360-2372`）で、token graph の hit 経路（s2 カーネル, `session.cpp:868-873`）
    には native 版が無い（`expert_source.cpp:5322-5324` のコメント）。
  - `session_loop`（`session.cpp:587`）を native pack で有効化すると 1 -> 96 launch の退行。
- 96 graph launch という数字は canonical pack（`session_loop`）のもので、本番には当てはまらない。
- 代替（C1b）: verifier の host 項を削る。`--stats` の "verify window" 行（`generate.cpp:20810`）で
  測る。既存スイッチ: `STRATA_VERIFY_DEVICE_PLAN=1`（既定 off、`verify.cpp:1292-1296`）、
  `STRATA_QFUSE=1` / `STRATA_VERIFY_QDEDUP=1`（既定 off、`verify.cpp:180/292`）。

### C2: hit 判定の device 化
- 本番（native pack の verifier）は既に実装済みの opt-in:
  `STRATA_VERIFY_DEVICE_PLAN=1`（`src/core/verify.cpp:1292-1296`、`2248`）と
  `native_route_resident`（`:2236`）。`all_resident_`（`:1267-1291`）は全 expert が VRAM にある層だけ
  zero-doorbell graph になる（P40 24GB では成立しないと推定）。
- canonical 経路: `expert_source.cpp:5317-5360` がホストで 10 id を走査し、`session.cpp:868-873` が
  device 版（`moe_hit_select` + `quantize_q8_0_scaled` + `moe_hit_grouped_s2_dev`）を graph に入れる。
- 既定 off の理由はコード内の測定メモ（`verify.cpp:1294-1296`）: 「exact, but neutral on RIBPC 1-2 GPUs」。
  P40 での効果は未測定。まず env で A/B する。

### C3: shared expert gate/up の pair 化
- 本番（native pack の verifier）は `shared_expert_multi` を `lfuse` ビット 2 付きで呼び、
  `STRATA_LFUSE_PAIR` は既定 on（`src/core/verify.cpp:179-180`, `2301-2302`）。**既に有効**。
- 残るのは canonical の単一トークン経路（`src/core/layer.cpp:499` -> `shared_expert.cu:436-445`）だけ。
  そこでは gate と up を別々に launch する（`shared_expert.cu:225-231` の pair は multi 用）。
- コメント: "S26 STRATA_LFUSE (gate_deferred): gate and up in one launch where the pair applies (bitwise the two calls)"。

### C4: q8_1 を norm/combine の epilogue へ
- 本番（verifier）は既存 opt-in: `STRATA_QFUSE=1`（`src/core/verify.cpp:292`）と
  `STRATA_VERIFY_QDEDUP=1`（`:180`）。`gr_read_group` が消費側の q8_1 を書く（`:1601`, `:2176`）、
  native pack は `quantize_q8_1_rows`（`:2313-2331`）。既定 off。
- コメント（`verify.cpp:181`）: "activation q8_1 images written by their producers (the GDN output norm) - the same bytes"。
- canonical の単一トークン経路（`src/core/layer.cpp:276`, `:376`）は未融合のまま。

### C5: norm+rope 融合
- 本番（verifier）は既に `native_qsa_rms_norm_rope` を使う（`src/core/verify.cpp:1815, 1860, 1934, 1946`）。
- 残るのは canonical の単一トークン経路（`src/core/layer.cpp:1147-1149` の norm + rope 2 発）。
- `native_qsa_rms_norm_rope` の定義は `src/kernels/cuda/native_rope.cu:220`（head_dim 128/256、n_rot 64）。
  このモデルは head_dim 256 / n_rot 64 なので条件を満たす。
- ビット一致の根拠は **prompt 経路の別関数** `rms_rope_rows` に対する `src/prefill/qsa_nr_parity.cu`
  （bit-for-bit）。decode 用 `native_qsa_rms_norm_rope` と norm+rope のビット一致は未確認。

### C6: q/gate split
- `src/core/layer.cpp:1258` の `cudaMemcpy2DAsync` は「per_head[:, :head_dim] を取り出す」ための
  2-D コピー。`q_full` は [n_head][2*head_dim] なので、norm/rope 側が stride 2*head_dim を読めば
  コピー自体が不要になる。
- 12 層 × 1 node。中身は 24 head × 256 float = 24 KB/層。

### C7: KV streaming
- `--kv-resident 32768` < `--max-context 131072` なので `kv_mode == 1`（`src/core/layer.cpp:1000-1027`）。
- `kv_stream_resolve` は resolve + copy の 2 発（`src/kernels/cuda/kv_stream.cu:230-233`）で、
  選択された block を pinned host から VRAM slot へ戻す。12 層 × 毎トークン。
- 期待幅: 起動 24/token と PCIe 帯域。QSA 5.3% と「QSA attention 13.6%（prefill）」の一部。
- リスク: page table の不整合は静かに attention を壊す。既存の `kva_test` / `routed_stream_test` で
  検証する。

### C8: bf16 入力の再利用
- `src/core/layer.cpp:493`（`f32_to_bf16_bulk(x, b.x_bf16)`、shared expert 用）と
  `layer.cpp:409-411`（router 用、`!native_bf16_projections` のときだけ）。
- prefill 側は `m.mixed_bf` を gr_norm/gr_mix が作っている（`prefill.cpp:4040` 以降）。
  decode でも gr_read の出力に bf16 像を持たせれば 1-2 起動/層が消える。

### C9: STRATA_FUSE_DECODE（本番では no-op）
- `include/strata/kernels/fuse_decode.hpp:9` の `fuse_decode()` は env のみで on（既定 off）。
- 呼び出しサイト（`grep -rn "fuse_decode()" src/`）: `src/core/layer.cpp:265, 327, 375, 484, 1166, 1198`、
  `src/kernels/cuda/shared_expert.cu:336`、`src/kernels/cuda/s2_expert_grouped.cu:613, 1309`。
- **`src/core/verify.cpp` には呼び出しが無い** -> 本番（native pack の decode window）には効果ゼロ。
  canonical の `session_loop` と s2 hit 経路にだけ効く。
- 前回の「層あたり 2-5 起動削減（100-240/token）」は canonical 経路の推定で、本番には当てはまらない。

### C10-C12: prefill
- C10: `Gemm::bf16` は Pascal で `bf16_path()==2` になり、**呼び出しごとに重みを FP32 に変換**する
  （`src/prefill/gemm.cu:1103-1120`）。hc down/up/inject、router、indexer q/k、PLE key/value が該当。
  重みは層ごとに固定なので、FP32 像をキャッシュすれば変換 1 発 + 帯域が消える。
  前例は quantized 重みの FP16 キャッシュ（`gemm.cu:1427-1490`、`STRATA_PF_W16_CACHE_MIB`）。
- C11: `STRATA_DENSE_F32=1` の分岐は投影ごとに `f16_to_f32(X)` を n_sl 回やる
  （`gemm.cu:1200-1210`）。同じ X を qkv/gate/out が読むので、層で 1 回にまとめられる。
  変換は exact なのでビット一致（`gemm_f32_parity` の実績）。
- C12: expert gate/up は report 13.4%。MMQ のタイル（`mul_mat_q_case`）と `ws::` / `fused::` の
  選択は既に env で切れる（`STRATA_PF_FUSED`, `STRATA_MMQ_*`）。まず A/B で現行最良を確認する。

### C13-C15
- C13: `qsa_fuse_gate()` は env 依存（`prefill.cpp:576`）。既定化には「bitwise でない」ことの
  許容判断が要る（`prefill.cpp:4401` のコメント）。
- C14: decode の PLE は 1 トークン 1 回ホストでハッシュ + テーブル gather をする
  （`layer.cpp:1593-1620`）。`--stats` の PLE 項（`generate.cpp:20982`）で実測できる。
- C15: `--spec 4` のウィンドウは T=5。expert 同期待ち 8.2%（report）はウィンドウ内の
  CPU pool 直列化。T を下げる / `--spec-min-p` を上げるのは既存スイッチだけで試せる。

## 3. 「既に最適化済み」で余地が無いと判断した項目

| 項目 | 根拠 |
|---|---|
| logits の D2H + NaN スキャン + sampler の同期 | `src/program/generate.cpp:19975-19985` のコメント: フラグで消すと **遅くなった**（大域 readback がパイプラインの flush を兼ねている）。`sampler.cu:245` の `cudaDeviceSynchronize` も設計 |
| doorbell を memcpy ではなく kernel で publish | `src/core/session.cpp:718-726` のコメント: event record は capture 内で落ちるため kernel にした |
| `parts` staging を pinned + `copy_from_mapped` kernel に | `src/core/session.cpp:849-880` のコメント: copy-engine node は WDDM の submission を分割する（67 flushes/token） |
| hc read の融合（6 kernel -> 2） | `src/kernels/cuda/fused_gr.cu:3071-3085`（gr_down + gr_up）。`STRATA_HC_PERSIST` も既存 |
| gr_write の折り込み | `src/core/layer.cpp:1564` は fused 時に gr_write を呼ばない（fused_gr_read の `apply` が折り込む） |
| expert hit の device 判定（単一トークン graph 内） | `src/core/session.cpp:868-873` に実装済み。native pack 側は verifier の `device_plan_`（`verify.cpp:1292-1320`） |
| shared expert の gate/up pair | 本番は `STRATA_LFUSE_PAIR` 既定 on（`verify.cpp:179-180`, `2301-2302`） |
| QSA の norm+rope 融合 | 本番は `native_qsa_rms_norm_rope` 使用済み（`verify.cpp:1815, 1860, 1934, 1946`） |
| 入力 q8_1 の生産側書き込み | 既存 opt-in `STRATA_QFUSE` / `STRATA_VERIFY_QDEDUP`（`verify.cpp:180, 292`）。既定 off なので余地は「有効化の判断」だけ |
| kv append の host mirror を別 kernel にしない | `src/core/layer.cpp:1226-1235` のコメント: append カーネルが同じ launch で host copy も書く |
| MMQ の direct read / PF_FUSED / DEQUANT_SIDE / w16 cache | 既存スイッチ（`src/prefill/moe_mmq.cu`、`prefill.cpp:4527`、`gemm.cu:1477`、`gemm.cu:1427`） |
| q36 の GDN 融合（`fused_gdn_conv_l2` 相当）を decode に使わない判断 | `src/qwen35/q36_engine.cu:1990-1996` のコメント: 4-tap の和が FMA に縮約され 1 ulp 違い、Q8_1 丸めで near-tie が反転しうる。MTP の完全一致を優先 |

## 4. 実装難度の目安

| 難度 | 候補 |
|---|---|
| 低（env / スイッチのみ） | C2, C4, C9（本番 no-op）, C13, C15 |
| 中（1 カーネルの置換） | C3, C5, C6, C8（いずれも canonical 経路のみ） |
| 中-高（graph 構成の変更） | C1b, C7 |
| 高（epilogue 融合 + ビット一致の作り込み） | C4（canonical への拡張） |
| 高（prefill の GEMM 経路） | C10, C11, C12 |
| 不能/不要 | C1（`c1-native-graph.md`） |

## 5. 検証の共通手順（推奨）

1. **ビット一致**: `--dump-layers`（`src/core/session.cpp:640-670` の dump、`layer.cpp:1463` の
   `dump_slot`）で層ごとの residual / attention 出力を比較する。`--dump-halves` もある
   （`src/core/session.cpp:327-380` の `session_replay_stages_per_layer`）。
2. **logits 一致**: `--dump-logits`（`src/program/generate.cpp`）で 248,320 float を比較。
3. **KL**: `tools/qwen35/kl.py` は q36 用だが、同じ手法（参照実装との分布比較）が使える。
4. **ベンチ**: `~/bench.py`（A-B-A、BENCH_SEED=42、各アーム 3 回、中央値、5% 閾値）。
5. **既存テスト**: `tests/core/host_graph_launch_test.cu`、`tests/core/dma_batch_parity.cu`、
   `tests/cuda/hc_q8_stream_test.cu`、`tests/cuda/prefill_swiglu_quant_test.cpp`、
   `src/prefill/gemm_f32_parity.cu` / `gemm_w16_parity.cu` / `gemm_bf16_parity.cu`。

## 6. 注意（推定と未確認）

- 期待幅はすべて **推定**。実測していない。
- C1 の blocker は特定済み（`c1-native-graph.md` §1-§2）: root commit `01529aaa` の plan v0.3 P6 で、
  native pack は verify window だけで decode する設計。判断は「不能/不要」。
- C2/C4 は本番では既存 opt-in（既定 off）なので、まず env で A/B する。C3/C5 は本番で既に有効。
  canonical 経路への拡張は本番の速度に影響しない（canonical 運用時のみ）。
- C7 は `--kv-resident` の値を変えるだけで試せる A/B が先。カーネル改造の前に設定で測る。
- C4 は本番では `STRATA_QFUSE` / `STRATA_VERIFY_QDEDUP` の有効化判断が先（既定 off、ビット一致は
  コード内コメント「the same bytes」）。canonical 経路への拡張は別作業で、q36 の
  `k_moe_out_small` / `k_add_rms_q` をそのまま移植できるかは未確認（q36 は別モデル・別レイアウト）。
- 本番コンテナは作業時点で停止していた（`docker exec` 不可）。ログ由来の数値は 2026-11-02 の
  ctx 32768 の run のもので、131072 の現行設定とは一致しない可能性がある。
