# decode-kernel-map: Qwen3.8-Flash-Next の decode 1 ステップ

作業ツリー: `~/work/wt/codemap`（branch `docs/codemap`）
対象モデル: `/var/lib/docker/volumes/compose_strata-data/_data/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`

## 0. 前提の訂正（最初に読むこと）

**`src/qwen35/q36_engine.cu` は本番モデルのエンジンではない。**

- 本番 GGUF の `general.architecture` は `qwen4exp`（GGUF ヘッダを直接読んで確認。`qwen4exp.block_count=48`,
  `embedding_length=2560`, `expert_count=512`, `expert_used_count=10`, `expert_feed_forward_length=640`,
  `hyper_connection.count=4`, `hyper_connection.low_rank=320`, `full_attention_interval=4`,
  `attention.head_count=24`, `head_count_kv=2`, `key_length=256`, `ssm.inner_size=6144`, `ssm.state_size=128`,
  `ssm.group_count=16`, `ple.layers=[1]`）
- `qwen35` エンジン（`src/qwen35/`）は `qwen35moe`（Qwen3.6-35B-A3B）専用。
  `include/strata/artifact/model_family.hpp:3`（"Strata was written for one model (Qwen3.8-Flash-Next, `qwen4exp`)"）、
  `src/program/generate.cpp:3345-3357`（family switch は `ModelFamily::Qwen35Moe` のときだけ `qwen35::run` へ）
- したがって本番の decode/prefill 経路は `src/program/generate.cpp` -> `src/core/session.cpp` ->
  `src/core/layer.cpp` -> `src/kernels/cuda/*` である。以下はこの経路の地図。
  q36 エンジンの呼び出し列は付録（§9）に、依頼どおり別モデル向けとして記載する。

### モデル形状（この地図で使う値）
| 項目 | 値 | 根拠 |
|---|---|---|
| 層数 | 48（GDN 36 + QSA 12） | GGUF `block_count=48`, `full_attention_interval=4` |
| n_embd | 2560 | GGUF |
| n_head / n_head_kv / head_dim | 24 / 2 / 256 | GGUF |
| expert | 512、top-10 | GGUF |
| n_ff（expert / shared） | 640 / 640 | GGUF |
| ssm conv channels C | 10240（q 2048 + k 2048 + v 6144） | `src/core/layer.cpp:293-295` のコメント |
| hc / hc_lr | 4 / 320 | GGUF |
| vocab（`output.weight` の行数） | 248,320 | `src/program/generate.cpp:19976-19979` |
| QSA 層 | 4 層ごと（12 層） | `is_qsa_layer()` |

### 本番設定（依頼文のとおり。値は未変更）
`--max-context 131072 --kv int8 --kv-resident 32768 --spec 4 --spec-min-p 0.5 --ple-io ram --prefill auto`
`STRATA_DENSE_F32=1`、`--pack`（native expert pack）、`--expert-cache auto`、`--mtp /data/Strata-data/mtp/rt`。
現行プリセットは `/data/state/strata/launch/docker.json` で確認した（`--batch` は無い）。

> 注: `--spec 4` があるため、本番の decode は「1 トークンずつの graph」ではなく
> **MTP ドラフト + 検証ウィンドウ（T=5 行）** が主経路（`src/core/verify.cpp:3220 Verifier::run`）。
> 本ドキュメント §1-§8 は「decode 1 ステップ（T=1）」の基準形を地図化し、§8 でウィンドウ経路との差を述べる。

## 1. 1 トークンの外枠（ホスト側の順序）

`src/program/generate.cpp:19900-20030` のトークンループが 1 ステップの外枠。

1. `ple_stage_token()`（定義: `src/core/layer.cpp:1611`）: PLE の n-gram ハッシュとテーブル gather は**ホスト**で行い、
   `emb_dev`（160 float）へ H2D。graph の外。
2. `embed_row()`（`src/core/layer.cpp:1345`）: `embedding_gather_kernel`（`src/kernels/cuda/elementwise.cu:137`）1 発。graph の外。
3. `stage_token()`（`src/core/session.cpp:191`）: QSA 12 層ぶんの `step` / `pos` を pinned host に書き、graph 内の H2D が読む。
4. `session_loop()`（`src/core/session.cpp:587`）: 層ごとに pre/post の 2 グラフを launch し、doorbell を poll し、
   CPU expert pool を回す。§2-§6。
5. `lm_head()`（`src/core/layer.cpp:1412`）: `lm_head_mix`（gr_read 4 発）+ `quantize_q8_K` 1 発 + `native_mmvq` 1 発。
6. logits の D2H（993 KB、ブロッキング）+ 248,320 float のホスト NaN スキャン（`src/program/generate.cpp:19975-19985`）。
7. `sample_tokens`（`src/program/generate.cpp:20002`、`src/kernels/cuda/sampler.cu:245` で `cudaDeviceSynchronize`）
   + 4 バイトのブロッキング readback（`src/program/generate.cpp:20005`）。

## 2. 層あたりの呼び出し列: GDN 層（36 層）

capture される pre グラフの中身（`src/core/session.cpp:200-300` が `block_layer_pre` を捕捉）。
`block_layer_pre` は `src/core/layer.cpp:1473`。

| # | 呼び出し | file:line | 起動数 |
|---|---|---|---|
| 1 | `fused_gr_read`（hc_attn 半分） | layer.cpp:1556 -> fused_gr.cu:3071 | 2（gr_down, gr_up） |
| 2 | `gemv_quantized(attn_qkv)` = `native_quantize_q8_1` + `native_mmvq` | layer.cpp:276 -> native_mmvq.cu:2596/2464 | 2 |
| 3 | `fused_gdn_conv_l2`（conv + SiLU + L2 norm） | layer.cpp:285 -> fused_gdn.cu:139 | 1 |
| 4 | `fused_gdn_ab`（alpha/beta 射影 + beta sigmoid + gate） | layer.cpp:322 -> fused_gdn.cu:150 | 1 |
| 5 | `gemv_quantized(attn_gate)`（x_q8_1_ready=true で quantize 省略） | layer.cpp:362 | 1 |
| 6 | `fused_gdn_step_norm`（recurrence + out norm） | layer.cpp:365 -> fused_gdn.cu:164 | 1 |
| 7 | `gemv_quantized(ssm_out)` = quantize + mmvq | layer.cpp:376-383 | 2 |
| 8 | `fused_gr_read`（hc_ffn 半分） | layer.cpp:1579 | 2 |
| 9 | `bf16_gemv_fp32_mmvf`（router、512 行） | layer.cpp:425 -> native_bf16.cu:478 | 1 |
| 10 | `native_router_top10` | layer.cpp:422 -> native_router.cu:224 | 1 |
| 11 | `doorbell_publish`（x/ids/weights を mapped pinned へ） | layer.cpp:432 -> elementwise.cu:380 | 1 |
| 12 | `f32_to_bf16_bulk`（shared expert の入力） | layer.cpp:493 -> elementwise.cu:174 | 1 |
| 13 | `shared_expert` 内: `native_quantize_q8_1` + gate mmvq + up mmvq + `native_swiglu_quantize_q8_1` + down mmvq | layer.cpp:499 -> shared_expert.cu:387-500 | 5 |
| 14 | `bf16_gemv_fp32_mmvf`（shared の scalar gate）+ `launch_sigmoid_scale_rows` | shared_expert.cu:476-480 | 2 |
| 15 | （graph 外・ホスト）expert hit 判定 + `cudaMemsetAsync(hit_out)` + slot/dst の H2D 2 発 + `quantize_q8_0_scaled` + `moe_hit_grouped_s2` | expert_source.cpp:5307-5420 | 3 kernel + 3 driver |
| 16 | （graph 外）pool の結果を `parts_dev` へ H2D（pinned から） | session.cpp:742 | 1 copy |
| 17 | `moe_hit_add` | expert_source.cpp:5450 付近 -> s2_expert_grouped.cu:1359 | 1 |
| 18 | post グラフ: `native_moe_combine` | layer.cpp:1626 -> native_moe.cu:168 | 1 |

- 合計（pre グラフ内）: **23 起動**、post グラフ: **1 起動**、graph 外の hit 経路: 3 kernel + 3 driver。
- 最終層だけ post で `gr_write` が 1 発増える（`layer.cpp:1635`、fused 時は R の materialize のみ）。

## 3. 層あたりの呼び出し列: QSA 層（12 層）

`qsa_layer`（`src/core/layer.cpp:1144`）。pre グラフ内。

| # | 呼び出し | file:line | 起動数 |
|---|---|---|---|
| 1 | `bf16_gemv_fp32_mmvf`（indexer.k_proj） | layer.cpp:1208 | 1 |
| 2 | `gemv_quantized(attn_k)` = quantize + mmvq | layer.cpp:1214 | 2 |
| 3 | `gemv_quantized(attn_v)`（ready=true） | layer.cpp:1215 | 1 |
| 4 | `native_qsa_rms_norm_weighted`（k）+ `native_rope_apply`（k） | layer.cpp:1147,1149 | 2 |
| 5 | `kv_append_q8_step`（INT8 K/V の page 書き込み。mirror は同一 kernel 内） | layer.cpp:1241 -> kv_q8.cu:187 | 1 |
| 6 | `native_qsa_indexer_append`（indexer の append + pool + rotate） | layer.cpp:1243 -> native_qsa_indexer.cu:289 | 1 |
| 7 | `gemv_quantized(attn_q)`（12288 行、ready=true） | layer.cpp:1254 | 1 |
| 8 | `cudaMemcpy2DAsync`（q_full -> qcur の前半 head_dim 取り出し） | layer.cpp:1258 | 1 copy node |
| 9 | norm + rope（q） | layer.cpp:1147,1149 | 2 |
| 10 | `bf16_gemv_fp32_mmvf`（indexer.q_proj） | layer.cpp:1261 | 1 |
| 11 | norm + rope（indexer q） | layer.cpp:1147,1149 | 2 |
| 12 | `qsa_block_scores` | layer.cpp:1270 -> qsa_select.cu:2089 | 1 |
| 13 | `qsa_block_topk` | layer.cpp:1271 -> qsa_select.cu:2905 | 1 |
| 14 | `kv_stream_resolve`（`--kv-resident 32768` < ctx 131072 なので mode 1） | layer.cpp:1276 -> kv_stream.cu:214 | 2（resolve + copy） |
| 15 | `qsa_decode_attn_step` | layer.cpp:1280 -> qsa_decode_attn.cu:1211 | 1 |
| 16 | `native_qsa_gate_apply` | layer.cpp:1312 -> native_qsa.cu:164 | 1 |
| 17 | `quantize_q8_K`（attn32） | layer.cpp:1315 | 1 |
| 18 | `gemv_quantized(attn_output)`（quantize は 17 で済み、mmvq のみ） | layer.cpp:1317 | 1 |

- pre グラフ内: **23 起動**（うち 1 は memcpy node）。post は GDN と同じ 1 発。
- 注意: `kv_stream_resolve` の copy kernel（`kv_stream.cu:232`）は **トークンごとに選択 block を pinned host から VRAM へ戻す**。
  12 層 × 毎トークンで PCIe トラフィックが発生する（`--kv-resident` を増やすか、resolve を間引く余地）。

## 4. 全体で 1 回のもの

| 処理 | file:line | 起動/コピー |
|---|---|---|
| PLE のホスト gather（1 トークン） | layer.cpp:1593-1620 | ホスト + H2D 160 float |
| `embed_row` | layer.cpp:1345 -> elementwise.cu:137 | 1 |
| PLE の device 半分（層 1 の pre グラフ内）: `ple_block` + `ple_history_advance` | layer.cpp:1511,1525 | 2 |
| `lm_head_mix`（gr_read） | layer.cpp:1391 -> gr.cu:425-446 | 4 |
| `quantize_q8_K` + `native_mmvq`（output.weight） | layer.cpp:1424-1432 | 2 |
| logits D2H + NaN スキャン | generate.cpp:19975-19985 | 1 copy（993 KB、ブロッキング） |
| `sample_tokens` + 4B readback | generate.cpp:20002-20010 | 1 kernel + 2 sync |
| `stage_token`（pinned 書き込みのみ） | session.cpp:191-198 | 0 |
| expert cache の fill（miss 時のみ） | expert_source.cpp:5334 | miss 数だけ H2D |

## 5. graph の内と外

### 5.1 どの graph が使われるか

> **訂正（2026-11-07、`c1-native-graph.md`）**: 本番（native pack）は `session_loop` ではなく
> **Verifier のウィンドウ経路**で、1 window = 1 graph launch（`src/core/verify.cpp:2748` で capture、
> `:3369` で launch、`:3887` で commit）。下の「96 グラフ」は **canonical pack** の話。

- `--pack`（native expert pack）があると **単一トークン graph も層グラフも作られない**:
  `src/program/generate.cpp:9519-9522` の条件 `(hit_fn == nullptr || thits.on()) && !native_pack && !multi_gpu`、
  および `generate.cpp:9000` の `!native_pack`（plan v0.3 P6「native pack は verify window だけで decode」）。
  本番は `--pack` があるため `native_pack == true`（`generate.cpp:4621`）。
- native pack の decode は `generate.cpp:19830` で token ループを抜け、spec/verify ループへ入る。
  1 ラウンド = `Verifier::run`（`verify.cpp:3220`）がウィンドウ全体を 1 graph で launch。
- canonical pack では層ごとの pre[l] / post[l] の 2 グラフ × 48 層 = 96 グラフを 1 トークンで launch する
  （`src/core/session.cpp:200-300` で capture、`session.cpp:587-800` で replay）。
- 単一トークン graph（`session_capture_token`, `session.cpp:849`）は canonical pack の別経路。
  native pack で使えない理由は「native pack の GPU expert 実装が verifier 側にしかない」こと
  （`src/core/expert_source.cpp:5322-5324`、`verify.cpp:2360-2372`）。詳細は `c1-native-graph.md`。

### 5.2 pre[l] グラフに入るもの
`block_layer_pre` の全 kernel（§2/§3 の 1-14, 18）と、層 1 の PLE device 半分。
doorbell の ring（`doorbell_publish`）もグラフ内。

### 5.3 post[l] グラフに入るもの
`moe_combine`（+最終層の `gr_write`）のみ。`block_layer_post` は `layer.cpp:1622`。

### 5.4 グラフに入らないもの
- `embed_row`、`lm_head`、sampler、logits readback（`generate.cpp` 側で eager launch）
- `ple_stage_token`（ホスト）
- `stage_token`（ホストの pinned 書き込み）
- **CPU expert pool**（`session.cpp:732`）、`parts_dev` への H2D（`session.cpp:742`）
- **expert hit 判定と hit kernel**（eager 経路: `expert_source.cpp:5307` 以降）。単一トークン graph 経路では
  `moe_hit_select` / `quantize_q8_0_scaled` / `moe_hit_grouped_s2_dev` が graph 内に入る（`session.cpp:868-873`）。
- expert cache の slot fill / swap の H2D
- QSA の `native_flash_attn_short` の status readback（`layer.cpp:1296-1300`、該当時のみ）

## 6. ストリーム・イベント・同期点

| 同期/待ち | file:line | 理由 |
|---|---|---|
| doorbell の spin（`cudaEventQuery(probe)` を毎周） | session.cpp:695-717 | mapped pinned の書き込みは driver 進入でしか host 可視にならない（コード内の実験メモ）。1 層 1 回の driver entry |
| `cudaMemcpyAsync(parts_dev, y_miss)` | session.cpp:742 | pool の結果を GPU へ。pinned なので非同期だが driver は通る |
| `cudaStreamSynchronize(cs)`（最終） | session.cpp:797 | 1 トークン 1 回 |
| hit 判定のホストループ（10 id、`cache->slot_of`、miss なら `fill_slot`） | expert_source.cpp:5317-5360 | 層ごとに CPU が 10 回の探索 + 必要なら H2D |
| logits D2H（993 KB、ブロッキング） | generate.cpp:19975-19985 | サンプラ前の flush を兼ねる（コード内の実測メモ） |
| `cudaDeviceSynchronize`（sampler） | sampler.cu:245（`sample_tokens(nullptr)` 経由） | 設計上の同期 |
| 4B readback の `cudaStreamSynchronize` | generate.cpp:20005-20008 | サンプル結果の取得 |
| PLE の H2D（160 float） | layer.cpp:1618-1620 | graph 外。stream 順序のみ |
| `kv_stream_resolve` の copy | kv_stream.cu:232 | stream 順序のみ。PCIe 帯域を消費 |
| expert cache の fill（miss） | expert_source.cpp:5334 | compute stream 上。pool は ring を見てから走る |
| `cudaEventQuery(ev_route)` の busy-wait | q36 エンジンのみ（q36_engine.cu:2226） | 本番経路ではない |

## 7. 起動回数の合計（静的カウント、**推定**）

> この節は **canonical pack の T=1 基準形**（`session_loop` 経路）の数字。本番の native pack は
> Verifier のウィンドウ経路で 1 window = 1 graph launch であり、ここでの「1 トークンあたり」は当てはまらない
> （`c1-native-graph.md` §3）。canonical 運用時の比較用として残す。

前提: T=1、canonical 経路（native pack ではない）、fused_gr / fused_gdn / native_* すべて on、
`STRATA_FUSE_DECODE` 未設定、expert cache に resident がいる（hit 経路が動く）。数え方は §2/§3 の表の合計。

| 区分 | 1 トークンあたり |
|---|---|
| pre グラフ内 kernel（GDN 36 × 23） | 828 |
| pre グラフ内 kernel（QSA 12 × 23） | 276 |
| post グラフ内 kernel（48 × 1 + 最終層 gr_write 1） | 49 |
| graph 外 hit 経路 kernel（48 × 3） | 144 |
| head（6）+ embed（1）+ sampler（1）+ PLE device（2） | 10 |
| **kernel 起動 合計（推定）** | **約 1,307** |
| cudaGraphLaunch | 96 |
| graph 外の driver 呼び出し（hit: memset + 2 H2D + 2 kernel + add = 6 × 48） | 約 288 |
| cudaMemcpyAsync（parts 48 + PLE 1 + logits 1） | 50 |

- この 1,307 という数字は **静的カウントからの推定**であり、実測ではない。`--stats` の
  `LAYERS` 項（`generate.cpp:20982`）と突き合わせて検証する必要がある。
- 参考: 既存レポートの decode GPU ステージ 49.0 ms/window を 1,307 で割ると平均 37 µs/起動。
  P40 の空 kernel launch が 5-10 µs 程度なので、**起動オーバーヘッドだけで 6-13 ms/token** になりうる
  （推定。実測は未取得）。

## 8. 本番の `--spec 4` 経路（verify window）との差

- `Verifier::run`（`src/core/verify.cpp:3220`）が T=5 行（4 ドラフト + 1）のウィンドウを 1 回で流す。
  層ごとの列は `Verifier::record_window`（`src/core/verify.cpp:1336-2732`）が組み、`Verifier::capture`（2748）が graph 化する。
- ウィンドウ経路は T 行ぶんのカーネル（`*_multi` 系）を使うため、**起動数は T=1 の約 1.2-1.5 倍**で
  5 トークンを処理する（推定）。レポートの「1 window 52-61 ms、うち expert 同期待ち 8.2%」はこの経路の値。
- したがって §7 の 1,307 は「T=1 の基準形」であり、本番スループットの分母ではない。
- ドラフト（MTP）側は `src/core/mtp.cpp` / `MtpDrafter`。ウィンドウの検証とコミットは
  `Verifier::commit`（`verify.cpp:3852`）と `commit_rows`。

## 9. 付録: `src/qwen35/q36_engine.cu`（qwen35moe 用、本番モデルでは未使用）

依頼の「decode 1 ステップの呼び出し列」をこのファイルについても記録する（他モデル向けの参考）。

- 入口: `process()`（q36_engine.cu:2434）。n==1 なら `graph_run(gs().g1, ...)`（q36_engine.cu:2459）で
  `forward(1, ...)` を捕捉・再生。`--no-graph` なら eager。
- `forward`（q36_engine.cu:2154）: `iq_embed_rows` -> `k_rms` -> 48 層の `layer()` -> `head_rows`。
- GDN 層（`layer()` の `!L.attn` 側、q36_engine.cu:1962-1999）の起動:
  `prep`(quantize_q8_1) 1, `apply(qkv)` 1, `apply(zg)` 1, `k_gdn_ab` 1, `k_conv_seq` 1, `k_gdn_seq` 1,
  `mm(ssm_out)` 2, `k_add_rms_q` 1, `f32mm(router_ext)` 1, `k_route_group` 1,
  `native_expert_grouped` 2-3, `k_sh_gu` 1, `k_moe_out_small` 1 -> **約 15 起動**（T=1、fast_k on）
- QSA 層（q36_engine.cu:2000-2050）: `apply(wq/wk/wv)` 3, `k_attn_prep` 1, `k_attn_part` 1, `k_attn_merge` 1,
  `mm(wo)` 2, `k_add_rms_q` 1, router 2, experts 2-3, shared 1, `k_moe_out_small` 1 -> **約 15 起動**
- 全体 1 回: `head_rows` の `mm(head)`（T=1 で 1 発）+ `k_argmax` 2 発。
- spec 経路 `spec_step`（q36_engine.cu:2680）: ドラフト D 回 × `graph_run(gm1)`（MTP 1 層）+ 検証
  `graph_run(gv[T])` 1 回 + `commit(a+1)`（GDN 層数 × 2 起動、q36_engine.cu:2216）。
- 同期点: `process()` の `cudaStreamSynchronize`（q36_engine.cu:2436, 2478）、`spec_step` の各ドラフト後の
  sync（q36_engine.cu:2716, 2730）、prefill MMQ 経路の層ごとの sync（q36_engine.cu:2041）。
- このエンジンは **prefill でも層ごとに `cudaStreamSynchronize`** を入れて `max_rows` を host に読む
  （q36_engine.cu:2039-2041）。本番モデルでは起きないが、同種の設計を流用する場合は注意。

## 10. 未確認事項
- `--stats` の実測内訳（PLE / embed / LAYERS / head / readback / sample）は本番ログに残っていない（`--stats` 未使用）。
- `native_pack` で単一トークン graph が使えない正確な理由（コメント以上の裏取りなし）。
- w16 FP16 weight cache（`gemm.cu:1427-1490`）は現行プリセットでは off（`--batch` なし、`generate.cpp:8180`）。
- expert cache の resident 数（`generate.cpp:9450-9475` のログ行は本番ログに無い）。
