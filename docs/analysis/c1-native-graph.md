# C1: native pack で単一トークン CUDA graph を使えるか

作業ツリー: `~/work/wt/codemap`（branch `docs/codemap`）。読み取りのみ。
関連: `decode-kernel-map.md`（§5 の訂正を含む）、`optimization-candidates.md`（C1 節を更新）。

## 0. 結論

**判定: 不能（というより不要）。** 本番（native pack）の decode は既に「ウィンドウ全体で 1 回の graph launch」であり、
`generate.cpp:9518-9522` の `!native_pack` は「native pack は verify window だけで decode する」という設計
（plan v0.3 P6）の一部。`session_loop` を native pack に有効化すると **1 launch -> 96 launch の退行**になる。

- 前回の `optimization-candidates.md` C1 の「96 graph launch -> 1」は **canonical pack の話**であり、
  本番の native pack には当てはまらない。訂正する（§6）。
- 本番の decode 1 ラウンド = `Verifier::run`（`src/core/verify.cpp:3220`）が
  **ウィンドウ全体（全 48 層・全グループ）を 1 つの graph で launch**（`verify.cpp:2748` で capture、
  `verify.cpp:3369` で `cudaGraphLaunch`）+ commit graph（`verify.cpp:3887`）。
- 残る host 項は graph の中の「グループごとの計画待ち」（`wait_flag_ge` / `copy_i32_from_mapped`）と
  CPU pool。これは `--stats` の "verify window: wait for rings / pool / host / commit"（`generate.cpp:20810`）で測れる。

## 1. `!native_pack` の出所（コミットと差分）

- `git log -S "!native_pack" -- src/program/generate.cpp` の結果は 4 件:
  `01529aaa`（root）, `85118e31`, `ab97f174`, `7f97c35a`。
- `git rev-list --max-parents=0 HEAD` = `01529aaa82b9f5c58d2bb87264b7b3f7b823678f`
  （"Strata: Qwen3.8-Flash-Next on a 12-24 GB NVIDIA GPU + 64 GB RAM, one-click install"、2026-09-24、219 files / 50165 insertions）。
- **その root commit に既に条件がある**。`git show 01529aaa:src/program/generate.cpp` の 1230 行:
  `if (!o.no_capture && !native_pack) {   // plan v0.3 P6: a native pack runs verify windows only`
  同じファイルの 1705 行に token graph の条件 `... && !native_pack`。
- 後続 3 件は条件そのものではなく、他の `!native_pack` 用途を触っている:
  - `85118e31` "PR #44 follow-up: keep the default from 20 GB/s up, probe native packs only, ..."（PCIe probe の分岐）
  - `ab97f174` "PcieAuto for the canonical pack: probe the link and refine the share ..."（同じく PCIe）
  - `7f97c35a` "DFlash2 drafter ..."（エラーメッセージの `!native_pack ? "it needs a native pack"`）
- 条件に `&& !multi_gpu` を足したのは `915f74fe` "Multi-GPU: the session carve and the per-stage prompt loans"
  （`git log -L 9500,9540:src/program/generate.cpp` の差分に `- ... && !native_pack` /
  `+ ... && !native_pack && !multi_gpu) {   // a split's token graph cannot span stages`）。
- **したがって「導入コミット」は root import であり、単独の理由を書いたコミットは存在しない。**
  理由はコード内の plan v0.3 P6 のコメントと、後述の実装事実から読み取るしかない（推測は §2 で区別する）。

## 2. なぜ native pack で per-layer graph を作らないのか

### 2.1 コードに書かれている理由（事実）
- `src/program/generate.cpp:9000`: `if (!o.no_capture && !native_pack) {   // plan v0.3 P6: a native pack runs verify windows only`
- `src/program/generate.cpp:19830`: `if (native_pack) { spec_pos = pos; break; }` — token ループを抜け、
  **必ず** spec/verify ループへ入る（`--spec 0` でも `spec_pos >= 0` なので `spec_loop_runs` は真、
  `include/strata/spec/spec_start.hpp:14-16`）。
- `src/program/generate.cpp:20374`: `bool first_window = native_pack;` — 最初のウィンドウは最後のプロンプト
  トークン 1 行。
- `src/program/generate.cpp:9189`: `--gpu-stages` の失敗メッセージ
  "a native (IQ) pack has no per-layer graphs to replay. Its decode window's GPU stages are printed per request
  by the serve path instead"。

### 2.2 技術的な理由（コードから確認できる事実）
1. **native pack の GPU expert 計算は verifier にしか無い。**
   canonical pack の token graph の hit 経路は s2 カーネル（`moe_hit_select` / `quantize_q8_0_scaled` /
   `moe_hit_grouped_s2_dev`、`src/core/session.cpp:868-873`）。native pack の GPU expert は
   `native_expert_grouped` + `NativeExpertLayout`（`src/core/verify.cpp:2360-2372`）で、APR/Hadamard 折り込み
   （`verify.cpp:2313-2331`）と device plan（`verify.cpp:1296-1320`, `2248`）を伴う。
   `src/core/expert_source.cpp:5322-5324` のコメント:
   "a native pack's experts run in the verify windows, so this is the backstop, never the path"。
   -> `block_layer_pre` 側（token graph）には native expert の実装が無い。
2. **native pack の decode は元々 window 前提。** `Verifier` は T 行ぶんの per-group 計画
   （`plan_`, `p_ptr`, `p_start` など、`verify.cpp:2330-2400`）と、`T == 1` の自己 commit
   （`verify.cpp:1356` の `self_commit`）を持つ。T=1 も同じ graph で回る。
3. **canonical の `session_loop` は CPU pool 前提。** `--no-capture` と expert が排他
   （`generate.cpp:9010-9020` のコメント）。`session_token` には pool hook が無い。

## 3. native pack の decode の実際の起動数

| 処理 | file:line | 起動数 |
|---|---|---|
| ウィンドウ graph の capture | `src/core/verify.cpp:2748`（`record_window` を丸ごと捕捉） | 1 回（T ごと、初回のみ） |
| ウィンドウ graph の launch | `src/core/verify.cpp:3369`（`cudaGraphLaunch(exec_[sel_long_][T], cs_)`） | **1 / ラウンド** |
| prelaunch（任意） | `src/core/verify.cpp:3212` | 0-1 / ラウンド |
| commit graph の capture / launch | `src/core/verify.cpp:2914`, `3887` | 1 / ラウンド |
| グループごとの host 計画待ち（graph 内 node） | `src/core/verify.cpp:2388-2400`（`wait_flag_ge` / `copy_i32_from_mapped`） | グループ数 G ぶん（node） |
| CPU pool | `src/program/generate.cpp:20648` の `ver.run(..., decode_pool_fn, ...)` 経由 | host 側 |

- graph の node 数を数える仕組みは既にある: `STRATA_VERIFY_NODES=1`（`verify.cpp:2795-2830` で
  node 種別ごとの件数を stderr に出す）。**実測はしていない**（GPU 不使用の制約）。
- 参考: 既存レポートの decode は「1 window 52-61 ms、GPU-reach wait 36-40 ms、ホスト側 4.4-8.4 ms」。
  このホスト項の内訳が `--stats` の wait/pool/host/commit で見える。

## 4. 有効化した場合に壊れる/劣化する点（実装案 A の検討）

「`!native_pack` を外して `session_capture` + `session_loop` を native pack でも使う」案のリスク。

| 論点 | 何が起きるか | 回避策 | 検証 |
|---|---|---|---|
| graph に焼き込まれるポインタ | `block_layer_pre` は `ss.qsa_states[...]`, `ss.gdn_state`, `bb.*`, `db->*` の**アドレス**を捕捉する（`session.cpp:200-300`）。本エンジンは `SessionGraphs gr` を 1 組しか作らない（`generate.cpp:9006`）。セッションごとの graph は q36 の `GSet`（`src/qwen35/q36_engine.cu:1172-1175`）に相当する仕組みが本エンジンには無い | セッション数ぶんの `SessionGraphs` を持つ（未実装） | `tests/core/host_graph_launch_test.cu`、`--batch` で 2 セッション |
| expert ポインタ | native pack の GPU expert は `native_expert_grouped`（blob の device ポインタ）。token graph の hit 経路は s2 専用（`session.cpp:868-873`） | native 用の hit kernel を新設するか、verifier の grouped 経路を呼ぶ | `tests/kernels/native_expert_parity.cpp`, `native_grouped_parity.cpp` |
| KV ページ | QSA state の `page_table` / pool ポインタは捕捉時に固定。`--kv-resident 32768` の streaming は content を device で読むので graph 化自体は可能 | 変更不要（内容は device 読み） | `kv_stream_parity.cpp`, `kv_q8_parity.cpp` |
| PLE | host の hash/gather は graph の外（`layer.cpp:1611`）、device 半分は graph 内（`layer.cpp:1511/1525`）で canonical と同じ | 変更不要 | `ple_parity.cpp`, `--stats` の PLE |
| session 切替 | 本エンジンの `session_loop` は 1 セッションぶんの `SessionGraphs` しか持たない（`generate.cpp:9006`）。`--batch` の decode は verifier の slot 経路（`capture_batch`, `verify.cpp:3981`）で、`session_loop` とは別 | 未確認（canonical の batch で session_loop が使われるか） | `tests/core/batch_rows_test.cpp`, `verify_batch_parity.cpp` |
| spec/verify との共存 | `session_loop` と `Verifier` が `ss.R` / doorbell / `d_parts` を奪い合う。verifier の commit と session_loop の state 更新は別経路 | どちらか一方に統一する必要がある | `spec_verify_parity.cpp`, `verify_parity.cpp` |
| 出力一致 | native expert の丸め（APR 折り込み、q8_1 rows）を token graph 側で再現しないと**静かに違う** | verifier のコードを共有する | `--dump-layers`（`session.cpp:640-670`）と logits 比較 |
| 速度 | 1 launch -> 96 launch、かつ window の device plan を失う | - | `--stats` の LAYERS と wait/pool |

## 5. 実装案（2 つ）

### 案 A: 条件を緩めて canonical の graph 経路を native pack にも使う（非推奨）
- 変更: `generate.cpp:9000` の `!native_pack` を外す。`generate.cpp:9518-9522` の token graph 条件も外す。
  `session.cpp:868-873` の hit ノードを native 版に差し替える（新規カーネル or verifier の grouped 呼び出し）。
- 出力一致: verifier と同じ native expert 演算を呼べば原理的には一致。ただし `block_layer_pre` の
  `moe_route` は canonical の router/doorbell 経路で、native pack の APR/Hadamard 折り込みは verifier 側にしかない
  （`verify.cpp:2313-2331`）。移植が必要。
- 推定効果: **マイナス**。graph launch 1 -> 96、device plan 喪失。実装コストも高い。
- 必要なテスト: `host_graph_launch_test.cu`、`native_expert_parity.cpp`、`verify_parity.cpp`、`--dump-layers`。

### 案 B: verifier の host 項を削る（本命。C1 の実質）
- 対象: `verify.cpp:2388-2400` のグループ計画待ち、`generate.cpp:20810` の wait/pool/host/commit。
- 既存スイッチで試せるもの:
  - `STRATA_VERIFY_ALL_RESIDENT=1`（既定 on、`verify.cpp:1267-1291`）: 全 expert が VRAM にある層だけ
    zero-doorbell graph。P40 24GB では全 resident にならないので効かない（推定）。
  - `STRATA_VERIFY_DEVICE_PLAN=1`（既定 off、`verify.cpp:1292-1296`）: 層ごとの計画を device で作る。
    コメントに「exact, but neutral on RIBPC 1-2 GPUs: off by default」とあり、**測定済みで中立**。
  - `STRATA_QFUSE=1` / `STRATA_VERIFY_QDEDUP=1`（既定 off、`verify.cpp:180/292`）: 入力の q8_1 を
    生産側で書く（後述 §7）。
  - `STRATA_LFUSE` / `STRATA_LFUSE_PAIR` / `STRATA_LFUSE_GATE`（既定 on、`verify.cpp:179-180`, `1376-1378`）:
    既に有効。
- 新規実装の候補: グループ数 G を減らす（`Verifier::run` の `G`）、計画の H2D を 1 回にまとめる、
  `wait_flag_ge` を group 先読みで 1 node にまとめる。**効果は未測定**。
- 必要なテスト: `verify_parity.cpp`、`batch_rows_parity.cpp`、`resident_plan_parity.cpp`、`--stats` の
  "verify window" 行。

### 案 C（補助）: T=1 専用の軽量 graph
- `Verifier` は `T == 1` の self commit を持つ（`verify.cpp:1356`）。最初の 1 トークンや `--spec 0` の
  1 行ウィンドウはこの経路で足りる。追加の graph を作らずに済む（既存）。
- 効果: 初回ウィンドウの capture を 1 回減らす程度。**小**。

## 6. `optimization-candidates.md` C1 の訂正

前回の記述:
> C1: native pack でも単一トークン graph を使う（96 graph launch -> 1）

訂正:
- 96 graph launch は **canonical pack**（`session_loop`, `session.cpp:587`）の数字。
- native pack（本番）は `Verifier` の **1 window = 1 graph launch**（`verify.cpp:2748/3369`）で、
  C1 の目的は既に達成されている。
- `!native_pack` は「native pack は verify window だけで decode する」設計（plan v0.3 P6、root commit
  `01529aaa` から存在）。外すと退行する。
- したがって C1 は「不能/不要」。残る作業は **案 B**（verifier の host 項の削減）に置き換える。

## 7. C2-C5 の実装サイトとパリティ（コードで確認済み）

**重要な訂正:** C3/C5 は本番（native pack の verifier）で既に有効。C2/C4 は既存の opt-in スイッチとして
実装済みで、既定 off。したがって「新規カーネルを書く」のではなく「既存スイッチの評価 + canonical 経路への拡張」が作業になる。

| # | 実装サイト（本番経路 = verifier） | canonical 経路（未実装の残り） | パリティテスト | ビット一致の見込み | 順序 |
|---|---|---|---|---|---|
| C2 | `verify.cpp:1292-1320`（`device_plan_`）、`2236`（`native_route_resident`）、`2248` | `session.cpp:868-873`（s2 hit nodes） | `resident_plan_parity.cpp`, `route_resident_parity.cu`, `s2_expert_grouped_parity.cpp` | 実装済み・コメントで「exact」 | 1（env A/B のみ） |
| C3 | `verify.cpp:2301-2302`（`shared_expert_multi` + lfuse pair）、`shared_expert.cu:225-231` | `layer.cpp:499` -> `shared_expert.cu:436-445`（単一トークン） | `shared_expert_parity.cpp` | コメントで「bitwise the two calls」 | 2（canonical のみ） |
| C4 | `verify.cpp:180/292`（`STRATA_QFUSE`/`QDEDUP`）、`2313-2331`（`quantize_q8_1_rows`）、`1601/2176`（`gr_read_group` が q8_1 を書く） | `layer.cpp:276,376`（`native_quantize_q8_1`） | `quantize_act_parity.cpp`, `elementwise_parity.cpp`, `verify_parity.cpp` | 既存の fused は「the same bytes」とコメント（`verify.cpp:181`） | 3（env A/B -> canonical 拡張） |
| C5 | `verify.cpp:1815/1860/1934/1946`（`native_qsa_rms_norm_rope`） | `layer.cpp:1147-1149`（norm + rope の 2 発） | `qsa_nr_parity.cu`（prompt 経路の融合は bit-for-bit）, `rope_parity.cpp`, `qsa_parity.cpp` | decode 用 `native_qsa_rms_norm_rope` の bit 一致は未確認（`qsa_nr_parity` は `rms_rope_rows` 側） | 4（canonical のみ） |

- C2 の既定 off 理由はコード内の測定メモ（`verify.cpp:1294-1296`）で「neutral on RIBPC 1-2 GPUs」。
  P40 での効果は**未測定**。
- C4 の `STRATA_QFUSE` は「activation q8_1 images written by their producers (the GDN output norm)」
  （`verify.cpp:181`）。既定 off なので、まず env で A/B する。
- C5 の `qsa_nr_parity.cu` は **prompt 経路**の `rms_rope_rows`（`STRATA_QSA_FUSE_NR`）を対象にしており、
  decode 経路の `native_qsa_rms_norm_rope` とは別関数。混同しないこと。

## 8. `STRATA_FUSE_DECODE=1`（C9）の実態

- 定義: `include/strata/kernels/fuse_decode.hpp:9`（env のみ、既定 off）。
- 呼び出しサイト（`grep -rn "fuse_decode()" src/`）:
  - `src/core/layer.cpp:265`（GDN: `quantize_act_images` 1 パス）
  - `src/core/layer.cpp:327`（GDN: `project_bf16_pair`）
  - `src/core/layer.cpp:375`（ssm_out 入力の q8_1 を 1 パス）
  - `src/core/layer.cpp:484`（shared expert 入力の 3 種を 1 パス）
  - `src/core/layer.cpp:1166`（QSA: `quantize_act_images`）
  - `src/core/layer.cpp:1198`（QSA: indexer の q/k を `project_bf16_pair`）
  - `src/kernels/cuda/shared_expert.cu:336`（`s_gemv_pair_silu`、`!native_gate && !native_up` のときだけ）
  - `src/kernels/cuda/s2_expert_grouped.cu:613, 1309`（s2 hit 経路の融合）
- **本番（native pack の verifier）では `fuse_decode()` を呼ぶ箇所が無い**（`src/core/verify.cpp` に出現しない）。
  したがって C9 は本番 decode に対して **no-op**（効果ゼロ）。canonical の `session_loop` と s2 hit 経路にだけ効く。
- 前回の「層あたり 2-5 起動削減（100-240/token）」は canonical 経路の推定であり、本番には当てはまらない。訂正。

## 9. `--stats` で取れる項目（コードで確認済み）

| 項目 | file:line | 内容 |
|---|---|---|
| per token | `generate.cpp:20975` | 壁時計 ms/token（embed, layers, head, sample を含む） |
| token host phases | `generate.cpp:20982-20987` | PLE / embed / LAYERS / head / readback / sample と合計 |
| PLE io report | `generate.cpp:20989` | PLE テーブルの I/O |
| CPU expert pool | `generate.cpp:20996-20998` | ms/token、positions、dispatches |
| pool phases | `generate.cpp:21006` | wait-park / drain / re-park |
| expert blobs | `generate.cpp:21009` | 読んだ blob 数 |
| per-GPU routed entries | `generate.cpp:21012` | ルーティングされた entry 数 |
| hit work DONE | `generate.cpp:21020` | pool が返った時点で GPU の hit 計算が終わっていた層数 |
| cache hit rate | `generate.cpp:21022` | admitted / refused / 充填率 |
| KV cells | `generate.cpp:21031` | VRAM セル数、grow/trim、expert の退避 |
| wait for rings / pool | `generate.cpp:21036` | ms/token、flush 回数 |
| rings seen MID-GRAPH | `generate.cpp:21046` | オーバーラップの指標 |
| ring latency / after the ring | `generate.cpp:21049`, `21059` | 層あたりの前半/後半 |
| **verify window** | `generate.cpp:20810` | wait for rings / pool / host / commit ms/round、CPU experts 数 |
| **pool multi** | `generate.cpp:20816-20819` | gate/up, quantize, down ms/round、GB/s |
| window cost（T 別） | `generate.cpp:20795-20797` | T ごとの ms/round（`STRATA_SPEC_T_CYCLE`） |
| GPU stages per window width | `generate.cpp:20799-20804` | `STRATA_VERIFY_PROFILE` / `STRATA_ROUTE_STATS` |

- 本番（native pack）の decode は spec ループなので、**"verify window" 行が C1/案 B の主指標**。
  "token host phases" は canonical 経路の指標。

## 10. 未確認
- `STRATA_VERIFY_NODES=1` で数えたウィンドウ graph の実 node 数（GPU 不使用のため未取得）。
- `STRATA_VERIFY_DEVICE_PLAN=1` の P40 での効果（コード内の測定は RIBPC の 1-2 GPU）。
- `STRATA_QFUSE=1` / `STRATA_VERIFY_QDEDUP=1` の P40 での効果とビット一致（env で A/B 可能）。
- 案 B の「グループ数削減」の実装可能性（`record_window` の G の決め方は未読）。
- canonical 経路（`session_loop`）が本番で使われる構成があるか（`--pack` 無しの canonical 運用時のみ）。
