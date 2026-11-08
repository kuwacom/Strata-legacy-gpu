# CPU expert pool の 1 層先読み（FreeToken #677 型）

作業ブランチ: `perf/cpu-lookahead`（fork HEAD 9f7d977d）。作業ツリー `~/work/wt/lookahead`。
対象: P40 の decode window で GPU が CPU expert pool の計算を待つ時間（profile の `waitCPU`）。

**結論: 実装は動くが効果なし。本番相当の A-B-A で decode +0.8%（A 同士の差 0.2%、ブロック間の揺れ 7%）、
waitCPU と pool の CPU 時間は不変。採用しない。**

## 1. CPU pool の現在の流れ

decode は verify window（`--spec 4`、T は 1..6）で回る。1 層あたりの順序は次のとおり。

| # | 場所 | 何が起きるか |
|---|---|---|
| 1 | `src/core/verify.cpp:2262` `doorbell_publish_res` | GPU が router GEMV + top-10 を終え、`xm`（MoE 入力）・`ids`・`weights` を mapped pinned host memory（`m_x_`/`m_ids_`/`m_w_`）へ書き、`m_seq_` に ring 値を置く |
| 2 | `src/core/verify.cpp:2275` `stamp(l, 17)` | router 完了。ここから GPU は共有 expert（`stamp(l,18)` = `:2327`）へ進む |
| 3 | `src/core/verify.cpp:5251` `Verifier::service` | host の service ループ。`h_seq_` をポーリングし、ring が来たら `pool(...)` を呼ぶ |
| 4 | `src/core/verify.cpp:5318` → `src/core/expert_source.cpp:4865` `expert_pool_dispatch_multi` | 1 層ぶんの予約・発行・計算 |
| 4a | `src/core/expert_source.cpp:4880` `d.lookahead->submit(...)` | RouterLookahead に「層 l+1 の expert 予測」を依頼（非同期、別スレッド） |
| 4b | `:4899`..`:5091`（`c0`..`c1`） | distinct expert を VRAM hit / PCIe / CPU に分類し、`P.publish` (`:5046`) で **plan を GPU へ publish**（`m_flagA_`）、`P.fetch` で PCIe DMA を開始 = `ms_plan` |
| 4c | `:5105`（`c1`..`c2`） | 活性化の量子化 = `ms_actq` |
| 4d | `:5272`（`c2`..`c3`） | `ExpertJobMulti` の構築（CPU expert のみ）= `ms_jobs` |
| 4e | `:5277` `d.pool->run_split_multi(...)`、`:5296`（`c3`..`c4`） | pool の実計算 = `ms_run` |
| 5 | `src/core/verify.cpp:2397` `wait_flag_ge(m_flagA_)` | GPU が plan を待つ = `waitA`（`stamp(l,19)` `:2382`） |
| 6 | `src/core/verify.cpp:2384` `stamp(l,20)` | VRAM 常駐 expert の計算（`grouped`） |
| 7 | `src/core/verify.cpp:2407` / `:2418` | PCIe 分の待ち / 計算（`waitB` / `PCIe grp`） |
| 8 | `src/core/verify.cpp:2429` `wait_flag_ge(m_flag_, ring, cs)` | **GPU が CPU の行を待つ = `waitCPU`**（`stamp(l,23)` `:2431`）。`fuse_rows` 有効時は `copy_rows_from_mapped_after` に折り込まれる（`:2433`） |
| 9 | `src/core/verify.cpp:2479` `stamp(l,24)` | copy + combine |

pool 側は `src/kernels/cpu/pool.cpp:1326` `run_split_multi` が **phase 3（gate/up 行）と phase 4（down 行）の 2 回** `run_phase`（`pool.cpp:1295`）を回す。各 phase は
`wait_parked`（`pool.cpp:1128`）→ `begin_batch`（`pool.cpp:1102`）→ drain → `wait_done`（`pool.cpp:1149`）→ `wait_parked`。
worker は `kSpinBeforeSleep` = 20 ms（`include/strata/kernels/cpu/pool.hpp:282`）までスピンし、その後 futex sleep（`pool.cpp:991`）。
つまり 1 層あたり **2 回の wake/park サイクル**があり、これが job 数に依らない固定費になる。

## 2. なぜ既存の RouterLookahead が CPU pool に効かないか

`RouterLookahead::run()`（`src/core/expert_source.cpp:3199`）は予測した expert を `src_->warm(...)`（`:3002`）/ `warm_scored`（`:1127`）に渡すだけ。

1. `FileExpertSource::warm` は `complement_ready_ && resident_blob(index) != nullptr` なら即 return（`:3016`-`:3018`）。
   P40 の CPU expert は RAM arena（pinned/mlocked）に常駐しているので、**warm は no-op**。
2. warm が実際にやるのは `madvise(MADV_WILLNEED)` か pread による **file tier（experts.bin）のページ先読み**だけ（`:3020`-`:3044`）。
3. `RouterLookahead::start` は `src->warms()` を要求する（`:3071`）。CPU pool の job queue（`ExpertPool`）には一切触らない。
4. CPU pool の job は `expert_pool_dispatch_multi` が **現層の ids**（`m_ids_`、`stamp(l,17)` でしか確定しない）から `d.jobs_multi` を作るときにしか発行されない（`:5272` 付近）。

したがって予測は「ファイルのページ先読み」にしかならず、`waitCPU` には効かない。ここが候補 #3 の隙間。

## 3. 実装（最小、40 行、1 ファイル）

`src/core/expert_source.cpp` のみ。`STRATA_POOL_PREFETCH=1` で opt-in（既定 off、パリティは自明に保たれる）。

- `pool_warm_ahead()`: 予測された次層の expert について `FileExpertSource::resident_blob(layer, e)`（`include/strata/core/expert_source.hpp:825`、read として数えない）を取り、blob を 64 バイトずつ読んで捨てる。1.38-2.33 MB の blob を 1 コアで逐次読むと、そのコアの L2（4 MiB）と共有 L3（16 MiB）に載る想定。
- 予算は `STRATA_POOL_PREFETCH_MB`（既定 8 MiB）。
- 呼び出しは `RouterLookahead::run()` の `dj == 0` の末尾。`submit()` は dispatch の先頭（`:4880`）なので、この read-ahead は **現層 l の pool 計算と並行**に走り、次層 l+1 の pool が読むときには L3 に載っている想定。
- 新しいスレッド・キャッシュ・ストリームは作らない。file tier の既存先読み（`warm`/`warm_scored`）は触らない。
- 出力に影響しない（読むだけ。expert の選択・重み・計算は不変）。

パッチ: `patches/cpu-lookahead.diff`（58 行、`src/core/expert_source.cpp` のみ）。
イメージ: `strata:fork-9f7d977-lookahead`（engine 0.1.57、sm_61、ビルド 519 s、パッケージ 97 s、イメージ 377 s）。

## 4. 切り分け実験（base イメージ、P40、512K、spec=4、BENCH_SEED=42）

`STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1` で 4K prose を 2 回。`--expert-cache auto`（10315 slots）と `--expert-cache 4800`（60% 構成）を比較。
数値は 1 リクエストぶんの per-window 平均（`strata decode timing` と `strata decode GPU stages` の最終行）。waitCPU は GDN 層 + QSA 層の合計。

| expert-cache | decode ms/window | CPU ms/window | entries/layer | waitCPU ms | waitA ms | waitB ms | VRAM hits ms |
|---|---|---|---|---|---|---|---|
| auto (10315) #1 | 57.80 | 7.60 | 1.19 | **3.28** | 2.01 | 1.70 | 10.45 |
| auto (10315) #2 | 51.86 | 3.65 | 0.46 | **0.91** | 2.03 | 1.91 | 9.57 |
| 4800 (≈6240) #1 | 69.64 | 20.01 | 3.31 | **10.61** | 2.05 | 6.75 | 8.72 |
| 4800 (≈6240) #2 | 62.42 | 10.24 | 1.46 | **2.05** | 2.10 | 8.12 | 9.27 |

読み方:

- **waitCPU は CPU expert 数（= pool の計算量）にほぼ比例する。** entries/layer 0.46 → 1.19 → 1.46 → 3.31 に対し waitCPU 0.91 → 3.28 → 2.05 → 10.61 ms。pool の CPU 時間 3.65 → 7.60 → 10.24 → 20.01 ms と同方向。
- **waitA は expert 数に依らず約 2 ms で一定。** これは「host が ring に気づいて plan を publish するまでの固定遅延」で、予測では消せない（router の結果が要る）。
- expert-cache を 60% に落とすと decode は 41.15 → 34.45 tok/s（−16%）と 38.93 → 36.07（−7%）。**CPU expert が増えると decode が落ちる**ので、pool は確かに critical path 上にいる。
- ただし **本番構成（auto, 10315 slots）では entries/layer が 0.46-1.19、waitCPU は 0.9-3.3 ms/window（全体の 1.8-5.7%）**。研究ノートの 3.04 ms（6.2%）は IQ3_S・8225 slots・2.37 entries/layer の別構成の値で、この構成の実測ではない。
- waitCPU が pool の計算量に比例することは「予測で消せる固定待ち」ではなく **pool の処理（DRAM 帯域）そのもの**であることを示す。したがって効くのは「同じ計算を速くする」方向だけ。

## 5. A-B-A（本番相当: expert-cache auto、profiler off）

`strata:fork-9f7d977-base`（A）/ `strata:fork-9f7d977-lookahead` + `STRATA_POOL_PREFETCH=1`（B）。
512K、spec=4、BENCH_SEED=42、4K prose を各 3 回、1 本目除外、中央値（2 本の平均）。
engine の実効設定は全アーム `--expert-cache auto`、`/proc/<engine>/environ` に `STRATA_VERIFY_PROFILE` なしを確認済み。

| アーム | r1 | r2 | r3 | 中央値 (r2,r3) |
|---|---|---|---|---|
| Q-A1 (base) | 45.31 | 42.67 | 41.78 | **42.23** |
| Q-B (patch) | 46.11 | 42.46 | 42.54 | **42.50** |
| Q-A2 (base) | 45.50 | 42.34 | 41.91 | **42.13** |

A の平均 42.18 → B 42.50 = **+0.77%**。A 同士の差は 0.24%。

機構の測定（`STRATA_DECODE_TIMING=1`、profiler off）: B の pool の CPU ms/window と waitCPU は A と同じ範囲。

**参考: 60% 構成 + profiler ON での A-B-A**（state 汚染により `--expert-cache 4800` と `STRATA_VERIFY_PROFILE=1` が混入したブロック。
A-B-A の形は同じなので相対比較は成立する）:

| 構成 | A1 base | B patch | A2 base | A 平均 | B の差 |
|---|---|---|---|---|---|
| 4K code | 41.20 | 41.65 | 41.21 | 41.20 | +1.1% |
| 4K prose | 33.19 | 34.17 | 34.13 | 33.66 | +1.5% |
| 24K code | 41.58 | 42.36 | 41.86 | 41.72 | +1.5% |
| 24K prose | 34.82 | 35.40 | 35.78 | 35.30 | +0.3% |
| 平均 | | | | | **+1.1%** |

このブロックの A-A の揺れ: 4K code 0.0%、4K prose +2.8%、24K code +0.7%、24K prose +2.8%。**B の差は A-A の揺れより小さい**。
pool の CPU ms/window（12 リクエスト平均）: A1 10.89 / B 10.74 / A2 10.66。waitCPU: A1 1.726 / B 1.752 / A2 1.723。
**pool の CPU 時間も waitCPU も動いていない。**

prefill（24K、同ブロック）: code A 494.5 → B 493.0（−0.3%）、prose A 494.5 → B 492.2（−0.5%）。

**順序の交絡について**: 別ブロックで base と patch を 2 アームだけ比べたところ B が +6.3% 高く出たが、
同じ base イメージでもブロック間で 39.35 と 42.23（−7%）と動くため、これはブロック間ドリフトと判断した。
A-B-A を取った上の 2 ブロックでは +0.8% / +1.1% で、いずれも A-A の揺れの内側。
（64 MiB 予算の prefetch も B と同じ値で、予算は律速していない = `want` が小さいため判定に使えない。）

## 6. 品質・パリティ

- CPU expert の bitwise パリティ: `pool_tasks_test` 360 比較すべて一致（`RC=0`）。`pool_test` は CPU に AVX512-VNNI が無いため SKIP。`expert_layout_test` はデータ未指定で usage 終了（未実行）。パッチは pool に触らないので当然の結果。
- 固定 4 プロンプト（`~/opt-work/src-build/answers.py`、greedy 200 tokens）を各アームで 2 回:
  - A1 内 2 回: 3/4 一致、A2 内 2 回: 3/4 一致、**A1 vs A2（同一イメージ・別起動）: 4/4 一致**
  - B 内 2 回: 2/4 一致、A1 vs B: 2/4、A2 vs B: 2/4
  - B の自己一貫性（2/4）と A-B の差（2/4）が同じ大きさなので、差をパッチに帰属できない。
  - 原因（推定）: エンジンは adaptive tier（expert swap）を実測時間で決めるため、タイミングが変わると expert 配置が変わり出力が変わる。パッチ自体は読み取りのみで数値を変えない。
- logits KL は未測定（パッチが数値を変えない設計であることと、上記のタイミング依存で KL を分離できないため）。

## 7. 判定

**採用しない。**

- 本番相当の A-B-A: decode **+0.77%**（A 同士 0.24%、ブロック間 7%）。5% 閾値に届かない。
- 60% 構成（pool がより重い条件）でも +1.1% で、pool の CPU 時間と waitCPU は不変。
- pool が最も重い条件で効果ゼロなので、pool の負荷が小さい本番構成で正になることはない。
- 実装コストは小さい（40 行）が、opt-in で既定 off のまま持つ理由もない。

**なぜ効かなかったか（推定、未確認を含む）**:

1. prefetch は 1 コアで数 MB を逐次読みする。1 層あたり pool の計算は 0.3 ms 程度しかないので、read-ahead は pool と DRAM 帯域を奪い合い、相殺する。実測で pool の CPU 時間が変わっていないのはこの相殺と整合する。
2. 共有 L3 は 16 MiB（E5-2680 v4、inclusive）。次層 + 現層の blob で L3 をほぼ埋め、19 worker が各自 1/19 を読む形では L3 ヒットに転ばない可能性がある。
3. `resident_blob` が非 null を返すことは直接計測していない。起動ログの arena ロード（39.97 GiB）から `complement_ready_` は true と推定されるが、**prefetch が実際に何バイト読んだかは未確認**。
4. 予測（`want`）は「VRAM に無い expert」だけを含む。hit 率 94% では 1 層あたり 1-3 個しかなく、予測が外れれば 0 個になる。予測精度（recall）も未測定（`STRATA_LOOKAHEAD_STATS` の出力がログに現れなかった）。

**代わりに効く候補**（切り分けの数値から）:

1. **pool の固定費を削る**: `run_split_multi` は 1 層 2 phase × 48 層で wake/park を 96 回する。entries/layer 0.46 でも CPU 3.65 ms/window（= 76 µs/層）かかっており、job が少ない層は固定費が支配的。2 phase を 1 wake にまとめる／phase 間で park しない、が最短。見込みは waitCPU の小さい側（0.9-2 ms/window）で 1-2%。
2. **waitA（約 2 ms/window、expert 数に依らず一定）**: host が ring に気づいて plan を出すまでの遅延。service ループのポーリング粒度と `ms_plan`（0.22-0.25 ms/window）が候補。ここは確実に削れる。
3. **PCIe share の調整**: 60% cache では waitB が 6.7-8.1 ms/window に膨らむ。`STRATA_PCIE_MIN1` / `STRATA_PCIE_BALANCE` で CPU と PCIe の分担を実測 fit にする（既存 opt-in、未 A/B）。本番構成では PCIe 0.15-0.17 experts/layer と小さいので効果は限定的。

## 8. 付録: 測定の汚染と復元

- 共有 preset `/data/config/strata/launch-presets/docker.json` の `extra` に、先行タスクと本タスクの `--set` が蓄積していた
  （`--expert-cache 4800`, `STRATA_DECODE_TIMING`, `STRATA_LOOKAHEAD_STATS`, `STRATA_POOL_PREFETCH`, `STRATA_VERIFY_PROFILE`）。
  これは `/data/state` の外にあり全コンテナで共有されるため、fork も本番も継承した（profiler ON / expert-cache 4800 の原因）。
- `strata launch --delete-preset docker` ではこのファイルは消えない。各アーム起動の直前に本番ベースライン
  （`--ple-io ram`, `--prefill auto`, `--rope-scale 2`, `--rope-scaling yarn`, `--spec 4`, `--spec-min-p 0.5`, `STRATA_DENSE_F32=1`）へ戻す手順を入れて測定した。
- 本番も同じ汚染を継承して起動していたため、`docker.choices.json` / `docker.json` / `launch-presets/docker.json` から該当キーを除去し、
  `strata launch --delete-preset docker` → compose `up -d --force-recreate` で作り直した。
  復元後: `--expert-cache auto`、`--max-context 524288`、`STRATA_*` は `STRATA_DENSE_F32=1` のみ、healthy。
- バックアップ: `~/work/results/lookahead/docker.choices.json.bak-1107-034314`、`docker.json.bak-1107-034314`。
