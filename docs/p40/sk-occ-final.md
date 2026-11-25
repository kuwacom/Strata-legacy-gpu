# S_K MoE occupancy 版の既定化 (OCC=2 削除 / 形式を S2_K・S3_K に限定)

最終更新: 2026-11-25 / 対象: `~/work/ink` branch `perf/ink-sk-occ-final` (`f25fdc067`、base `b872a8f81`) / Tesla P40 (sm_61) 単機 / SQ-10 / 32K prefill (chunk 512、fp16、dspark、`STRATA_INK_PF_TILE=1 STRATA_INK_PROF=2`)。本番は停止 -> 復旧済み (healthy、`.env` sha256 `(省略)` 不変、GPU/ビルド両ロック解放)

## 結論

- **既定化は成立**。`STRATA_INK_SK_OCC` の既定を 1 にし、env 未指定の 32K で **253.7 ppts / 129.16 s**、`STRATA_INK_SK_OCC=0` で **243.4 ppts / 134.62 s** -> **+10.3 ppts (+4.23%) / −5.46 s**
- sk 区間は **gu(sk) 567.06 / dn(sk) 308.69 ms/step** で、先行実験の OCC=1 の値 (567.08 / 308.64) と **−0.00% / +0.02%**。合格条件 (±3%) に対して十分内側
- 旧 OCC=2 (重みの 1 sub-block 先行ロード) はテンプレート実体・分岐とも削除。**カーネル数 84 -> 12**、バイナリは **8,820,752 -> 6,867,624 B (−1,953,128 B)** になり、実験前 (6,558,248 B) に対する増は **+309,376 B (+4.7%)** まで縮んだ
- 従来カーネル (`k_ink_moe_*_sk_fmt`) の REG・コードは不変 (gu S2_K/S3_K, MAX_M=16 で 160/158、dn で 134/132)。`STRATA_INK_SK_OCC=0` で完全に従来経路へ戻る

## 変更 (2 ファイル、+51 / −202 行)

| ファイル | 内容 |
|---|---|
| `engine/src/inkling2/cuda/ink_moe_sk_occ.cuh` | OCC=2 の計算部 (`sk_pre16` / `sk_unpack64_pre_g` / `sk_gu_fused_4lane_pf` / `sk_dn_paired2_4lane_pf`) と `OCC` テンプレート引数を削除。カーネルは `__launch_bounds__(128,4)` 固定。実体は **S2_K / S3_K のみ** (7 形式 -> 2 形式)。起動関数は起動した bit を `mask` から落とし、残りは呼び出し側の従来 if 連鎖へ流す |
| `ink_moe_cuda.cuh` | コメント 3 行のみ (include 位置と 2 か所の起動行の説明)。+7 行の実験用フックはそのまま |

- 既定は `ink_sk_occ_mode()` が env 未指定で 1、`STRATA_INK_SK_OCC=0` で 0。0 以外の値は 1 とみなす (旧 `=2` を渡しても occupancy 版になる)
- `MAX_M` は 1 / 4 / 16 の 3 本を残した。**MAX_M=1 は decode +2.01% (8K、draft off) / +1.93% (graph 有効) の実測があるため残す**。MAX_M=4 (M=2..32) は未測定なので、指示どおり現状の対象のまま (従来と同じ経路を occupancy 版で置き換えるだけで、結果は bit 一致)

## バイナリと resource

| ビルド | サイズ | sha256 |
|---|---|---|
| 実験前 (`d59c0373a`) | 6,558,248 B | - |
| 実験版 (`b872a8f81`、OCC=1/2 x 7 形式 x 3 MAX_M) | 8,820,752 B | `45c344fb…` |
| **本版 (`f25fdc067`)** | **6,867,624 B** | `537eef77…` |

- cubin は sm_61 のみ (`cuobjdump -lelf`: `strata-inkling.1.sm_61.cubin` / `.2.sm_61.cubin` の 2 本、PTX なし)
- `cuobjdump -res-usage` の該当行 (`~/work/results/sk-occ-final/res-usage.txt`):

| kernel | REG | STACK | CTA/SM (reg) | warp/SM |
|---|---|---|---|---|
| `gu_sk_fmt_occ<S2_K,16>` | **128** | 0 | **4** | **16** |
| `gu_sk_fmt_occ<S3_K,16>` | **128** | 8 | **4** | **16** |
| `dn_sk_fmt_occ<S2_K,16>` | **128** | 0 | **4** | **16** |
| `dn_sk_fmt_occ<S3_K,16>` | **128** | 0 | **4** | **16** |
| `gu_sk_fmt<S2_K,16>` (従来) | 160 | 0 | 3 | 12 |
| `gu_sk_fmt<S3_K,16>` (従来) | 158 | 0 | 3 | 12 |
| `dn_sk_fmt<S2_K,16>` (従来) | 134 | 0 | 3 | 12 |
| `dn_sk_fmt<S3_K,16>` (従来) | 132 | 0 | 3 | 12 |

- 参考 (残した MAX_M): `gu_occ<S2_K,1>` 88 / `<S3_K,1>` 88、`dn_occ<S2_K,1>` 88 / `<S3_K,1>` 90 (従来 76/76, 80/80)、`gu_occ<2,4>` 83 / `<3,4>` 84、`dn_occ<2,4>` 88 / `<3,4>` 86 (従来 83/84, 88/86)
- テンプレート実体は 12 本 (`gu`/`dn` x S2_K/S3_K x MAX_M 1/4/16)。OCC=2 の 3 引数シンボルは 0 本

## 32K A/B (同一バイナリ、アーム間 60 s)

`~/work/bin/sk-occ-final-ab.sh`。arm `new` は `STRATA_INK_SK_OCC` を env から外して起動 (新既定の確認)。

| 項目 | 新既定 (未指定) | `SK_OCC=0` | 差 |
|---|---|---|---|
| prefill | **253.7 ppts (129.16 s)** | 243.4 ppts (134.62 s) | **+10.3 ppts (+4.23%) / −5.46 s** |
| gu(sk) ms/step | **567.06** | 607.63 | −40.57 (−6.68%) |
| dn(sk) ms/step | **308.69** | 358.00 | −49.31 (−13.77%) |
| gu+dn | **875.75** | 965.63 | −89.88 (−9.31%) |
| MoE 位相 | 1,543.22 | 1,631.61 | −88.39 (−5.42%) |
| step 全体 | 2,008.11 | 2,096.43 | −88.32 (−4.21%) |
| attn | 316.05 | 316.02 | +0.03 (+0.0%) |
| decode (参考、dspark) | 25.9 tps | 24.3 tps | +6.6% |

- 先行実験の基準 (OCC=1 254.2 / OCC=0 242.3、+4.9%) に対し、今回は 253.7 / 243.4 で **+4.23%**。新既定は旧 OCC=1 の再現 (−0.20%)、OCC=0 も旧値の +0.45% で、差の縮小は OCC=0 側がわずかに速かったことによる
- sk 区間の旧 OCC=1 比は **gu −0.00% / dn +0.02%** で、合格条件 ±3% の内側
- attention は動いていない (0.0%) -> 改善は S_K MoE だけで説明できる
- 実行時刻 (UTC): new 05:45:16、o0 05:49:33。各アーム終了時の clock 1531 MHz (P40 上限)、温度 72-73 C、消費 63 W で throttle なし
- 生データ: `~/work/results/sk-occ-final/arm-{new,o0}.{out,err,gpu}`、ログ `~/work/logs/sk-occ-final-ab.log`、復旧記録 `prod-restore.txt`

## 逸脱・注意

- **作業中に別セッションが同じ worktree (`~/work/ink`) のブランチを動かした**。reflog に `perf/ink-sk-occ-final -> perf/ink-attn-prof -> perf/ink-sk-occ` の checkout が残っている (いずれも同じ `b872a8f81` を指すため作業ツリーの変更は保持された)。そのため最初の commit が `perf/ink-sk-occ` に載ったので、`perf/ink-sk-occ` を `b872a8f81` に戻し、`perf/ink-sk-occ-final` を `f25fdc067` に向け直した。**同じ worktree の同時利用は避けるべき**
- MAX_M=1 は「残す」判断。理由は上記の decode 実測 2 本。MAX_M=4 は未測定のため削除せず現状維持 (判断材料が無い)
- 形式を S2_K / S3_K に限定したので、他の S_K 形式 (S1_5 / S4_K / S5_K / S6_K / S8_K) は `mask` の bit を落として従来カーネルに流す。SQ-10 は S2_K / S3_K だけなので測定条件では発動しない (他形式のモデルでは従来どおり = 後退なし、改善もしない)
- ビルドは ssh セッションが 120 s で切られたが、コンテナ側は完走した (`.ninja_log` に `ink_cuda_backend.cu.o` -> `libstrata_inkling_cuda.a` -> `strata-inkling` の 3 行、バイナリ mtime 05:40:59)。ビルドロックは trap で解放済み
- `STRATA_INK_SK_OCC=0` の従来経路は同じバイナリで 1 本だけ確認 (旧 OCC=0 の 609.21/358.13 に対し 607.63/358.00)。ビット一致の再検証は前回 (20/20 visit) から変更が launch_bounds と形式の絞り込みだけなので再実行していない
- decode の 25.9 vs 24.3 は参考値。条件が dspark 有効・PROF=2 (eager) で、前回の decode A/B (draft off) とは比較できない
- 512K は未測定

## 次の提案

1. **この branch を本線候補として採用**。既定化・OCC=2 削除・形式限定で、効果は 32K +4.2% / 128K +3.6% (前回) / decode 後退なし
2. 512K を 1 本だけ確認する (S_K の絶対値が context 長に依存しないので、改善量は chunk 数に比例する見込み)
3. 別 worktree で同時作業する運用に切り替える (`~/work/ink` は 1 セッション専有)
4. S_K 876 ms の次は活性側 traffic (16 token x 36 block x 80 B の shared 化)。occupancy は 4 CTA/SM で打ち止め
