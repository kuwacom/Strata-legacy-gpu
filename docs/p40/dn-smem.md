# S_K dn の活性 shared ステージ (STRATA_INK_SK_DN_SMEM)

調査日: 2026-11-25 / 対象: `~/work/ink-ap` branch `perf/ink-dn-shared` (`f417d78ee`、base `perf/ink-final` `84f727c0e`) / バイナリ sha256 `ea9e0a7e…` / Tesla P40 (sm_61) 単機 / SQ-10 / 32K prefill (chunk 512、fp16、dspark、`STRATA_INK_PF_TILE=1 STRATA_INK_PROF=2`)。本番は停止 -> 復旧済み (healthy、`.env` sha256 `(省略)` 不変、GPU/ビルド両ロック解放)

## 結論

- **成立**。`dn_sk_fmt_occ` の MAX_M=16 で活性 (q8_1、16 token x 18 block x 80 B = 22.5 KiB/CTA) を shared に 1 回だけコピーする版を追加し、`STRATA_INK_SK_DN_SMEM=1` で切り替える
- **dn(sk) 308.60 -> 196.66 ms/step (−111.94 ms、−36.3%)** で、同一アームの重複差 (0.07 ms、0.02%) を 2 桁上回る。gu(sk) は 567.02 -> 566.38 (−0.1%) で動いていない
- **32K prefill 258.6 -> 274.2 ppts (+15.6 ppts、+6.03%) / 126.73 -> 119.50 s (−7.23 s)**、**128K prefill 203.0 -> 212.4 ppts (+4.63%) / 645.83 -> 617.10 s (−28.73 s)**
- **数値は不変**: 同一プロセス内で smem=1 と 0 の `y_pair` を **20/20 visit で diff 0 / 9,437,184 float** (S2_K / S3_K 両方、mask 0xc)
- REG は 128 -> **89/90**、shared **23,040 B**、STACK/LOCAL 0。shared が律速になり **4 CTA/SM を維持** (レジスタ上は 5 まで行ける)
- 既定は `STRATA_INK_SK_DN_SMEM=0` (従来どおり)。env を設定しない限りカーネルも挙動も変わらない

## 変更 (1 ファイル、+100 / −9 行)

`engine/src/inkling2/cuda/ink_moe_sk_occ.cuh` のみ。

| 追加 | 内容 |
|---|---|
| `ink_sk_dn_smem_mode()` | `STRATA_INK_SK_DN_SMEM` を 1 回だけ読む (既定 0)。0 以外は 1 |
| `k_ink_moe_dn_sk_fmt_occ_smem<FMT,MAX_M>` | 本体は `k_ink_moe_dn_sk_fmt_occ` と同一。反復の先頭で `h_sk` の n_e 行 (最大 16 x 1,440 B) を `s_hq[16][18]` に uint4 で複製し、`hp[m] = s_hq[m]` にして同じ `sk_dn_paired2_4lane` を呼ぶ。反復間は `__syncthreads()` でコピーと読みを分離 |
| `launch_ink_moe_dn_sk_one` / `_ex` | smem の有無を明示して S2_K / S3_K を起動する。`MAX_M==16` のときだけ smem 版を選ぶ |
| `launch_ink_moe_dn_sk_occ` | 起動対象 bit をまとめて `_ex` に渡す形に整理 (起動するカーネル・bit の落とし方は従来と同じ) |

- MAX_M=1 / 4 の経路は従来から活性を shared に置いている (`SM_M=MAX_M`)。smem 版は `MAX_M==16` 専用なので、decode (M=1) と M<=32 の経路はカーネル・shared とも不変
- コピー元は `h_sk + (row0 + off_m + m) * (kInkInter/64)`。`h_sk` は token 順ではなく **pair スロット順** ([P,18]) なので `tok_idx` ではない (最初 `tok_idx` で書いて bit 比較で 100% 不一致になり、この誤りに気づいた)
- 従来カーネル `k_ink_moe_dn_sk_fmt_occ` と `k_ink_moe_*_sk_fmt`、既存 env の意味は変更していない

## resource と occupancy

`cuobjdump -res-usage` (sm_61、最終バイナリ)。全体は `~/work/results/dn-smem/res-usage-final.txt`

| kernel | REG | STACK | SHARED | LOCAL | CTA/SM (計算値) |
|---|---|---|---|---|---|
| `dn_sk_fmt_occ<S2_K,16>` (従来) | 128 | 0 | 0 | 0 | 4 |
| `dn_sk_fmt_occ<S3_K,16>` (従来) | 128 | 0 | 0 | 0 | 4 |
| **`dn_sk_fmt_occ_smem<S2_K,16>`** | **90** | **0** | **23,040** | 0 | **4** |
| **`dn_sk_fmt_occ_smem<S3_K,16>`** | **89** | **0** | **23,040** | 0 | **4** |
| `gu_sk_fmt_occ<S2_K,16>` (参考) | 128 | 0 | 0 | 0 | 4 |
| `gu_sk_fmt_occ<S3_K,16>` (参考) | 128 | 8 | 0 | 0 | 4 |

- 4 CTA/SM の根拠: shared は 96 KiB/SM なので `98,304 / 23,040 = 4.26 -> 4`。REG 90 は `65,536 / (12,288/CTA) = 5.33` なので、**shared が律速でも 4 CTA/SM は維持**される (P40 の unified L1/shared は 4 CTA で 90 KiB を shared に割り当て、L1 は 6 KiB になる)
- 実 occupancy は ncu で確認していない (従来実験と同じ。計算値)
- `hp[16]` のポインタ配列が shared の即値オフセットに畳まれ、REG が 128 -> 90 に下がった。spill は出ていない (STACK 0 / LOCAL 0)
- cubin は sm_61 のみ (`cuobjdump -lelf`: `strata-inkling.1.sm_61.cubin` / `.2.sm_61.cubin`、PTX なし)
- バイナリサイズ 6,894,400 -> 6,943,136 B (+48,736 B、smem 版 2 形式分)

## bit 検証 (検証専用コードはコミットに含めない)

`launch_ink_moe_dispatch_only` の dn 起動直後に一時フックを入れ、同一プロセスで「smem 版 -> 従来版」を同じ入力で流し、`y_pair` を D2H して float を bit 比較した (20 visit、prompt 4096、M=512)。

| 比較 | 対象 | 一致 |
|---|---|---|
| smem -> 従来 | `y_pair` (dn) | **20/20 visit で diff 0 / 9,437,184 float** |

- 各 visit の mask は 0xc で、S2_K と S3_K の両方を含む
- 出力は (pair, row) ごとに 1 CTA が書くだけで atomic 加算が無く、起動順に依存しない
- 検証ビルド (フック入り) は `~/work/results/dn-smem/verify.err`、最終バイナリはフックを削除して再ビルドしたもの

## 性能 A/B (同一バイナリ、アーム間 60 s)

`~/work/bin/dn-smem-ab.sh`。`STRATA_INK_SK_DN_SMEM` を明示して 32K を 0 / 1 / 0、128K を 0 / 1 で回した。値は PROF PREFILL ブロックの平均 (`moe-prof-parse.py`)。

### 32K (chunk 512、fp16、dspark、gen 64、64 blocks)

| 項目 | 0 (k0a) | **1 (k1v)** | 0 (k0b) | 1 の差 (0 の平均比) |
|---|---|---|---|---|
| gu(sk) ms/step | 567.02 | **566.38** | 566.13 | −0.03% |
| dn(sk) ms/step | 308.60 | **196.66** | 308.67 | **−111.94 (−36.3%)** |
| gu+dn | 875.62 | **763.04** | 874.80 | −112.18 (−12.8%) |
| MoE 位相 | 1,543.55 | **1,430.75** | 1,543.16 | −112.61 (−7.3%) |
| step 全体 | 1,972.85 | **1,860.21** | 1,972.20 | −112.31 (−5.7%) |
| attn | 448.55 | 448.79 | 448.01 | +0.11% |
| gu(sf4) / dn(sf4) | 419.16 / 236.35 | 419.05 / 236.23 | 419.56 / 236.36 | 0.0% |
| **prefill** | **258.6 ppts (126.73 s)** | **274.2 ppts (119.50 s)** | **258.7 ppts (126.67 s)** | **+15.55 ppts (+6.01%) / −7.20 s** |
| decode (参考、dspark) | 24.8 tps | 24.3 tps | 26.0 tps | −1.7% (2 本の 0 が 24.8/26.0 でノイズ内) |
| 実行時刻 (UTC) | 17:12:31 | 17:16:45 | 17:20:02 | |
| clock / 温度 / 消費 | 1531 MHz / 73 C / 63.7 W | 1531 / 76 / 65.3 | 1531 / 76 / 65.5 | throttle なし |

- 0 の重複差は dn +0.07 ms (+0.02%) / prefill +0.1 ppts (+0.04%) / attn −0.12%。dn の短縮 (−36.3%) はその 1,600 倍
- attention・SF4 側・router/bin は動いていない (attn +0.11%、sf4 0.0%)。改善は S_K dn だけで説明できる
- decode は dspark 有効の参考値で、0 の 2 本が 24.8 / 26.0 と振れるため判定に使わない

### 128K (chunk 512、fp16、draft off、gen 16、256 blocks)

| 項目 | 0 (l0a) | **1 (l1v)** | 差 |
|---|---|---|---|
| gu(sk) ms/step | 567.56 | 568.03 | +0.08% |
| dn(sk) ms/step | 309.12 | **196.80** | **−112.32 (−36.3%)** |
| gu+dn | 876.68 | 764.83 | −112.85 (−12.9%) |
| MoE 位相 | 1,545.01 | 1,432.56 | −112.45 (−7.3%) |
| step 全体 | 3,211.81 | 3,087.22 | −124.59 (−3.9%) |
| attn | 1,531.30 | 1,531.52 | +0.01% |
| **prefill** | **203.0 ppts (645.83 s)** | **212.4 ppts (617.10 s)** | **+9.4 ppts (+4.63%) / −28.73 s** |
| decode (draft off) | 12.5 tps | 12.5 tps | 0.0% |
| 実行時刻 (UTC) | 17:23:18 | 17:35:13 | |

- dn の短縮量は 32K と 128K で同じ (−112 ms/step)。S_K の絶対値が context 長に依存しないという前提と整合する
- 128K の prefill 改善 (+4.63%) が 32K (+6.01%) より小さいのは、attn が step の 50% を占めて分母が大きいため

### 帯域の目安

dn の重み読みは 195 MB/visit x 48 = 9.36 GB/step (調査メモ §3.3 の見積り)。308.60 ms -> 30.3 GB/s が 196.66 ms -> **47.6 GB/s** (346 GB/s の 8.8 -> 13.8%)。まだ DRAM floor (~27 ms) の 7 倍で、短縮分は L1/L2 の再読みが消えたことによるものと解釈する (ncu 未計測)

## 逸脱・注意

- **検証版を 2 回作り直した**。1 回目はコピー元を `tb.tok_idx[...]` で引いていた (h_sk は pair スロット順なので誤り)。bit 比較で 20/20 すべて ~100% 不一致になり、`row0+off_m+m` に直して 20/20 diff 0 を確認した。**性能測定は修正後のバイナリのみ**で行った (誤版の値は使っていない)
- 1 回目の検証ビルドは比較前に SIGSEGV (exit 139) で落ちた。検証フックの `cudaMalloc` + `cudaMemcpyAsync` を `std::malloc` + 同期 `cudaMemcpy` に変え、`<cstdio>` / `<cstring>` を namespace の外 (ink_moe_cuda.cuh の先頭) に移したら通った。**どれが原因かは未確定** (検証コードのみの話で、カーネル・本番経路には影響しない)
- 32K の decode は dspark 有効で 0 の 2 本が 24.8 / 26.0 tps。smem 版 24.3 はその範囲内だが、decode の判定には使っていない。128K の draft off では 12.5 / 12.5 で差なし
- 既定は 0 のまま。`STRATA_INK_SK_DN_SMEM=1` を設定したときだけ smem 版が動く (本線既定化は 512K と decode の確認後)
- smem を 22.5 KiB/CTA 使うので L1 は 6 KiB に減る。今回の dn は改善したが、gu を同じ手でやると 45 KiB/CTA で 4 CTA/SM が崩れるため対象外 (指示どおり)
- 実 occupancy は ncu 未計測 (計算値)。`__launch_bounds__(128,4)` と REG 90 / shared 23,040 から 4 CTA/SM
- 512K は未測定

## 次の提案

1. **`STRATA_INK_SK_DN_SMEM=1` を既定化する**。32K +6.0% / 128K +4.6% / 数値不変。ただし 512K を 1 本確認してから
2. gu 側は「8 token ずつ 2 回に分けて 11.5 KiB/CTA」なら 4 CTA/SM を保てるが、重みの pass が 2 -> 4 に増えて重み traffic が 2 倍になる。gu は重み 58.75 KB/CTA なので、活性 45 KiB を削るより重みを削る方が先
3. dn の残り 196.7 ms は DRAM floor の 7 倍。次は `#pragma unroll 1` の s ループに 2 段の明示プリフェッチを入れる (shared ステージで REG が 90 まで下がったので、128 に戻す余地がある)
4. `s_hq` を 8 token x 2 段のリングバッファにすると shared が 11.5 KiB になり 5 CTA/SM (REG<=102) の目が出るが、重み pass が増えるので割に合わない見込み
