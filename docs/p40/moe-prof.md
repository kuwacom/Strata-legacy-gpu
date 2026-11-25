# Inkling MoE 位相の内訳 (S_K 区間を独立タグ 6/7 で分離)

調査日: 2026-11-25 / 対象: `~/work/ink` (branch `perf/ink-moe-prof`, commit `08c919db3`) / バイナリ sha256 `67d895f9…` / Tesla P40 (sm_61) 単機 / SQ-10 / 32K prefill。GPU は共用ロック内で使い、本番は復旧済み (healthy、`.env` sha256 `(省略)` 不変)

## 結論

- S_K の gu / dn を独立タグで分離した。**MoE 位相 1,737.72 ms/step = router 3.79 + bin 1.09 + SF4 (15 expert) 758.18 + S_K 967.59 + reduce8 7.08**。内訳の合計は位相に **0.00%** で一致 (イベントが区間を重複なく分割するため、恒等的に一致する)
- **S_K : SF4 = 1.28 : 1**。重みの 0.5% (15 expert / 3072) が MoE の 43.6%、S_K (3057 expert) が 55.7%
- これまで「S_K 983 ms」として `moe=` の残りに一括計上していたものは、実測で **gu(sk) 609.57 + dn(sk) 358.02 + reduce8 7.08** だった。reduce8 は事前見積の約 1 ms ではなく **7.08 ms**
- 計測コードは性能を変えていない: prefill **230.9 ppts** (基準 231.9 から **-0.43%**、合格 ±3%)。MoE 位相の総和もベースライン比 +0.08%

## 変更 (追加のみ 7 行)

| ファイル | 箇所 | 内容 |
|---|---|---|
| `engine/src/inkling2/cuda/ink_moe_cuda.cuh` | `:3632` の次 | タグ 6/7 が S_K 用である旨を 1 行追記 (既存行は残した) |
| 同上 | `:3795` / `:3799` (int8 TC 分岐) | `ink_moe_mark(6)` / `ink_moe_mark(7)` |
| 同上 | `:3804` / `:3809` (P40 の dp4a 分岐) | `ink_moe_mark(6)` / `ink_moe_mark(7)` |
| `engine/src/inkling2/cuda/ink_runner.cuh` | `:1970` | `[PROF moe sk parts] gu(sk)=… dn(sk)=…` の 1 行を追加 |

- タグ 6/7 -> `prof_ms[14]/[15]` は従来未使用 (`prof_ms[16]` のうち 14/15 が空き)。既存タグ 0..5 の意味は変えていない
- 区間は「直前の mark から今回の mark まで」なので、**gu(sk) は入力 x の q8_1 量子化を、dn(sk) は活性 h の q8_1 量子化を含む**。空きタグが 2 個しかないため量子化だけの分離はしていない

## 実測 (32K prefill、M=512 x 64 chunk の平均)

| 区間 | ms/step | 最小..最大 | MoE 位相に占める割合 |
|---|---|---|---|
| router + topk (tag0) | 3.79 | 3.76..3.84 | 0.2% |
| bin (tag1) | 1.09 | 1.03..1.13 | 0.1% |
| gu(sf4/mx8) (tag2、15 expert) | 479.14 | 419.11..537.00 | 27.6% |
| dn(sf4/mx8) (tag3、15 expert) | 279.04 | 238.48..305.08 | 16.1% |
| gu(sk) (tag6、x 量子化込み) | 609.57 | 604.76..613.98 | 35.1% |
| dn(sk) (tag7、h 量子化込み) | 358.02 | 352.07..365.00 | 20.6% |
| reduce8 (`moe=` の残り) | 7.08 | 7.06..7.10 | 0.4% |
| **合計** | **1,737.72** | | **100.0%** |

- 内訳の和 1,737.73 と位相 1,737.72 の差は 0.00%
- 位相の全項目 (res_ctrl 66.10 + dense_qkv 55.76 + sconv_kv 10.88 + attn 315.94 + dense_wo 16.92 + MoE 1,737.72 + unembed 0) の和も total 2,203.33 に -0.00%
- reduce8 が `moe=` の残りに入るのはコード経路で確認した: prefill は `emit_48_visits(M, 0, …)` (`ink_runner.cuh:3736`) で mode=0 のため `fuse_moe_sconv` が false になり、`launch_ink_moe` の `run_reduce8` が true (`:3547` / `:3550`)
- 参考: 重み 1 個あたり SF4 は 50.5 ms (4.48 MB)、S_K は 0.32 ms (3057 expert)。**約 160 倍** (重み 1 個あたりの比較。層あたりの visit 数は未確認)
- SF4 758.18 ms は step 全体 (2,207.30、warmup を除く 63 chunk 平均) の 34.3%、S_K 967.59 ms は 43.8%、MoE 位相は 78.8%

## ベースライン (ba6bfb870 のタイル attn バイナリ) との比較

| 項目 | ベースライン | 今回 | 差 |
|---|---|---|---|
| MoE 位相の総和 | 1,736.37 | 1,737.72 | +0.08% |
| gu(sf4/mx8) | 477.92 | 479.14 | +0.26% |
| dn(sf4/mx8) | 279.34 | 279.04 | -0.11% |
| router+topk / bin | 3.78 / 1.07 | 3.79 / 1.09 | +0.3% / +1.9% |
| `moe=` (残り) | 974.26 | 7.08 | — |
| ベースライン `moe=` の再構成 | 974.26 | 974.67 (gu(sk)+dn(sk)+reduce8) | **+0.04%** |

- ベースラインの `moe=` 974.26 ms が、今回の 3 区間の和 974.67 ms にほぼ一致する = **タグは区間を分けただけで、時間を動かしていない**。±0.3% は同一バイナリ内の隣接 chunk 間のばらつき (moe の min..max は 967..983) と同じ大きさ

## prefill / decode

| アーム | prefill | 秒 | decode (tps agg) |
|---|---|---|---|
| base (`ba6bfb870`、tiled) | 231.9 ppts | 141.30 | 24.4 |
| moe-prof (`08c919db3`、tiled) | **230.9 ppts** | 141.90 | 25.1 |
| 差 | **-0.43%** | +0.4% | +2.9% (判定対象外) |

- 実行条件は ab-attn の tiled アームと同一: SQ-10、`--max-context 131072 --parallel 1 --chunk 512 --kv fp16 --draft dspark --prompt-len 32768 --gen 64`、`STRATA_INK_PF_TILE=1 STRATA_INK_PROF=2`。起動ログに `tiled prefill attention is off` は無い
- 搬送は 96 個の追加イベント記録 (48 visit x 2) のみで、カーネルは 1 個も変わっていない

## ビルド検証

- インクリメンタルビルド 113s (configure 1s、docker 合計 149s)。ログ `~/work/logs/ink-moe-prof-build-20261125.log`
- cubin は `sm_61` のみ 2 エントリ (`cuobjdump -lelf`)
- `cuobjdump -res-usage` は従来と同じ: `gu_sk_fmt<2,16>` REG 160 / `<3,16>` 158、`dn_sk_fmt<2,16>` 134 / `<3,16>` 132。STACK 0、SHARED 0 (使用実体は変わっていない)
- ベースラインのバイナリは `~/work/build-ink-attn-baseline/strata-inkling` (sha `2ce2532…`) に退避してある

## 未確認

1. **量子化の分離**: gu(sk) 609.57 のうち x の q8_1、dn(sk) 358.02 のうち h の q8_1 が何 ms かは未計測。分離にはタグがあと 2 個要る (`prof_ms[16]` に空きが無い)。h の量子化 (P x 2304) は x (M x 3072) より大きいので、dn(sk) が重い側に寄る理由の一部と推測するが、証拠は無い
2. **reduce8 7.08 ms の内訳**: 8 slot の strided 和 (37.7 MB read + 4.7 MB write = 42.4 MB) で 6.0 GB/s。遅い理由 (coalescing か 8 回 read か) は未計測
3. 15 個の SF4 expert が何層に散っているかは未確認。したがって「1 層あたりの SF4 コスト」は出せない
4. SF4 758 ms の内訳 (グローバル重み読み / `__hfma2` 1/64 律速 / 空 block の走査) は未計測
5. ncu / deviceQuery は GPU ロックの制約で未使用。occupancy は prereq の計算値 (gu 3 CTA/SM、dn 3 CTA/SM)

## 次の提案

1. **15 SF4 expert を S3_K に再量子化する**。今回の実測で SF4 は MoE の 43.6% (step の 34.3%) を 0.5% の重みで食っていると確定したので、削減余地は最大 ~758 ms/step。重みは 15 x 4.48 MB = 67 MB なので S3_K 化の増分は小さい。まず `.strata` を作り 128K で A/B
2. S_K 967.59 ms は重み 31.9 GB/step (prereq の推定) を 33.0 GB/s で読んでいる計算になり、まだ帯域律速ではない。`MAX_M=16` の occupancy を 3 -> 4 CTA/SM に上げる実験 (`__launch_bounds__(128,4)`、REG<=128) を次に
3. reduce8 は 7.08 ms (MoE の 0.4%) なので優先度は低い。S_K の後ろに回す
4. SF4 経路を残す場合は 1 launch にまとめる (`g16.y = max_act` -> SF4 expert 数)。現状は 97% が空 return の block
5. 量子化の分離が要るなら `prof_ms` を 16 -> 18 に広げてタグ 8/9 を足す (runner 側の変更が増えるので、必要になったときだけ)
