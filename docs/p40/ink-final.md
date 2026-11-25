# Inkling 最終構成 (PV shape 2 + GQA pack 既定) の 32K / 512K 測定

- 日時: 2026-11-25 14:23-15:49 UTC
- 対象: `~/work/ink-ap` (branch `perf/ink-final`、`84f727c0e` = `perf/ink-attn-gqa` `3a2d76de0` + GQA pack 既定化)
- バイナリ: `build/src/inkling2/strata-inkling` sha256 `dde1017fdb79a5342bf91c2174424d8e660fdf3f534037960727adf044a35da2` (sm_61 の cubin のみ 2 本)
- ホスト: <host> / Tesla P40 24GB (sm_61, 1531 MHz) / SQ-10
- 本番: 停止 -> 復旧済み (healthy、`.env` sha256 `(省略)` 不変、GPU ロック解放済み)

## 結論

- **構成は env 未指定の既定で確定**: `STRATA_INK_PF_TILE=1` を明示するだけで、PV SHAPE=2 / GQA_PACK=1 / OCC=1 / SF4 compact が全部効く (起動ログで確認)
- **32K: prefill 258.1 ppts (126.96 s)、decode 25.1 tps**。過去値 253.7 から **+4.4 ppts (+1.73%)**
- **512K: prefill 111.8 ppts (4,691.47 s)**、過去の 97.6 ppts / 5,373.52 s から **+14.2 ppts (+14.55%) / -682.0 s (-12.69%)**、完走
- **512K の decode は 5.2 tps で、draft off の過去値 7.5 tps より -30.7% -> この項目は不成立**。今回は `--draft dspark` で回しており、512K では speculative の検証パスが重い (過去値は draft off。同一条件の draft off は未測定)
- K/V は **VRAM のみ** で host に溢れていない (host KV のログが 1 行も出ず、elastic growth は 524,560 tokens まで VRAM 内)
- VRAM ピークは **20,037 MiB** (過去 17,531 MiB から +2,506 MiB)。空き 4,403 MiB を残して完走

## 測定条件

| 項目 | A (32K) | B (512K) |
|---|---|---|
| 引数 | `--bench --native --gpu 0 --max-context 131072 --parallel 1 --chunk 512 --kv fp16 --draft dspark --prompt-len 32768 --gen 64` | `--bench --native --gpu 0 --max-context 524288 --parallel 1 --chunk 512 --kv k8v4 --kv-offload off --draft dspark --prompt-len 524288 --gen 256` |
| env | `STRATA_INK_PF_TILE=1 STRATA_INK_PROF=2` (PV_SHAPE / GQA_PACK は未指定 = 既定) | 同左 |
| timeout | 1,800 s | 18,000 s |
| 所要 | 153 s (14:23:33 -> 14:26:06) | 4,748 s (14:27:06 -> 15:46:14) |

- 起動ログ: `PV SGEMM shape = 2` / `GQA pack = 1 (pack_kv 3 -> 1 copy, QK/PV batch=8)` / `tiled prefill attention is off` は出ていない
- アーム間は 60 s 待機、VRAM は 5 秒間隔で `vram{A,B}.csv` に記録

## 結果

| 項目 | 今回 | 過去値 | 差 |
|---|---:|---:|---:|
| A 32K prefill | **258.1 ppts (126.96 s)** | 253.7 ppts (129.16 s) | **+4.4 ppts (+1.73%)** |
| 参考: 32K の旧基準 | - | 231.9 (タイル attn) / 178.6 (素の fork) | +11.30% / +44.51% |
| A 32K decode (dspark) | **25.1 tps** | 25 前後 (dspark) | 同等 |
| B 512K prefill | **111.8 ppts (4,691.47 s)** | 97.6 ppts (5,373.52 s) | **+14.2 ppts (+14.55%) / -682.0 s (-12.69%)** |
| B 512K decode | **5.2 tps (49.09 s / 256 token)** | 7.5 tps (draft off、8.54 s / 64 token) | **-2.3 tps (-30.67%) 不成立** |
| B attention 合計 | 2,906.5 s (62.1%) | 3,430.15 s (65.9%) | -523.7 s (-15.27%) |
| B VRAM ピーク | 20,037 MiB (空き 4,403) | 17,531 MiB (空き 6,909) | +2,506 MiB (+14.30%) |
| B 温度ピーク | 79 C | 75-76 C | +3-4 C |
| クロック | 全アーム 1531 MHz | 1531 MHz | 同じ (throttle なし) |

- ppts はバイナリ側の表示 (小数 1 桁)。秒は bench の生値で、`prompt/gen` から計算し直すと A 258.1 / B 111.75 で一致する
- 512K の attention は prefill の 62.1% (1,024 chunk)。32K は 18.0 s / 14.2% (64 chunk)
- B の prefill 内訳 (最終 chunk 付近): total 6,923 ms/step、attn 5,182、res_ctrl 66.1、dense_qkv 54.7、moe 7.1、dense_wo 16.9

## K/V の host 溢れなし (確認方法)

- `--kv-offload off` で `kv_offload_mode = -1` (VRAM のみ)。起動時に `Host KV Offload ON` / `Auto Host KV Offload` / `Auto-sizing max_ctx` のいずれも出力されていない
- モデルロード行は `experts=3072 VRAM / 0 host (hyb=0, promo=0, pf_buf=0MB)`
- `Elastic global KV grown` は 64 回で **524,288 -> 524,560 tokens** まで VRAM 内で伸びて止まった (max_ctx=524,560)
- VRAM 使用量の推移: 起動 13.5 GiB -> 18.0 GiB (chunk 776) -> 19.5 GiB (chunk 953) -> 20.0 GiB (完走時)

## 未確認・注意

1. 512K の decode が遅い原因は未確定。dspark の受理率はエンジンがログを出さないため**不明** (`accept` の出力はソースに無い)。過去値 7.5 tps は draft off / gen 64 なので、条件が違う (draft dspark / gen 256)。同一条件で draft off を取るには 512K prefill をもう 1 回 (約 78 分) 回す必要があり、未測定
2. VRAM ピークが過去より +2.5 GiB 増えた理由は未確認。dspark 分は 78 MiB しかない。KV を 524,560 tokens まで伸ばし切ったこと (過去は同じ max_ctx だが伸び切り方が違う可能性) と、elastic growth の二重確保が候補
3. 32K の decode 25.1 tps は dspark 有効。過去の dspark 値と同等で、GQA pack 既定化による decode への影響は見えていない (decode は per-token 経路でタイル attention を使わない)
4. 測定は `STRATA_INK_PROF=2` (eager、CUDA Graph 無効) の上。過去値も同じ条件
