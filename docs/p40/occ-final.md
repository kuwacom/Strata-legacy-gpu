# S_K MoE occupancy の最終ビルド検証 (decode 後退なし / 128K prefill +3.6%)

調査日: 2026-11-25 / 対象: `~/work/ink` branch `perf/ink-sk-occ` (`b872a8f81`、SF4 起動統合 `d59c0373a` の上) / バイナリ sha256 `45c344fb…` (指示値と一致、ビルド不要) / Tesla P40 (sm_61) 単機 / SQ-10 / 本番は停止 -> 復旧済み (healthy、`.env` sha256 `(省略)` 不変、GPU/ビルド両ロック解放)

## 結論

- **decode は後退していない**。8K context・prompt 512・gen 512・draft off で OCC=0 **19.90 tps** -> OCC=1 **20.30 tps** (+2.01%)。同一アームの 2 本は完全に同値 (ばらつき 0.00%) で、差はそれを超える。`STRATA_INK_PROF` を切った graph 有効モードでも **20.70 -> 21.10 (+1.93%)** と同じ方向・同じ幅
- **128K prefill の改善は再現**。OCC=0 **190.4 ppts (688.29 s)** -> OCC=1 **197.2 ppts (664.75 s)** で **+3.57% / −23.54 s**。位相別でも total 2,682.13 -> 2,590.19 ms/step (−91.94) と一致し、差はすべて S_K MoE (attention は −0.1% で不変)
- 過去基準 182.8 ppts (717.21 s、旧バイナリ・dspark) との比は、**同一条件 (dspark) の OCC=1 で 195.8 ppts = +7.11% (−47.75 s)**、draft off の OCC=1 で **+7.88%**
- **`STRATA_INK_SK_OCC=1` は既定にしてよい**。decode 後退なし・32K/128K とも向上・bit 一致 (前回 20/20) が揃った
- **MAX_M==1 の除外は不要**。理論上は REG 76/80 -> 88/90 で 6 -> 5 CTA/SM に落ちるが、実測の decode は +2.0% だった

## 測定条件

| 項目 | 内容 |
|---|---|
| バイナリ | `~/work/ink/build/src/inkling2/strata-inkling` sha256 `45c344fbfa0a32bf89686c43be0986eb5d970b6f527bd80e9320101066642e15` |
| ランタイム | `LD_LIBRARY_PATH=~/work/bin/lib-sm61`、`CUDA_VISIBLE_DEVICES=0` |
| モデル | `<models>/inkling2/Inkling-2-Mini-SQ-10.strata` |
| 共通 env | `STRATA_INK_PF_TILE=1 STRATA_INK_PROF=2`、`STRATA_INK_SK_OCC=0/1` |
| 共通引数 | `--bench --gpu 0 --parallel 1 --chunk 512 --kv fp16 --draft off` |
| decode | `--max-context 8192 --prompt-len 512 --gen 512`、各アーム 2 本 |
| 128K | `--max-context 131072 --prompt-len 131072 --gen 64`、各アーム 1 本 |
| 順序 | dec-o0a -> 60 s -> dec-o1a -> 60 s -> dec-o0b -> 60 s -> dec-o1b -> 60 s -> pf128-o0 -> 60 s -> pf128-o1 (アーム交互) |
| 追加 | PROF 無しの decode 1 対 (dec2-o0/o1)、dspark 条件の 128K 1 本 (pf128-o1-dspark) |
| 生データ | `~/work/results/occ-final/{dec-*,dec2-*,pf128-*}.{out,err}`、ログ `~/work/logs/occ-final-ab{,2}.log` |
| スクリプト | `~/work/bin/occ-final-ab.sh` (本 A/B)、`occ-final-ab2.sh` (追加)、`occ-final-parse.py` (集計) |
| GPU 状態 | 開始時に他プロセス無し (本番コンテナのみ -> 停止)。各アーム終了時の clock はすべて 1531 MHz (P40 の上限)、温度 48 -> 74 C、消費 59-64 W で throttle の兆候なし |

## decode A/B (8K context、draft off)

`STRATA_INK_PROF=2` (eager、graph 無効):

| arm | OCC | decode tps | 512 token の所要 |
|---|---|---|---|
| dec-o0a | 0 | 19.90 | 25.77 s |
| dec-o1a | 1 | 20.30 | 25.24 s |
| dec-o0b | 0 | 19.90 | 25.77 s |
| dec-o1b | 1 | 20.30 | 25.24 s |
| **OCC=0 平均** | 0 | **19.90** (ばらつき 0.00%) | |
| **OCC=1 平均** | 1 | **20.30** (ばらつき 0.00%) | |

`STRATA_INK_PROF` 無し (graph 有効、本番に近い):

| arm | OCC | decode tps | 512 token の所要 |
|---|---|---|---|
| dec2-o0 | 0 | 20.70 | 24.78 s |
| dec2-o1 | 1 | 21.10 | 24.25 s |

- OCC=1 の差は **+0.40 tps (+2.01% / +1.93%)** で、両モードで同じ値。同一アームの 2 本が同値なので run-to-run ばらつきは 0.1 tps の表示分解能未満 (<0.5%)
- PROF 有無で絶対値が 19.9 -> 20.7 tps と 4% 違う (eager の launch オーバーヘッド)。**相対差はモードに依存しない**
- 絶対値が 32K の記録 (25.1-25.7 tps) より低いのは `--draft off` のため。dspark 有無を含めて絶対値を過去値と比べてはいけない
- 参考: 同一条件の 1 ステップあたりの短縮は 25.77 -> 25.24 s (0.53 s / 512 token = 1.04 ms/token)

## 128K prefill A/B (131072 token、gen 64、chunk 512、draft off)

| arm | OCC | prefill ppts | 秒 | 対 OCC=0 | gen decode tps |
|---|---|---|---|---|---|
| pf128-o0 | 0 | 190.4 | 688.29 | - | 13.3 |
| pf128-o1 | 1 | 197.2 | 664.75 | **+3.57% (−23.54 s)** | 13.4 |

- PROF の total 平均 2,682.13 -> 2,590.19 ms/step、256 chunk で 686.6 -> 663.1 s。bench の実測差 −23.54 s と一致 (0.0 s)。**位相の合計と GPU timeline の両方が同じだけ縮んでいる** (移動ではない)
- gen 64 の decode は 13.3 -> 13.4 tps で不変 (prefill の変更が decode を動かしていない)

## 位相別 (256 chunk の平均、ms/step)

| 区間 | OCC=0 | OCC=1 | 差 | 差% |
|---|---|---|---|---|
| res_ctrl | 66.24 | 66.15 | −0.09 | −0.1% |
| dense_qkv | 54.76 | 54.79 | +0.03 | +0.0% |
| sconv_kv | 10.85 | 10.87 | +0.01 | +0.1% |
| **attn** | **898.18** | **897.41** | **−0.77** | **−0.1%** |
| dense_wo | 16.59 | 16.56 | −0.03 | −0.2% |
| reduce8 (`moe=`) | 7.09 | 7.10 | +0.01 | +0.1% |
| router + topk | 3.78 | 3.77 | −0.01 | −0.2% |
| bin | 1.59 | 1.63 | +0.04 | +2.3% |
| gu(sf4/mx8) | 419.84 | 420.18 | +0.34 | +0.1% |
| dn(sf4/mx8) | 235.53 | 235.44 | −0.09 | −0.0% |
| **gu(sk)** | **609.26** | **567.32** | **−41.95** | **−6.9%** |
| **dn(sk)** | **358.41** | **308.98** | **−49.44** | **−13.8%** |
| **合計** | **2,682.13** | **2,590.19** | **−91.94** | **−3.4%** |

attn と MoE の寄与:

| 内訳 | 差分 (ms/step) | 全体 −91.94 に対する寄与 |
|---|---|---|
| attention (attn のみ) | −0.77 | 0.8% |
| **MoE S_K (gu+dn)** | **−91.38** | **99.4%** |
| MoE SF4 (gu+dn) | +0.25 | −0.3% |
| その他 (router/bin/reduce8/dense/sconv/res) | −0.04 | 0.0% |

- **attention は動いていない**。128K の改善は S_K MoE だけで説明できる
- 128K の S_K は 967.67 -> 876.29 ms/step。32K の記録 (967.34/968.22 -> 875.72) と**同一の絶対値**で、S_K 区間が context 長に依存しないことを確認した。そのため改善率は 32K の +4.9% より小さく +3.6% になる (分母が attn で膨らむため)
- 1 visit あたり gu 12.69 -> 11.82 ms、dn 7.47 -> 6.44 ms (48 visit/step 換算)。前回 32K の 12.69 -> 11.81 / 7.46 -> 6.43 と一致
- S_K の実効帯域 (前回と同じ 31.9 GB/step の見積り): 33.0 -> 36.4 GB/s

## 過去基準 182.8 ppts との比較

| 条件 | ppts | 秒 | 対 182.8 |
|---|---|---|---|
| 過去実測 (旧バイナリ、dspark) | 182.8 | 717.21 | - |
| 本ビルド OCC=0、draft off | 190.4 | 688.29 | +4.16% |
| 本ビルド OCC=1、draft off | 197.2 | 664.75 | **+7.88% (−52.46 s)** |
| **本ビルド OCC=1、dspark (同一条件)** | **195.8** | **669.46** | **+7.11% (−47.75 s)** |

- 同一条件 (dspark) で比べても **+7.11%**。うち S_K occupancy が −3.5 pt、残り (+3.6 pt) は SF4 起動の絞り込みなど本セッションで A/B していない変更分
- dspark の有無そのものは prefill を **195.8 vs 197.2 = −0.71%** にする (位相合計では 2,604.84 vs 2,590.19 = +0.57%)。位相別では attn 910.78 vs 897.41 (+1.5%) で、他は不変 -> drafter の tap 取得が attention に上乗せされるため。S_K (877.14) と SF4 (655.92) は draft off の 876.29 / 655.62 と同一

## 逸脱・注意

- **指示の 2 アーム A/B に加えて 3 本を追加した** (dec2-o0 / dec2-o1 / pf128-o1-dspark)。理由は (1) 128K の基準 182.8 が dspark 条件で、draft off の数値と直接は比べられない、(2) decode が PROF の eager モードだけで判断されるのを避ける。いずれも本番復旧込みで 1 セッション (05:08-05:27 UTC) に収めた
- 128K は指示どおり各 1 本。位相の再現性は過去実測で ±0.25% だが、今回の 128K は同一アームの反復を取っていない。差 +3.57% はそれを 1 桁上回る
- decode の「ばらつき 0.00%」は 0.1 tps 表示の分解能による。実際の再現差は 0.5% 未満とみている
- PROF=2 は eager 起動 (graph 無効) を強制するため、decode の絶対値は本番より 4% 低い。相対差は graph 有効時と一致した (追加測定)
- `STRATA_INK_SK_OCC` は MAX_M=1/4/16 の全経路に効く。MAX_M=1 の REG は実測 76/80 -> 88/90 で、計算上の CTA/SM は 6 -> 5 に落ちる (`~/work/results/sk-occ/res-usage.txt`)。それでも decode は +2.0% だった
- MAX_M=4 (M=2..32) の経路は今回測っていない (未測定)
- 128K の 512K への外挿は未測定。S_K が 18.4% (967.67 ms/chunk ÷ 5.25 s/chunk) とすると、512K の 97.6 tok/s は +1.7% 程度の見込み (見積り、測定していない)
- 検証コード (bit 比較) は前回のバイナリと同じ sha なので再実行していない。前回 20/20 visit で `h_f32` / `y_pair` が diff 0

## 次の提案

1. **`STRATA_INK_SK_OCC=1` を既定にする**。今回の decode +2.0% / 128K +3.6% / 32K +4.9% と bit 一致で、既定化を止める理由がない。既定化の際は OCC=2 (先行ロード) を削り、MAX_M=16 の S2_K / S3_K に絞るとバイナリ増 (+2.26 MB) を抑えられる
2. **MAX_M==1 の除外は不要**。理論 occupancy は下がるが実測は改善。production の serve で decode を 1 本だけ確認すれば十分
3. 512K で 1 本確認する (見積り +1.7%)。128K と同じく S_K の絶対値が変わらないなら、改善量は chunk 数に比例する
4. 残る S_K 876 ms は依然 MoE の 56%。occupancy は 4 CTA/SM で打ち止め (5 CTA/SM には REG<=102 が必要) なので、次は活性側の traffic (16 token x 36 block x 80 B を shared に置く) の番
