# QSA value カーネルの占有率改善（採用）と MMQ タイル表の拡張（不採用）

取得日: 2026-11-10。P40 1 枚、本番 v5（`strata:fork-v5`）を基準に A-B-A。
関連: `docs/references/gap-analysis.md`（SASS 実測と候補の全体像）。

## 1. 見つけたもの

`cuobjdump -res-usage` で `ps_value_d4_kernel`（QSA prompt attention の value 側）が
**151 レジスタ / 64 スレッド / 6 CTA per SM（占有率 18.75%）** と分かった。
`__launch_bounds__` の minBlocks は未指定で、レジスタが占有を決めていた。

`__launch_bounds__(64, 8)` を指定するとレジスタ上限が 128（64 x 8 x 128 = 65,536）になり、
8 CTA が載る（25%）。演算も和の順序も変えないので bitwise のまま。

## 2. 検証したもの（採用）

- `qsa_pattn_parity`: **468 cases, 0 differ**（bitwise）
- A-B-A（A1/A2 = v5、B1 = `strata:fork-tiles`。B1 には下記のタイル表変更も入っている）

| 指標 | A1 | A2 | B1 | B1 vs A |
|---|---:|---:|---:|---:|
| prefill 24K (tok/s) | 689.6 | 688.0 | 693.7 | +0.7% |
| prefill 128K (tok/s) | 696.0 | 694.9 | **705.5** | **+1.45%** |
| `qsa attn`（128K、ms） | 25,287 | - | **22,534** | **-10.9%** |
| GPU timeline（128K、ms） | 194,647 | - | 192,159 | -1.3% |

- 位相の減少（2,753 ms / 194,647 ms = 1.4%）が end-to-end の +1.45% と一致する
- 品質: needle 32K / 128K とも found（677.0 / 710.5 tok/s）

## 3. 採用しなかったもの（タイル表）

同じビルドに、IQ2_S / IQ2_XS の J=48 行を I=128 にする overlay 変更を入れていた
（J=64 は 51,200 B で 48 KiB を超えるが、J=48 なら 49,088 B）。

- 128K の測定では `qsa attn` だけが動いており、MoE 側の位相に差は無い。
  大きなチャンクでは expert あたりの行数が 64 を超えるため J=64 が選ばれ、J=48 の行は使われない
- J=48 が選ばれるのは小さいプロンプトや末尾のチャンクだけ。その経路は今回の A/B で
  一度も踏んでおらず、49,088 B / 49,152 B という余裕 64 B の状態も未確認
- したがって **v6 には launch_bounds だけを入れた**（`perf/qsa-value-occ`、commit `c9eec7a9`）。
  タイル表の変更は `perf/mmq-tiles`（commit `78767395`）に記録として残す

## 4. v6 の確認

`strata:fork-v6`（launch_bounds のみ）:

| 指標 | 値 |
|---|---:|
| prefill 24K | 692.0 tok/s |
| prefill 128K | 704.7 tok/s |
| decode 4K code / prose | 48.7 / 41.5 tok/s |
| decode 24K code / prose | 50.1 / 43.1 tok/s |
| needle 24K | found、TTFT 32.71 s / 700.3 tok/s |

本番反映後のスモーク: 24K needle が TTFT 33.38 s / 686.1 tok/s で found、healthy。

## 5. 学び

- `ps_value_d4` は「並列度を上げる」方向では 2 回失敗した（128 スレッド化は -5.2%、DP4A 化は +4.3% 遅）。
  今回効いたのは **同じスレッド数・同じ命令列のまま、レジスタ上限を下げて CTA 数を増やす**方向だった
- したがって、このカーネルは「命令数は足りているが、同時に走る CTA が足りない」状態だった
- 同種の余地は `__launch_bounds__` の minBlocks が未指定のカーネルに残っている
  （`native_gu_multi` / `native_down_multi` / `native_mmvq_il` / `gr_down_staged` / `s2_expert_grouped` の
  per-expert カーネル）。SASS のレジスタ実測とセットで 1 つずつ試す価値がある

