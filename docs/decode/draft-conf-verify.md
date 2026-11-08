# draft-conf 検証（strata:fork-v3 / engine 0.1.57 / 512K / spec=4 / dflash2）

## 目的

`--draft-conf 0.5`（先行計測で decode が prod 比 +7.7%、base 比 +4.7% と報告されたが、
base のドリフト +2.8% があり再確認が必要）を A-B-A で再検証し採用可否を確定する。

先行データ: `~/work/results/dknobs/v3/`（prod=conf 0.7、base=conf 0.7+TIMING、dc05=conf 0.5）。
本検証の生ログ: `~/work/results/dknobs/v3b/`（ブロック d1）。

## 条件

- A = conf 0.7（本番値、a1/a2）、B = conf 0.5（b）。`STRATA_DECODE_TIMING=1` を全アームに付与
- `bench-decode.py 4000,24000 3 256 code,prose`、`BENCH_SEED=42`、1 本目除外、中央値、5% 閾値
- cpuset なし（decode は CPU experts 律速、32 worker）。state コピー毎で preset 汙染なし
- answers/klprobe（固定 4 プロンプトの greedy 回答と top-5 logprobs）を a1/b とも 2 pass
- spec 3/5 は 1 パスのみ

## 速度（新規実測 v3b、decode_tps 中央値）

| 条件 | a1 (0.7) | a2 (0.7) | b (0.5) | b vs a1 | A 揺れ (a1 vs a2) |
|---|---|---|---|---|---|
| 4K code | 53.82 | 51.31 | 55.20 | +2.6% | -4.7% |
| 4K prose | 45.53 | 45.24 | 44.35 | -2.6% | -0.6% |
| 24K code | 49.58 | 51.77 | 53.45 | +7.8% | +4.4% |
| 24K prose | 44.85 | 44.38 | 44.29 | -1.2% | -1.1% |

- b vs a1 は -2.6〜+7.8% とばらけるのに対し、A 自己揺れも -4.7〜+4.4%。
  **条件ごとの差はすべて A 揺れの範囲内**で、5% 阈値を満たす安定な差はない
- 先行の「+7.7%（prod 比）」は再現しなかった（24K code の +7.8% は A2 が +4.4% 揺れているため
  A 揺れ込み。先行計測は A-A 対比がなくドリフトを速度差として引用した可能性が高い）

### timing 内訳（18 window 以上の行の中央値）

| arm | tokwin | ms/window | wait (GPU-reach) | draft | 実効 tps (=tokwin/ms*1000) |
|---|---|---|---|---|---|
| a1 (0.7) | 2.36 | 48.83 | 37.04 | 4.46 | 48.3 |
| b (0.5) | 2.64 | 53.56 | 39.52 | 4.54 | 49.3 |
| a2 (0.7) | 2.43 | 48.80 | 37.11 | 4.44 | 49.8 |

- conf 0.5 は 1 window あたりの採択 token を +12% 増やす一方、verify が長くなり
  ms/window も +10% 増（GPU-reach wait 37.0 -> 39.5）。**実効 gain は約 +2% で A 自己揺れ
  （a2 は実効 +3.1%）に埋もれる**
- spec 3 / spec 5（1 パス）: 実効 tps 45.7 / 46.3 で spec 4（48.3）より劣。変更不要

## 品質

### 固定 4 プロンプト（temp 0、max 200）の greedy 回答

| プロンプト | a1 vs b | a1 自己 (pass1 vs pass2) |
|---|---|---|
| 1 首都・17*23 | 一致 | 一致 |
| 2 Fibonacci コード | 一致 | 一致 |
| 3 空が青い理由 | 4 行差 | 6 行差 |
| 4 素数 5 個と検証 | 4 行差 | 14 行差 |

- **出力は不変ではない**（conf 0.5 は verify を完全に経由しない受諾があるため想定通り）。
  ただし差分は A 自己自己揺れ（6/14 行）以下で、誤答・事実誤認はどちらにも見つかっていない

### logprobs 一致（klprobe: 4 プロンプト × greedy 24 token の top-5）

- top-1 token 一致率: **146/154 = 94.8%**
- top-1 logprob 差: 中央値 0.0001、最大 0.0884（一致 token のみ）
- greedy 内容の一致は 8 プロンプト中 6（pass1/pass2 のペア数分）

## 判定

- **速度: 5% 阈値を満たす差は確認できない**。実効 gain 約 +2% は A 自己揺れ（±4〜5%）以下。
  先行の +7.7% は再現しない（A-A ドリフトの混入と判断）
- **品質: 実質同等**。出力は不変ではないが、差は A 自己揺れ以下。top-1 一致率 94.8%、
  誤答なし。ただし「出力が完全に同じ」を前提にする運用はできない
- **リスク: 中**。速度便益が誤差以下なのに出力非不変のリスクを取る価値がない
- **採用可否: 不採用（本番は conf 0.7 を維持）**
- 参考（採用する場合の STRATA_SET）:

```
STRATA_SET="ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 spec=4 draft-conf=0.5"
```

## 逸脱・注意

- spec3/spec5 は 1 パスのみ（時間制約）。1 pass の 4 条件で spec4 劣（-4〜-5%）だが統計は薄い
- ANS2=2 を全アームに適用してしまい、b/a2 も 2 pass 取った（結果的に B 自己揺れデータが増えて有利）
- ブロック終了時の trap で restore_prod が 2 回走った（22:26 と 22:30 に lock release ログ）。
  最終状態は healthy・preset 汚染なし・ロック解放済みで問題なし
- a2 の bench 行が 12 行あるのに最初の集計で 0 行と出た（再実行で正常。初回実行の読み取り時の
  一時的問題。ファイル自体は 21:51 時点で完成済み）
- 本番復元後の確認: `--draft-conf 0.7 --spec 4 --draft dflash2 --max-context 524288` を
  engine cmdline で直接確認済み。healthy・ロック解放済み

