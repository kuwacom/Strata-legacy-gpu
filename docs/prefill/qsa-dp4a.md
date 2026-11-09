# QSA の QK を DP4A にする（opt-in、不採用）

取得日: 2026-11-09 / 10。作業ツリー `~/work/wt/qsa3`（branch `perf/qsa-dp4a`）。
イメージ `strata:fork-dp4a` / `strata:fork-dp4a-fix`。P40 1 枚。

## 1. 仮説

sm_61 の prefill で `qsa attn` は約 184 ms/1k（GPU 時間の 11.6%、128K）。
スコア側 `ps_score_x_kernel<1>` は int8 の K をいったん float に widen して FP32 FMA している
（`dequant8`、DP4A は未使用）。P40 は `__dp4a` が使え、理論上 FP32 の 4 倍のスループットなので、
QK を DP4A にすれば attention が速くなるはず、というのが出発点。

## 2. 実装（opt-in `STRATA_QSA_DP4A=1`）

- クエリ（float, HD=256）を 64 dim ごとに 1 スケールで int8 に量子化（`q8` と `qs[4]`）
- セルごとに 64 dim を 16 dword の `__dp4a` で int32 累積し、
  `sum_g (int32_g * qs[g] * sks[cell][g])` を float で合成
- マスクされたセルの扱いは float 経路と同一
- 既定（env 無し）は従来の float 経路のまま。`qsa_pattn_parity` は env 無しで
  **468 cases, 0 differ**（退行ゼロ）
- shared は 20.5 KiB -> 23.7 KiB。48 KiB に対して余裕があり、occupancy は変わらない

## 3. 最初の失敗と修正

最初のビルドは、24K 以上のプロンプトで応答が空になった（`prompt 0`、TTFT 約 140 s）。
engine ログの末尾は `prefill mmq: set_bounds: misaligned address`。起動時の 53 トークンの
ヘルスチェックだけは通っていた。

原因は `score_chunk_dp4a` の読み出し。K と Q の 64 バイトを `int4`（16 バイト、16 バイト境界必須）で
読んでいたが、位置は `g * 64 + 4 * j` で 4 バイト間隔なので `j >= 1` が 16 の倍数にならない。

修正: `int4` を 4 バイトの `int` 16 回に置き換えた（演算は同じ、順序も同じ）。
- 修正後スモーク: 24K needle が `prompt 22904 / TTFT 34.21 s / 669.5 tok/s` で found、
  engine ログの `misaligned` は 0 件
- 修正コミット `0903bf97`

## 4. 速度（A-B-A、24K x2 + 128K x1、`prefill=32768` ほか本番 v4 と同じ設定）

| 指標 | A1 (env 無し) | B1 (DP4A=1) | A2 (env 無し) | B1 vs A |
|---|---:|---:|---:|---:|
| prefill 24K (tok/s) | 659.5 | 662.1 | 664.8 | +0.0% |
| prefill 128K (tok/s) | 672.4 | 669.8 | 674.8 | **-0.6%** |

位相別（128K、ms/1k）:

| 位相 | A1 | B1 | A2 |
|---|---:|---:|---:|
| qsa attn | 25,328 (12.6%) | **26,429 (13.1%)** | 25,296 (12.6%) |
| qsa select | 7,789 | 7,792 | 7,784 |

- DP4A にすると `qsa attn` 自体が **+4.3% 遅くなる**
- 品質: needle 32K / 128K とも found（prefill 641.7 / 673.1 tok/s）

## 5. 判定: 不採用

- DP4A は P40 で FP32 と同じ命令スロットを使う（一次情報: NVIDIA の Mixed-Precision CUDA 8 と
  `pascal-gemv.md`）。したがって「int8 x int8 の 4 MAC/命令」は、FP32 FMA の 4 命令を 1 命令に
  するだけで、パイプラインの吞吐は上がらない
- スコアの内側和は 64 項で、K の widen と `fmaf` はメモリ待ちに隠れていたとみられる
- さらに、チャンクごとにクエリを量子化するコスト（1 クエリあたり最大 33 回の重複）が乗る
- 結果として、attention フェーズが遅くなった

残る `qsa attn` の改善は、DP4A ではなく「value 側の命令数を減らす」「セル方向の和を分割して
並列度を上げる（bitwise でなくなる）」の方向になる。value カーネルの 128 スレッド化も
issue-bound で退行した（`docs/prefill/qsa-value-reject.md`）。

## 6. 生ログ

`~/work/results/dp4a/`（`block-d1.log` = 修正前、`block-d2.log` = 修正後、
`bench-A1/B1/A2.jsonl`、`engine-*.log`、`smoke24-sm1.jsonl`）。

