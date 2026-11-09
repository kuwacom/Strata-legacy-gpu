# QSA value カーネルの並列度上げ（不採用）

取得日: 2026-11-09。作業ツリー `~/work/wt/qsa2`（branch `perf/qsa-attn`、commit `02f0673f`）。
イメージ `strata:fork-qsa2`。P40 1 枚、fork-v3 相当（PATTN + DENSE_MMQ）。

## 1. 仮説

sm_61 の prefill では `qsa_prompt_attn_batch` が cc<70 で false を返し、`qsa_pattn_batch` に落ちる。
実行カーネルは `ps_score_x_kernel<1>` と `ps_value_d4_kernel`。

`ps_value_d4_kernel` は 64 スレッド / CTA で、1 CTA が (KV head, クエリ) の 1 組。
CTA 数は約 214（`n_head_kv=2` x バッチ約 107）しかなく、60 SM に対して 1 SM あたり 7.1 ワープ
（スレッド枠の 11%）しか埋まらない。ここを 128 スレッドにして 12 head を 2 群に分ければ、
ワープ数が 2 倍（22.3%）になり、1 スレッドの `acc[4][12]+a[4][12]`（96 本）も 48 本に減る。

## 2. 実装

`src/kernels/cuda/qsa_pattn.cu` のみ（+48 / -28）。

- `PSV_NT=128`、`PSV_DG=HD/4=64`、`PSV_NHG=2`、`PSV_HG=G/2=6`（`static_assert` 付き）
- `d0 = 4*(t % 64)`、`h0 = (t/64)*6`、`acc[4][6]` / `a[4][6]`
- 重みは `spt[cl] + h0` から自分の 6 本だけ読む（`h0` はワープ内で一様 = 放送）
- `srow` の書き込みを `t < CHUNK`（=64）に限定
- CTA 数は変えないので、並列度の上限は 22.3%

## 3. パリティ

`qsa_pattn_parity`（sm_61）: **468 cases, 0 differ、29,724,792 個の非ゼロ出力すべて一致 -> PASS**。
`gemm_f32_parity --quick`: 134 sizes bitwise OK。
算術の順序を変えないので bitwise である。仮説どおりの実装にはなっている。

## 4. 速度（A-B-A、24K x2 + 128K x1）

| 指標 | A1 (fork-v3) | B1 (fork-qsa2) | A2 (fork-v3) | B1 vs A |
|---|---:|---:|---:|---:|
| prefill 24K (tok/s) | 629.2 | 596.7 | 627.1 | **-5.2%** |
| prefill 128K (tok/s) | 622.8 | 599.9 | 622.4 | **-3.7%** |

A 同士のドリフトは -0.3% / -0.06% なので、退行は誤差ではない。

## 5. 判定: 不採用

- 仮説（latency-bound）は外れ。このカーネルは **issue-bound** だった
- スレッドを増やすと、V の 32-bit ロード・int8 の展開・重みの共有メモリ読みが head 群ごとに
  重複する。机上でも命令数が約 5,950 -> 7,170（+20%）になる見積りだった
- 並列度 2 倍の効果を命令 +20% が上回った

## 6. 次に試すなら

- 1 スレッド 4 dim x 12 head のまま、**CTA をクエリ方向に分割**する（セル方向の和の順序は
  保てないため bitwise ではなくなる）
- もしくは `ps_score_x_kernel` の `UNR` を振る（bitwise のまま並列度を上げられる可能性）
- value カーネルは約 184 ms/1k（qsa attn 全体）の 4 割程度と推定。ここを触るより、
  QK を DP4A にする案（`perf/qsa-dp4a`）の方が期待幅が大きい

生ログ: `~/work/results/qsa2/`（`parity-qsa_pattn_parity.log`、`bench-A1/B1/A2.jsonl`、`block-i1.log`）。

