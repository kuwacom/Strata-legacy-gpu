# hyper-connection read の down/inject で活性変換を共有する

取得日: 2026-11-09。作業ツリー `~/work/wt/hcpair`（branch `perf/hc-pair`、commit `02e8c663`）。
イメージ `strata:fork-hc` -> `strata:fork-v4`。

## 1. 見つけた無駄

Pascal では BF16 の GEMM が `bf16_path()==2`（FP32 経由）になり、`Gemm::bf16` は呼び出しごとに
重みと活性の両方を FP32 に変換してから `cublasSgemm` を呼ぶ（`src/prefill/gemm.cu:1090-1140`）。

hyper-connection read は 1 つの half で同じ活性 `xn16` を 2 回使う:

- `hc_*_down`（N=320, K=10240）
- `hc_*_inject`（N=4, K=10240）

`STRATA_COLD_TRACE=1`（T=8192）の定常値:

| 呼び出し | ms |
|---|---:|
| hc_down | 10.14 |
| hc_inject | 3.35 |
| hc_up（K=320 なので変換は小さい） | 5.82 |

inject は N=4 なので GEMM 自体はほぼ無く、**活性の FP32 変換（約 500 MB）と FP32 像の読み（336 MB）**
で時間を使っている。half は 96 回あるので、変換を 2 回やるのは純粋な重複。

## 2. 変更

`Gemm::bf16_pair(X, W1, W2, Y1, Y2, T, N1, N2, K, ldy1, ldy2, ldx)` を追加し、
1 回の活性変換で 2 つの GEMM を回す。

- `include/strata/prefill/gemm.hpp`: 宣言と、2 本目の重み用バッファ `tc_w2_`
- `src/prefill/gemm.cu`: 実装。Pascal の path 2 のときだけ共有し、それ以外は `bf16()` を 2 回呼ぶ
- `src/prefill/prefill.cpp`: hc read で down と inject を `bf16_pair` にまとめる。
  条件（両方 `Bf16InF32`、`q8_scale` なし、同じ K、`xn16_lo` なし、同じ `ldx`）を満たさないときは
  従来どおり 2 回呼ぶ

`gr_silu` は down の出力 `lo` にしか触らないので、inject を down と同じタイミングで実行しても
数値は変わらない。

## 3. bitwise の根拠

1. `bf16_to_f32` はビットの上位 16 bit シフトのみ（一対一）。同じ入力から同じ FP32
2. スライス境界 `x_rows = max(1, min(T, kXSliceElems / K))`、`t0 += x_rows`、`n = min(x_rows, T-t0)` は
   既存 `bf16` と同一
3. 重みのタイル分割 `w_rows` も初期値が同じで、`grow()` が最初の試行で成功する
4. `cublasSgemm` の引数（`OP_T/OP_N`、`m=N_i`、`k=K`、`lda=ldb=K`、`beta=0`）と stream が同一。
   カーネル選択は形で決まるので発行順に依存しない。`Y1`/`Y2` は互いに素
5. 非 Pascal（path 0/1）と VRAM 不足時は必ず従来の 2 回呼びに落ちる

## 4. 検証

- `gemm_bf16_parity`: **0 failures**（`hc inject T=300 N=4 K=10240` のケースを含む）
- A-B-A（24K x2 + 128K x1、`draft-conf=0.7`、`prefill=32768`）で、hc-pair 単体の寄与は
  chunk 長と SIMT の効果に埋もれて分離できていない。合成後の最終構成は
  `docs/prefill/chunk-profile.md` のとおり prefill 24K +7.1% / 128K +9.9%
- 期待値は机上で約 210 ms / 8192 トークン（1 half あたり inject が 3.35 -> 1.0-1.2 ms）

## 5. 注意

- cold trace の行が `bf16` 2 行から `bf16_pair` 1 行に変わるため、過去のログと直接比較できない
- env のキルスイッチは無い（同一バイナリでの A/B は不可）
- `include/strata/kernels/qsa_pattn.hpp` など他のヘッダの説明は変更していない

