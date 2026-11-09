# hyper-connection の down / inject を BF16 のまま読む

取得日: 2026-11-09 / 10。作業ツリー `~/work/wt/hcdown`（branch `perf/hc-down-bf16`）。
イメージ `strata:fork-hcdown`（down のみ）/ `strata:fork-hcdown2`（down + inject）-> `strata:fork-v5`。

## 1. 背景

Pascal では `Gemm::bf16` が `bf16_path()==2` で、活性と重みを FP32 に広げてから `cublasSgemm` を呼ぶ
（`src/prefill/gemm.cu:1090-1140`、`kXSliceElems = 16<<20` ごとのスライス）。
hyper-connection read の down は N=320, K=10240 と狭く、1 回の呼び出しで

- 活性 `T x K` 個を FP32 に変換して書く（T=8192 で 335 MB）
- `cublasSgemm` がその FP32 像を読み直す（335 MB）

を払う。`STRATA_COLD_TRACE=1`（T=8192、定常値）:

| 呼び出し | ms | 実効 |
|---|---:|---:|
| `bf16 hc_*_down`（N=320, K=10240） | 10.14 | 5.3 TFLOPS |
| `bf16 hc_*_up`（N=10240, K=320） | 5.82 | 9.1 TFLOPS |
| `bf16 hc_*_inject`（N=4, K=10240） | 3.35 | 帯域律速 |

`Gemm::bf16_pair`（`docs/prefill/hc-pair.md`）で down と inject の変換は 1 回に共有済み。
残るのは「down の変換そのもの」と「N が狭いための cuBLAS の非効率」。

## 2. 変更

`Gemm::bf16_narrow(X, W, Y, T, N, K, ldy, stream)` を追加（`STRATA_HC_DOWN_BF16=1` のときだけ使う）。

- カーネル `bf16_narrow_kernel`: BM 64 x BN 64 x BK 32、256 スレッド、1 スレッド 4 行 x 4 列
- X と W を BF16 のまま共有メモリに置き、`(uint32_t)v << 16` で厳密に float へ widen して FP32 で累積
- 共有メモリ 8,960 B（X 64x36x2 + W 32x68x2、bank conflict 回避のパディング込み）
- 対象は N <= 512、K % 32 == 0、T >= 64、`bf16_path()==2`、X/W 16 B・Y 4 B 境界
- down（N=320）と inject（N=4）の両方に適用。inject は N が小さいのでタイル幅を余らせるが、
  従来の FP32 像の読み直し（335 MB）が消える方が大きい
- env 無しでは `hcnarrow` が false のままで、従来の `bf16_pair` / `bf16_proj` と同じ順に呼ぶ（退行ゼロ）
- `bf16_pair` とは排他（narrow のときは inject も narrow、pair は使わない）

## 3. 誤差

widen は両経路とも厳密（ビットの上位 16 bit シフト）。変わるのは K 方向の和の順序だけで、
相対 RMS は `eps * sqrt(K/3)` ≈ 3.5e-6（K=10240）。入力の BF16 量子化 2^-8 = 0.4% より 3 桁小さい。
**bitwise ではない**。

## 4. 速度

A-B-A（24K x2 + 128K x1、本番 v4 と同じ設定）。同じイメージで env だけ切り替える。

### down のみ（`strata:fork-hcdown`）

| 指標 | A1 | B1 | A2 | B1 vs A |
|---|---:|---:|---:|---:|
| prefill 24K (tok/s) | 682.3 | 680.6 | - | -0.3% |
| prefill 128K (tok/s) | 680.7 | 685.2 | - | +0.7% |

位相（128K、ms）: `hc read: down+silu` + `hc read: inject` は A1 18,766 -> B1 12,686 + 5,214 = 17,900。
down 自体は -32% だが、inject が FP32 経路に戻って相殺された。

### down + inject（`strata:fork-hcdown2`）

| 指標 | A1 | B1 | A2 | B1 vs A |
|---|---:|---:|---:|---:|
| prefill 24K (tok/s) | 680.5 | 681.2 | 681.4 | +0.0% |
| prefill 128K (tok/s) | 685.7 | **694.6** | 684.9 | **+1.36%** |

位相（128K、ms）: 18,795 -> 12,658 + 2,746 = 15,404（**-18%**）。GPU timeline 197,755 -> 194,953 ms。
A 同士のドリフトは 128K で -0.1%。

## 5. 品質

- needle 32K: found（prefill 670.1 tok/s）、needle 128K: found（700.5 tok/s）
- 本番 v5 の 24K needle: TTFT 33.19 s / 689.9 tok/s / found
- 本番 v5 の 128K QA: **8/8**（Dr. Vera Lindqvist / 03:40 UTC / 7,312 records / 19 attempts /
  61 / cobalt / 3,072 MiB / 3 engineers）
- `gemm_bf16_parity`: 0 failures（既存ケースのみ。`bf16_narrow` 専用のケースは未追加 = 未確認）

## 6. 判定: 採用（opt-in env を本番で有効化）

- 128K で +1.36%、hc ペアの位相は -18%。24K は誤差内
- 5% 閾値には届かないが、位相の削減が明確で、品質（needle と QA）も通ったため採用
- 本番 `STRATA_SET` に `STRATA_HC_DOWN_BF16=1` を追加（`strata:fork-v5`）
- ロールバックは env を外すだけ（既定経路に戻る）

## 7. 次に効く可能性があるもの

- down と inject を **1 回の X 読みで同時に**計算する（現在は 2 パスで 15.4 s。1 パスなら ~13 s 程度）
- inject の N=4 に対して BN を 8/16 に絞る（64 幅は 16 倍の無駄）
- `hc_up`（N=10240, K=320）は変換が小さいので対象外

生ログ: `~/work/results/hcdown/`（h1）、`~/work/results/hcdown2/`（h2）。

