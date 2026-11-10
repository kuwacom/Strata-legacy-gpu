# v7: write-back のローカルメモリ、dense MMQ の FP16 活性、K=640 の MMQ、MoE の SQ

取得日: 2026-11-11。P40 1 枚。基準は `strata:fork-v6`（128K prefill の GPU timeline 192,062 ms、137,344 tokens）。
比較は位相内訳（`STRATA_PREFILL_TIMING=1`）の同一位相どうし。同じ画像の再測で位相の振れは ±0.25% 以内。

## 1. hyper-connection の write-back がローカルメモリに落ちていた（-41%、-3.61 s）

`gr_write_norm_rs_kernel`（`src/prefill/kernels.cu`）は、1 スレッドが 10 要素を持ち回るために
`float v[GRW_PER]`（GRW_PER = 2560/256 = 10）を使う。ループが

```cuda
int k = 0;
#pragma unroll
for (int d = threadIdx.x; d < N; d += 256, ++k) { ... v[k] = x; ... }
```

だと、`d` の開始値が `threadIdx.x` なので ptxas は反復回数を定数と見なせず、`v[k]` の添字が実行時値に
なって配列がローカルメモリに置かれる。SASS では `STACK:40`、`LDL` 6 本 / `STL` 3 本、4 B 刻みのポインタ
更新（`IADD32I R5,R5,0x4`）として出ていた。1 トークン・1 ハーフあたり 81,920 B のローカル往復で、
全体では 1.08 TB。

`k` を数える形（`for (int k = 0; k < GRW_PER; ++k) { const int d = threadIdx.x + k * 256; ... }`）に
すると添字が展開後に定数になり、`v[]` はレジスタに載る。`d` は昇順のままで、演算も順序も変わらない
（bitwise）。

| 位相 | v6 | v7 | 差 |
|---|---:|---:|---:|
| `hc write` | 8,709 ms | **5,099 ms** | **-3,610 ms (-41.4%)** |
| GPU timeline | 191,965 ms | 188,425 ms | -3,540 ms (-1.84%) |
| 128K prefill | 706.1 tok/s | **718.6 tok/s** | +1.8% |

実測帯域は 171 GB/s -> 292 GB/s に上がり、`gr_mix_r`（298 GB/s）と同じ天井に乗った。
同じ形のループを持つ `gr_write_cvec_norm_rs_kernel`（steered layer 用、既定では走らない）も直した。

## 2. dense MMQ の活性を FP16 から直接量子化（-2.5 s）

dense 射影の MMQ 経路は、X が w16（FP16）で来たときに `f16_to_f32` で FP32 に広げてから
`quantize`（q8_1）していた。1 起動あたり T x K を 2 往復する。q8_1 の量子化は FP16 の値をそのまま
読めるので、往復を消して FP16 から直接書く。値も縮約順も同一（構成上 bitwise）。

| 位相 | 前 | 後 | 差 |
|---|---:|---:|---:|
| gdn | 27,000 | 26,312 | -2.55% |
| qsa proj | 12,882 | 12,224 | -5.11% |
| gdn out proj | 11,348 | 10,481 | -7.64% |
| router+shared | 10,268 | 9,209 | -10.31% |

## 3. 共有 expert の down 射影（K=640）を dense MMQ に載せる（-2.3 s）

`ffn_down_shexp` は N=2560 x K=640 で、dense MMQ の `K % 256 != 0` の条件で MMQ から落ちていた
（`K` タイル 256 が 640 を割り切れない）。MoE 側が使っている `overread_bytes` と同じ論法で、
読み越しが活性の 512 値パディング（スケール 0）に収まることを `mmq::k_in_padding` で判定して許可する。
重みは末尾 256 B をゼロにした控えを渡す。

| 位相 | 前 | 後 | 差 |
|---|---:|---:|---:|
| router+shared | 9,209 | 7,820 | **-15.1%** |
| 他 3 位相 | - | - | ±40 ms（ノイズ内） |

活性を int8 に丸める点でこれまでと同じ種類の非 bitwise 変更（dense MMQ の採用時と同じ品質ゲート）。

## 4. MoE の SwiGLU と down 用 q8_1 を 1 パスに（SQ、-1.3 s）

MoE は 1 (token, expert) 行あたり GU を 5,120 B 書き、swiglu が 5,120 B 読み 2,560 B 書き、
quantize が 2,560 B 読んで 4,608 B 書く（計 19,968 B）。既存の `mmq::swiglu_quant` で swiglu と
quantize を 1 パスにすると H の往復 5,120 B/行が消える。

| 位相 | 前 | 後 | 差 |
|---|---:|---:|---:|
| gemm gate/up | 28,620 | 27,430 | **-4.16%** |
| GPU timeline | 192,088 | 190,749 | -1,339 ms (-0.70%) |

`STRATA_MMQ_SQ=1` で有効になる。

## 5. 実測して捨てたもの（同じ 128K 1 アームで判定）

| 候補 | 対象 | 結果 | 理由 |
|---|---|---|---|
| `STRATA_MMQ_JWIDE=48`（J=48 に I=128） | gemm gate/up | **+16.5%** | I=128 が入るのは IQ2_XS/XXS だけで、最多の IQ2_S と IQ3_* は I=64 のまま ntx が 10->13 に増える |
| MMQ write-back への GLU 融合（`STRATA_MMQ_GU=1`） | gemm gate/up | **+76.9%** | アキュムレータが 32 -> 64 float/thread になり 128 reg を超えてスピル、tile_y の staging も 2 -> 4 回 |
| `STRATA_TOPK_SAMPLE=1` | qsa select | ±0.05% | `counted` が偽なので sample 経路に入らない |
| `STRATA_TOPK_ACTIVE_ANY=1` + `SAMPLE` | qsa select | **+26.9%** | P40 では sample カーネルが遅い |
| `STRATA_SELECT_SIMT=2` | qsa select | **+8.7%** | マイクロベンチの 1.49x は本番で再現しない |
| `STRATA_SELECT_SIMT=0` | 全体 | -1.9% | 既定の 1 が最良 |

## 6. 学び

- 位相の再現性が ±0.25% しかないので、**1 アーム 8 分で 0.3% の差は判定できる**。
  大きなビルドを伴う候補でも、まず 128K 1 アームで白黒つけるのが正しい
- `-41%` のような大きな当たりは「アルゴリズム」ではなく「コンパイラが配列をレジスタに載せ損ねた」
  ところから出た。SASS の STACK/LDL/STL を見る価値は高い
- 他実装（llama.cpp）の実測値は、その機械のボトルネック（DGX Spark は帯域、P40 は命令）が違うと
  そのまま移らない。GLU 融合は上流で +5〜17% だが、P40 ではレジスタ圧で負ける
