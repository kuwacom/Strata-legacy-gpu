# v8: inject 専用カーネル、mix が BF16 像を読む、IQ2_S/IQ2_XS の I=128 @ J=48

取得日: 2026-11-11。基準は `strata:fork-v7`（128K prefill の GPU timeline 182,758 ms、137,344 tokens）。
位相の再現性は ±0.25%。128K 1 アームで判定し、勝ったものだけ 24K / decode / needle / QA で確認した。

## 1. inject 専用の 8 トークンカーネル（-1,859 ms、-67%）

`hc_*_inject`（N = 4、K = 10,240、BF16 重み）は `bf16_narrow_kernel` の 64x64 タイルで掛けていた。
N = 4 なので 64 列のうち有効なのは 4 列だけで、FMA の 16 分の 15 が無駄。命令律速で 1 呼び 5.7 ms
（メモリ下限は 2.2 ms）だった。

`bf16_narrow8_kernel`（`src/prefill/gemm.cu`）は 1 ワープが 8 トークンを担当し、1 レーンが `uint4`
（k 8 値）を持ち、**重み 1 値を 8 トークンで使い回す**。X（T x 10,240 BF16）を 1 度だけ読み、
k の和は 32 レーンの butterfly で決定的な順序に畳む。`REG:100 STACK:0 SHARED:0`、スピルなし。

| 位相 | v7 | v8 | 差 |
|---|---:|---:|---:|
| `hc read: inject` | 2,756 ms | **897 ms** | **-67.5%** |

実測 270 GB / 0.897 s = **301 GB/s** で、このカードの実用上限に乗った（旧 98 GB/s）。

## 2. mix が BF16 像を読む（-895 ms、-20%）

`gr_mix_r_kernel` は `R`（4 x 2560 の FP32 残差）から `R * rs * w` を計算し直していたが、同じ値は
すでに `xn16` に BF16 で入っている（down と inject が読んでいるのと同じ像）。`gr_mix_r16_kernel`
（`src/prefill/kernels.cu`）は BF16 を読んで `w` を掛け直す。読みが 1.08 TB -> 540 GB になる。

| 位相 | v7 | v8 | 差 |
|---|---:|---:|---:|
| `hc read: mix` | 4,525 ms | **3,630 ms** | **-19.8%** |

3 つの消費者（down / inject / mix）が同じ BF16 像を見るようになる方向で、`mixed` の
`R` 経路との相対 RMS は **9.3e-4**（BF16 の丸め 2^-9 相当）。int8 量子化の丸めより小さい。

## 3. IQ2_S / IQ2_XS を I=128 のタイルで回す（`gemm gate/up` -5.4%）

MoE の gate/up は IQ2_S と IQ2_XS が起動数の 42% を占めるが、この 2 型だけ shared メモリの都合で
I=64 のタイルだった（sm_61 の 48 KB に I=128/J=64 は 51,200 B で入らない）。

`iter_k`（K タイル）を縮めて入れる案は成立しない: `mmq_get_nbytes_shared` の `nbs_x` は
`tile_x_sizes = I * MMQ_TILE_NE_K` で **I だけで決まり K_vram を引数に取らない**うえ、
`static_assert(K_vram % 256 == 0)` があり、IQ2_S の `qk` は 256 なので K を割れない。

そこで J（トークンタイル）を 48 にすると I=128 が 49,088 B で入る。`switch_J` は「タイル数の
最小化」なので、J=64 の行を残すとそちらが選ばれ続ける。**J=64 の行を削除**して J=48 を選ばせた。

| arm | IQ2_S/IQ2_XS | 実測 smem | `gemm gate/up` | GPU timeline |
|---|---|---:|---:|---:|
| v7 再測 | J=64, I=64 | 30,336 B | 27,455 ms | 182,661 ms |
| **採用** | **J=48, I=128** | **49,088 B** | **25,936 ms (-5.5%)** | **181,338 ms (-0.72%)** |
| 不採用 | J=32, I=128 | 46,976 B | 28,343 ms (+3.2%) | 184,003 ms (+0.73%) |

J=32 は ntx（列タイル数）が 56 -> 84 に増えて重みタイルの再読が利得を上回る。

## 4. まとめ（v7 -> v8）

| 位相 | v7 | v8 | 差 |
|---|---:|---:|---:|
| `gemm gate/up` | 27,463 | 25,970 | -5.44% |
| `hc read: inject` | 2,756 | 897 | -67.45% |
| `hc read: mix` | 4,525 | 3,630 | -19.78% |
| 他 | - | - | ±25 ms（ノイズ内） |
| **GPU timeline** | **182,758 ms** | **178,510 ms** | **-2.32%** |
| 128K prefill | 740.4 tok/s | **758.1 tok/s** | +2.4% |

品質: `gemm_f32_parity` 5,201 sizes bitwise / `gemm_bf16_parity` 0 failures（inject の N=4 も含む）/
`qsa_pattn_parity` 468 cases 0 differ / `prefill_swiglu_quant_test` ok /
needle 24K・32K・128K found / QA 128K 8/8（v7 と同一の回答）。

## 5. 実測して捨てたもの

| 候補 | 対象 | 結果 | 理由 |
|---|---|---|---|
| dense 重みを int8 へ事前展開 + 素の DP4A GEMM | gdn / qsa proj | **+82.9% / +65.2%** | 純 dp4a の実測上限は 35.5 TFLOPS なのに、この GEMM は 6.1 TFLOPS。`IDP.4A` 512 に対し `LDS.U.32` 192 で 4 B のスカラー LDS のレイテンシに止まる。本番 MMQ は 10〜15 TFLOPS 出ており、「64 トークンごとの unpack が損」という見立ては否定された |
| attention のチャンクまとめ（CPC） | qsa attn | **+5.7%**（bitwise は維持） | 同一バイナリ内では cpc1 -> cpc4 が -6.5% なので狙いは効くが、ループ版のチャンク本体が v7 より 3% 以上遅く、漸近値でも負ける。チャンク境界の `__syncthreads()` 1 本が位相 ~1.5 s に相当する |
| IQ2_S/IQ2_XS の J=32 | gemm gate/up | +3.2% | ntx 56 -> 84 |
| `iter_k` を 128/192 に | - | 実装不可 | `static_assert(K_vram % 256 == 0)`、`nbs_x` は I のみで決まる |

## 6. 次の候補（未着手）

1. **attention: q を shared に staging**（ループ化しない）。33 個のチャンク CTA が同じ q 12 KB を
   warp ごとに読み直している（CTA あたり 96 KB）。shared を 20,992 -> 33,280 B にしても 2 CTA は維持できる
2. **inject を mix に融合**（同じハーフで `xn16` を 2 回読んでいる）。270 GB が消えて -0.9 s
3. **`hc read: up`（9.3 s、cuBLAS FP32 の 78%）** を SIMT に寄せる（`gated` の書+読 1.08 TB が消える）
4. **`hc read: down+silu`（12.7 s、FP32 ピークの 58〜68%）の 8x8 レジスタタイル**
