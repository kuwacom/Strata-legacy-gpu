# hc-fuse2: inject を mix に融合（-0.49 s / timeline -0.54 s）、down の 8x8 タイルは不採用

取得日: 2026-11-11。基準は `strata:fork-v8`（`main` = `a9433270`、128K prefill の GPU timeline 178,510 ms、
137,344 tokens）。A/B は同じイメージの `STRATA_PREFILL_TIMING` 行で取った（`phaseDiff.py`）。

## 1. inject を mix に融合（`STRATA_HC_MIXINJ=1`）

`hc read: inject`（`bf16_narrow8_kernel`、N = 4）と `hc read: mix`（`gr_mix_r16_kernel`）は、同じハーフで
**同じ xn16 の行（1 トークン 10,240 値 = 20,480 B）を 2 回**読んでいる。inject の K = 10,240 は
xn16 の列そのもの（`k = c * 2560 + d`）なので、mix のカーネルに 4 列の内積を足せば 1 回の読みで両方が出る。

`gr_mix_r16_inj_kernel`（`src/prefill/kernels.cu`、1 ブロック = 1 トークン、256 スレッド）:
- スレッド tid が全 4 ストリームの `d = tid + 256 m`（m = 0..9）を持つ。xn16 の 10,240 値はスレッドの
  分割になり、inject の部分和は「自分の 10 列 × 4 ストリーム」の内側で閉じる
- `mixed` は `gr_mix_r16_kernel` と同じ c 昇順・同じ `fmaf` なので**ビット不変**（`gr_mixinj_parity`）
- `inj` は k の和の順だけが変わる（ワープ内 butterfly → 8 ワープの部分和を昇順）。ビットは narrow カーネルと違う
- 資源: `REG:40 STACK:0 SHARED:128`（6 ブロック/SM = 48 ワープ）

### A/B（同一イメージ・同一セッション、ホスト側の静かな回）

| 位相 | z（素の SET） | q（MIXINJ=1） | 差 |
|---|---:|---:|---:|
| `hc read: inject` | 897 ms | **4,041 ms** | （融合カーネル） |
| `hc read: mix` | 3,631 ms | **1 ms** | （同上） |
| **inject + mix** | **4,528 ms** | **4,042 ms** | **-486 ms（-10.7%）** |
| GPU timeline | 178,667 ms | **178,131 ms** | **-536 ms（-0.30%）** |
| 128K ttft | 181.23 s | **180.76 s** | -0.47 s |

他の位相は ±0.3% 以内（`hc read: up` -93 ms と `host grouping` -212 ms は同じビルドのゆらぎ）。
v8 の log とも比べると inject + mix は 4,527 -> 4,042 ms、timeline は 178,510 -> 178,131 ms（-379 ms）。
融合カーネルは同じ 2 回目の測定で 4,041 / 4,039 ms（-0.05%）と再現する。

融合カーネルは 1.08 TB を 4.04 s = 267 GB/s。mix 単体は 298 GB/s なので、inject の重み読み
（80 KB/token、全体 1.05 TB を L2 から）と 4 列 × 10,240 の FMA が 410 ms ぶん乗っている計算になる
（分離していたときの inject は 897 ms だったので、差し引き -487 ms）。

### ホスト段取りのゆらぎについて

`ple` / `embed+steps` / `host grouping` はホスト側の段取り（PLE 行の用意、expert の grouping）で、
**同一アームの 2 回でも ±0.6〜0.9 s 動く**（DOWN88 の 2 回で `ple` 2,513 対 1,877）。GPU の位相は
±0.2% で再現するので、timeline の A/B は「ホスト段取りを除いた GPU 合計」も併記して読むこと。

## 2. down の 8x8 レジスタタイル（`STRATA_HC_DOWN88=1`）は不採用

`bf16_narrow88_kernel`（`src/prefill/gemm.cu`）: 同じ 64x64 タイルを 8x8 出力/スレッド・64 スレッドで。
4x4 は X タイルを 16 個の tx 群が、W タイルを 16 個の ty 群が読み直すので、8x8 は LDS と BF16 の widen を
半分にする（4 k 値あたり 256 FMA / 16 LDS / 32 widen 対 64 / 8 / 16）。k の和の順は昇順のままで、
`gemm_bf16_parity` の narrow ケースは **0.000e+00（bitwise）**、`REG:168 STACK:0 SHARED:8960` でスピルなし。

しかし実測は **12,692 -> 13,613 ms（+7.3%、2 回目 13,611 で再現）**。1 ブロック 2 ワープ（5 ブロック/SM =
10 ワープ）になり、4x4 の 4 ブロック/SM（32 ワープ）からレイテンシ隠しが落ちる分が、命令密度の
+15% を上回った。コードは残していない（負けたので戻した）。

8x8 を採るなら 128 スレッド（64x64 を 4x8 で覆う、20 ワープ。W 行を 72 要素に詰めて `LDS.128`）か、
widen を共有メモリへの書き込み時に 1 度だけ行う形（float で持つ、共有 16.6 KB、16 ワープ）で測り直すこと。

## 3. 品質ゲート

- `gemm_bf16_parity` 0 failures（素 / `STRATA_HC_DOWN88=1` / `STRATA_HC_INJ8=1` の 3 通り）
- `gr_mixinj_parity` 0 failures（T = 1, 2, 7, 31, 63, 64, 65, 127, 128, 129, 600, 1000, 2048, 4099, 8192, 32768）:
  `mixed` と BF16 / FP16 像はビット一致、`inj` は narrow カーネルと 1.5e-7 相対、FP64 に対しては
  融合側の方が narrow より良いか同じ（例: T = 32768 で 1.07e-7 対 2.66e-7）
- `gemm_f32_parity` 5,201 sizes bitwise
- needle 24K / 128K found、QA 128K は v8 と同じ回答（v8 は 8 問目が途中で切れていた）
- 本番は復旧して healthy

## 4. 次の候補

1. 融合カーネルの重み読みをベクタ化する（320 スレッド・1 スレッド 8 列 = `uint4` 1 本）。重みの
   L2 読みの命令数が 1/8 になり、4.04 -> 3.8 s 前後（推定 -0.2 s）
2. down を 4x8（128 スレッド、W 行 72 要素で `LDS.128`）で測り直す（推定 -1.0 s）
3. `hc read: up`（9.3 s）を `gr_upmix` 系の SIMT に寄せる（`gated` の書+読 1.08 TB が消える、推定 -2.8 s。
   FP32 ピークの 80% 以上が要る）
