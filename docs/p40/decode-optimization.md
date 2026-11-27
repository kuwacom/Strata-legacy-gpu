# decode 最適化 (sm_61 / P40)

測定日: 2026-11-27 / Tesla P40 (sm_61) 単機 / Inkling-2-Mini SQ-10 / `inkling` ブランチに
4 施策を積んで既定を整えた統合確認。本番は停止 -> 復旧済み (healthy、`.env` sha256 `(省略)` 不変、
GPU/ビルド両ロック解放)。

## 結論

- decode の律速は **unembed (full logits) と global attention** の 2 つだった。8K/32K/128K では
  unembed が 25 ms/token で文脈長に依らず一定、256K では global attention が 46 ms/token で最大
- 4 施策を入れて **8K 20.3 -> 33.0、128K 13.8 -> 21.6、256K 10.4 -> 16.3 tok/s**。すべて bit 一致で、
  既定は sm_61 で有効、`env=0` で従来経路に戻せる
- 統合ビルドは error / undefined reference 0、警告 405 行 (従来と同数)、cubin は sm_61 のみ
- 256K の VRAM ピークは 21,223 MiB / 24,576 MiB で、host スピルなし

## decode の区間別内訳

計測用に `STRATA_INK_DEC_PROF=1` を追加した (既定 off = ゼロオーバーヘッド)。区間は event で
挟み、host 側の unembed も `STRATA_INK_PROF=2` の行に出る。計測オーバーヘッドは 8K で
20.3 -> 20.2 tok/s の +0.5% に収まった。ステージの和は全体 ms/token に対して
-2.07 / -2.05 / -1.60 / -1.10% で閉じた。

| stage | 8K | 32K | 128K | 256K | 何を測っているか |
|---|---:|---:|---:|---:|---|
| res_ctrl | 3.796 | 3.788 | 3.790 | 3.788 | post-attn ShortConv + 残差 + mlp res mix + RMSNorm + controllers |
| dense_qkv | 5.082 | 5.089 | 5.091 | 5.094 | wq/wr/gate/sink/wk/wv (+global の k/v dyn proj) |
| attn_g | 0.383 | 5.853 | 22.535 | 46.043 | rel_table + splitk + merge (global 8 visit、KV 全長読み) |
| attn_swa | 1.668 | 1.907 | 1.912 | 1.898 | rel_table + splitk + merge (SWA 40 visit、window 1024) |
| dense_wo | 1.688 | 1.704 | 1.703 | 1.703 | wo_ud |
| router | 1.024 | 1.033 | 1.035 | 1.026 | gate GEMV + topk8 |
| gu_sk | 4.237 | 4.269 | 4.290 | 4.275 | S_K gate+up + x q8_1 quant |
| dn_sk | 2.267 | 2.283 | 2.300 | 2.299 | S_K down + h q8_1 quant |
| gu_sf4 | 0.776 | 0.852 | 0.849 | 0.859 | SF4/MX8 gate+up |
| dn_sf4 | 0.485 | 0.528 | 0.519 | 0.528 | SF4/MX8 down |
| bin + moe_rest | 0.653 | 0.681 | 0.684 | 0.660 | MoE bin + 残余 |
| reduce_sconv | 0.189 | 0.197 | 0.198 | 0.191 | MoE reduce + post-MLP ShortConv + 残差 |
| out_norm + embed | 0.048 | 0.046 | 0.047 | 0.066 | output depth mix + final RMSNorm + embed |
| **unembed** | **25.056** | **25.085** | **25.065** | **25.080** | full logits GEMV + softcap |
| **gpu_sum** | **47.735** | **53.689** | **70.394** | **93.874** | event 区間の和 |

- unembed は文脈長に依存せず約 25 ms で一定。global attention だけが p0 に比例して伸びる
  (0.38 -> 5.85 -> 22.54 -> 46.04 ms)
- 1 token あたりの重み読みは約 3.1 GB で文脈長に依らない。帯域上限 (346 GB/s) に対する効率は
  8K 19% / 32K 23% / 128K 30% / 256K 36% で、短い文脈ほど launch と固定費が効く
- CUDA Graph の利得は 8K +4.4% (20.3 -> 21.2)、128K +2.9% (13.8 -> 14.2)。長文脈では
  launch (eager で 1,326 launch/token) より帯域が律速になる

## 施策: unembed の FP32 常駐

現行の head は毎 token、S6_K の unembed (200,058 x 2,304) を 66 タイルに分けて
S_K -> BF16 -> FP32 の 2 変換をかけ、SGEMM して softcap していた。ここを初回だけ通して
FP32 を常駐させ、以降は同じ SGEMM だけを回す。

- env `STRATA_INK_HEAD_F32`。既定は **sm_61 / 単機 (`ep2`・`ls` が false) / full logits 経路
  (`--draft off`) のとき 1**。sm_70+、複数 GPU、dspark (screen 経路)、VRAM 確保失敗は従来経路
- bit 一致の根拠: 同じタイル幅 (3072)・同じ 2 カーネル・同じ lda/ldY・同じ compute type で
  FP32 を作るので、SGEMM に渡るバイト列が従来と同一。同一プロセスで `=0` と `=1` を 20 step
  交互に走らせ、full logits (200,058 要素) の memcmp が全 step 一致、GPU top-128 の key も一致
  (argmax 変化 0)。M=512 の prefill head も needle の 32K で一致した
- head は 25.10 -> 6.61 ms/token (-18.49 ms、-73.7%)。decode は 8K 20.3 -> 32.4 (+59.6%)、
  32K 18.0〜18.1 -> 27.3 (+50.8〜51.7%)。256K は 10.4 -> 12.9 tok/s (+24%)
- VRAM は +1,758.3 MiB。256K で 19,463 -> 21,223 MiB となり 24 GB 内に収まる
- dspark は top-128 screen 経路で full logits を作らないため対象外。既定では確保もしない
  (確保ログ 0 行、`STRATA_INK_HEAD_F32=0 (dspark (screen 経路))`)

## 施策: decode global attention の 1 ウェーブ化

従来の `k_ink_flash_gqa_splitk` は REG 64 で 8 CTA/SM、30 SM で 240 枠 < 256 CTA となり
2 ウェーブになっていた。端数の 16 CTA は 1 CTA/SM = 4 warp で MLP が足りない。

- env `STRATA_INK_DEC_ATTN=0|1|2`。既定は sm_61 で **2**。適用は M=1 / mode 0 / global /
  `kv_fmt==0` (fp16) / `host_cap==0` のときだけで、SWA・M>1・k8v4・host offload は従来経路
- VAR=1 は古い split の relpos 参照を除去 (`rt[1024]` は必ず `+0.0f` なので CTA 先頭で 1 回
  読んで加算順を保つ)。VAR=2 は VAR=1 + 次 key の K/V を 1 段先読みする
- `__launch_bounds__(128, 9)` で REG 56 -> 9 CTA/SM -> 270 枠 >= 256 CTA = 1 ウェーブ
- bit 一致の根拠: kernel-level parity で、従来との partial m/l/O と merge 出力が長さ 1〜262144 の
  全 288 比較で完全一致 (`TOTAL=288 BAD=0 DETBAD=0`)。決定性も variant 同士で一致
- 途中で 1 ulp 差が出た。原因は 4 項 dot の FMA 融合位置が NVCC によって従来と別の項に
  移っていたこと (SASS で `FMUL qy*ky` から始まっていた)。`__fmaf_rn` で従来と同じ融合順
  (`mul(x) -> fma(y) -> fma(z) -> fma(w)`) に固定し、score も `__fmaf_rn(warp_sum, scale, rt)`
  にして解消した
- 128K の attn_g 22.554 -> 15.058 ms (-33.2%)、decode 18.6 -> 21.6 (+16.1%)。
  256K は 45.811 -> 29.766 ms (-35.0%)、12.9 -> 16.2 (+25.6%)。8K は 0.382 -> 0.306 ms で悪化なし
- 256K 1 token の global attention は 8.59 GB の読みで、187.5 -> 288.6 GB/s (P40 実測 333 GB/s の 87%)

## 施策: res_ctrl の depth mix 共有

SWA 層は `rq = is_swa ? 0 : c` なので c=0,1,2 の src mix が同じ stp・同じ q_res[0]・同じ count
から作られ、`d_mixed` slot0/1/2 に同一バイト列が 3 回書かれていた。msg と src を別 CTA に分け、
SWA の src を 1 CTA だけが計算する。

- env `STRATA_INK_DEC_RC=0|1`。既定は **1**。共有するのは M=1 decode のときだけで、
  sm_70+ / tree / DSpark / prefill / M>1 は従来経路
- 重複は SWA 20 visit x 2 = 40 mix/token、`d_mixed` slot1/2 の書き込み 368,640 B/token。
  launch 数は visit あたり 1 のまま (1,194 -> 1,194)
- bit 一致の根拠: 同一プロセス・同一状態から rc と従来を両方走らせ、`d_msgs` / `d_mixed` /
  `d_ctrl_cat` 3 slot / apply 出力を D2H で比較し、24 token x 24 core visit = 1,056 比較で不一致 0
  (max 差 0)。ss/dot の還元順・softmax・重み付き和・RMS は従来と同一
- res_ctrl は 8K 3.784 -> 3.584 ms (-0.200、-5.3%)、32K 3.794 -> 3.618 ms (-0.176、-4.6%)。
  end-to-end は 8K 32.7 -> 33.0 tok/s (+0.9%)、CUDA Graph で 35.0 -> 35.3 (+0.86%)。
  32K の end-to-end はアーム間ばらつき (±0.4 ms) に埋もれて判定不能

## 統合確認 (新既定)

`inkling` に 4 施策を積み、既定を sm_61 向けに整えたビルドで通した。env は
`STRATA_INK_PF_TILE=1 STRATA_INK_PROF=2` のみで、`HEAD_F32` / `DEC_ATTN` / `DEC_RC` は未指定。
起動ログで既定が効いていることを確認した。

```
[InkCUDA] DEC_ATTN variant = 2 (fast path: M=1 global fp16 KV)
[InkCUDA] STRATA_INK_DEC_RC=1 (M=1 decode の depth mix を共有)
[InkCUDA] STRATA_INK_HEAD_F32=1 (sm_61 単機 + full logits 経路): unembed FP32 を初回の full logits head で常駐
```

| アーム | prefill ppts | decode tok/s | ms/token | VRAM ピーク MiB |
|---|---:|---:|---:|---:|
| 8K (prompt 512, gen 256, draft off) | 288.9 | **33.0** | 30.30 | 13,255 |
| 32K (prompt 32,768, gen 64, draft dspark) | **274.1** | 25.0 | 40.00 | 13,167 |
| 8K dspark smoke (gen 32) | 315.6 | 25.1 | 39.84 | 12,015 |
| 128K (prompt 131,072, gen 256, draft off) | 212.4 | **21.6** | 46.30 | 17,127 |
| 256K (prompt 262,144, gen 256, draft off) | 163.7 | **16.3** | 61.35 | **21,223** |
| needle 32K (serve, GEN 16) | - | - | - | 12,263 |

- 8K の prefill 288.9 は本番停止直後の最初のアームで冷えているため比較対象外。同じ条件の
  過去値は 315.6 (下の dspark smoke が同値)
- 期待値との差: 8K 33.0 (期待 33)、128K 21.6 (21.6)、256K 16.3 (16.2)。すべて ±5% 内
- 32K prefill 274.1 は前回の統合値 274.3 に対して -0.07% (合格 ±2%)
- needle 32K は `7741`、16 token の id 列も過去と一致
- 256K は host スピルなし (`experts=3072 VRAM / 0 host`、`Auto-sizing max_ctx` なし、
  KV は 8,192 -> 262,416 まで elastic に成長)。GPU は最大 79 C / 1,531 MHz / 222 W で throttle なし
- 起点は 8K 20.3 / 32K 18.0 / 128K 13.8 / 256K 10.4 tok/s

### ビルド

| 項目 | 内容 |
|---|---|
| ツリー | `~/work/integ` (branch `inkling`) をコンテナに read-only で mount、ビルド先 `~/work/integ-build/dec` |
| 条件 | `nvidia/cuda:12.9.1-devel-ubuntu24.04`、`--memory=10g --memory-swap=10g`、`JOBS=4`、`CMAKE_CUDA_ARCHITECTURES=61-real`、`STRATA_PORTABLE=OFF -march=native`、ターゲット `strata strata-inkling` |
| 結果 | configure 17 s + build 529 s (docker 合計 599 s)。`error:` 0、`undefined reference` 0、警告 405 行 (従来と同数)、nvcc warning 68 |
| 成果物 | `strata-inkling` 7,014,224 B (sha256 `b736e0bb…`)、`strata` 49,022,856 B |
| cubin | `strata-inkling.{1,2}.sm_61.cubin` の 2 本のみ (sm_61 以外 0)。`strata` は 58 エントリすべて sm_61 |
| env スイッチ | `STRATA_INK_DEC_ATTN` / `STRATA_INK_DEC_RC` / `STRATA_INK_HEAD_F32` が `strings` に存在 |

- 前回の統合ビルド (`c1aca006c`) の `strata-inkling` は 6,943,136 B。今回 +71,088 B は
  decode 用カーネル (dec splitk の 2 インスタンス) の分

## env スイッチと既定

| スイッチ | 既定 | 効果 | 従来に戻す |
|---|---|---|---|
| `STRATA_INK_HEAD_F32` | sm_61 単機 + full logits で 1 | unembed の FP32 常駐 | `=0` |
| `STRATA_INK_DEC_ATTN` | sm_61 で 2 | decode global attention の 1 ウェーブ化 | `=0` |
| `STRATA_INK_DEC_RC` | 1 | res_ctrl の depth mix 共有 | `=0` |
| `STRATA_INK_DEC_PROF` | 0 | decode のステージ別 prof | 未指定 |
| `STRATA_INK_PF_TILE` | 1 | タイル型 prefill attention | `=0` |

- `DEC_ATTN` は sm_61 以外では 0 のまま。`DEC_RC` は sm_70+ では使われない
- 起動時に `DEC_ATTN variant = N`、`STRATA_INK_DEC_RC=<0|1>`、`STRATA_INK_HEAD_F32=<0|1>` の
  1 行ずつが出る

## ブランチの積み順

`inkling` (`c1aca006c`) の上に次の順で積んだ。author は元のまま。

1. `68b1474bc` perf: unembed の FP32 重みを常駐させて展開を省く
2. `e20c546e4` perf: unembed の FP32 常駐を sm_61 単機の既定にする
3. `303afcb13` update: decode ステージ別 prof を追加
4. `894f5c3a4` perf: decode の global attention を 1 ウェーブ化し relpos を省く
5. `136628a0f` perf: decode の res_ctrl で同一 depth mix を共有
6. `perf: decode の global attention と res_ctrl 共有を既定にする` (既定の整理)

- decode prof は `303afcb13` / `bb6e8e496` / `d74c4b708` が同一パッチ (patch-id `0206e163d`) なので
  1 回だけ入れた
- `ink_runner.cuh` で衝突が 2 回出た。どちらも「同じ位置に別々のメンバ / env ブロックを足す」
  形だったので、両方を残す最小の解決にした (メンバ 2 つ、env ブロック 2 つを並べ、閉じ括弧を
  それぞれに補った)

## 注意

- 32K の decode 27 tok/s は unembed 常駐と res_ctrl 共有を入れた時点の値で、今回の 32K アームは
  dspark (gen 64) のため直接比較していない
- dspark は unembed 常駐の対象外なので、dspark の decode はこの施策では動かない
- 512K は今回測っていない。global attention が p0 に比例する以上、512K では 256K よりさらに
  `DEC_ATTN` の寄与が大きくなる見込み (未確認)
- 8K の prefill 288.9 は冷えた最初のアームの値で、定常値ではない
