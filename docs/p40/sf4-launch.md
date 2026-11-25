# SF4/MX8 MoE 起動の絞り込み (実在 expert だけに grid を張る)

調査日: 2026-11-25 / 対象: `~/work/ink` (branch `perf/ink-sf4-launch`、`perf/ink-moe-prof` = `08c919db3` から分岐) / バイナリ sha256 `8d3475e6…` / Tesla P40 (sm_61) 単機 / SQ-10 / 32K prefill。GPU は共用ロック内で使い、本番は復旧済み (healthy、`.env` sha256 `(省略)` 不変、両ロック解放)

## 結論

- **仮説は「記述としては正しい」**: SF4/MX8 の起動は `grid = (72, max_act=128)` で、実際に演算するのは層の SF4 expert 数 k だけで、**block の 98.40% が空 return** していた
- **時間の支配要因でもあった**: 実在 expert のみに絞ると **SF4 区間 761.84 -> 656.04 ms/step (−105.80 ms、−13.9%)**、**prefill 231.3 -> 242.9 ppts (+5.0%)**。同一バイナリ・同一条件の A/B
- 重み・量子化・カーネル演算は無変更。**同一プロセス内の bit 比較で 20/20 visit が完全一致** (下記)
- 残る SF4 656 ms は「15 expert の実演算」で、律速は起動ではなく occupancy (72 CTA/層 = 2.4 CTA/SM)。次の手はそこ

## 仮説の検証 (起動の決まり方と空 return 率)

### grid の決まり方 (`ink_moe_cuda.cuh` の 2B else 分岐)

```
dim3 g16(kInkInter / 16, max_act);      // kInkInter=1152 -> x=72
max_act = min(M * 8, 128);              // chunk 512 -> M=512, P=4096 -> max_act=128
```

- `blockIdx.y` = `ai` は `tb.active[]` (bin カーネルが作る「cnt>0 の expert」の compact リスト) の添字
- 空 return は 2 段:
  1. `if (ai >= *tb.nact) return;` — 未ルーティング expert (nact は層あたり 37..50 を実測。合成プロンプト次第)
  2. `if (ed.tier > 1 || (ed.gu_code != FMT_SF4 && ed.gu_code != FMT_MX8)) return;` — 形式不一致 (支配的)

### 層ごとの SF4 expert (Section 3 を parse、main model 24x128)

| 層 | SF4 expert id | k | visit/step |
|---|---|---|---|
| 1 | 84 | 1 | 1 |
| 2 | 7 | 1 | 1 |
| 8 | 123 | 1 | 3 |
| 10 | 24 | 1 | 3 |
| 11 | 100, 74 | 2 | 3 |
| 15 | 67, 58 | 2 | 3 |
| 16 | 64, 92 | 2 | 3 |
| 17 | 124, 104, 3, 41, 63 | 5 | 3 |
| 合計 | 15 expert | | **20 visit** |

- visit 数は `core_start=6, core_end=18, num_passes=3` (24 層 -> 48 visit)。SF4 層のうち 8/10/11/15/16/17 が 3 visit、1/2 が 1 visit
- `Σk = 2x1 + 3x(1+1+2+2+2+5) = 41` expert-visit/step

### 空 return 率

```
1 launch の CTA 数            = 72 x 128 = 9,216
1 launch で演算する CTA       = 72 x k
空 return 率 (k の層)          = 1 - k/128
step 全体 (gu 20 launch)       = 1 - Σk / (20 x 128) = 1 - 41/2,560 = 98.40%
step の CTA 数                 = 20 x 9,216 = 184,320  (演算するのは 72 x 41 = 2,952)
```

- dn も同じ grid・同じ k (SQ-10 は SF4 の gu/dn が同 code) なので同じ 98.40%
- gu+dn で **1 step あたり 362,736 CTA が空 return** していた

## 変更 (3 ファイル、+62 −11 行)

| ファイル | 内容 |
|---|---|
| `ink_moe_cuda.cuh` | bin カーネルが層ごとの `active_gu` / `active_dn` (実在 SF4/MX8 expert の compact リスト) と件数を作る。2B の起動で `grid.y = min(max_act, info.n_gu_fmt)` とし、カーネルには `e_list` / `n_list` を渡す |
| `ink_moe_fp4_sm120.cuh` | `InkMoeTables` に `active_gu` / `active_dn` / `nact_gu` / `nact_dn` を追加 |
| `ink_runner.cuh` | 層の SF4/MX8 expert 数 `n_gu_fmt` / `n_dn_fmt` を host で数える (2 箇所の loader) |

- `STRATA_INK_SF4_COMPACT` は既定 1、`0` で従来 grid (`max_act` + `tb.active`/`tb.nact`) に戻る。既存 env の意味は不変
- カーネルの演算本体・重み・量子化・書き込み先は無変更 (引数が 2 個増え、expert の引き方だけが変わる)
- リスト作りを prefix sum でなく **thread 0 の直列走査 (128 回)** にしたのは下記「逸脱・注意」の理由

## 正しさ

### 同一プロセス内 bit 比較 (検証専用コード、コミットに含めず)

compact 起動の直後に同じ入力を従来 grid でもう一度走らせ、`h_f32` と `y_pair` を D2H で比較 (先頭 20 visit = 1 chunk 分の全 SF4 visit):

| 比較 | 結果 |
|---|---|
| compact -> 従来 (20 visit) | **h_diff=0 / y_diff=0 が 20/20** |
| compact -> compact (再実行) | 常に 0 (カーネル自体は決定的) |

- 比較した visit の内訳: `active_gu` は `[7]` `[24]` `[100]` `[58,67]` `[64,92]` … と層の SF4 expert と一致 (legacy 側の `active` から拾った SF4 と同一集合)。nact_gu も 0..3 で実測と一致
- 出力は (token, feat) / (pair, row) ごとに 1 CTA が書くだけで atomic 加算が無いため、起動順・block の割当が変わっても bit 一致する (実測がそれを裏付け)

### トークン列の A/B (参考)

同一入力・greedy・`--gen 64` (1024 id のパターン入力):

| 比較 | 一致トークン | logits hash |
|---|---|---|
| A/A (COMPACT=0 を 2 回) | 26/64 | DIFF |
| A/A (COMPACT=1 を 2 回) | 26/64 | DIFF |
| A/B (0 vs 1) | 26/64 | DIFF |

- **このエンジンは run-to-run で非決定的** (A/A でも logits hash 65/65 が違う)。したがってトークン一致は判定に使えない。A/B の一致率が A/A と同じ 26/64 であることは「差がノイズと同程度」を意味するが、証拠としては上の bit 比較の方が強い
- 参考: 別入力 (4096 id の乱数) では A/B のトークンが 64/64 一致したが、その入力は 4 トークンの繰り返しに落ちる退化出力なので証拠にしない

## 性能 A/B (同一バイナリ、32K、chunk 512、fp16、dspark、`STRATA_INK_PF_TILE=1 STRATA_INK_PROF=2`、アーム間 60 s)

| 項目 | COMPACT=0 | COMPACT=1 | 差 |
|---|---|---|---|
| gu(sf4/mx8) | 481.03 | 419.67 | **−61.36 (−12.8%)** |
| dn(sf4/mx8) | 280.81 | 236.37 | **−44.44 (−15.8%)** |
| **SF4 計** | **761.84** | **656.04** | **−105.80 (−13.9%)** |
| gu(sk) | 609.92 | 609.68 | −0.04% |
| dn(sk) | 357.92 | 358.04 | +0.03% |
| router + bin | 3.76 + 1.58 | 3.77 + 1.58 | ±0 |
| reduce8 (`moe=`) | 7.07 | 7.07 | 0 |
| MoE 位相 | 1,742.08 | 1,636.17 | −105.91 (−6.1%) |
| step 全体 | 2,206.71 | 2,100.68 | −106.03 (−4.8%) |
| prefill | 231.3 ppts (141.67 s) | **242.9 ppts (134.89 s)** | **+5.0%** |
| decode (参考) | 25.3 tps | 25.8 tps | +2.0% |

- S_K 区間 (967.8 ms)・router・bin・reduce8 は動いていない = 変更が SF4 起動だけに閉じている
- 1 expert-visit あたり: gu 11.73 -> 10.24 ms、dn 6.85 -> 5.77 ms
- 削減 105.8 ms は「空 CTA の起動・退役のコスト」で、事前見積 (2·dn − gu ≈ 79 ms) と同じ桁。**97% が空でも、時間の 86% は実演算** (656 ms) だった

## ビルド・実 resource

- インクリメンタルビルド 113 s (configure 2 s)。ログ `~/work/logs/ink-sf4-launch-final-build.log`
- `cuobjdump -lelf`: `sm_61` のみ 2 エントリ
- `cuobjdump -res-usage` (従来と同一):
  - `k_ink_moe_gu_sf4_mx8<4>` REG **93** / SHARED 18,432 / STACK 0
  - `k_ink_moe_dn_sf4_mx8<4>` REG **79** / SHARED 9,216 / STACK 0
  - `k_ink_moe_gu_sk_fmt<2,16>` 160 / `<3,16>` 158、`dn_sk_fmt<2,16>` 134 / `<3,16>` 132 (不変)
  - `k_ink_moe_bin` REG 32 / SHARED 2,560 (不変)

## 逸脱・注意

- **最初の実装は prefix sum で失敗した**: `k_ink_moe_bin` の既存 Hillis-Steele に 2 配列 (`s_fgu`/`s_fdn`) を足す方式では、リストが壊れた (多数 thread が index 0 に書き、`nact_gu` が 1 に固定) -> compact 起動が何も実行せず、自己検証で h/y が大量不一致になった。原因は特定できていない (5 配列 + `#pragma unroll` + `__syncthreads` の組み合わせ)。**thread 0 の直列走査に置き換えて解決**。同種の拡張をするときは自己検証を先に入れること
- 検証専用の自己検証コード (約 60 行) は**コミットに含めていない**。再現するなら `ink_moe_cuda.cuh` の 2B else 分岐の直後に「h_f32/y_pair を D2H で退避 -> 従来 grid で再実行 -> 比較」を入れ、`cudaMallocHost` のバッファは P に応じて再確保する
- 自己検証中に **device pointer を host で `*p` 参照して segfault** した (1 回)。device の値は `cudaMemcpy` で取ること
- `nact` (層あたりの active expert 数) は入力依存 (測定プロンプトで 37..50)。空 return 率の式 `1 - k/128` は nact に依存しない
- 性能は各アーム 1 本 (指示どおり)。位相の再現性 ±0.25% に対し差は 4.8〜5.0% なので判定に足る
- 測定バイナリは `8d3475e6…` (自己検証なしの最終ビルド)。自己検証入りのバイナリ (`6582944` B) は残していない

## 次の提案

1. **occupancy**: 残る SF4 656 ms は 72 CTA/層 = 2.4 CTA/SM (容量 5) の実演算。`MAX_M=16` 版や block 256 thread 化で CTA あたりの仕事を増やす/減らす実験。空 return はもう無いので、次はここ
2. `reduce8` 7.07 ms と bin 1.58 ms は据え置き (MoE の 0.5%)
3. S_K 967.8 ms は依然 MoE の 59%。`__launch_bounds__(128,4)` (REG<=128) の occupancy 実験は未着手
4. 15 SF4 expert を S3_K に再量子化する案 (prereq の提案 2) は、今回の削減で SF4 が 656 ms になったため優先度は据え置きでよい
