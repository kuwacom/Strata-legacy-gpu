# Inkling-2-Mini prefill attention 改善計画 (sm_61 / Tesla P40)

- 日時: 2026-11-25
- ホスト: <host> / Tesla P40 24GB (sm_61, 1531 MHz, 30 SM, GDDR5 346 GB/s, L1+shared 96 KB/SM, L2 3 MB)
- コード: `~/work/build-base` (upstream 0.2.21 / engine 0.1.67)
- 入力: `~/work/results/inkling-128k.md` (32K 179.6 / 64K 139.0 / 128K 94.6 tok/s)、`~/work/inkling-512k-plan.md` (512K の VRAM/時間モデル)
- 制約: 本計画は読み取りと計算のみ。GPU・ビルド・本番は使っていない。数値はすべて「実測から導出」か「推定」かを明記する

## 結論

128K prefill の 66% は global attention 8 visit の KV 再読みで、カーネルは **333 GB/s (spec 346 GB/s の 96%) / 1.0 TFLOPS (FP32 ピークの 8.5%)** で走っている。演算強度 3 FLOP/B は機械平衡 34 FLOP/B の 11 分の 1 で、完全なメモリ律速。原因は `k_ink_flash_gqa_gqa3` が **(kv_head, token) 単位の CTA** で、chunk 512 行のあいだ KV を 1 度も再利用しないこと。

最も費用対効果が高いのは、**既存の cuBLAS タイル型経路 (`ink_runner.cuh:3310`) を sm_61 向けに FP32 (SGEMM) で開ける**こと。分岐条件は `cc>=70 || cc==60` で sm_61 が落ちるが、落ちている理由は「FP16 が 1/64 レート」であり、dtype を FP32 にすれば理由は消える。タイル経路は 256 行ぶんの KV を 1 回の pack で共有するので、KV 読みが 256 分の 1 になり、ボトルネックが FP32 GEMM (844 TFLOP) に移る。

推定効果 (すべて推定、±20%):

| prompt | 現在 (実測) | 変更後 | 倍率 |
| ---: | ---: | ---: | ---: |
| 32K | 179.6 tok/s | **235-255 tok/s** | 1.35x |
| 128K | 94.6 tok/s | **162-196 tok/s** | 1.7-2.1x |
| 512K | 3.6-4.4 h | **83-121 分** | 1.8-3.2x |

---

## 1. 現行の prefill attention 構造 (数値付き)

### 1.1 sm_61 が選ぶ経路

`run_visit()` (`ink_runner.cuh:3122`) の 7 番、attention の分岐は 3 本。

| 条件 | 行 | カーネル | sm_61 prefill (M=512) |
| --- | --- | --- | --- |
| `M <= 16 || mode == 2` | `ink_runner.cuh:3298-3309` | `k_ink_flash_gqa_splitk` + `k_ink_flash_gqa_merge` | 使わない (decode/verify 用) |
| `mode == 0 && M >= 128 && (cur_cc >= 70 || cur_cc == 60)` | `ink_runner.cuh:3310-3394` | `k_ink_pf_pack_q/kv` + cuBLAS + `k_ink_pf_softmax_*` + `k_ink_pf_accum_tile` | **条件不成立で落ちる** |
| それ以外 | `ink_runner.cuh:3395-3401` | `k_ink_flash_gqa_gqa3` (grid `(8, M)`, block 128) | **これが動く** |

補足:
- `kInkPfKvTile = 2048` (`ink_runner.cuh:1820`)、作業領域は `ensure_pf_attn()` (`:3084-3118`) が確保する。**sm_61 でも `:2357` の `m.sm_cc >= 60` に該当するので 114 MiB は確保済みで、一度も使われていない** (分岐だけが閉じている)。
- 直前の `k_ink_rel_table` (`ink_attn.cuh:316-338`) は全 visit で走る。grid `(24, M)` で 1 行あたり 24 x 1025 x 16 FMA = 39.4 万 FMA。1 chunk (512 行 x 48 visit) で 9.7e9 FMA = 1.65 ms。無視できる。

### 1.2 `k_ink_flash_gqa_gqa3` の資源と 1 CTA の仕事

定義は `ink_attn.cuh:642-736`。中身は `q` を float4 で 3 head 分 (`:665-667`)、rel table を 3 本 (`:668-670`) 持ち、キーを 4 warp で stride 4 に分担するループ 1 本 (`:684-706`)。

| 項目 | 値 | 根拠 |
| --- | --- | --- |
| grid | `(kInkKVHeads=8, M)` -> M=512 で **4096 CTA** | `:3396` |
| block | 128 thread = 4 warp | `:642` `__launch_bounds__(128)` |
| shared | **2080 B** (`sm_m[4]` 16 + `sm_l[4]` 16 + `sm_a[4][128]` 2048) | `:708` |
| レジスタ | 常駐 `a[3][4]=12` + `qv0..2=12` + `mw/lw=6` + アドレス類 -> 40-64 と推定 (ptxas -v 未取得) | `:665-681` |
| occupancy | thread 上限 2048/128 = 16 CTA/SM、reg 64 なら 8 CTA/SM、shared は律速でない -> 8-16 CTA/SM (32-64 warp) | 計算 |
| 1 CTA のキー数 | その行の `qp+1` (global visit)。128K 末尾 chunk で **131,072 key** | `:672-673` |
| 1 key あたりの読み | 32 lane x 8 B = **K 256 B + V 256 B = 512 B** (FP16)。8 B/lane は 2 sector で完全 coalesced | `ink_kv_load4` `:64-74` |
| 1 CTA の総読み | 131,072 x 512 B = **67 MB** (128K 末尾 chunk、global visit 1 回) | 計算 |
| 1 key あたりの演算 | 3 head x (128 FMA QK + 128 FMA AV) = 768 FMA = 1536 FLOP -> 512 B で **3.0 FLOP/B** | `:689-705` |
| 1 key あたりの命令 | 約 91 warp-inst (LDG 2、cvt 8、QK 3x(4 FFMA + 5 SHFL + 5 FADD)、softmax/acc 3x13) | `:684-706` を数えた推定 |

要点: **同じ K/V を、chunk 512 行 x 8 kv_head の CTA がそれぞれ独立に読み直す**。行をまたぐ共有は L2 任せで、CTA は位置とともにずれるのでほぼ効かない (1.5 参照)。SWA visit (40 本、window 1024) も同じカーネルで `qp-1023..qp` を読む (`:672`)。

### 1.3 TC タイル経路が sm_61 で除外される理由 (P40 の FP16 1/64)

- 分岐は `cur_cc >= 70 || cur_cc == 60` (`ink_runner.cuh:3310`)。sm_61 はここに入らない。
- 仮に入れた場合、`:3312` の `use_fp16_pf = (cur_cc < 80)` で **FP16** が選ばれ、`:3314` の `pf_algo` は `cc<70` なので `CUBLAS_GEMM_DEFAULT` (TC なし) になる。
- GP102 (P40) の FP16 は FP32 の **1/64**: 11.76 TFLOPS / 64 = **0.184 TFLOPS**。128K の attention GEMM は 844 TFLOP なので、これだけで 4,600 s かかる (現行 847 s より悪い)。
- sm_60 (P100) は FP16 が 2:1 (19 TFLOPS) なので `cc==60` を通すのは正しい。sm_70+ は TC。**つまり除外条件そのものは正しく、sm_61 を通すには dtype を FP32 にする変更が別に要る**。分岐を反転させるだけでは 5 倍以上遅くなる。

### 1.4 128K の理論値と実測

グローバル visit 8 本、SWA visit 40 本 (window 1024)、head_dim 128、Q head 24 / KV head 8。

| 項目 | 計算式 | 128K 全体 |
| --- | --- | --- |
| (token, position) ペア数 | N^2/2 | 8.59e9 |
| global の KV 読み | 8 visit x 8 head x 128 dim x 2 B x 2 (K,V) = 32,768 B/ペア | **282 TB** |
| SWA の KV 読み | 40 x 1024 x 4096 B x N | **22 TB** |
| 合計読み | - | **304 TB** |
| global の FLOPs | 24 head x 128 dim x 2 x 2 = 98,304 FLOP/ペア | **844 TFLOP** |
| SWA の FLOPs | 40 x 1024 x N x 12,288 | 66 TFLOP |
| 合計 FLOPs | - | **910 TFLOP** |

実測 (128K の累積 5 点、`~/work/results/inkling-128k.md` の区間値から係数を決めたもの):
`T(n) = 4.12e-3 * n + 4.93e-8 * n^2` 秒 (n = token 数)。n=131072 で 1,387 s、実測 1,386 s。8K/32K/64K/96K でも ±2% で合う。

この式の二次項 847 s を attention とみなすと (一次項 540 s は MoE/weight/SWA などの定常部):

| 資源 | 理論上限 | 実測 (847 s で割る) | 飽和度 |
| --- | --- | --- | --- |
| DRAM 帯域 | 346 GB/s | **333 GB/s** (282 TB / 847 s) | **96%** |
| FLOPs | 11.76 TFLOPS (FP32) | **1.00 TFLOPS** | 8.5% |
| 命令発行 | 1.84e11 warp-inst/s (30 SM x 4 sched x 1.53 GHz) | 約 1.5e10 (8%) | 8% |
| shared | 2080 B/CTA、96 KB/SM | 律速でない | - |
| レジスタ | 8 CTA/SM まで | 律速でない | - |

**ボトルネックは DRAM。** 演算強度 3.0 FLOP/B に対し P40 の機械平衡は 34 FLOP/B で、11 倍メモリ寄り。したがって FMA を速くしても、命令を減らしても効かない。**KV を何度読むかを減らす以外に道がない**。

SWA は 22 TB = 66 s (128K) で、全体の 4.8%。ただし一次項に効くので 32K では 17 s / 182 s (9%)。

### 1.5 反証データ 1 件 (要再測)

FP8 KV はバイト数が半分なのに prefill が 18% 遅い (128K 77.4 tok/s)。バイト律速なら 1.8 倍速くなるはずで、これは 1.4 の結論と矛盾する。可能性は 2 つ。

1. FP8 のデコード命令 (e4m3x2_to_f32x2 x4 + scale 2 本の追加ロード) が per-key の依存鎖を伸ばし、レイテンシ律速に落ちている。この場合 FP16 も DRAM 96% ではなく、別の律速 (レイテンシ) がある。
2. アーム b は別セッションが起動した個体で、同時刻に 256K/512K の chain が VRAM を奪っていた記録がある (`inkling-128k.md` の逸脱欄)。測定が汚染されている。

**step 0 として 32K を fp16 / fp8 で 1 本ずつ取り直す (各 3 分) こと。** ここで fp8 が 1.5 倍以上速くなれば 1.4 の DRAM 説が確定し、遅いままならレイテンシ説も混ぜて考える必要がある。ただし後述の推奨案はどちらでも効く (バイトも命令数も減る)。

---

## 2. 改善案の比較

| # | 案 | 変更 | 期待 (128K) | 期待 (512K) | 規模/難度 | 主なリスク | 判定 |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| A | **既存 cuBLAS タイル経路を FP32 で sm_61 に開放** | 2 ファイル ~150 行 | 94.6 -> **162-196** | 3.6-4.4 h -> **83-121 分** | 中 / 中 | cuBLAS SGEMM の実効 (3-6 TFLOPS の幅)、S/P の VRAM 往復 | **推奨** |
| B | sm_61 専用タイル型 attention 新設 (FP32 SIMT) | 新規 ~400 行 + 30 行 | 94.6 -> 180-195 | 85-115 分 | 大 / 高 | 新規実装の正しさ (rel-pos の Toeplitz 化、sink、SWA mask)、到達 FLOPS が読めない | 次点 (A が 3 TFLOPS を下回ったら) |
| C | 既存 `gqa3` の R 行グループ化 (R=2-4) | ~120 行 | 94.6 -> 130-175 | 2.4-3.2 h | 小 / 低 | レジスタ増で occupancy 低下。DRAM 説が外れると無効 | 1 日で打てる保険 + DRAM 説の判定実験 |
| D | QK^T の DP4A 化 (k8v8 KV) | ~250 行 | A/B の上に +5-10% | 同 | 中 / 中 | 精度 (4 章)、KV 形式を k8v8 に固定する必要 | 第 2 段 (A の後) |
| E | prefill に split-K / flash-decoding | ~80 行 | **悪化** | 悪化 | 小 / 低 | KV 読みが n_split 倍になる | 却下 |
| F | chunk 幅 / visit 順 / SWA ring の見直し | ~10 行 | **ほぼ 0** | 0 | 小 / 低 | - | 却下 |
| G | 既存ループの回帰改善 (ベクトル幅・バンク) | ~20 行 | 0-3% | 同 | 小 / 低 | - | 却下 (既に 8 B/lane・完全 coalesced) |

各案の根拠:

- **A**: タイル経路は Q_CH=256 行ごとに KV を 1 回 pack して 24 head ぶん使い回す (`ink_runner.cuh:3320-3355`)。KV 読みは 32 KB/ペア -> 128 B/ペア (256 分の 1)。残るコストは FP32 GEMM 844 TFLOP と S/P の VRAM 往復。FP32 なら 11.76 TFLOPS が使える。
- **B**: A と同じタイル化を手書きする。S/P を VRAM に置かないぶん A より 20-40 s 速いが、SIMT で出せるのは FP32 ピークの 25-35% (2.9-4.1 TFLOPS) で、cuBLAS SGEMM と同等以下になりやすい。実装量は 2.5 倍。
- **C**: `gqa3` の CTA を (kv_head, m) から (kv_head, m/4) にし、1 回の `ink_kv_load4` を 4 行で共有する。バイトは 1/4、命令は行あたり 91 -> 84 でほぼ変わらない。**つまり純粋にバイトだけを減らす実験**で、DRAM 説の判定に使える。R=4 はレジスタ 160 で 3 CTA/SM (12 warp) まで落ちるので R=2 が安全。
- **D**: QK は全 FLOP の半分 (422 TFLOP)。DP4A の実測上限 35.5 TFLOPS なら 12 s だが、AV が FP32 のまま (422 TFLOP / 3.5 TFLOPS = 121 s) なので全体では効かない。A の後、GEMM が律速になってから。
- **E**: M=512 で CTA は既に 4096 個 (30 SM あたり 136)。並列度は足りている。split-K は KV 読みを n_split 倍にするので、DRAM 律速の現状では必ず悪化する。M=1 decode 用の仕組み (`:3298`) を prefill に持ち込む理由はない。
- **F**: 二次項は chunk 幅に依存しない。chunk M で `sum_c M * (c*M) = N^2/2`。SWA の窓の重なりだけが M に依存し、M=512 で 512/1535 = 33%、M=256 で 20% だが、SWA 自体が 128K で 66 s なので差は 8 s。visit 順を変えても読みは減らない。
- **G**: ロードは 8 B/lane (float4 相当の half4) で 2 sector、完全 coalesced。shared は 2080 B でバンク衝突の起きる構造でない (最終マージのみ)。伸びしろは数 %。

---

## 3. 推奨案 A: 既存 cuBLAS タイル経路の FP32 化

### 3.1 変更点 (ファイル・関数・行)

**`engine/src/inkling2/cuda/ink_attn.cuh`**

| # | 箇所 | 変更 | 行数 |
| --- | --- | --- | ---: |
| 1 | `pf_cvt_from_f32` `:741-751` | `template <bool USE_FP16>` を `template <int PFDT>` (0=bf16, 1=f16, 2=f32) に。PFDT==2 は float をそのまま返す | ~15 |
| 2 | `k_ink_pf_pack_q` `:753-764` | 出力を float に (PFDT==2)。呼び出し側は 3 分岐 | ~12 |
| 3 | `k_ink_pf_pack_kv` `:766-840` | 同上。**3 head への複製書き (`:834-839`) はそのまま** (24 head ぶんに 3 倍書き) | ~15 |
| 4 | `k_ink_pf_softmax_rel_sink` `:843-956` | probs を float で書く (`pr_row` の型を PFDT で分岐、`:859, :911, :954`) | ~20 |
| 5 | `k_ink_pf_softmax_tile_online` `:994-1065` | 同上 (`:1012, :1056`) | ~15 |

**`engine/src/inkling2/cuda/ink_runner.cuh`**

| # | 箇所 | 変更 | 行数 |
| --- | --- | --- | ---: |
| 6 | `ensure_pf_attn` `:3084-3118` | `pf_elem = (cc == 61 ? 4 : 2)` を先頭で決め、`q_b/kv_b/p_b` (`:3092-3098`) を 4 B で確保。cc は `m.ls_active ? m.stage_cc[cur_stage] : m.sm_cc` | ~12 |
| 7 | 分岐 `:3310` | `(cur_cc >= 70 || cur_cc == 60 || cur_cc == 61)` に。`pfd` (0/1/2) を決める 3 行を追加 | ~4 |
| 8 | dtype `:3312-3314` | `use_fp16_pf` を `pfd` に置換。`pf_dt = (pfd==2) ? CUDA_R_32F : (pfd==1 ? CUDA_R_16F : CUDA_R_16BF)`、`pf_algo` は cc<70 で既に `CUBLAS_GEMM_DEFAULT` | ~6 |
| 9 | 呼び出し 6 箇所 `:3317-3318, :3325-3326, :3341-3345, :3367-3368, :3377-3381` | `if (use_fp16_pf) A<true> else A<false>` を 3 分岐に (ヘルパ 1 本にまとめると差分が縮む) | ~30 |
| 10 | `k_ink_pf_online_init` / `unpack_o_gate` / `accum_tile` `:3354, :3363, :3390` | 変更不要 (S/O/l は元から float) | 0 |

合計 ~130-150 行、2 ファイル。`d_pf_q/k/v/p` の宣言 (`:1821`) は `__nv_bfloat16*` のままでよい (cudaMalloc は void** を受けるので byte 数だけ変えればよい)。

VRAM: 114 MiB -> **約 240 MiB** (q 6.3 + k 25 + v 25 + s 50 + p 50 + o 3 + その他)。128K の残りは 8.0 GB あるので影響なし。`ensure_pf_attn` の `+192 MiB` マージンも維持される。

### 3.2 期待効果 (推定)

タイル経路の 128K の内訳 (計算):

| 項目 | 量 | 時間 |
| --- | --- | ---: |
| pack_kv (FP16 KV 読み 8 MB + FP32 3 倍複製書き 50 MB) / tile-visit | 58 MB | - |
| GEMM1 読み (K 25 + Q 3) + S 書き 50 | 78 MB | - |
| softmax 読み S 50 + 書き P 50 | 100 MB | - |
| GEMM2 読み (V 25 + P 50) + O 3 | 78 MB | - |
| accum 12 + その他 | 12 MB | - |
| 合計 / tile-visit | ~270 MB | - |
| 128K 全体の tile-visit 数 (8 visit x 平均 32 tile/chunk x 256 chunk) | 65,536 | **54 s** (326 GB/s) |
| GEMM 844 TFLOP (SGEMM 実効 3-6 TFLOPS) | - | **141-281 s** |

合計 **190-330 s** (一部は重なる)。現在の 847 s から 2.6-4.5 倍。

`T(n) = 3.61e-3 * n + c * n^2` (SWA も 1/256 になるので一次項は 4.12 -> 3.61 ms/token) に c = 1.16e-8 (200 s 相当) を入れると:

| prompt | 現在 | 推定 | tok/s |
| ---: | ---: | ---: | ---: |
| 32K | 182.4 s | 131 s | **179.6 -> 250** |
| 128K | 1,386 s | 673 s | **94.6 -> 195** |
| 512K | 13,050-15,700 s | 5,080 s (85 分) | 40-36 -> 103 |

SGEMM が 6 TFLOPS なら 195 s -> 128K 668 s (196 tok/s)、512K 83 分。3 TFLOPS なら 335 s -> 128K 808 s (162 tok/s)、512K 121 分。**幅を持たせて 128K 162-196 tok/s、512K 83-121 分**と見る。

### 3.3 検証方法と合格条件

計測は既存の計測手順に乗せる (`CONTEXT.md` のロック、本番停止、温度 60 秒待ち)。

1. **step 0 (前提の確認、各 3 分)**: 32K を `--kv fp16` と `--kv fp8` で 1 本ずつ。fp8 が 1.5 倍以上速ければ DRAM 説確定。差がなければ 1.5 のレイテンシ説を併記して先へ進む。
2. **位相**: `STRATA_INK_PROF=2` で `[PROF PREFILL-M512 ... attn=...]` 行 (`ink_runner.cuh:1915, 1958, 3700-3704`、phase 3 = attention、`:3293-3403` が区間)。
   - 合格: 128K の attn 合計が **914 s -> 300 s 以下**、かつ最終 chunk の attn が最初の chunk の 1.5 倍以内 (今は 8K 区間 37 s -> 128K 区間 139 s で 3.8 倍)。
   - 参考: レポートの `STRATA_PREFILL_TIMING=1` は現行コードに見当たらない (未確認)。`STRATA_INK_PROF=2` を使う。
3. **全体**: 32K 182.4 s -> **140 s 以下**、128K 1,386 s -> **700 s 以下** (目標 650 s)。`--bench` の tok/s で 32K 234 以上、128K 187 以上。
4. **品質**: needle 24K/32K/128K 全通過、128K QA 8/8、`--bench` の greedy 出力が現行と 60/64 token 以上一致 (FP32 化で数値は良くなる方向。順序が変わるので bitwise 一致は期待しない)。
5. **副作用**: 128K で `Host KV Offload` / `spilled` が出ないこと (VRAM +130 MiB)。decode 側は分岐に入らないので不変 (`:3310` は `mode == 0 && M >= 128`)。

### 3.4 リスクと代替

- cuBLAS SGEMM が 3 TFLOPS を下回る -> 案 B (手書きタイル) に切り替えるか、`k_ink_pf_pack_kv` の 3 倍複製をやめて batch=8 x 3 回の GEMM に変えて pack を 1/3 にする (54 s -> 36 s)。
- FP32 化で `--verify-hash` 系のハッシュが変わる。これは想定内 (BF16 pack が FP32 になる)。品質ゲートは needle/QA で取る。
- `M >= 128` の条件は chunk 512 で満たすが、末尾 chunk が 128 未満だと旧経路に落ちる (正しさは保たれる、速度だけ旧)。

---

## 4. DP4A 案の精度 (スケール) の扱い

前提: 現行の計測は FP16 KV で、**K は int8 で保存されていない**。DP4A を使うには `--kv k8v8` (`ink_main.cpp:886`) にして K/V を int8 にする必要がある (decode の KV も同じ形式になる点に注意)。

- **K (既存)**: `ink_kv_store_128` (`ink_attn.cuh:140-145`) が (slot, pos, head) ごとに `sc = amax/127` を FP32 で保存。読みは `ink_kv_load4` (`:88-98`)。
- **Q (新規)**: Q は既に `kInkQkClip = 22.627` でクリップ済み (`ink_attn.cuh:190`, `k_ink_post_qkv_fused`)。行 (token, head) ごとの対称スケール `sq = amax|q|/127` を 1 個持てばよい。**スケールは 1 個/行/head で足りる**が、精度を上げたければ 32 dim ごとのグループスケール (4 個/head) にする。
- **積**: `S = sc_qk * sq * sk * sum(q_i k_i)`。int32 の桁: 128 x 127 x 127 = 2.07e6 < 2^31 で**オーバーフローしない**。累積中のスケール操作は不要、最後に 1 回だけ掛ける。
- **誤差**: 一様量子化の誤差は `delta = amax/127`。128 次元の内積の相対誤差は最悪 `sqrt(2)/127 = 1.1%`、典型 0.3%。スコア (1/sqrt(128) スケール後で ±20-30) の絶対誤差は 0.05-0.3 -> exp で 5-35% の重み歪み。**キーが 1 本だけ突出している needle では安全だが、近接キーが競合する retrieval では危ない。**
- **対策**: (a) Q を 32 dim グループスケールにする (誤差 1/4)、(b) ゲートは needle 24K/32K/128K + 128K QA 8/8、(c) 1 つでも落ちたら QK は FP32 に戻し、DP4A は使わない。
- **期待**: QK が 422 TFLOP / 35.5 TFLOPS = 12 s、AV は FP32 で 121 s。A の後段で 128K 673 s -> 620 s 程度 (7%)。**単独では推奨しない**。
- なお `ink_caps.hpp` の `ink_cap_int8_tc(major >= 8)` は P40 で false。DP4A は TC ではなく `dp4a_i8` (`ink_cuda.cuh:384-394`) の SIMT 経路で、sm_61 はハードウェア dp4a が有効。

---

## 5. 却下した案の理由 (再掲)

- **split-K / flash-decoding を prefill へ**: M=512 で 4096 CTA。並列度は足りており、KV 読みが n_split 倍になるので DRAM 律速の現状では悪化する。
- **chunk 幅**: 二次項は `sum_c M*(cM) = N^2/2` で M に依存しない。SWA の重なりだけが M 依存で、128K で最大 8 s。
- **visit 順 / SWA ring**: 読みの総量が変わらない。ring は 1024 で正しく回っており (`ink_attn.cuh:685` の `kp % cap_pos`)、改善余地はない。
- **既存ループのベクトル幅・バンク**: 既に 8 B/lane・完全 coalesced、shared 2080 B。伸びしろは数 %。
- **sm_61 で分岐条件だけ反転**: FP16 1/64 で 5 倍以上遅くなる。dtype の FP32 化が必須 (1.3)。

---

## 6. 未確認・注意

- GPU・ビルド・本番は使っていない。**すべての性能数値は実測からの導出か推定**で、実機確認はしていない。
- SGEMM の実効 (3-6 TFLOPS) は P40 での実測がない。ここが推奨案の最大の不確定要素。
- レジスタ数は ptxas -v を取っていない (40-64 はソースからの推定)。
- L2 の再利用率は測っていない。1.4 は「再利用ゼロと仮定すると spec の 96% に一致する」という整合性の議論で、直接測定ではない。FP8 の逆転 (1.5) が解ければ確定する。
- 512K の現在値は 3.6 h (`inkling-512k-plan.md` の 13,050 s) と 4.4 h (本計画の外挿 15,700 s)、観測「5 時間」の 3 つがある。変更後の 83-121 分はこの幅に対する推定。
- 512K は `--kv k8v4` 前提 (`inkling-512k-plan.md`)。k8v4 の V は int4 なので、A の FP32 pack はそのままデコードできる (`ink_attn.cuh:822-827`) が、V の精度 (SNR 18 dB) は別問題。
- DP4A は KV 形式を k8v8 に固定する。decode の速度・精度への影響は未評価。
