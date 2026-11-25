# Inkling-2-Mini prefill MoE (gu/dn) - Tesla P40 (sm_61) 向け設計調査

調査日: 2026-11-25 / 対象: `~/work/ink` branch `perf/ink-attn-sm61` (HEAD `ba6bfb870`)、読み取りのみ
実測値の出所: `~/work/results/attn-ab/tiled.err` (SQ-10, PREFILL-M512) と `base.err`

## 0. 前提

- ビルドは sm_61 のみ: `~/work/ink/build/CMakeCache.txt` の `CMAKE_CUDA_ARCHITECTURES=61-real` (コンテナ内パス `/work/strata/...` = `~/work/ink`)
- 能力判定: `engine/include/strata/core/ink_caps.hpp:16` の `ink_cap_int8_tc(major)=major>=8`、`:17` の `ink_cap_wmma_f16(major)=major==7`。P40 は major=6 なので
  `use_sm120_tc` / `use_fp16_tc` / `use_wmma_tc` はすべて false (`engine/src/inkling2/cuda/ink_moe_cuda.cuh:3676-3680`)
  -> 2A (WMMA) と 2B 上段 (SF4/MX8 の TC) は不成立。**S_K の dp4a 経路 (2C) が本命**
- モデル: `Inkling-2-Mini-SQ-10.strata`。expert codec 内訳 (`attn-ab/tiled.err:1`):
  `2/2: 585, 3/2: 1814, 3/3: 658, 15/15: 15` -> **3057/3072 が S2_K/S3_K**、SF4 は 15 expert だけ (0.5%)
- 定数 (`ink_cuda.cuh:159-175`): `kInkVisits=48`, `kInkHidden=2304`, `kInkInter=1152`, `kInkExperts=128`, `kInkTopK=8`
  48 visit = layer 0-5 + core 6-17 を 3 回 + 18-23 (`ink_cuda.cuh:244-250`, `kInkCoreStart=6/KEnd=18/KPasses=3`)
- 実測 (ms/step, 48 visit 合計): router+topk 3.76-3.81 / bin 1.03-1.10 / **gu 444-548 (代表 490)** / **dn 250-310 (代表 285)** / gu(gguf)=dn(gguf)=0
  `moe=` は 983 ms/step で、内訳合計 706 ms との差 277 ms は reduce8 + 計測区間外 (6. 参照)

## 1. prefill MoE の呼び出し経路

```
ink_runner.cuh:3545  launch_ink_moe(st, cublas, m.sec4_stage(cur_stage), m.d_experts[ip], m.moe_info[ip],
                                    gate_w, gate_b, d_moe_in, d_moe_raw, M, *cur_moe_ws,
                                    route_scale, (mode==2), !fuse_moe_sconv)
  └─ ink_moe_cuda.cuh:3864  launch_ink_moe
      ├─ :3880 launch_ink_moe_router (定義 :3637)
      │    ├─ M が 16 以下: k_ink_router_gemv_smallM <...> (:3647-3651)
      │    ├─ M が 16 超  : cublasGemmEx(gate_w[128,2304] x x[512,2304], FP32) (:3653-3659)
      │    ├─ k_ink_router_select_topk8 (grid M, block 32) (:3662, 定義 :211)  sigmoid+bias の exact top-8, w = p*route_scale/psum
      │    └─ ink_moe_mark(0) (:3662)  -> PROF tag0
      ├─ :3883 launch_ink_moe_dispatch_only (定義 :3666)
      │    ├─ k_ink_moe_bin (grid 1, block 128) (:3688, 定義 :278)  128 expert の counting sort + tile list (device 側, host sync なし)
      │    ├─ ink_moe_mark(1) (:3689)
      │    ├─ 2A WMMA (sm_70/75 のみ :3692-3720) : P40 では丸ごと skip
      │    ├─ 2B SF4/MX8 (:3722-3780) : 15 expert のみ。MAX_M は 1 か 4 (:3771-3779)
      │    ├─ 2C S_K (:3785-3810)  ★P40 の本体
      │    │    ├─ k_ink_quant_q8_1(x_f32 -> ws.xq_sk) (:3788, 定義 :1099)
      │    │    ├─ launch_ink_moe_gu_sk_mask MAX_M=16 (:3803, 定義 :3601)
      │    │    │    -> k_ink_moe_gu_sk_fmt FMT_S2_K|S3_K,16  grid (1152/32, max_act) block 128 (:3606-3612, 定義 :2109)
      │    │    │       └─ sk_gu_fused_4lane (:1829) -> sk_unpack64_g_to_u32 (:1733) + dp64_i8_smem (:1810)
      │    │    │          └─ SwiGLU (situ_f32) + inv_sq を掛けて h_f32 へ (:2162)
      │    │    ├─ k_ink_quant_q8_1(h_f32 -> ws.hq_sk, P=4096 行) (:3804)
      │    │    └─ launch_ink_moe_dn_sk_mask MAX_M=16 (:3808, 定義 :3616)
      │    │         -> k_ink_moe_dn_sk_fmt grid (1152/32, max_act) block 128 (:2171)
      │    │            └─ sk_dn_paired2_4lane (:1902) -> y_pair[pid] = Σ * topk_w[pid] (:2225)
      │    └─ 2D GGUF dp4a (:3812-3861) : SQ-10 では未使用 (gu(gguf)=0)
      └─ :3887-3888 k_ink_moe_reduce8 grid (2304/256, M) (定義 :352)  8 slot の決定的加算 -> d_moe_raw
ink_runner.cuh:3552  run_sconv_uniform(d_moe_raw, sconv_mlp, ..., d_cur)  (post-MLP ShortConv4 + residual add)
```

- SwiGLU は独立カーネルではなく **gu カーネル内** (`:2162` の `situ_f32(ag[m], au[m])`)
- `fuse_moe_sconv` は prefill (mode 0, M=512) では false なので reduce8 は必ず走る (`ink_runner.cuh:3544-3549`)
- 参考: EP2 (2 GPU) は `:3483` router -> `:3523` dispatch_only -> `:3525` reduce8_tier0、hybrid は `:3536` `launch_ink_moe_hybrid` (`ink_hybrid.cuh:2472`)

## 2. 現行 gu / dn の形 (P40 で実際に走るカーネル)

### 2.1 タイル形状と M

| 項目 | gu (`k_ink_moe_gu_sk_fmt FMT,16`) | dn (`k_ink_moe_dn_sk_fmt FMT,16`) |
|---|---|---|
| grid | `(kInkInter/32=36, max_act<=128)` = 最大 4608 CTA (`:3800`) | `(kInkInter/32=36, max_act)` = 最大 4608 CTA (`:3807`) |
| block | 128 threads (32 feature x 4 lane) | 128 threads (32 row-pair x 4 lane) |
| CTA の担当 | 32 feature (gate 行 + up 行の 2 行/feature) | 32 row-pair = 64 出力行 (K=1152) |
| 1 expert の M | **平均 32 token** -> `MAX_M=16` なので 2 反復 (`off_m += 16`, `:2131`) | 同じ 2 反復 (`:2200`) |
| K の進め方 | 9 sub-block x 4 group x 64 weight = 2304 (`:1840` の `for s<9`) | 9 sub-block (2 行分で 2304 weight, `:1913`) |

- `MAX_M` はステップ全体の M で選ばれる: `M==1 -> 1`, `M<=32 -> 4`, それ以外 `-> 16` (`:3800-3803`)。**chunk 512 では必ず 16**
- `MAX_M<=4` のときだけ活性を shared に置く (`SM_M=MAX_M`)。`MAX_M=16` では `SM_M=1` で **活性はグローバルから直接読む** (`:1994-2009`, `:2138-2148`)

### 2.2 重みの読み方と dequant の位置

- 重みは **shared にステージしない**。各 lane が自分の 64-weight group を直接 global から読む:
  S3_K は `pl + g*16` の 4 x u32 + `pl+64+g*8` の 2 x u16 (`sk_unpack64_g_to_u32`, `:1738-1760`)
- dequant (unpack) は **レジスタ内**。`sk_unpack64_g_to_u32` が 2/3/4/5/6/8 bit を int8 x 4 (`w[16]`) に展開し、その場で dp4a
- group scale は `eff = f16(blk) * blk[2+g]/255` として per-token の epilogue で適用:
  `acc = fmaf(eff, xb.d*isum - HALF*xb.s, acc)` (`:1876`, `HALF=ink_sk_half(FMT)=1.5/3.5`, `ink_cuda.cuh:232-240`)
  非対称 offset (q-1.5 / q-3.5) を `xb.s = d*Σq` で補正する形。Qwen fused の「offset を活性側の和に畳む」と同じ発想 (`moe_fused.hpp:25-26`)
- 重みの再読み出しは「(expert, 16 token tile) ごとに 1 回」= 1 expert あたり 2 回

### 2.3 DP4A の使い方

- `dp64_i8_smem` (`:1810-1827`): 64 weight を 16 dp4a で 1 つの int32 に積む。活性 64 値は `xb.qs` から uint4 x 4 = 4 ロード
- `dp4a_i8` は `ink_cuda.cuh:384-386` で `__CUDA_ARCH__==610 || >=750` のとき **実 __dp4a**。sm_61 は 610 なので HW 命令
- gate と up は活性 `xb` を共有して 2 回 dp4a (`:1874`, `:1885` の順次 unpack で w[16] を再利用)

### 2.4 共有メモリと occupancy

| | gu | dn |
|---|---|---|
| shared (MAX_M=16) | `s_xq[1][36]` x 80 B = **2,880 B** (`:2136`, `:1988`) | `s_hq[1][18]` x 80 B = **1,440 B** (`:2196`, `:2058`) |
| shared (MAX_M=4 のとき) | 11,520 B | 5,760 B |
| launch_bounds | `(128, 2)` (`:2109`) | `(128, 2)` (`:2171`) |
| 目標 occupancy | 2 CTA/SM = 8 warp = 64 warp の **12.5%** | 同 |
| 実 register 数 | **未確認** (cuobjdump がホストに無い) | 未確認 |
| 備考 | S1_5 のみ `(128,2)`、MAX_M=1 は 6、MAX_M<=4 は 5 | 同 |

- 128 thread x 16 token 分の `ag[16]+au[16]` = 32 float の accumulator を持つため register 圧は高い。`:2107` のコメントが「汎用版は REG:255 STACK:32..72 まで落ちた」と記録
- `#pragma unroll 1` が s ループに付いており (`:1840`, `:1863`, `:1913`, `:1930`)、**sub-block 間のプリフェッチが無い** -> global load の latency がそのまま出る

### 2.5 SF4/MX8 経路 (15 expert のみ、参考)

`k_ink_moe_gu_sf4_mx8 4` (`:1355`) / `k_ink_moe_dn_sf4_mx8 4` (`:1537`)。shared は `s_x[4][1152] half2` = 18,432 B。重みは `sf4_decode16_h2` / `mx8_decode16_h2` (`:1310` / `:1327`) で half2 に展開し `dot16_h2_2acc` (`:1341`) の **__hfma2**。P40 の FP16 は 1/64 なのでこの経路は構造的に不利 (expert 数が 15 なので実害は小)

## 3. 32K prefill (chunk 512, 128 expert top-8) の量と律速判定

### 3.1 expert あたりトークン数

```
tokens/expert = M * topk / E = 512 * 8 / 128 = 32
```
(一様分布の平均。実際は多項分布でばらつくが bin は expert 毎に ceil する。合計は 4096 で一定)

### 3.2 1 expert あたりの重み量 (実測 codec 内訳で重み付け)

```
S3_K: gu = 2*1152 行 * 9 block * 102 B = 2,115,072 B = 2.017 MB
      dn = 1152 pair * 9 block * 102 B = 1,057,536 B = 1.008 MB
S2_K: gu = 1,451,520 B = 1.384 MB,  dn = 725,760 B = 0.692 MB
SF4 : gu = 2*1152*(2304/2 + 2304/16) = 2.986 MB, dn = 1152*(576+72) = 0.746 MB
加重平均 (585/1814/658/15) : gu = 1.993 MB, dn = 0.797 MB   <- ログの mean MB (2.18/2.85/3.18/4.48) と整合
```
(block バイト数は `ink_cuda.cuh:219-228`, gu の行は `:1985` の `gr/ur = 2*feat, 2*feat+1`、dn の pair は `:2054` の `pair_b = 9*SB`)

### 3.3 重み読み出し量 (passes = ceil(32/MAX_M))

| MAX_M | passes | gu / visit | dn / visit | 48 visit 合計 |
|---|---|---|---|---|
| 4 (現行の M<=32 側) | 8 | 1,946 MB | 778 MB | 127.7 GB |
| **16 (現行 chunk512)** | **2** | **487 MB** | **195 MB** | **31.9 GB** |
| 32 (1 pass の理想) | 1 | 243 MB | 97 MB | 16.0 GB |

```
gu/visit = 128 expert * 1.993 MB * 2 = 486.6 MB
step     = (486.6 + 194.6) MB * 48 = 32.7 GB  (切り捨て 31.9 GB)
```

### 3.4 律速判定 (P40: 346 GB/s, DP4A 実測 35.5 TFLOPS)

```
実測 gu+dn = 490 + 285 = 775 ms/step
実効帯域   = 31.9 GB / 0.775 s = 41 GB/s = 346 GB/s の 12%
DRAM floor = 31.9 GB / 346 GB/s = 92 ms
MAC 量     = 512*8*(2*1152*2304 + 2304*1152) * 48 = 1.565 TMAC (= 3.13 TFLOP)
DP4A 数    = 1.565e12 / 4 = 391e9 命令
DP4A floor = 1.565e12 MAC / 17.75e12 MAC/s = 88 ms   (35.5 TFLOPS を 2 op/MAC と解釈)
           = 44 ms                                   (35.5e12 MAC/s と解釈した場合)
issue floor = 0.445 instr/MAC * 1.565e12 / 5.87e12 lane-instr/s = 119 ms
              (5.87e12 = 30 SM * 128 lane * 1.53 GHz, 0.445 = 16 dp4a + 4 load + unpack/16 + fma の見積)
```
- **帯域律速ではない** (12%)。**DP4A 律速でもない** (9-18 倍の余裕)。**命令発行律速でもない** (6.5 倍の余裕)
- 残る説明は latency/occupancy: 8 warp/SM、活性はグローバル読み、s ループが `unroll 1` でプリフェッチ無し、`ag[16]/au[16]` による register 圧
  - CTA 反復あたりの実測: gu は 48*128*2*36 = 442,368 CTA 反復を 490 ms。2 CTA/SM 同時なら 1 反復 ≒ 66 us = 101k cycle。命令から見た下限は 6.8k cycle -> **約 15 倍の stall** (推測: 内訳は未計測)
  - MAX_M=4 (8 pass, 127.7 GB) でも DRAM floor は 369 ms で実測 775 ms を下回る。つまり tile をどちらにしても「stall 支配」で説明できる
- 結論: **現行 775 ms の主因は「重み 2 pass 読み」ではなく「8 warp/SM + プリフェッチ無し + グローバル活性読み」**。したがって
  1. 重み 1 回読みあたりの token 数を増やす (32 に近づける) と同時に、
  2. warp 数を増やし (256-512 thread/CTA, 2-4 CTA/SM)、
  3. 重み/活性を shared にステージして明示プリフェッチする
  の 3 点セットでしか効かない (1 だけでは 3.3 の表のとおり 2 倍止まり)

## 4. Qwen 側 (MMQ / fused / weight-stationary) から移植できる技法と sm_61 48KB 判定

| # | 技法 | 根拠 (file:line) | sm_61 48KB 判定 |
|---|---|---|---|
| 1 | **weight-stationary (1 thread = 1 重み行, TPP token をレジスタ accumulate)** | `Strata-legacy-gpu/include/strata/prefill/moe_ws.hpp:5-8`, `src/prefill/moe_ws_gu_iq3.cu:28,91-101,119-188` | **条件付き成立**。TPP=8 なら活性 8*2880=23 KB -> 2 CTA/SM、TPP=16 は 45 KB -> 1 CTA/SM。重み行数/CTA は 128 か 256 (`NT=128`, `PIECES=GU_ROWS/NT`) |
| 2 | **u (expert あたり行数) でタイルを切り替える** | `moe_ws_down.cu:620-644` (u が 5 未満: 128x8, 20 未満: 256x16, 以上: 256x32), `fewrows.hpp:21-26` (1..12 行は量子化カーネルへ) | **成立**。Inkling は現在 M (ステップの token 数) で選んでおり (`:3800`)、`tb.cnt[e]` (expert の実 token 数) で選ぶべき。cnt=32 の層と cnt が小さい層が混在する |
| 3 | **SwiGLU + 出力の int8 量子化を gu の epilogue で融合** | `mmq/moe_mmq.cu:72-118` (swiglu_quant_kernel), `moe_fused.hpp:20-22` | **成立** (5. 参照)。64 feature を 1 CTA が持つ必要あり |
| 4 | **重みを shared にステージして多 token で再利用** | `moe_ws_gu_iq3.cu:98-101,119-188` (`wsm[NT*CWS]`), `moe_ws.cuh:40-44` (cta_work) | **成立**。S3_K で 128 行 x 32 weight = 4 KB 程度。活性と合わせて 20-40 KB |
| 5 | **明示プリフェッチ (全 load を store の前に発行)** | `moe_ws_gu_iq3.cu:9-13` のコメント, `load_w/store_w` (:126-160) | **成立**。現行の `unroll 1` を外して 2 段パイプラインにするのが最小変更 |
| 6 | **int8 積和を 2 x mad.wide.s16 でエミュレート (sm_60 用)** | `moe_ws.cuh:20-33` (`split2`/`mad2`) | **不要**。sm_61 は実 `__dp4a` (`ink_cuda.cuh:384-386`)。むしろ 4 倍得 |
| 7 | **float 変換を magic number で回避 (`kMagicI/kMagicF`)** | `moe_ws.cuh:35-37` | **成立**。Pascal の I2F は quarter rate。現行 `:1876` の `(float)isum` を置換可能 |
| 8 | **K 方向のタイル設定 (I=128/J=64/K_vram=256, 256 thread, occ 2)** | `mmq-config-pascal-dp4a.cuh:272`, `mmq.cuh:9,165-183,256` | **成立**。J=64 は token 64 を要求するので、Inkling では J=32 (cnt=32 に一致) + I=128 行が自然。SRAM は重み 128 行 x 256 weight (S3_K で 13 KB) + 活性 32 token x 256 int8 (8 KB) で約 21 KB -> 2 CTA/SM |
| 9 | **stream-k / split-K** | `mmq.cuh:165-183` (`stream_k`), `moe_mmq.cu:451-460` | **不成立 (不要)**。1 expert の作業が小さく、K=2304 を割る利益が無い |
| 10 | **64 token tile の prefill カーネル** | `ink_moe_prefill.cuh:17-18,24,47-58` | **不成立**。`__CUDA_ARCH__` が 800 以上の mma.m16n8k16 前提 (sm_61 に tensor core 無し) |
| 11 | **expert ポインタをカーネル引数で渡す (gather 無し)** | `moe_fused.hpp:14-22,47-52` (`kMaxBatch=128`) | **成立**。Inkling は既に `InkExpertDev*` + `tb` 方式で同等 (`:1355` の引数) |

- 「I=128 タイル」は MMQ の Pascal 設定では **既定 config** (`mmq-config-pascal-dp4a.cuh:272` の `128, 64`) として存在し、型ごとの個別設定は I=64/J が 64 以下 (`:2-12`)。Inkling 側の既存 64 token tile は tensor core 前提なので、形状だけ借りて dp4a で書き直す形になる

## 5. SwiGLU 融合と gu 内 int8 量子化

### 5.1 現行のデータフローと削減量

```
gu -> h_f32 [P=4096, 1152] fp32  (書き: :2162)     18.9 MB/visit
   -> k_ink_quant_q8_1 (読み 18.9 MB + 書き 5.9 MB) (:3804, 定義 :1099)
   -> hq_sk [P, 18] (80 B/block) -> dn (:2225 で y_pair へ)
中間バッファ総交通量 = 18.9 + 18.9 + 5.9 + 5.9 = 49.6 MB/visit -> 2.4 GB/step
```
- 2.4 GB/step は 41 GB/s なら 58 ms、346 GB/s なら 7 ms。**単体では 775 ms の 1-7%** だが、カーネル 1 本 (73,728 block) と 18.9 MB の fp32 を消せる
- より大きいのは「gu が fp32 を書き、dn が int8 を読む」非対称を無くせること (gu の store が 1/4 になる)

### 5.2 可否: **可能 (bit-exact も可)**

条件は「1 CTA が 64 feature を所有すること」。現行は 32 feature/CTA なので 64 値グループの absmax が取れない:
- `:1986-1990` の `sub = tid>>2; g = tid&3; feat = blockIdx.x*32 + sub` -> 64 値グループが 2 CTA にまたがる
- `sk_gu_fused_4lane` は 4 lane で 1 feature を分担し、`situ_f32` の結果は `g==0` の lane だけが書く (`:2021-2024`)

### 5.3 変更点 (具体)

1. `k_ink_moe_gu_sk_fmt` (`:2109`) の CTA を **64 feature / 256 thread** に:
   - `sub = tid >> 2` (0..63), `g = tid & 3`, `feat = blockIdx.x * 64 + sub`, grid.x = `kInkInter/64 = 18`
   - 起動側 `launch_ink_moe_gu_sk_mask` (`:3601`) の grid を `kInkInter/64` に、block を 256 に
2. `situ` 後に CTA 内で 64 値の absmax: `__shared__ float s_amax[SM_M][64]` に 1 値ずつ書き、`__syncthreads()` 後に 64 値の max (2 warp 分の shuffle + shared)
3. 量子化式は `k_ink_quant_q8_1` (`:1106-1125`) と同一にする:
   ```
   d = amax / 127 ; inv = 127/amax ; q = clamp(rn(v*inv), -127, 127) ; s = d * Σq
   ```
   `inv_sq` (`:2133`) を掛けた**後**の値で量子化する (現行は h_f32 に書いた値がそのまま量子化されるため)
4. 書き込み先を `ws.h_f32` から `ws.hq_sk` に変更 (両方書けばデバッグ比較も可能)。dn 側 (`:2171`) は **変更不要**
5. 呼び出し側 `:3804` の `k_ink_quant_q8_1(h_f32 -> hq_sk)` を削除 (h_f32 が不要になるため)。`ws.h_f32` の確保 (`:77`) も dn が読まなくなるなら縮小可

- 注意: `h_f32` は他経路 (wmma / fp16tc / sm120) も使うため、S_K 経路専用のフラグを立てて分岐する必要がある (`launch_ink_moe_dispatch_only:3666` の引数追加が最小)
- 注意: 需要側の `sk_dn_paired2_4lane` は `h_ptrs[m][xg]` の 18 ブロックを前提 (`:1942`)。G64 のまま変えないこと

### 5.4 効果と限界

- 効果: 5.1 の 2.4 GB/step の削減 + kernel 1 本 + gu の store 4 倍減
- 限界: 3.4 のとおり律速は latency/occupancy なので、**これ単独では数 %**。4 章の 1/4/5 (weight-stationary + shared ステージ + プリフェッチ) と同時に入れて初めて意味が出る
- x 側の量子化 (`:3788`) は per-visit で 512 token 分だけなので既に十分小さい (1.5 MB write)

## 6. 未確認・注意

1. **prof tag の不整合 (要確認)**: 現行 source の S_K 分岐 (`:3799-3809`) には `ink_moe_mark(2)/(3)` が無い (mark は 2B の `:3731-3779` と 2D の `:3845/3860` のみ)。しかし実測 .err は gu=445-548 / dn=250-310 を出しており、しかもその大きさは S_K dp4a の計算 (10.2 ms/visit, 5.9 ms/visit) と一致する。**実測に使ったバイナリが HEAD と一致しない可能性**、または別経路の mark がある。数値の解釈 (S_K が 445/285 を占める) は量の計算から見て妥当と判断したが断定は不可
2. 「moe= の残り 277 ms」の内訳は未計測 (reduce8 の交通量は 37.7 MB read/visit で説明できない。prof 区間の切り方の問題もありうる)
3. 実 register 数・実 occupancy は未確認 (ホストに cuobjdump が無く、コンテナも使っていない)。`launch_bounds(128,2)` からの下限推定のみ
4. sm_61 の shared 96 KB/SM・48 KB/block はドキュメント値。deviceQuery は未実行 (GPU 不使用の制約)
5. FP16 1/64 は P40 の仕様。SF4 経路 (`:1355`) の half2 dot は不利だが、S_K 経路は int8 のみなので影響なし
6. 3.4 の CTA 反復あたり 66 us / 101k cycle は「2 CTA/SM 同時」を仮定した概算 (推測)。stall の内訳 (global latency / L1 miss / shuffle) は未計測

## 7. 次に測るべきこと (提案)

1. `k_ink_moe_gu_sk_fmt` の実 register 数と実 occupancy を cuobjdump で確認 (コンテナ内で `cuobjdump -res-usage libstrata_kernels.a`)
2. MAX_M=16 のまま活性を shared に載せる版 (SM_M=16, 45 KB) を作り、A/B で gu の ms を比較 (stall 仮説の直接検証)
3. `s` ループの `unroll 1` を外し、重み 2 段プリフェッチ + 256 thread/64 feature CTA にした版で 3.3 の 1 pass 化 (cnt 32 を 1 tile に) を狙う
