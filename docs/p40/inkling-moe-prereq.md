# Inkling MoE 最適化の前提確定 (prof tag / バイナリ来歴 / 実 resource と occupancy)

調査日: 2026-11-25 / 対象: `~/work/ink` (branch `perf/ink-attn-sm61`, HEAD `ba6bfb870`、作業ツリー clean)、読み取りのみ。GPU は未使用 (共用ロックに触れず、strata 系プロセスも起動していない)。コード・設定・compose は無変更。

## 結論 (親タスク 1-4 の対応)

| 親タスク | 結論 | 主な根拠 |
|---|---|---|
| 1. 実測ログのバイナリ | **`~/work/ink/build/src/inkling2/strata-inkling`** (sha256 `2ce2532…`、mtime 17:05:44)。`build-base` ではない。ab-attn の base/tiled 両アームとも ink 側 | `ab-attn.sh:5` / `run-512k.sh:4` の BIN、`base.err:5` の文字列が build-base に存在しないこと、build レポートの sha と現在の sha が一致 |
| 2. gu/dn タグの行き先 | `gu(sf4/mx8/sk)` / `dn(sf4/mx8/sk)` は **2B の SF4/MX8 専用**。S_K 経路 (2C) には mark が無く、S_K の時間は **`moe=` の残り**に入る。実測の gu=445.75 / dn=256.39 は **15 個の SF4 expert の時間**であって S_K ではない | `ink_moe_mark` 全 18 箇所、2C は `ink_moe_cuda.cuh:3784-3809` に mark 無し、x86 逆アセンブリ、`total = 位相 + parts` の数量一致 (64/64 行) |
| 3. 実 resource | `gu_sk_fmt<S2_K,16>` REG=160 / `S3_K` 158 / `dn_sk_fmt<S2_K,16>` 134 / `S3_K` 132、shared 0 / stack 0 / local 0 | `cuobjdump -res-usage` (arch = sm_61) |
| 4. 理論 occupancy | **3 CTA/SM = 12 warp/SM = 384 thread = 18.75%** (register 律速)。`__launch_bounds__(128,2)` は拘束していない (保証 2 CTA に対し実測 3) | 下記の計算式 |

**最重要**: 現行 775 ms とされていた「gu+dn = S_K」は誤り。MoE 位相の実測内訳は **全体 1,690 ms = router 3.80 + bin 1.04 + SF4 702.14 + S_K ほか 983.22** で、**0.5% の重み (15 expert) が MoE 時間の 41.5%、step 全体の 37.1%** を占める。S_K は 983 ms (31.9 GB -> 32.4 GB/s = 346 GB/s の 9.4%)。

## バイナリ来歴 (親タスク 1)

| バイナリ | mtime (+0000) | size | sha256 |
|---|---|---|---|
| `~/work/ink/build/src/inkling2/strata-inkling` | 2026-11-24 17:05:44.466518289 | 6,558,248 | `2ce25323062884f1dce2b847ef532754039d725cbaca9dfe44e547631f76f922` |
| `~/work/build-base/build/src/inkling2/strata-inkling` | 2026-11-24 10:29:21.710695410 | 6,521,432 | `bee32221360265a6bcfdeafb9464e6c2b6a2fccf3de4fd26757eda1b42070c26` |

- 実行スクリプトの BIN は 3 本とも ink 側: `~/work/bin/ab-attn.sh:5`、`~/work/bin/run-512k.sh:4`、`~/work/bin/ab-128k.sh:4` = `BIN=~/work/ink/build/src/inkling2/strata-inkling`
- 実行時刻 (`~/work/logs/ab-attn.log`): base 17:22:55 -> tiled 17:28:13 -> done 17:30:52。`tiled.err` mtime 17:30:49、`base.err` 17:26:57、`arm.err` (512K) 20:32:03。**すべてバイナリ mtime 17:05:44 より後**
- ビルドは 16:55:37 configure -> 16:55:54 build 開始 -> **17:05:44 終了** (`~/work/logs/ink-attn-build-20261125.log` 末尾 `[174/174] Linking CXX executable src/inkling2/strata-inkling`)。`~/work/logs/ink-attn-build-report.md` の成果物表の sha256 が現在値と一致 -> **17:05:44 以降に再ビルドされていない** (mtime も 17:05:44 のまま)
- `build-base` ではないことの直接証拠: `base.err:5` に `[InkCUDA] tiled prefill attention is off (STRATA_INK_PF_TILE=0)` がある。この文字列は `ink_runner.cuh:3100` のみに存在し、build-base 側の `ink_runner.cuh` では `grep -c` = **0**。つまり `STRATA_INK_PF_TILE` を解釈するタイル attention 入りバイナリ = ink 側が base アームでも走っている
- 両ツリーの `ink_moe_cuda.cuh` は **バイト一致** (`diff -q` で同一)。したがって MoE 部のコードは build-base でも同じ (差は `ink_runner.cuh` の +3,387 B のみ)
- 注意: `STRATA_INK_PROF=2` のログは `gu(gguf)=dn(gguf)=0.00` なので GGUF 経路は未使用、`has_sf4` は「層に 1 個でも SF4 expert があれば true」(`ink_runner.cuh:375`)。codec 内訳は `tiled.err:1` の `2/2: 585, 3/2: 1814, 3/3: 658, 15/15: 15` = **S2_K/S3_K が 3057、SF4 が 15**

## prof tag の実装と実測の読み方 (親タスク 2)

### 計測の仕組み (区間は「次の mark」に入る)

- `ink_cuda_backend.cu:315-318`: `g_ink_moe_hook = [](int tag, void* ctx){ …->prof_mark_end(8 + tag); }`
- `ink_runner.cuh:1942-1950`: `prof_mark_end(idx)` は「**直前の mark から今回の mark まで**」を `cudaEventElapsedTime` で測り `prof_ms[idx]` に足す (`prof_pairs.push_back({prof_prev, e, idx})`)
- `ink_runner.cuh:1968`: 出力は idx 8=router+topk / 9=bin / 10=gu / 11=dn / 12=gu(gguf) / 13=dn(gguf)。`moe=` は idx 5 で、MoE 位相の**残り** (mark の付かなかった区間)
- 位相の区切りは `ink_runner.cuh:3550` の `prof_mark_end(5); prof_mark_start();` (MoE 終了時) と、その手前の `:3445` (dense_wo 終了 -> MoE 開始)

### `ink_moe_mark` の全呼び出し (18 箇所、`ink_moe_cuda.cuh`)

| 行 | tag | 直前に起動するカーネル | P40 で実行? |
|---|---|---|---|
| 3635 | (定義) | — | — |
| 3662 | 0 | `k_ink_router_select_topk8` | する (router) |
| 3689 | 1 | `k_ink_moe_bin` | する (bin) |
| 3731 / 3733 | 2 / 3 | `sm120::moe_sf4_gu_tc` / `_dn_tc` | **しない** (sm_120 のみ) |
| 3744 / 3746 | 2 / 3 | `k_ink_moe_pf_fp16tc` gu / dn | **しない** (`use_fp16_tc`=int8 TC) |
| 3754 / 3758 / 3761 | 2 / 2 / 3 | `k_ink_moe_fp16tc_v2` sf4 gu / mx8 gu / 共通 dn | **しない** |
| 3764 / 3766 | 2 / 3 | `k_ink_moe_sf4_mx8_fp16tc` gu / dn | **しない** |
| 3772 / 3774 | 2 / 3 | `k_ink_moe_gu_sf4_mx8<1>` / `dn_sf4_mx8<1>` (else, M==1) | する (SF4 層のみ) |
| **3777 / 3779** | **2 / 3** | **`k_ink_moe_gu_sf4_mx8<4>` / `dn_sf4_mx8<4>` (else, M>1)** | **する (chunk 512 はこちら)** |
| 3845 / 3860 | 4 / 5 | GGUF gu / dn | **しない** (has_gguf=false) |

`P40` では `use_sm120_tc` / `use_fp16_tc` / `use_wmma_tc` がすべて false (`ink_caps.hpp:16-17` の `ink_cap_int8_tc(major>=8)` / `ink_cap_wmma_f16(major==7)`、`major=6`) なので、2B の実行経路は **`:3771` の else 一択**。

### S_K 経路 (2C) に mark は無い

- 2C 本体は `ink_moe_cuda.cuh:3784-3809`。P40 が通るのは `:3798` の else (3799-3807: `g32 = (36, max_act)`、`launch_ink_moe_gu_sk_mask<16>` -> `k_ink_quant_q8_1(h)` -> `launch_ink_moe_dn_sk_mask<16>`) で、**`ink_moe_mark` は 1 つも無い** (上の表の 18 箇所に 2C が無い)
- バイナリ側でも同じ (x86 逆アセンブリ、`launch_ink_moe_dispatch_only` = `0x10bc50`、size `0x5257`):
  - M>32 の S_K: `0x10ece3 call launch_ink_moe_gu_sk_mask<16>` -> `0x10c2c0 __cudaPushCallConfiguration` (h 量子化) -> `0x10d678` (M>32 分岐) -> `0x10d6ca call launch_ink_moe_dn_sk_mask<16>`。**この間に `g_ink_moe_hook` を読む `call *%rax` は無い**
  - M<=32 の S_K: `0x10c25a` (gu<4>) -> `0x10c346` (dn<4>) も同様に mark 無し
  - mask ランチャ自体 (`launch_ink_moe_gu_sk_mask<16>` = `0x108660`、`dn_sk_mask<16>` = `0x10ae90`) にも hook 呼び出しは無い (呼び出しは `__cudaPush/PopCallConfiguration` と `cudaLaunchKernel` のみ)
- **結論: S_K の時間は `moe=` (idx 5) の残りに入る。別タグには入らない。** `moe=` 983.22 ms には S_K の gu/dn + 活性 q8_1 量子化 (`:3787` / `:3804`) + reduce8 (約 1 ms) が含まれる
- 誤解の元: `ink_moe_cuda.cuh:3632` のコメントが「Tags 0..3 for SF4 / MX8 / **S_K** layers」と書いているが、S_K 用の mark は実装されていない (コメントが stale)

### 数量での裏取り (ログだけで閉じる)

`total` は `prof_ms[0..15]` の総和 (`ink_runner.cuh:1963`、print は `:1962-1969`)。もし parts が `moe=` の内側なら `total - 印刷された位相 = prof_ms[6]` (≈0) になるはず。実測は **total - 印刷位相 = parts に厳密に一致**した:

| 例 (tiled.err の 1 組目) | 値 |
|---|---|
| total | 1,894.78 |
| 印刷位相の和 (res_ctrl+dense_qkv+sconv_kv+attn+dense_wo+moe+unembed) | 1,187.79 |
| 差 | **706.99** |
| parts の和 (router+bin+gu+dn) | **706.98** |

64 行すべてで `|差 - parts| <= 0.02` ms (= 丸め)。→ **parts は `moe=` の外側**、`prof_ms[6]` (other) は 0.0 ms。つまり `moe=` は位相の残りであって位相全体ではない。

### 実測の再解釈 (代表値: `attn-ab/tiled.err` 1 組目)

| 内訳 | ms/step | 備考 |
|---|---|---|
| router+topk | 3.80 | tag0 |
| bin | 1.04 | tag1 |
| **gu(sf4/mx8/sk)** | **445.75** | tag2 = `k_ink_moe_gu_sf4_mx8<4>` (15 expert, 8 pass) |
| **dn(sf4/mx8/sk)** | **256.39** | tag3 = `k_ink_moe_dn_sf4_mx8<4>` |
| S_K gu/dn + h 量子化 + reduce8 | **983.22** | `moe=` の残り (31.9 GB -> 32.4 GB/s) |
| MoE 位相 合計 | **1,690.20** | step 1,894.78 の **89.2%** |

- SF4 の 702.14 ms は step の **37.1%**、MoE の **41.5%**。重みは 15/3072 expert = 0.5% なのに、この時間
- 妥当性の概算 (推測含む): SF4 gu は 1 expert あたり `32 token x 2*1152*2304 MAC = 170 MMAC`。`__hfma2` は sm_61 で FP32 の 1/64 (0.183 TFLOPS) なので、機械を満たせば 1.9 ms。しかし実際は **1 expert あたり 72 block しか出ない** (grid.x = `kInkInter/16`) ため 30 SM に 2.4 CTA/SM しか載らず、実効 ~15% -> 12 ms/expert 前後。15 expert x 1-3 visit = 180-540 ms で、実測 445 ms と同桁
- **survey の §0 / §3.4 の「gu+dn = 775 ms が S_K」「moe との差 277 ms が reduce8」は要修正**。S_K は 983 ms、SF4 が 702 ms、reduce8 は約 1 ms
- 512K も同じ (arm.err 1024 行、gu 410-442 / dn 240-255 / moe 982)。`attn-128k` の 3 本も同様

## 実レジスタ数・shared・local (親タスク 3)

`nvidia/cuda:12.9.1-devel-ubuntu24.04` (既存、pull なし) を `--gpus` なし・`--entrypoint cuobjdump` で使用:

```
sudo docker run --rm --entrypoint cuobjdump -v ~/work/ink/build/src/inkling2:/b:ro \
  nvidia/cuda:12.9.1-devel-ubuntu24.04 -res-usage /b/strata-inkling
```
出力は `arch = sm_61` (2 セクション、`strata-inkling` は sm_61 のみ)。`-res-usage` は静的 shared、起動時 dynamic shared は 0 (`<<<grid, 128, 0, st>>>`、`ink_moe_cuda.cuh:3606-3612`)。

| kernel (実体) | REG | STACK | SHARED | LOCAL | 備考 |
|---|---|---|---|---|---|
| `k_ink_moe_gu_sk_fmt<FMT_S2_K=2, 16>` | **160** | 0 | **0** | 0 | **chunk 512 の gu (実行される実体)** |
| `k_ink_moe_gu_sk_fmt<FMT_S3_K=3, 16>` | 158 | 0 | 0 | 0 | 同上 |
| `k_ink_moe_gu_sk_fmt<1, 16>` (S1_5) | 255 | 448 | 0 | 0 | 未使用 (mask に S1_5 無し)。spill あり |
| `k_ink_moe_gu_sk_fmt<2..8, 4>` | 80-96 | 0 | 11,520 | 0 | M<=32 用 |
| `k_ink_moe_gu_sk_fmt<2..8, 1>` | 76-79 | 0 | 2,880 | 0 | M==1 用 |
| `k_ink_moe_dn_sk_fmt<2, 16>` | **134** | 0 | **0** | 0 | **chunk 512 の dn** |
| `k_ink_moe_dn_sk_fmt<3, 16>` | 132 | 0 | 0 | 0 | 同上 |
| `k_ink_moe_dn_sk_fmt<1, 16>` (S1_5) | 255 | 40 | 0 | 0 | 未使用 |
| `k_ink_moe_dn_sk_fmt<2..8, 4>` | 86-88 | 0 | 5,760 | 0 | M<=32 用 |
| `k_ink_moe_dn_sk_fmt<2..8, 1>` | 80 | 0 | 1,440 | 0 | M==1 用 |
| `k_ink_moe_gu_sk<16>` / `dn_sk<16>` (mask==0 版) | 255 | 48 | 0 | 0 | SQ-10 では未使用 |
| `k_ink_moe_gu_sf4_mx8<4>` | 93 | 0 | 18,432 | 0 | **702 ms を食っている側** |
| `k_ink_moe_dn_sf4_mx8<4>` | 79 | 0 | 9,216 | 0 | 同上 |

- `MAX_M=16` は shared 宣言 (`s_xq[1][36]` = 2,880 B、`ink_moe_cuda.cuh:2136` / `:1988`) が `if constexpr (MAX_M <= 4)` でしか使われないため **DCE されて 0 B**。survey §2.4 の「shared (MAX_M=16) = 2,880 B」は宣言値であって実値は 0
- `-res-usage` の LOCAL=0 / STACK=0 は **spill 無し**を意味する。`k_ink_moe_sk_fmt` の `:2107` のコメントにある「汎用版は REG:255 STACK:32..72」は mask==0 版 (`k_ink_moe_gu_sk<16>` = 255/48) の話で、format 特化版は解消済み

## 理論 occupancy (親タスク 4)

P40 (sm_61): 64K reg/SM = **65,536**、96 KB shared/SM、2,048 thread/SM = 64 warp、32 block/SM、本カーネルは **128 thread/CTA = 4 warp/CTA**。register の割当粒度は 256 reg/warp。

```
登録数による CTA/SM = floor(65536 / (ceil(REG*32/256)*256 * 4))
``` 

| kernel | REG | per warp (256 単位) | per CTA (4 warp) | CTA/SM (reg) | CTA/SM (shared) | CTA/SM (thread 16 / block 32) | 確定 CTA/SM | warp/SM |
|---|---|---|---|---|---|---|---|---|
| `gu_sk_fmt<2,16>` | 160 | 5,120 | 20,480 | floor(65536/20480)=**3** | ∞ (0 B) | 16 / 32 | **3** | **12** (18.75%) |
| `dn_sk_fmt<2,16>` | 134 | 4,352 | 17,408 | floor(65536/17408)=**3** (3.76) | ∞ | 16 / 32 | **3** | **12** (18.75%) |
| `gu_sf4_mx8<4>` (参考) | 93 | 3,072 | 12,288 | 5 | floor(98304/18432)=5 | 16 / 32 | **5** | 20 (31.25%) |
| `gu_sk_fmt<2,4>` (参考) | 83 | 2,816 | 11,264 | 5 | floor(98304/11520)=8 | 16 / 32 | **5** | 20 (31.25%) |
| `gu_sk_fmt<2,1>` (参考) | 76 | 2,560 | 10,240 | 6 | floor(98304/2880)=34 | 16 / 32 | **6** | 24 (37.5%) |

- **chunk 512 (M=512) で走る `MAX_M=16` は 3 CTA/SM = 12 warp = 384 thread = 18.75%**。48 visit x 2 format で 1 launch あたり `36 x 128 = 4,608` block なので、30 SM x 3 CTA = 90 並列 -> **51.2 波**
- 参考の整合: `MAX_M<=4` は 5 CTA、`MAX_M==1` は 6 CTA で、`ink_moe_cuda.cuh:2109` / `:2171` の `__launch_bounds__(128, (FMT==S1_5 ? 2 : (MAX_M==1 ? 6 : (MAX_M<=4 ? 5 : 2))))` の数値と**一致**する (コンパイラは指定どおりに詰めている)
- **`launch_bounds(128,2)` の影響**: 指定は「最低 2 CTA/SM」。このときの register 上限は `floor(65536/(2*128)) = 256 -> 255`。実使用は 160 / 134 で上限には達しておらず、**拘束していない** (spill も無い)。保証は 8 warp/SM だが実際は 3 CTA 取れているので 12 warp。**4 CTA/SM (16 warp = 25%) に上げるには REG<=128 が必要** (`__launch_bounds__(128,4)` 相当。spill 覚悟の実験になる)
- shared は 0 B なので **96 KB は全部 L1 として使える** (sm_61 は shared と L1 が同じ 96 KB を分け合う)。S_K カーネルは重みを L1 に置けない代わりに L1 をキャッシュに全振りできている
- SF4 側は occupancy 5 CTA/SM と足りているのに遅い。理由は **1 expert あたり 72 block しか発行されない**こと (grid = `(kInkInter/16, max_act)`、`ink_moe_cuda.cuh:3770`)。30 SM x 5 = 150 CTA の容量に対し 72 block = 2.4 CTA/SM -> **machine の ~48% が遊ぶ**。かつ `MAX_M=4` で 8 pass に分かれる

## 未確認

1. S_K の 983.22 ms のうち、活性 q8_1 量子化 (`:3787` / `:3804`、合計 ~25 MB の traffic なので 1-2 ms 見込み) と reduce8 (37.7 MB read、~1 ms 見込み) の実分離は**未計測**。mark を足すまで確定できない
2. SF4 の 702 ms の内訳 (グローバル重み読み vs `__hfma2` 1/64 律速 vs 空 block の走査) は**未計測**。上の 12 ms/expert は概算 (推測)
3. 15 個の SF4 expert が何層に散っているかは**未確認** (モデルの expert directory を parse していない。sidecar manifest は無い)。層数が分かれば 1 step あたりの SF4 expert-visit 数が確定し、per-expert コストを実測から逆算できる
4. deviceQuery / 実測 occupancy (ncu 等) は GPU 不使用の制約により**未実行**。65,536 reg / 96 KB / 2,048 thread / 32 block は sm_61 のドキュメント値
5. `k_ink_moe_gu_sk_fmt<1,16>` (S1_5) の STACK=448 は未使用経路なので放置可。SQ-10 の mask は S2_K/S3_K のみ

## 次の提案

1. **S_K 区間を独立計測する最小の変更 (推奨、変更は未実施)**。空きタグは 6/7 だけ (`8+tag` -> idx 14/15。現行 print は idx 8-13 のみ) なので既存の意味を壊さない:
   - `ink_moe_cuda.cuh:3802` の直後 (gu_sk の 3 分岐の後) に `ink_moe_mark(6);`
   - `ink_moe_cuda.cuh:3807` の直後 (dn_sk の 3 分岐の後) に `ink_moe_mark(7);`
   - M>=24 の int8 TC 分岐 (`:3796-3797`) にも同じ 2 行を入れて対称にする (P40 では通らないが将来 sm_80 で効く)
   - `ink_runner.cuh:1968` の隣に 1 行追加して `prof_ms[14]` / `prof_ms[15]` を `gu(sk)=` / `dn(sk)=` として出す
   - これで `moe=` が「reduce8 + 量子化 + 2B の空回り」に純化し、S_K gu / S_K dn / reduce8 が 3 分離できる。差分は 4-6 行
2. **15 個の SF4 expert を S3_K に再量子化する** (または S_K カーネルで処理する)。期待削減は ~700 ms/step (step の 37%)。重みは 15 expert x 4.48 MB = 67 MB なので、S3_K 化しても +67 MB 程度。精度影響は 0.5% の expert のみ。まずは「SF4 層だけ S3_K に差し替えた .strata」を作って 128K で A/B
3. SF4 経路を残す場合の最低限の手当て: 1 launch で全 SF4 expert をまとめる (現状は層ごとに 72 x 128 block を発行し、97% が空 return)。`g16.y = max_act` を「SF4 expert 数」に詰めるだけで空回りが消える
4. occupancy 実験: `MAX_M=16` の `launch_bounds` を `(128,4)` にして REG<=128 に落とす (12 -> 16 warp)。spill が出たら `(128,3)` (REG<=170、現状 160 でちょうど 3) で据え置き。判定はビルド後に `cuobjdump -res-usage` の REG/STACK と 128K の gu(sk) ms
5. `MAX_M=16` の 51 波を減らす方向 (block を 256 thread にして 32 feature -> 64 feature/CTA、CTA 数を半減) は 2 の後で。まず S_K の実測分離 (1) を行い、983 ms のうち何が dominant かを確定してから触る
