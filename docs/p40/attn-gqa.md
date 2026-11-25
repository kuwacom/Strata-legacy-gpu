# Inkling タイル attention の GQA pack 削減 (sm_61 / FP32 SGEMM 経路)

- 日時: 2026-11-25
- 対象: `~/work/ink-ap` (branch `perf/ink-attn-gqa`、base `perf/ink-attn-pv` `ad12f69d3`)、`engine/src/inkling2/cuda/{ink_attn.cuh,ink_runner.cuh}`
- バイナリ: `build/src/inkling2/strata-inkling` sha256 `bda418830ba6b93fd06aba9698e3707cd4e092cd68a6aa93d4fde140936f3d1d` (sm_61 の cubin のみ 2 本)
- ホスト: <host> / Tesla P40 24GB (sm_61, 1531 MHz) / SQ-10
- 本番: 各計測の前後に compose で停止 -> 復旧。最終状態 healthy、`.env` sha256 `(省略)` 不変

## 結論

- **`STRATA_INK_PV_SHAPE` の既定を 2 にした** (0 で従来の batch=24 形に戻る)。shape 1 (m/n 入れ替え) はコードごと削除した
- **`STRATA_INK_GQA_PACK=1` を追加** (既定 0)。pack_kv の GQA 3 複製をやめて K/V を 8 KV head ぶんだけ書き、QK と PV を batch=8 で回す。**QK も batch=8 (n=3C) で成立**した
- 128K FP16 の 1 本比較で **prefill 197.7 -> 202.4 ppts (+2.4%)**、attn= **226.80 -> 211.63 s (-6.7%)**
- 内訳: **pack_kv 29.9 -> 12.8 s (-57%)**、QK 59.7 -> 57.2 s (-4.2%)、softmax 55.6 -> 55.7 s (不変)、**PV 71.3 -> 75.5 s (+5.9%)**
- 数値は **bit 一致** (S / P / PV 出力 / 最終 attention 出力 / 生成 token)。16K fp16・16K k8v4・32K needle のいずれも同一
- K/V の作業領域が 2 本あわせて **-32 MiB** (24 -> 8 slab。`kInkPfKvTile`=2048、FP32 で 1 slab = 1 MiB)。他の scratch は増減なし

## 形状 (書き下し)

`cublasGemmStridedBatchedEx`、算術型は全て `CUBLAS_COMPUTE_32F` / `CUDA_R_32F`、algo は `CUBLAS_GEMM_DEFAULT` (sm_61 は Tensor Core 無し)。C=256 (Q_CH)、K_t は tile 長、M=512。

| 段 | GQA_PACK | transA/B | m | n | k | lda / strideA | ldb / strideB | ldc / strideC | batch |
|---|---|---|---|---|---|---|---|---|---|
| QK | 0 | OP_T / OP_N | K_t | C (256) | 128 | 128 / K_t x 128 | 128 / M x 128 | K_t / C x K_t | 24 |
| **QK** | **1** | OP_T / OP_N | K_t | 3C (768) | 128 | 128 / K_t x 128 | 128 / 3C x 128 | K_t / 3C x K_t | **8** |
| PV | 0 (=shape 2) | OP_N / OP_N | 128 | 3C (768) | K_t | 128 / 3 x K_t x 128 | K_t / 3C x K_t | 128 / 3C x 128 | 8 |
| **PV** | **1** | OP_N / OP_N | 128 | 3C (768) | K_t | 128 / K_t x 128 | K_t / 3C x K_t | 128 / 3C x 128 | 8 |

- QK を batch=8 にできる根拠: B (Q) は head ごとに `[C, 128]` が連続で、同じ KV head の 3 head は `3*kh + rep` の隣接 3 本。よって B を `[8, 3C, 128]` に並べ替えれば **n=3C の 1 呼び出し**で 3 head ぶんを積める。`pf_pack_q` に gqa 用の並べ方 (`(h/3)*3C + (h%3)*C + (m-q0)`) を追加し、q ブロック単位で pack する
- S (`d_pf_s`) と P (`d_pf_p`) のレイアウトは **従来と同一** (batch kh の n 方向 3 head は、head 番号順の連続領域にそのまま一致する)。softmax / accum / unpack は無変更
- PV は V の batch ストライドだけが `3 x K_t x 128` -> `K_t x 128` に変わる (V が 8 slab しかないため)
- GQA_PACK=1 では `STRATA_INK_PV_SHAPE=0` は成立しない (K/V が 8 head しかない) ので、`pf_pv_shape` を 2 に強制する

## 変更 (env 2 本 + カーネル 2 本の引数、+94/-97 行)

| ファイル | 箇所 | 内容 |
|---|---|---|
| `ink_runner.cuh` | `:1834` | `pf_pv_shape` の既定を 2 に、`pf_gqa_pack` / `pf_gqa_cap` を追加 |
| 同上 | `:2218` | `STRATA_INK_PV_SHAPE` は 0/2 のみ受理 (1 は「削除した」と 1 行出して 2 のまま)、`STRATA_INK_GQA_PACK=1` の読み取り、起動ログ 2 行 |
| 同上 | `:3216-3252` | `ensure_pf_attn` の再確保条件に `pf_gqa_cap` を追加、GQA_PACK=1 では K/V を 8 head ぶんだけ確保 |
| 同上 | `:3459-3512` | `pf_pack_q` を q ブロック引数付きに、`pf_pack_kv` に `ncopy` を追加、QK を `pf_qk_gemm` に集約 (batch=8 / n=3C と従来の 2 分岐)、PV の shape 1 分岐と転置を削除 |
| 同上 | `:3540-3590` | 短い経路 / 長い経路で、gqa のときだけ q ブロック先頭で `pf_pack_q` を呼ぶ |
| `ink_attn.cuh` | `:762-790` | `k_ink_pf_pack_q` に gqa 用の書き込み先 (引数 +3)、`k_ink_pf_pack_kv` に `ncopy` |
| 同上 | `:1106` | shape 1 専用だった `k_ink_pf_ot_to_o` を削除 |

- 追加カーネルは無し。`ncopy=3` のときの書き込み順・値は従来と同一 (`h0 = kh*ncopy`)
- REG (cuobjdump): `pack_q` 17/20/20、`pack_kv` 32/32/32 (PFDT=0/1/2)。`pack_kv` は従来と同じ 32
- 起動ログは `[InkCUDA] PV SGEMM shape = 2` を常時 1 行、GQA_PACK=1 のときだけ `GQA pack = 1 ...` を追加

## 数値 (bit 一致)

検証は 2 通り。どちらも 16K prompt (K_len > 2048 で長い経路に入る) を使った。

1. 同一プロセス内で GQA_PACK=0 の rank と 1 の rank を両方実行し、S / P / PV を D2H 比較 (検証フック入りバイナリ `732c7bf7…`、フックは検証後に削除)

| 比較 | 要素数 | bitwise | max rel err | max abs | RMS rel err |
|---|---|---:|---|---:|---:|
| S (QK 後) | 12,582,912 | yes | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| P (softmax 後) | 12,582,912 | yes | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| PV 出力 | 786,432 | yes | 0.000e+00 | 0.000e+00 | 0.000e+00 |

2. 最終 attention 出力 (`d_attn_out`, 16K x 24 x 128 = 6,291,456 B) を GQA_PACK=0/1 で別プロセスにダンプしてバイト比較: **sha256 一致** (`0b62e585…`)。生成 8 token も同一

| 経路 | GQA_PACK=0 vs 1 |
|---|---|
| 16K fp16 (`--kv fp16`) 生成 8 token | 同一 id |
| 16K k8v4 (`--kv k8v4 --kv-offload off`) 生成 8 token | 同一 id |
| 32K needle fp16 生成 16 token | 同一 id (`7741`) |

- 合格条件 (max rel err <= 1e-5) に対して 0。QK の n を 3C にしても k 方向の累算順は変わらないため、cublas が同じ系列で積む限り差が出ない
- 上の 3 行は最終バイナリ (`bda41883…`、フック削除後) で取り直した。16K fp16 / 16K k8v4 の生成 8 token と 32K needle の生成 16 token が GQA_PACK=0/1 で同一 (`~/work/results/attn-gqa/final/`、同ディレクトリの `prod-restore.txt` が healthy / `.env` 不変)
- S / P / PV の bit 比較 (1) と最終出力ダンプ (2) は検証フック入りのバイナリ `732c7bf7…` で取った。フックを削除した最終バイナリは本番経路のコードが同一で、上記のとおり生成物も一致する
- 32K needle は `~/work/logs/needle-32k.ids` (深度 49.9%) を `--serve` の `GEN 16` に流し、`needle.py` のデコードが **`7741`** (ORCHID-7741) を含むことを確認

## 性能 (128K FP16 KV、256 chunk、各 1 本、アーム間 60 s 待機)

`STRATA_INK_PF_TILE=1 STRATA_INK_PROF=2 STRATA_INK_ATTN_PROF=1`、`--max-context 131072 --kv fp16 --prompt-len 131072 --gen 16 --chunk 512 --draft off`。

| アーム | GQA_PACK | prefill ppts | prefill s | attn s | pack_kv s | QK s | softmax s | PV s | acc s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gq0 | 0 | 197.7 | 662.92 | 226.80 | 29.9 | 59.7 | 55.6 | 71.3 | 5.2 |
| **gq1** | **1** | **202.4** | **647.60** | **211.63** | **12.8** | 57.2 | 55.7 | 75.5 | 5.2 |
| gq0b | 0 | 197.8 | 662.48 | 226.50 | 29.8 | 59.7 | 55.6 | 71.1 | 5.2 |

- GQA_PACK=0 の重複 (gq0 / gq0b): prefill 197.7 / 197.8、attn 226.80 / 226.50 (+-0.13%)。**+2.4% / -6.7% はこのばらつきを超える**
- 段の変化: pack_kv **-17.1 s (-57%)**、QK **-2.5 s (-4.2%)**、PV **+4.2 s (+5.9%)**、softmax 不変、attn= 合計 **-15.2 s**
- 実効 (FLOP は 128K で QK/PV とも 464 TFLOP): QK 7.77 -> 8.11 TFLOPS、PV 6.52 -> 6.15 TFLOPS
- decode は全アーム 12.5 tps。GPU は全アーム 1531 MHz / 73-74 C / 64 W で throttle 無し

位置別 (ms/chunk):

| 位置 | gq0 pack_kv | gq1 pack_kv | gq0 QK | gq1 QK | gq0 PV | gq1 PV | gq0 sum | gq1 sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 序盤 | 43.89 | 18.77 | 92.93 | 89.00 | 118.26 | 124.09 | 374.72 | 351.77 |
| 中盤 | 117.04 | 50.34 | 234.25 | 224.24 | 279.50 | 296.23 | 889.20 | 829.71 |
| 末尾 | 189.75 | 81.71 | 374.50 | 358.60 | 439.15 | 466.74 | 1399.46 | 1303.74 |
| 全体 | 116.61 | 50.15 | 233.34 | 223.42 | 278.35 | 295.02 | 885.79 | 826.55 |

経路別 (s):

| 経路 | gq0 pack_kv | gq1 pack_kv | gq0 QK | gq1 QK | gq0 PV | gq1 PV | gq0 合計 | gq1 合計 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 28.08 | 12.12 | 54.24 | 51.92 | 62.05 | 66.18 | 199.00 | 184.92 |
| SWA | 1.77 | 0.72 | 5.49 | 5.28 | 9.21 | 9.34 | 27.76 | 26.67 |

- pack_kv は global -56.8% / SWA -59.3% で、書き込み 3 -> 1 にほぼ比例する (読みは同じ、書が 1/3)
- QK は batch=24 -> 8 で **K の読みが 24 MiB -> 8 MiB (3 分の 1)**。演算律速 (平衡 34 に対し FLOP/byte 43) だったので伸びは -4.2% に留まる
- PV は V の読みが shape 2 の時点で既に 8 MiB (複製を読まない) なので、転送量は変わらない。**それでも +5.9% 遅くなった理由は未確認**。切り分け用マイクロベンチ (`pvbench5/6`、V を 8 slab 密に置くか 24 slab 疎に置くかだけを変えた同形状) では密配置が **0.5-1.9% 遅い**だけで、エンジン内の +5.9% の大半は説明できていない (候補: K/V の作業領域が 24 -> 8 slab になったことによるアドレス配置 / L2 の当たり方の差。未確認)
- それでも attn= は -6.7%。pack_kv の削減が PV の悪化を上回る

## needle (32K、最終バイナリ)

| GQA_PACK | 生成 16 token | 判定 |
|---|---|---|
| 0 | `7741` を含む | 通過 |
| 1 | 0 と同一 id | 通過 |

## 未確認・注意

1. PV が +5.9% 遅くなった原因は未解明 (上記)。V を 8 slab だけ密に置く現方式に対し、8 slab を疎 (stride 3) に置く案は未測定。マイクロベンチ差は 2% 以内なので、上限は prefill +0.5 ppts 程度と見積もる
2. sm_70+ の BF16 / FP16 経路 (pfd=0/1) は同じ分岐を通るが未測定。`kInkHeads == 3 * kInkKVHeads` を満たすモデルでのみ GQA_PACK=1 が効く (満たさない場合は shape 0 に落ちる)
3. `STRATA_INK_GQA_PACK` の既定は 0 のまま (指示どおり)。効果は +2.4% なので既定 1 への切替を提案する
4. 計測は `STRATA_INK_PROF=2` (eager) と `STRATA_INK_ATTN_PROF=1` の上。3 アームすべて同条件なので相対比較は有効
5. 検証フック (`STRATA_INK_GQA_CMP` / `STRATA_INK_GQA_DUMP`) は検証後に削除済み。コミットに含まれない
