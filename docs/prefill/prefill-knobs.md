# prefill 起動ノブの A/B（Tesla P40 / 512K 構成）

作業ツリー: `~/work/wt/pknobs`（branch `docs/prefill-knobs`、ソース変更なし）
イメージ: `strata:fork-9f7d977-base`（engine 0.1.57-linux-x86_64-cuda-cuda12）
本番: `~/opt-work/prod.env`（512K、IQ3_XXS、engine 0.1.52）
生ログ: `~/work/results/pknobs/`

## 目的と計測条件

512K 構成の prefill（TTFT）を env と起動フラグだけで速くできるかを A/B で確かめる。特に
`STRATA_KV_PREFETCH`（KV streaming のアップロードを別 stream で重ねる）と `STRATA_KVA`
（late-layer 近似）を優先した。

計測に使った起動設定（本番 512K 相当）:

```
--max-context 524288 --kv int8 --kv-resident 32768 --rope-scaling yarn --rope-scale 2
--spec 4 --ple-io ram --prefill auto
STRATA_DENSE_F32=1
```

- エンジンログで確認した実効値: `prompt chunk auto: 8192 tokens, a 384-slot ring`、
  `KV streaming: 32768 of 524288 cells per QSA layer in VRAM, the K/V in 6.19 GiB of pinned RAM`
- fork イメージの Terminal が独自に足した env: `STRATA_GDN_TC=1`、`STRATA_GDN_CONVL2=2`
  （どちらも A/B の両アームで同じ）。drafter は Terminal 既定の DFlash2（main / quality アーム）
- プロンプトは `~/Strata/docs/*.md` の連結。`BENCH_SEED=42`、`max_tokens=1`、
  TTFT は最初の content/reasoning トークンまで。prefill tok/s = usage.prompt_tokens / TTFT
- A と B は同じ run index で同じプロンプト（`BENCH_SEED=42` の固定シード列）。1 本目除外は
  せず、各アームの全 run を記録して中央値で比較（run 数が 2〜3 のため）

## prefill 関連 env / フラグの一覧

| 名前 | 何をするか | bitwise | sm_61 (P40) | 既存実装 |
|---|---|---|---|---|
| `STRATA_KV_PREFETCH=1` | KV streaming 時、次の QSA layer の host prefix の stage を `m.kv_copy` stream で先行発行し、現 layer の計算と重ねる。`p0 > 0` のチャンクだけ | コピー内容は同一だが、docs は pure-upstream の bitwise parity を「未確認」と明記 | 可（arch guard なし） | `src/prefill/prefill.cpp:3233-3253, 3555-3582, 4280, 4440, 6189` |
| `STRATA_KVA=1` | late layer（既定 split 36、suffix 2048）を state-only 近似。GDN recurrence / QSA K/V / indexer key だけ作り、attention 出力・MoE を飛ばす | いいえ（近似） | 可（CUDA、Qwen3.8-Flash-Next、1 GPU、batch なし、cold read、画像なし、DFlash2 drafter なし） | `include/strata/prefill/kva.hpp`, `src/prefill/prefill.cpp`, `src/prefill/kernels.cu` |
| `STRATA_KVA_SPLIT` | 最初に近似する layer（既定 36） | いいえ | 可 | 同上 |
| `STRATA_KVA_SUFFIX` | 完全計算する末尾位置（既定 2048） | いいえ | 可 | 同上 |
| `--prefill N` / `auto` / `auto:N` | 1 チャンクのトークン数。今回の auto は 8192 | 8192 以下は CUDA で exact、8192 超は not bit-exact（`DETAILS.md:1243`） | 可 | `src/program/generate.cpp` の chunk 決定 |
| `STRATA_PREFILL_CPU_SHARE` | chunk が小さいとき CPU pool が expert の一部を計算。既定 auto、上限 3072 | いいえ（last bits 変化、KL ~0.004） | 可だが、24K/128K の chunk 8192 は対象外（1024 未満で効果、2K/4K/16K は unchanged、`DETAILS.md:1002`） | `src/prefill/prefill.cpp:262-308, 2862-2889, 3594` |
| `STRATA_PATTN_SM60=1` | sm_6x の prompt QSA attention を score/value kernel で実行。int8 K/V、attention gate fusion off のときだけ | docs と `qsa_pattn_parity` は「bit for bit qsa_decode_attn_batch」。今回の 24K answer は on/off で完全一致 | 可（`STRATA_EXPERIMENTAL_SM60` build のみ。fork base は該当） | `src/kernels/cuda/qsa_pattn.cu`, `src/prefill/prefill.cpp:4417` |
| `STRATA_IQ_STAGE_TABLES=1` | IQ3_XXS / IQ3_S の codebook を shared memory に stage | はい | 可（Pascal）。P100 では end-to-end 変化なし、3090 Ti では 3-13% 遅い | `src/kernels/cuda/iq_kernels.cu:3435` |
| `STRATA_MMQ_KQ=1` | K-quant の prompt expert を MMQ 経由で。CUDA では `libstrata_kq.so` を開く。cc 8.0+ は既定 on、sm_61 は opt-in | MMQ の bits（FP16 prompt path とは別経路） | 可（`libstrata_kq.so` は fork base に存在） | `src/prefill/moe_mmq.cu:41`, `src/prefill/mmq_kq_loader.cpp` |
| `STRATA_PF_W16_CACHE_MIB` | dequant 済み FP16 重みを VRAM にキャッシュ。`--batch` なしでは既定 off | engine の scratch では cached/uncached が同じ GEMM（`gemm_w16_parity`）。sliced form とは not bit-exact | 可（CUDA） | `src/prefill/gemm.cu:1405-1450` |
| `STRATA_PF_HCDOWN=1` | hyper-connection の down/inject を 1 つの WMMA GEMM に融合 | rounding-level | 不可（gfx11 の `strata_pf_hcdown_bf16`） | `src/prefill/prefill.cpp:556-570`, `src/prefill/wmma_gemm.cu` |
| `STRATA_PF_GEMM` / `STRATA_PF_PAD` | padded X の FP16 GEMM | `PF_PAD` は same bits | 不可（gfx11 の padded product のみ） | `src/prefill/gemm.cu:1356-1387`, `wmma_gemm.cu:308` |
| `STRATA_PF_FUSED=1` | fused int8 prompt experts。CUDA は 80 SM 以上で既定 | 経路による（fused の bits） | 不可（P40 は 30 SM） | `src/prefill/moe_fused.cu:678-693` |
| `STRATA_PF_STEP_SYNC=1` | 各 layer 後に compute/copy stream を待つ診断 | はい（診断） | 可 | `src/prefill/prefill.cpp:3912-3928` |
| `STRATA_PF_SWITCH_MIN_T` | rounding-level の prompt switch を有効にする最小 chunk | — | 可 | `src/prefill/prefill.cpp:536-540` |
| `STRATA_DENSE_F32=1` | 大きい dense GEMM を FP32 の `cublasSgemm` で | cuBLAS build 依存（`gemm_f32_parity`） | 可（sm_61 の opt-in、本番も設定済み） | `src/prefill/gemm.cu:1195-1215` |
| `STRATA_DEQUANT_SIDE=1` | dense dequant を side stream へ | はい | 可（CUDA） | `src/prefill/gemm.cu:1477-1495` |
| `STRATA_PLE_COMPACT=1` / `STRATA_PLE_COPY_STREAM=1` / `STRATA_EMBED_VRAM=1` | PLE 行を raw で上げて card で dequant、専用 stream、embedding を VRAM に | はい | 可 | `DETAILS.md:1546` |
| `STRATA_PREFILL_TIMING=1` | prefill の phase 別 GPU 時間を記録 | 影響なし（profiling） | 可 | `src/prefill/prefill.cpp:2985-3030` |

補足: `STRATA_KV_HOST_DMA` は既定 off（`prefill.cpp:4204-4213`）。`STRATA_HC_UPMIX` は
CUDA sm_80+ / gfx11 向け。`STRATA_CVEC_FUSE` は bitwise だが今回未測定。

## 計測結果

### 24K（A = KV_PREFETCH off、B = on）

| run | prompt tokens | A prefill tok/s | A TTFT s | B prefill tok/s | B TTFT s | 差 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24,592 | 488.7 | 50.32 | 498.4 | 49.34 | +1.99% |
| 2 | 24,716 | 486.9 | 50.76 | 491.1 | 50.33 | +0.86% |
| 3 | 25,003 | 484.7 | 51.59 | 488.9 | 51.14 | +0.87% |
| 中央値 | — | **486.9** | **50.76** | **491.1** | **50.33** | **+0.86%** |

TTFT 中央値の差: -0.43 s（-0.85%）。5% 閾値未満 -> 差なし。

### 128K（A = KV_PREFETCH off、B = on）

| run | prompt tokens | A prefill tok/s | A TTFT s | B prefill tok/s | B TTFT s | 差 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 137,333 | 480.3 | 285.92 | 484.4 | 283.53 | +0.85% |
| 2 | 129,432 | 484.1 | 267.37 | 487.4 | 265.57 | +0.68% |
| 中央値 | — | **482.2** | **276.65** | **485.9** | **274.55** | **+0.77%** |

TTFT 中央値の差: -2.10 s（-0.76%）。5% 閾値未満 -> 差なし。

### 521K（480K 目標、B = KV_PREFETCH on のみ）

| アーム | prompt tokens | TTFT s | prefill tok/s |
|---|---:|---:|---:|
| B | 520,966 | 1738.68 | **299.6** |

目標 480,000 トークンに対して 3.6 文字/トークンで文字数を決めたため、実際は 520,966 トークン
（docs の実測は約 3.3 文字/トークン）になった。A アームは 60 分ブロックに収まらないため未測定。
本番 512K 構成の既知値 480K 約 303 tok/s（engine 0.1.52）と比べると -1.1% だが、engine と
プロンプト長が違うので A/B の差としては扱わない（参考値）。

### KV_PREFETCH が効いた量の説明（推定）

128K・chunk 8192 では、2 チャンク目以降の 12 QSA layer が各 `[0, p0)` prefix を上げる。
prefix の総セル数は `8192 * (0+1+...+15) = 983,040`、int8 K/V を約 1 KB/cell とすると
12 layer 分で約 11.8 GB。PCIe 3.0 x16 を約 12 GB/s とすると約 1 秒で、prefill 286 秒の
0.35% 程度。実測 +0.77% と整合する。521K では 64 チャンクで約 198 GB -> 約 16.5 秒、
prefill 1739 秒の 0.95% 程度。RTX 3080 で報告された 4-5% より小さいのは、P40 の計算が
相対的に遅く、KV アップロードが critical path の小さい部分だから、という推定。

## 品質・出力一致

### PATTN_SM60

- Q2（`STRATA_PATTN_SM60=1`、`STRATA_IQ_MT_MIN=1`、`STRATA_PREFILL_CPU_SHARE=0`、
  `STRATA_ROUTE_TAIL_SKIP=0`）と Q3（同じ env で PATTN off）の 24K greedy answer は
  **バイト単位で完全一致**（189 文字、diff 空）。
- Q2 のエンジンログ: `strata pattn: STRATA_PATTN_SM60=1: the prompt path's QSA attention on the
  sm_6x score / value kernels (bit for bit qsa_decode_attn_batch; int8 K/V)`。
- 前提条件も確認: `attention gate in the attention epilogue off`（gated=false）なので
  PATTN 経路に入っている。
- 速度は今回測っていない（出力一致の 1 回確認のみ）。

### KVA

- Q1（`STRATA_KVA=1`）は **KVA が適用されなかった**。エンジンログ:
  `strata prefill: KVA configured but NOT applied: the DFlash2 drafter reads the late layers'
  hidden states (prompt 141612 positions, min_prompt 4096, suffix 2048, split 36)`。
  fork イメージの Terminal 既定が DFlash2 drafter のため。KVA は MTP か drafter off のときだけ動く。
- そのため Q1 の 128K needle は実質 KVA off と同じ: `found: true`、answer `QV-4821-ZT`、
  wall 329.37 s（Q0 は 329.38 s）。24K answer も Q0 と完全一致。
- MTP での KVA on/off 再試験は `runKva.sh` を用意したが、GPU ロックが別タスク
  （`dknobs-decode`、10:21 取得、83.6 GB のモデルダウンロード中）に取られ、30 分待っても
  空かなかったため未実行（未確認）。

## 判定

| 候補 | prefill 差（24K / 128K） | TTFT 差（128K） | 品質影響 | リスク | 判定 |
|---|---|---|---|---|---|
| `STRATA_KV_PREFETCH=1` | +0.86% / +0.77% | -2.10 s | 未確認（コピーのみ） | 追加 stream/event、bitwise 保証なし | **差なし**（P40 では採用推奨しない） |
| `--prefill` chunk 8192 -> 16384 | 未測定 | 未測定 | 8192 超は not bit-exact | 出力が変わる | **未測定 / 条件付き候補** |
| `STRATA_PREFILL_CPU_SHARE` | 24K/128K は対象外（chunk 8192） | — | 短文では last bits 変化 | 長文では無関係 | **差なし（構造上）** |
| `STRATA_KVA=1` | 未測定（DFlash2 で turn away） | 未測定 | 近似、split 36 で品質 tradeoff | DFlash2 と併用不可 | **未確認 / 条件付き保留** |
| `STRATA_PATTN_SM60=1` | 未測定 | 未測定 | 24K answer 完全一致 | sm_6x build 限定、gate fusion off 必須 | **条件付き（速度次第）** |
| `STRATA_IQ_STAGE_TABLES=1` | 未測定 | 未測定 | bitwise | P100 で効果なし、3090 Ti で遅い | **未測定** |
| `STRATA_MMQ_KQ=1` | 未測定 | 未測定 | MMQ の bits | 別経路、sm_61 では opt-in | **未測定** |
| `STRATA_PF_W16_CACHE_MIB` | 未測定 | 未測定 | engine scratch では bitwise | VRAM を食う、`--batch` なしでは既定 off | **未測定 / 候補** |

総合: 今回の P40 512K 構成では、`STRATA_KV_PREFETCH` は 24K/128K とも 5% 閾値を大きく下回り、
521K でも本番既知値と同等（299.6 vs 303、参考）だった。**env だけで prefill を大きく速くする
候補としては、KV_PREFETCH は採用推奨しない**。chunk 16384、KVA（MTP）、PATTN の速度、
PF_W16_CACHE は未測定で、次の候補。

## リスク

- `STRATA_KV_PREFETCH`: 効果が小さいのに stream/event が増える。bitwise は未確認。無効のままが安全。
- `STRATA_KVA`: 近似なので長文の抽出・agent ループで品質が落ちうる（docs の split 24 の結果）。
  DFlash2 drafter と併用すると黙って無効になる（今回確認）。MTP か off が前提。
- `--prefill 16384+`: CUDA では not bit-exact。品質確認なしに本番へ入れない。
- `STRATA_PATTN_SM60`: sm_6x build のみ。gate fusion off のときだけ。出力は一致したが、
  速度改善は未確認。
- `STRATA_PREFILL_CPU_SHARE`: 1024 未満の chunk で last bits が変わる。長文には効かない。
- `STRATA_DENSE_F32=1` は本番設定。cuBLAS build 依存の bitwise 性（`gemm_f32_parity`）は
  今回再確認していない。

## 再現コマンド

```bash
# main: A (KV_PREFETCH off) -> B (KV_PREFETCH=1)、24K x3 / 128K x2 / 521K x1
bash ~/work/wt/pknobs/scripts/runPknobs.sh main

# quality: KVA (DFlash2 のまま) と PATTN の出力一致
bash ~/work/wt/pknobs/scripts/runPknobs.sh quality

# KVA を MTP で再試験（未実行）
bash ~/work/wt/pknobs/scripts/runKva.sh
```

単発のベンチ:

```bash
STRATA_URL=http://127.0.0.1:18080 BENCH_SEED=42 STRATA_API_KEY=$KEY \
  python3 ~/work/wt/pknobs/scripts/prefillBench.py 128000 2 1 prose
```

## 未確認・逸脱

- A-B-A ではなく A -> B の 2 アーム。60 分ブロックに A-B-A と 480K を両立できなかった。
  24K/128K は同一シードの paired prompt で比較し、中央値と 5% 閾値で判定した。
- 480K は B のみ、実際は 520,966 トークン。A アームと本番 303 tok/s は参考値で、A/B 差ではない。
- GPU ロックは main ブロックで 73 分保持した（60 分超過）。内訳: 521K の prefill が 29 分、
  復元時に fork コンテナを消す前に本番を起動したため本番が RAM 不足で再起動を繰り返し、
  healthy 待ちが伸びた。以降の quality ブロックは 32 分で収めた。復元手順は
  `runPknobs.sh` の `restore_prod` で fork を先に消すよう修正済み。
- KVA は DFlash2 drafter のため適用されず、MTP での再試験は `dknobs-decode` の stale lock と、
  その後 `pgdn-main` の takeover（10:53:51）でロックが空かず、30 分以上待っても取れなかったため未実行。
- chunk サイズ、CPU_SHARE の実測、IQ_STAGE_TABLES、MMQ_KQ、PF_* は時間とロックの都合で未測定。
- 温度は main 中 46 -> 77 C、clocks は 1531 MHz のまま（89 C の throttle には未到達）。
- 本番復元: quality ブロック終了時に `prod.env` と一致、preset `docker.json` に計測用 `--set` なし、
  healthy を確認。その後 `dknobs-decode` がロックを取り本番を止めたが、そのタスクはモデル未指定で
  83.6 GB のダウンロードに入り、10:21 以降コンテナもエンジンも残さず停止した（stale lock）。
  10:55-10:57 に `prod.env` で本番を復元し、healthy（engine 0.1.52、`--max-context 524288`、MTP、
  `STRATA_DENSE_F32=1`）と preset clean を確認した。
- その後 `pgdn-main` が 10:53:51 に stale lock を takeover し（`~/work/results/pgdn/lock-takeover.txt`）、
  10:59 頃に本番を止めて自分の計測アームを開始した。本番は pgdn のブロック終了時に pgdn の trap が
  復元する。このタスクの終了時点では本番は pgdn の管理下にあり、healthy の最終確認はできていない。
- `/tmp/strata-gpu.lock` は `pgdn-main` 名義で、pgdn の計測が動いている（`strata-pgdn` コンテナ）。
  他タスクのロックなので解除していない。

## 生ログ

- `~/work/results/pknobs/main.out` — main ブロックの全出力
- `~/work/results/pknobs/A-24k.jsonl` / `A-128k.jsonl` — A アーム
- `~/work/results/pknobs/B-24k.jsonl` / `B-128k.jsonl` / `B-480k.jsonl` — B アーム
- `~/work/results/pknobs/Q0-*.jsonl` / `Q1-*.jsonl` / `Q2-answer.jsonl` / `Q3-answer.jsonl` — 品質
- `~/work/results/pknobs/{A,B,Q0,Q1,Q2,Q3}/engine-environ.txt` — 実効 env
- `~/work/results/pknobs/{A,B,Q0,Q1,Q2,Q3}/state/strata/launch/docker.engine.log` — エンジンログ
- `~/work/results/pknobs/prod-restore.txt` / `prod-preset.json` — 本番復元の確認
