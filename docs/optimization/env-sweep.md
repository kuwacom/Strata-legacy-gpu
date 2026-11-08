# env sweep: 本番経路の opt-in スイッチ A/B（P40 / sm_61）

作業ツリー `~/work/wt/env`（branch `perf/env-sweep`、ソース変更なし）。計測は 2026-11-07 01:53-02:50 UTC、
`ssh llm-02` の Tesla P40 1 枚。イメージ `strata:fork-9f7d977-base`（engine 0.1.57）、本番と同じ 512K 構成
（`--max-context 524288 --kv int8 --kv-resident 32768 --ple-io ram --rope-scaling yarn --rope-scale 2 --spec 4 --spec-min-p 0.5`）。

## 対象 env の意味・bitwise 可否・sm_61 可否

| env | 何をするか | 実装 | bitwise | sm_61 | P40 実測 |
|---|---|---|---|---|---|
| `STRATA_HC_PERSIST=1` | verifier の hyper-connection read（norm + down + inject + up + mix）を 80 block の persistent kernel 1 発にまとめる。2 つの grid barrier を atomic counter + `clock64()` timeout で張る。起動時に T=1..8 を実際に走らせて常駐を判定し、通った T までを使う | `src/core/verify.cpp:1587`（`gr_read_group` から `fused_gr_read_persist`）、`src/kernels/cuda/fused_gr.cu:3296-3778` | **いいえ**（和の順序が違う。既定 read との差は起動時チェックで 1e-4 未満に束縛） | 可（warpSize 32、`clock64()` 分岐、`__nanosleep` は sm_70+ のみ） | **常駐成立**（後述） |
| `STRATA_VERIFY_DEVICE_PLAN=1` | 層ごとの expert 計画を device の `resident_plan` で作り、host の plan/doorbell 待ちを飛ばす（`skip_` 表）。`all_resident_` でないかつ `always_publish_` でないときだけ有効 | `src/core/verify.cpp:1292-1296, 2248, 2387-2419` | はい（コード内コメント "exact"） | 可 | 実施 |
| `STRATA_QFUSE=1` | 活性の q8_1 像を生産側（GDN output norm / HC read）で書く。消費側の quantize を省く | `src/core/verify.cpp:292, 1783, 1601, 2176, 2313-2331`、`src/kernels/cuda/fused_gr.cu:246` | はい（コメント "the same bytes"） | 可 | 実施 |
| `STRATA_VERIFY_QDEDUP=1` | expert 用 q8_1 像を先に作り shared expert がそれを読む | `src/core/verify.cpp:180, 2292` | はい（`quantize_q8_1_rows` と `native_quantize_q8_1` は同じバイト） | **単独では no-op** | 未実施（後述） |
| `STRATA_VERIFY_NODES=1` | capture した window graph の node 数と種別を stderr に出す（計測のみ） | `src/core/verify.cpp:2820`、`src/core/mtp.cpp:932` | 出力に影響なし | 可 | 実施 |
| `STRATA_DECODE_TIMING=1` / `STRATA_PREFILL_TIMING=1` | リクエストごとの decode / prefill 内訳を 1 行で出す（計測のみ） | `src/program/generate.cpp:17831`、`src/prefill/prefill.cpp:2998` | 出力に影響なし | 可 | 全アームで有効 |
| `--stats` | `strata generate` の per-stage 集計。**serve では出力されない**（`generate.cpp:20974` は generate 経路のみ） | `src/program/generate.cpp:20790-20830` | - | - | 代替として decode timing 行を保存 |

排他・相互作用（コードで確認）:

- `STRATA_HC_PERSIST` と `STRATA_QFUSE` は**排他**。`fused_gr_read_persist` は `a[0].q8_mixed != nullptr` のとき false を返す（`fused_gr.cu:3776-3778`）。同時に立てると QFUSE 側が勝って persistent read は走らない。
- `STRATA_QFUSE=1` は `self_commit` も落とす（`verify.cpp:1356`）。
- `STRATA_VERIFY_QDEDUP` は `!sh_fork` が条件（`verify.cpp:2292`）。`sh_stream_on()` は CUDA で既定 true（`verify.cpp:357-366`）なので、P40 の既定では `qdedup` は常に false。単独で立てても何も起きない。効かせるには `STRATA_SH_STREAM=0` が要る（共有 expert の fork 自体を止める別の変更）。

## HC_PERSIST の常駐判定

起動ログ（`~/work/results/env/arms/Bhc-persist/engine.log`）:

```
strata hc: CUDA0: STRATA_HC_PERSIST=1: the verifier's hyper-connection read runs as one persistent kernel
 (80 blocks on 30 multiprocessors, up to 5 tokens; 1 token counts timed out in the check;
  checked against the default read, max relative difference 1.4e-07)
```

- 80 block × 256 thread が 30 SM に常駐した（2.67 block/SM）。`max_t = 5` は `--spec 4` の窓（T≤5）をちょうど覆う。
- 起動時チェックは T=1..5 を既定 read と比較して最大相対差 1.4e-07。T=6 は barrier timeout（＝その T では常駐しない）で、T=6 の窓は既定 read にフォールバックする。
- **注意**: この run の lookup drafter は T=6 の窓も作る（`--lookup-max-t 12`）。T=6 は persistent read の対象外なので、HC_PERSIST の効果は T≤5 の窓に限られる。

### KL / 一致（`STRATA_LOGITS_DUMP`、4K プロンプト、max_tokens=1）

base（`A0-base-kl`）と HC（`Bhc-check`）の同じ位置の logits を比較（`kl.py`、record は pos で突き合わせ）:

| pos | T | top-1 一致 | KL(base→HC) | max abs logit diff |
|---:|---:|---|---:|---:|
| 52（ヘルスチェック由来の decode 窓） | 1 | はい | 3.71e-04 | 1.016 |
| 3848（4K リクエストの窓） | 6 | はい | 0.0 | 0.0 |
| 3854（4K リクエストの窓） | 1 | はい | 1.66e-03 | 0.479 |
| 48（ヘルスチェック） | 4 | はい | 1.89e-07 | 0.783 |
| 0, 6, 12, 18, 24, 30, 36, 42（ヘルスチェック） | 6 | はい | 0.0 | 0.0 |

- T=1 と T=4 は persistent read が走る窓で、**bitwise ではない**（KL 最大 1.66e-03、top-1 は全 12 窓で一致）。
- T=6 の窓は `max_t=5` を超えるため既定 read のままで、bitwise（差 0）。
- 参考: 起動時チェックの T=1..5 は最大相対差 1.4e-07。logits の KL 1.66e-03 は温度 0 の greedy を変えない範囲（top-1 一致）。
- 数値の位置は `pos`（窓の先頭トークン位置）。1 リクエストで複数窓が出るため、`pos` で対応付けた。

## A-B-A 計測

`BENCH_SEED=42`、temperature 0、code プロンプト、max_tokens 64。decode は 4K/24K を各 3 回、prefill は 24K を各 3 回。
各サイズの 1 本目を除外し、残り 2 本の中央値（3 回のうち 1 本目は warmup で 36-38 tok/s と低い）。
単位は tok/s。生値は `~/work/results/env/arms/<arm>/bench-*.jsonl`。

| アーム | 構成 | 4K decode | 24K decode | 24K prefill |
|---|---|---:|---:|---:|
| A1-base | base | 46.93 | 47.73 | 487.45 |
| Bhc-persist | `STRATA_HC_PERSIST=1` | 47.93 | 47.31 | 485.45 |
| A2-base | base | 46.91 | 45.00 | 486.00 |
| Bdp-plan | `STRATA_VERIFY_DEVICE_PLAN=1` | 46.26 | 47.67 | 484.40 |
| A3-base | base | 46.95 | 46.81 | 482.85 |
| Bq-qfuse | `STRATA_QFUSE=1` | 47.20 | 46.94 | 484.90 |

差（%）は各候補の前後 2 つの base の平均を基準にした（QFUSE は前の A3 のみ）。

| 候補 | 4K decode | 24K decode | 24K prefill | 5% 閾値 |
|---|---:|---:|---:|---|
| HC_PERSIST（A1-Bhc-A2） | +2.15% | +2.04% | -0.26% | 未満 |
| DEVICE_PLAN（A2-Bdp-A3） | -1.43% | +3.84% | -0.01% | 未満 |
| QFUSE（A3-Bq、A-B のみ） | +0.53% | +0.28% | +0.42% | 未満 |

**基準のばらつき**: base の 24K decode は A1 47.73 → A2 45.00 → A3 46.81 と 5.9%（最大-最小 2.73 tok/s）動いた。A2 の 24K は 1 本目 35.6 の後
46.8 / 43.2 で、43.2 も外れ値。したがってこの 3 本 × 2 本中央値では 5% 閾値を分解できない。上の差は「ノイズ内」と読むのが正しい。

### 窓あたりの内訳（`STRATA_DECODE_TIMING`、各アームの最後の 24K リクエスト）

| アーム | ms/window | verify | GPU-reach wait | per-layer host | draft |
|---|---:|---:|---:|---:|---:|
| A1-base | 47.11 | 42.64 | 37.73 | 2.36 | 4.25 |
| Bhc-persist | 47.61 | 42.86 | 37.05 | 3.20 | 4.33 |
| A2-base | 48.08 | 43.34 | 38.15 | 2.66 | 4.34 |
| Bdp-plan | 48.46 | 43.71 | 37.90 | 3.22 | 4.26 |
| A3-base | 47.19 | 42.67 | 37.71 | 2.45 | 4.25 |
| Bq-qfuse | 47.02 | 42.34 | 37.45 | 2.37 | 4.51 |

GPU-reach wait が 37-39 ms で全体の約 8 割。HC_PERSIST で GR read の起動と待ちが減っても、窓の wall は
GPU-reach wait（CPU expert の待ち）に支配されるため、差が 5% に届かないのは整合的（推定）。

## VERIFY_NODES: window graph の node 数・種別

`STRATA_VERIFY_NODES=1` を A1-base で有効にし、capture 時の stderr を記録（`arms/A1-base/engine.log`）。

| 窓 | nodes | edges |
|---:|---:|---:|
| T=1 | 2069 | 2163 |
| T=2 | 2130 | 2224 |
| T=3 | 2177 | 2271 |
| T=4 | 2186 | - |
| T=6 | 2205 | - |

- **T=5 は今回 capture されなかった**。draft policy が選んだのは 1,2,3,4,6 で、T=5 の graph は作られていない（未確認）。
- node 種別は**すべて kernel**（`memcpy` / `memset` / 型番号の node は 0）。CUDA 12.9 なので `cudaFuncGetName` で名前が出る。
- T=1 の内訳（上位、ログは 80 件で打ち切り）: 145 × native_mmvq、144 × verify_kernels、122 × native_bf16、
  3 × 97 × fused_gr、71 × native_mmvq、58 + 55 × iq_kernels、51 × `copy_i32_from_mapped_kernel`、
  48 × ずつ s2_expert_grouped / verify_kernels / native_router / shared_expert / iq_kernels / native_moe /
  `copy_rows_from_mapped_kernel` / `doorbell_publish_res_kernel`、40 + 36 × native_mmvq など。
- 起動キューの実態: 1 窓 = 1 graph launch（`verify.cpp:3369`）だが、その graph は約 2,069 node を含む。
  pre-Ampere の graph launch は node 数に比例する（研究ノートの 200 ns/node 推定だと 0.4 ms 程度）。

## 判定

| 候補 | 判定 | 根拠 |
|---|---|---|
| `STRATA_HC_PERSIST=1` | **差なし（ノイズ内）。常駐は成立** | 80 block / 30 SM、max_t=5、起動時チェック 1.4e-07。4K +2.15% / 24K +2.04% / prefill -0.26%。bitwise ではないが KL 最大 1.66e-03・top-1 12/12 一致。既定 off のまま、サンプル数を増やして再測定するのが妥当 |
| `STRATA_VERIFY_DEVICE_PLAN=1` | **差なし** | 4K -1.43% / 24K +3.84% / prefill -0.01%。コード内コメントの "neutral on RIBPC 1-2 GPUs" と同方向 |
| `STRATA_QFUSE=1` | **差なし（A-B のみ）** | +0.53% / +0.28% / +0.42%。HC_PERSIST と排他なので併用不可。A-B-A の closing A は時間切れ |
| `STRATA_VERIFY_QDEDUP=1` | **未実施（コード上 no-op）** | `!sh_fork` 条件により CUDA 既定（`STRATA_SH_STREAM` 未設定 = fork on）では `qdedup=false`。単独で立てても効果ゼロ |

## 再現コマンド

計測スクリプトは `~/work/results/env/scripts/`（`arm.sh` は engine 直起動の失敗版、`env-arm.sh` が本番経路）:

```sh
# GPU ロック + 本番停止（VRAM 0 / HugePages_Free >= 20464 まで待つ）
mkdir /tmp/strata-gpu.lock && echo "env-sweep $(date -Is)" > /tmp/strata-gpu.lock/owner
sudo docker exec strata sh -c "strata launch --stop --preset docker"
sudo docker stop -t 120 strata

# 1 アーム（Terminal が HTTP を出し、engine がコンテナ env を継承する）
printf "STRATA_HC_PERSIST=1\n" > /tmp/hc.env
~/work/results/env/scripts/env-arm.sh Bhc-persist /tmp/hc.env 0 0 1

# KL 用（logits dump、bench なし）
~/work/results/env/scripts/env-arm.sh Bhc-check /tmp/hc.env 1 0 0

# 本番復元（prod.env を読み直す）+ healthy 待ち + ロック解放
~/work/results/env/scripts/restore.sh
```

- `env-arm.sh <label> <env-file|-> <kl 0/1> <nodes 0/1> <bench 0/1>`。`STRATA_SET` と CONTEXT は本番と同じ。
- ベンチは `STRATA_URL=http://127.0.0.1:18080 BENCH_SEED=42 STRATA_API_KEY=<key> python3 ~/bench.py 4000,24000 3 64 code`。
- KL は `kl.py`（`pos` で record を突き合わせ、KL と top-1 を出す）。
- 生ログ: `~/work/results/env/arms/<arm>/`（`bench-*.jsonl`、`engine.log`、`engine-environ.txt`、`engine-cmdline.txt`、`gpu.txt`、`logits.bin`、`kl.json`、`kl-request.json`、`docker.log`）。

## 本番復元の確認

- `~/opt-work/prod.env` を読み直して `~/Strata/docker/compose/.env` にコピーし、preset を削除して compose で作り直した。
- `healthy`、image `ghcr.io/niko1221/strata:0.2.7-cuda12`、engine 0.1.52。
- engine の argv に `--max-context 524288 --kv int8 --kv-resident 32768 --ple-io ram --rope-scaling yarn --rope-scale 2 --spec 4 --spec-min-p 0.5`。
- 残留: `STRATA_DENSE_F32=1` のみ（コンテナ既定の DIR / DOCKER / PROFILE 系を除く）。`strata-env` コンテナは残っていない。
- `.env` は `~/opt-work/prod.env` と一致。GPU ロックは 02:49:57 に解放（取得 01:53:14 の 56 分 43 秒後）。

## 注意（推定・未確認）

- 差の % は 3 回 × 2 本中央値で、base 自体が 6% 動く。5% 閾値の判定は「ノイズ内」であり、有意差ではない。
- HC_PERSIST の KL は 4K プロンプト 1 本（窓 3 つ）のみ。長文・別プロンプトでの KL は未確認。
- T=5 の graph は capture されず、node 数は未確認。T=6 は persistent read の対象外（`max_t=5`）。
- QFUSE の A-B-A は closing A がなく、A-B の 1 ペアのみ。
- QDEDUP はコード上 no-op と判断したが、`STRATA_SH_STREAM=0` との組み合わせは未測定。
- `--stats` の "verify window" 行は serve では出ないため、`STRATA_DECODE_TIMING` の per-request 行で代替した。
- 最初のブロックで engine を直接 `docker exec` 起動する方式を試したが、engine 単体は HTTP を出さず（Terminal が出す）、
  ready 判定に失敗して約 10 分を消費した。2 回目のブロックは `env-arm.sh`（Terminal 経由）でやり直した。
- API キー: 最初に `~/opt-work/prod.env` をそのまま cat して 1 行表示してしまった。以後は `grep ^API_KEY=` で変数に入れるだけにした。
- 本番復元は 02:49:57 に healthy を確認してロックを解放した。その後 02:52:12 に別タスク `lookahead-all` がロックを取得し、本番を停止して `strata-fork` を起動している。2026-11-07 02:56 時点の prod 停止はこの別タスクによるもので、本タスクの残留ではない。
