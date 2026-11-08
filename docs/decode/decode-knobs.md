# decode-knobs: decode に関係する起動フラグ / env と A/B の設計・結果

作業ツリー: `~/work/wt/dknobs`（branch `docs/decode-knobs`、ソース変更なし）
対象: engine 0.1.57（image `strata:fork-9f7d977-base`）、Qwen3.8-Flash-Next IQ3_XXS、P40 (sm_61) 1 枚
本番相当: `--max-context 524288 --kv int8 --kv-resident 32768 --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp ... --ple-io ram --prefill auto --rope-scaling yarn --rope-scale 2` + `STRATA_DENSE_F32=1`

このファイルは「意味・bitwise・sm_61 可否・既存実測」をコードと既存ログからまとめ、その後に本ブロックで何が測れたか（測れなかったか）を書く。
推定は「推定」と明記する。数値の出典は行内に書く。

## フラグ / env 表

「bitwise」はモデル出力（greedy のトークン列）が変わるか。draft 系は候補が変わるだけで、採用トークンは verify を通るため出力は変わらない。
「sm_61」は P40 で有効か（アーキ既定で on か、opt-in か、不可か）。

| フラグ / env | 意味（コードの根拠） | bitwise | sm_61 | 既存実測・根拠 |
|---|---|---|---|---|
| `--spec N` | MTP/drafter の verify window の幅（トークン数）。draft 数は N-1。既定 2、setup は 4 を書く。suffix/lookup が当たると窓は N+2 まで伸びる（`generate.cpp:13780` の `S = o.spec`、`S_look`）。 | はい（draft は必ず verify され、採用トークンは greedy と一致） | 可（arch 制限なし） | 0.1.57 fork / 128K / dflash2 conf0.5 の 1 パス: spec6 = 37.87, 38.29, 44.17, 38.29 tok/s、spec4 = 43.13, 42.27, 49.78, 41.13 tok/s（`~/work/results/build/run-base*/bench-*.jsonl`）。spec4 が約 +13%（1 パスずつ、ノイズ大） |
| `--spec-min-p P` | MTP の窓を 1 draft ずつ伸ばす閾値。draft の確率が P 未満で止める（`window_rule.hpp:64`）。 | はい | 可 | 本番 0.5。0.1.57 の dflash2 では `--draft-conf` が優先される（下記） |
| `--draft mtp\|dflash2` | drafter の選択。既定 mtp。dflash2 は `--draft-model PACK` と `--mtp-draft-vocab` が要り、`--draft-fallback none` なら MTP 層を積まない（約 0.9 GiB が expert cache に回る）。 | はい（verify 経由） | 可（CUDA 実装あり） | 本番は MTP。0.1.57 fork の Terminal 既定は dflash2（`strata_control/drafting.py`） |
| `--draft-conf P` | dflash2 専用。窓を T=1 から始め、selector の確率の積が P 以上である限り伸ばす（`window_rule.hpp:66`）。`--spec-min-p` を置換する。0 なら `--spec-min-p` に戻る。 | はい | 可 | launcher は expert slot 数で決める（`drafting.py` の `CONF_STEPS`）: ≥12000 → 0.35、≥7500 → 0.5、≥3000 → 0.7、else 0.8。512K 本番の 10315 slots は 0.5。pknobs が 0.5/0.7 を A/B 中（2026-11-07 09:48 の lock owner の cmdline） |
| `--kv-resident N` | QSA 12 層それぞれで VRAM に置く KV cell 数。残りは pinned RAM。0 で全 VRAM。help は min 20480 だが、コードの下限は `STRATA_KV_RESIDENT_MIN`（既定 4096、`layer.cpp:774`）。512K では 32768 cells = 12 層 × 1056 B = 415 MB VRAM、残り 6.19 GiB が RAM。65536 で 831 MB、4096 で 52 MB。 | はい（同じバイトをどこに置くかだけ） | 可 | 本番 512K のログ: 129699 token で KV hit 83.74%（371.3 MiB を RAM から）、479998 token で 75.50%（152.1 MiB を RAM から）。480K で 89% という依頼文の数字は本番ログと合わない（実測 75.5%） |
| `--expert-cache auto\|N` | VRAM に置く expert blob 数。auto は空き VRAM から決める。512K 本番は 10315 slots / 16.73 GiB。1 slot 約 1.66 MB。 | はい（GPU/CPU のどちらで計算しても同じ値） | 可 | 本番 512K: 10315 slots。128K fork (dflash2): 10720 slots / 17.40 GiB。480K の decode hit 86.7% |
| `--pcie-frac`（既定 0.55 DMA for native pack） | PCIe で GPU に持ってくる expert の割合。0.55 が既定なので PCIe 経路は既定で使われる（`generate.cpp:879`）。 | はい | 可 | 本番 480K: 192 件を PCIe 経由（2.3% of 8485 routed）。PCIe 待ち 0.36 ms/window（research） |
| `STRATA_PCIE_MIN1` | 1 層の miss が 1-2 個だと share が 0 に丸まり PCIe lane が遊ぶので、miss ≥ 2 なら最低 1 個を PCIe に回す（`expert_source.cpp:4967`）。 | はい | 可 | P40 未測定。waitCPU 3.04 ms/window の一部を削る候補（推定） |
| `--pcie-mode dma\|kernel\|direct` | auto=2（staging copy kernel）、dma=0（copy engine）、direct=1。native pack の既定は auto（`generate.cpp:11639`）。 | はい | 可 | P40 未測定。PCIe 0.36 ms/window なので上限は小さい（推定） |
| `STRATA_MTP_BATCH_F16=1` | MTP の draft window のバッチ prefill を FP16 GEMM にする（既定は Q8_1×Q8_0 MMQ、`prefill.cpp:2201`）。draft の内部 K/V だけ。 | はい（draft のみ、verify で吸収） | 可だが P40 は FP16 1/64 なので不利（推定） | P40 未測定。dflash2 では MTP 層を積まないので無関係 |
| `STRATA_GR_FAST=1` | hyper-connection read の latency-hidden norm/up（gfx906 由来、CUDA でもビルドされる）。CUDA の既定は Volta + `STRATA_SM70_TABLE` のみ on（`fused_gr.cu:2627`）。 | はい（同じ和の順序、コメント明記） | 可（opt-in） | **P40 では T≤4 で遅い**（82.0 → 88.2 µs at T1、`docs/OLDER_GPUS.md:270`）。本番 T≈4-6 なので採らない |
| `STRATA_EXP_MODE` | VRAM expert のレイアウト。0 = CUDA、8 = shared + SwiGLU + q8_1 融合。sm_61 は既定で 8（`iq_kernels.cu:6200` 付近、`STRATA_SM61_TABLE=0` で CUDA レイアウト）。 | はい（bitwise equal とコメント） | 既定 on | P40 実測: 18-group call で IQ3_S 390 → 255 µs、IQ3_XXS 345 → 252 µs（bitwise equal）。明示設定は no-op |
| `STRATA_B6_MMVQ_ROWS=1` | 複数列 MMVQ で活性 fragment を列間で再利用する exact レイアウト（`native_mmvq.cu:1383`）。sm_60/sm_86 は既定 1、P40 は既定 0（`nv_defaults.cpp` に sm_61 ブロックが無い）。 | はい（exact） | opt-in | sm_60 で −1.3%、sm_86 で A4000 0 / 3060 −1.7%、0.1.45 で +0.2..2%（`docs/DETAILS.md:1110,1506`）。P40 未測定 |
| `STRATA_VERIFY_DEVICE_PLAN=1` | 全 expert が VRAM に無い層の plan を device で作る（`verify.cpp:1291`）。`all_resident_` でないときだけ有効。512K は partial residency なので効く。 | はい（exact、コメント「neutral on RIBPC 1-2 GPUs」） | 可 | P40 未測定。512K では 10315/24576 experts なので `all_resident_` は成立しない |
| `STRATA_DENSE_F32=1` | 大きい dense GEMM を FP16 operand の cuBLAS ではなく FP32 コピー + `cublasSgemm` でやる（`docs/OLDER_GPUS.md` の表）。本番が使用。 | 条件付き（cuBLAS build 依存。`gemm_f32_parity` で確認） | opt-in | 本番で使用中。prefill 側のフラグで decode には直接効かない |
| `STRATA_DECODE_TIMING=1` | リクエストごとに decode の内訳を 1 行出す（`generate.cpp:19074`）。GPU-reach wait / per-layer host（plan, actq, jobs, CPU）/ commit / draft / CPU experts / VRAM hits / PCIe。 | いいえ（計測のみ） | 可 | 本ブロックで全アーム on にする設計（比較を揃えるため） |

### sm_61 の既定（`nv_defaults.cpp`）

sm_61 用のブロックは無い（sm_60 / sm_86 / sm_121 のみ）。したがって P40 の既定はエンジン素の既定で、
`docs/OLDER_GPUS.md` の 3 つ（wide dequant、expert mode 8、`kIlRows_61` の interleaved MMVQ）はコード側の既定で on。
`STRATA_DENSE_F32=1` だけが opt-in で、本番が設定している。

## 測定計画（60 分 1 ブロック）

アームは起動フラグなのでエンジン再起動が要る。起動は 40 GiB のロードで 3-6 分（実績 175 s と 369 s）。
優先順は decode の主質問（drafter）→ KV 配分（480K）→ 余り。

| 順 | アーム | 起動 args の差分 | 測るもの |
|---|---|---|---|
| 1 | `base` | 本番と同一（MTP, spec 4, spec-min-p 0.5, kv 32768, cache auto） | 4K/24K × code/prose × 3、answers 4 本、timing |
| 2 | `df2` | `--draft dflash2 --spec 6 --draft-conf 0.5`、MTP を外す | 同上 |
| 3 | `kv64` | `base` + `--kv-resident 65536` | answers、480K 1 本、余れば 4K/24K 1 パス、timing |

- `BENCH_SEED=42`、1 本目除外、中央値、5% 閾値。
- `STRATA_DECODE_TIMING=1` を全アームで on（内訳は各アームの最初のリクエストから）。
- P40 はファンが無く 89 C で thermal slowdown（`docs/OLDER_GPUS.md`）。各アームの前後で温度と clock を記録し、1 本目を捨てることで高温側の比較にする。
- 480K は `/v1/messages/count_tokens` で実トークンを先に測り、524288 を超えないよう縮める（bench.py の 3.6 chars/token は markdown では甘い。実測約 2.86）。

### なぜ 480K を kv64 で 1 回だけか

本番ログに 0.1.52 の 479998-token run があり、baseline の TTFT 1532 s / 302.6 tok/s / KV hit 75.50% が既にある。
0.1.57 で同じ 480K をもう一度測ると 26 分かかり、base と kv64 の 2 本で 52 分になって他のアームが消える。
そこで 480K は kv64 の 1 本だけ測り、KV hit 率（設定で決まる量）を本番ログの 75.50% と比べる。
TTFT / decode の絶対値は engine 0.1.52 と 0.1.57 の差が混ざるので、その点は「注意」に書く。

## 結果（2026-11-07 のブロック）

### ブロックは 2 回失敗した（A/B の数値は未取得）

**1 回目（10:21-10:39、ロック取得済み）**: `run-decode-ab.sh` は prod の preset JSON をコピーして
`PRESET=<arm>` で起動する方式にした。prod を停止し、MemAvailable 56 GiB を確認して base を起動したが、
entrypoint は MODEL 未指定のため既定モデル（IQ3_S）の setup を始め、`IQ3_S` の **38.9 GB の部分ダウンロード**
（`.../models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf.part`、10:29-10:31）を作った。
モデル DL 禁止に触れるため即座にコンテナとスクリプトを停止した。エンジンは healthy にならず、decode は 1 本も走っていない。

**2 回目（10:58 開始、ロックは既に pgdn-main が保持）**: 起動方式を `strata launch` 直叩き（`--model IQ3_XXS` 明示、
`--drafting mtp|dflash2`、`--set spec=...`）に直したスクリプト `run-decode-ab2.sh` を用意したが、
10:53 に GPU ロックが pgdn-main に渡っており（私の 1 回目のスクリプトを kill した後にロックが解放された）、
スクリプトのロック判定が「既に保持」と誤認してアームを起動した。コンテナは 44 秒で
`strata: There is no saved preset called 'base'. Saved presets: docker.` で終了した
（`--preset base` は既存 preset を要求し、`--save-preset base` だけでは新規作成にならない）。
df2 も同じ理由で即終了し、GPU は使っていない。このため A/B の decode 中央値・timing 内訳・480K の KV hit は **未測定**。
修正は `--preset` を外して `--save-preset <arm>` だけにする（または preset を先に作る）。

ログ: `~/work/results/dknobs/run.log`（1 回目）、`~/work/results/dknobs/docker-run-base.log`。

### 既存実測（本番ログと 0.1.57 fork の 1 パス、参考値）

| 条件 | engine | ctx | decode tok/s | prefill tok/s | expert hit | KV hit | 出典 |
|---|---|---|---|---|---|---|---|
| 本番 MTP spec4 | 0.1.52 | 512K | 51.0（14 token） | 418.4（129699 token） | 94.9% | 83.74% | prod `docker.engine.log` |
| 本番 MTP spec4 | 0.1.52 | 512K | 34.0（14 token） | 302.6（479998 token） | 86.7% | 75.50% | prod `docker.engine.log` |
| fork dflash2 spec6 conf0.5 | 0.1.57 | 128K | 37.87 / 38.29 / 44.17 / 38.29 | 445.6 / 437.0 / 493.5 / 486.6 | 80.2%（短い answer の cold） | 96.30%（同） | `run-base/bench-*.jsonl` |
| fork dflash2 spec4 conf0.5 | 0.1.57 | 128K | 43.13 / 42.27 / 49.78 / 41.13 | 443.1 / 439.9 / 493.4 / 486.7 | 80.2% | 96.30% | `run-base-spec4/bench-*.jsonl` |
| fork base（他タスク pgdn の同一 image / 512K） | 0.1.57 | 512K | —（max_tokens=1） | 431.0 / 424.0 / 428.4 / 423.9（135K-143K token、TTFT 306-338 s） | — | — | `~/work/results/pgdn/block1.log`（参考。私のアームではない） |

注: 本番の decode は 14 token の生成なので bench の 256 token と比較できない。fork の expert/KV hit は 53 token の
answer 確認直後の行で、24K プロンプトの行ではない。したがって上の表は「桁と方向」の参考で、A/B の差としては使えない。

### 副産物

- 起動方式の知見: fork 0.1.57 で prod preset をコピーして `PRESET=` だけで起動すると、MODEL 未指定のため
  既定モデル（IQ3_S）の setup が走る。`--model IQ3_XXS` を明示するか、settings 経路（MODEL/CONTEXT）で起動する必要がある。
- `--drafting mtp|dflash2` は entrypoint の環境変数には無い。`strata launch` を直接叩くのが確実。
- `--set` は engine フラグ / STRATA_* env 専用で、`drafting` のような launch 設定には使えない（`--drafting` として engine に渡り拒否される）。
- `strata launch --preset NAME` は既存 preset を要求する。新規に作るときは `--save-preset NAME` だけにする。

## 判定

**不成立（A/B 未測定）**。フラグ表と起動方式の知見だけが成果。数値の捏造はしていない。

現時点で言えること（すべて既存データ / コードからの推定。P40 の 0.1.57 同一ブロック A/B ではない）:

| 候補 | 判定 | 理由 |
|---|---|---|
| dflash2 (`--draft-conf 0.5`) vs MTP spec4 | **条件付き** | 0.1.57 fork の 1 パスでは dflash2 spec4 が 43-50 tok/s、spec6 が 38-44 tok/s。MTP の 0.1.57 は未測定。prod 0.1.52 の 14-token decode 51 tok/s とは比較できない。drafter の採用は同一ブロック A/B が必須 |
| `--spec 6` vs `--spec 4`（dflash2） | **条件付き（spec4 優位の可能性）** | 128K の 1 パスで spec4 が約 +13%。n=1 なので 5% 閾値の判断は不能 |
| `--kv-resident 65536` | **条件付き** | 480K の KV hit は 75.50%（32768）。resident を 2 倍にすると resident pages は 8192 → 16384（480K の総 pages 約 120K）で hit は上がる見込み（推定）。VRAM は +416 MB（expert slot 約 250 個減）。TTFT への寄与は未測定 |
| `--kv-resident 4096` | **条件付き（逆方向）** | コードの下限 4096 は「4096/8192 でも 20480 と同じ tokens」というコメントあり（`layer.cpp:774`）。VRAM を 363 MB 空けて expert slot を約 220 増やせる。ただし 480K では hit が落ちる可能性 |
| `STRATA_B6_MMVQ_ROWS=1` | **条件付き** | bitwise、sm_60/sm_86 で既定。P40 だけ opt-in。既存の他アーキ実測は −1.7% 〜 +2%。P40 未測定 |
| `STRATA_PCIE_MIN1=1` / `--pcie-mode dma` | **差なしの可能性が高い** | bitwise だが上限は PCIe 0.36 ms/window（全体 52-56 ms の 0.6%）。PCIe 経由は 480K で 2.3% の expert だけ |
| `STRATA_GR_FAST=1` | **悪化** | P40 の T≤4 で 82.0 → 88.2 µs（既存実測）。本番 T≈4-6 |
| `STRATA_EXP_MODE` | **差なし（設定不要）** | sm_61 は既定で mode 8、bitwise equal、既に最速 |
| `STRATA_MTP_BATCH_F16=1` | **優先度低** | MTP のみ。P40 の FP16 は 1/64 |
| `STRATA_VERIFY_DEVICE_PLAN=1` | **優先度低** | exact だが他 GPU で neutral。512K の partial residency では効くが期待は小さい |

## 注意（逸脱）

- **IQ3_S の 38.9 GB 部分ダウンロードを作った**（`/var/lib/docker/volumes/compose_strata-data/_data/Strata-data/models/IQ3_S/`、10:29-10:31）。
  モデル削除禁止のため消していない。ディスクは 170G 空き。要判断（削除するならメインの指示で）。
- 本番の復元は pgdn-main の EXIT trap が 11:40:48 に実行し、11:50 時点で healthy・
  engine 0.1.52・`--max-context 524288 --mtp ... --spec 4 --spec-min-p 0.5` を確認した
  （`~/work/results/pgdn/restore.txt`）。その直後 11:50:21 に integrated-parity がロックを取り、
  本番を自分のブロックのため停止した（現在 strata は Exited）。**最終状態の確認は integrated の復元後**。
- 1 回目のスクリプトを kill した後、10:53 に pgdn-main が「stale lock takeover」としてロックを取得した
  （`~/work/results/pgdn/lock-takeover.txt`: owner dknobs-decode、31 分、プロセスなし、VRAM 0）。
  2 回目のスクリプトはロック所有者を確認せず「既に保持」と誤認するガードの穴があり、修正した（owner が自分か確認）。

## 再現コマンド

```
# 準備（GPU 不要）
python3 ~/work/wt/dknobs/scripts/mk-corpus.py ~/work/results/dknobs/corpus/dedup.md

# 1 ブロック（ロック取得 -> base -> df2 -> kv64 + 480K -> 本番復元）
bash ~/work/wt/dknobs/scripts/run-decode-ab2.sh 3480 base:3 df2:3 kv64:1
```

`run-decode-ab2.sh` は `strata launch` を直接叩き、`--model IQ3_XXS` を明示する（既定モデルの setup を避ける）。
アームは `arm:reps` で指定する。480K は `bench-long.py` を別途呼ぶ（`run-decode-ab2.sh` には未統合）。

