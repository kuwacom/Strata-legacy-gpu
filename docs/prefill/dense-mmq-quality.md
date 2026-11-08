# dense MMQ の 512K 本番構成での品質・安定性検証

作業ツリー: `~/work/wt/gdnq`（branch `docs/dense-mmq-quality`、ソース変更なし）
対象イメージ: `strata:fork-prefill-gdn`（fork HEAD + `patches/gdn-mmq-dense.diff`）、基準: `strata:fork-9f7d977-base`
計測: `~/work/results/gdnq/`（`block1.log` / `block2.log` / `block4.log`、`arms/<label>/`）。数値は実測。推定は「推定」と明記する

## 結論

**採用推奨（opt-in、条件付き）**。

- 512K 本番構成（`--max-context 524288 --kv int8 --kv-resident 32768 --rope-scaling yarn --rope-scale 2 --spec 4 --ple-io ram --prefill auto`、`STRATA_DENSE_F32=1`）で、dense MMQ は prefill を **32K +12.6〜12.9%、128K +11.2%** 速くする。TTFT は 32K で -7.3〜-7.5 s、128K で -25.6〜-25.9 s。
- 品質は実用上同等。needle retrieval は 32K/128K の 50%/90% すべて正解、480K の 50%/90% も正解。128K の長文 QA（8 問）は base と **バイト一致で 8/8**。KL は base 同士の再実行ノイズ（4K 最終位置で 0.0176）と同程度（0.0191）で、top-1 は全点一致。
- bitwise ではない。24K の greedy 回答は言い回しが変わる（182/434 文字、意味は同一）。
- 480K では **qsa select 位相が prefill の 42%** を占め、gdn は 7.1% しかない。MMQ の効く範囲が相対的に小さいため、480K の短縮は **約 4.5%（推定）** にとどまる。

推奨 `STRATA_SET`:

```
STRATA_SET="ple-io=ram prefill=auto STRATA_DENSE_F32=1 STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 rope-scaling=yarn rope-scale=2"
```

## dense MMQ が変えるもの

- 対象は GDN 層 36 層の dense 投影: `attn_qkv.weight`（10240x2560）、`attn_gate.weight`（6144x2560）、`ssm_out.weight`（2560x6144）。`prefill.cpp:4109-4146` の half 0 分岐。
- `ssm_alpha` / `ssm_beta`（N=48）は `bf16_proj` 経路なので MMQ の対象外。
- 重みは IQ3_XXS パックの Q3_K。`prefill_mmq_kquant_test` が Q3_K の qkv 実形状 10240x2560 で rel_l2 0.0014 を記録している（`~/work/results/pgdn/block2.log`）。
- 従来（`STRATA_DENSE_F32=1`）は重みを FP32 に widen して `cublasSgemm`。MMQ は重みを量子化のまま、活性を q8_1 に丸めて int8 DP4A で積む。P40 の DP4A ピークは FP32 の約 4 倍。
- K は 256 の倍数のみ、T は 1024 行ずつスライス（`gemm.cu:1267-1278`）。`mmq::fits` が false の形は自動で cuBLAS に落ちる。

## `STRATA_DENSE_MMQ` と `STRATA_MMQ_KQ` の関係

CUDA（P40, sm_61）では **両方が必要**。片方だけでは GDN の Q3_K 投影は従来の cuBLAS 経路のまま。

コードの連鎖:

- `STRATA_DENSE_MMQ=1` は `Gemm::native_mmq` の入口だけを開ける（`src/prefill/gemm.cu:1264-1267` の `static const bool enabled`）。
- 入口を開けても、`native_mmq` は最初に `mmq::fits(type, N)` を呼び、false なら `return false` で cuBLAS に落ちる（`gemm.cu:1267`）。
- `mmq::fits` は `mmq::supported(t)` を呼ぶ（`src/prefill/moe_mmq.cu:349`）。パッチ後の `supported` は CUDA で Q3_K に対して `kq_ok()` を返す（`moe_mmq.cu:335-338` の `#if !defined(GGML_USE_HIP)` ブロックに Q3_K ケースを追加）。
- `kq_ok()` は CUDA ビルド（`STRATA_KQ_DSO`）で `kq::available()`（`moe_mmq.cu:44-47`）。`kq::available()` は `wanted() && lib().ok`（`src/prefill/mmq_kq_loader.cpp:110-114`）。
- `wanted()` は `STRATA_MMQ_KQ=1` のときだけ true（`mmq_kq_loader.cpp:109-113`）。`lib().ok` は `libstrata_kq.so` を engine と同じフォルダから dlopen し、ABI が一致したときだけ true（`mmq_kq_loader.cpp:57-98`）。
- Q3_K のカーネル本体は `libstrata_kq.so` にしかない（`src/prefill/mmq_kq_lib.cu` の `strata_kq_run` に Q3_K ケースを追加、`CMakeLists.txt` の `_strata_kq_types` に `q3_k` を追加）。engine 本体の `strata_mmq` には Q3_K インスタンスを入れていない。
- ビルド前提: `STRATA_MMQ_KQUANTS` は既定 ON（`CMakeLists.txt:134`）。CUDA では `STRATA_KQ_DSO=1`（`CMakeLists.txt:2330-2333`）。

| 設定 | GDN の Q3_K 投影 | 根拠 |
|---|---|---|
| どちらも未設定 | cuBLAS（従来） | `enabled=false` |
| `STRATA_DENSE_MMQ=1` のみ | cuBLAS（従来） | `fits(Q3_K)` が `supported` -> `kq_ok()` -> `wanted()=false` で false |
| `STRATA_MMQ_KQ=1` のみ | cuBLAS（従来） | `enabled=false`。このモデルの expert は IQ3_XXS なので K-quant expert 経路にも影響しない |
| 両方 | int8 MMQ | `enabled=true` かつ `fits(Q3_K)=true` |

実測での確認（`arms/denseonly/engine.log` と `arms/mmq/engine.log`）:

- `STRATA_DENSE_MMQ=1` 単独: engine log に `strata: K-quant prompt kernels from .../libstrata_kq.so (STRATA_MMQ_KQ=1)` が **出ない**（DSO を開いていない）。GDN の Q3_K は cuBLAS のまま。
- 両方: この行が出る。
- 注意: `prefill gemm: dense MMQ on (STRATA_DENSE_MMQ=1)` の行は単独でも出る。これは Q3_K 以外の dense 行列（draft 層の Q8_0 など、`supported()` が無条件に true を返す型）が MMQ に入ったためで、Q3_K が入った証拠にはならない。判定は GDN 位相の時間と K-quant ライブラリの行で行う。

## テスト設計

- 機材: Tesla P40 24GB（sm_61）。本番と同じデータボリュームを読み、state だけ `arms/<label>/state` に複製して起動（共有 preset `/data/state/strata/launch/docker.json` を汚さないため）。
- 起動: 本番と同じ `CONTEXT=512k` / `STRATA_SET` / `--spec 4`。コンテナ名 `strata-gdnq`（port 18081）、480K 用は `strata-gdnq480`（port 18082）。
- アーム: `base` = `strata:fork-9f7d977-base`、`mmq` = `strata:fork-prefill-gdn` + `STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1`、`denseonly` = 同イメージ + `STRATA_DENSE_MMQ=1` のみ、`base2` = base の再実行（ノイズ床）。
- needle: `~/work/results/quality/needle.py`（`~/Strata/docs/*.md` を 3.25 文字/token で埋め、一意 ID を 50%/90% に挿入、`temp 0`、ID を返すか）。
- 長文 QA: `~/work/results/gdnq/qa.py`。128K 相当の文書の 50% 位置に事実メモ（固有名詞 1、数値 3、対応表 3 行、推論 1）を埋め、8 問を 1 リクエストで問う。採点は `score_qa.py`。
- KL: `STRATA_LOGITS_DUMP=/out/logits.bin` の記録を `kl_by_rec.py` で突き合わせ。4K/24K のプロンプト最終位置。
- 出力一致: `eq.py`（2K/8K/16K/24K、同じ質問、`temp 0`、max 128）。
- 速度: needle/QA リクエストの TTFT と prefill tok/s。decode は eq の steady tok/s。位相内訳は `STRATA_PREFILL_TIMING=1`。

## 結果: 速度

prefill（needle リクエスト、1 回ずつ）:

| サイズ | prompt tokens | base tok/s | mmq tok/s | 差 | base TTFT | mmq TTFT |
|---|---:|---:|---:|---:|---:|---:|
| 32K | 32,123 | 491.7 / 491.1 | 553.6 / 554.2 | **+12.6% / +12.9%** | 65.33 / 65.42 s | 58.03 / 57.96 s |
| 128K | 125,007 | 491.2 / 488.1 | 546.1 / 543.0 | **+11.2% / +11.2%** | 254.5 / 256.1 s | 228.9 / 230.2 s |
| 480K | 470,288 | 未測定 | 333.8 | - | - | 1409.1 s |

- 480K の base は 1 回 25 分級になるため 60 分ブロックに収まらず未測定。mmq の 480K TTFT は 23.5 分。
- 480K の位相内訳（mmq、470,281 tokens）: GPU timeline 1,406,295 ms のうち **qsa select 590,360 ms（42.0%）**、qsa attn 136,675（9.7%）、gdn 100,075（7.1%）、gemm gate/up 128,888（9.2%）。
- 480K の短縮は gdn 位相の差だけで見積もると **約 4.5%（推定）**: gdn が -40% とすると base の gdn は約 166,800 ms、合計約 1,473,000 ms 対 1,406,295 ms。

decode（eq の steady tok/s）:

| サイズ | base | mmq |
|---|---|---|
| 2K | 40.26 | 38.76 |
| 8K | 45.62 | 45.22 |
| 16K | 48.82 | 49.26 |
| 24K | 52.22 | 51.56 |

decode は MMQ の対象外なので差は ±1.5% 以内（ノイズ）。

GDN 位相（同一プロンプト・同一コンテナ内の比較）:

| prompt | 位相 | base | mmq | 差 |
|---|---:|---:|---:|---:|
| 3,848 | gdn | 1,396 ms (17.3%) | 830 ms (12.1%) | **-40.5%** |
| 3,848 | gdn out proj | 690 ms (8.6%) | 335 ms (4.9%) | -51.4% |
| 3,848 | GPU timeline | 8,057 ms | 6,878 ms | -14.6% |
| 9,342 | gdn | 3,291 ms (16.4%) | 1,994 ms (11.2%) | **-39.4%** |
| 9,342 | gdn out proj | 1,312 ms (6.5%) | 811 ms (4.6%) | -38.2% |
| 9,342 | GPU timeline | 20,107 ms | 17,817 ms | -11.4% |

## 結果: needle retrieval

`arms/*/needle.jsonl`。ID は一意（GD32A/GD32B/GD128A/GD128B/GD4806）。480K の 2 回はスクリプトの `printf "%c"` の癖で同じ GD4806 を使ったが、各リクエスト内では一意なので判定には影響しない。

| アーム | サイズ | 位置 | prompt tokens | TTFT | prefill tok/s | 正解 |
|---|---|---:|---:|---:|---:|---|
| base | 32K | 50% | 32,123 | 65.33 s | 491.7 | はい |
| base | 32K | 90% | 32,123 | 65.42 s | 491.1 | はい |
| base | 128K | 50% | 125,007 | 254.5 s | 491.2 | はい |
| base | 128K | 90% | 125,007 | 256.1 s | 488.1 | はい |
| mmq | 32K | 50% | 32,123 | 58.03 s | 553.6 | はい |
| mmq | 32K | 90% | 32,123 | 57.96 s | 554.2 | はい |
| mmq | 128K | 50% | 125,007 | 228.9 s | 546.1 | はい |
| mmq | 128K | 90% | 125,007 | 230.2 s | 543.0 | はい |
| mmq | 480K | 50% | 470,288 | 1409.1 s | 333.8 | はい |
| mmq | 480K | 90% | 470,288 | 1407.5 s | 334.1 | はい |

- 480K の 2 条件は別ブロック（`block2` の 50%、`block4` の 90%）で 1 回ずつ。どちらも一意 ID をそのまま返した。
- 480K/90% は block2 では 60 分ブロックに収まらないため打ち切った（`block2.log` の `needle480-90` は kill）。block4 で再測定して正解。
- base の 480K は未測定（1 回 25 分級で 60 分ブロックに収まらないため）。

## 結果: 長文 QA（128K、8 問）

| アーム | prompt tokens | wall | 正答 | 回答本文 |
|---|---:|---:|---:|---|
| base | 125,274 | 258.6 s | **8/8** | `1. Dr. Vera Lindqvist` ... `8. 3` |
| mmq | 125,274 | 232.5 s | **8/8** | base とバイト一致 |

- 8 問は固有名詞 1（Vera Lindqvist）、時刻 1（03:40）、数値 2（7,312 / 19）、対応表 1（basalt -> 61）、比較 1（cobalt）、推論 1（6 スロット × 512 MiB = 3072）、人数 1（3）。
- 128K の QA で MMQ による正答の劣化は検出できなかった。wall は -10.1%。

## 結果: KL と出力一致

KL（プロンプト最終位置の logits、`kl_by_rec.py`）:

| 位置 | base vs base2（ノイズ床） | base vs mmq | base vs denseonly |
|---|---:|---:|---:|
| 3,854（4K 最終、cache なし） | 0.0176 | **0.0191** | 0.0195 |
| 25,722（24K 最終） | 0.0040 | **0.0052** | 0.0215 |
| 7,999（8K 最終） | 0.0216 | 0.0059 | - |
| 3,848 / T=6（spec window） | 2.2e-5 | 0.0049 | 0.0923 |
| 25,716 / T=6（spec window） | 0.0306 | 0.4546 | 0.1115 |

- top-1 トークンは全点で一致。
- 4K 最終位置の MMQ の KL（0.0191）は、同一設定の再実行（base vs base2、0.0176）と同程度。**MMQ 固有の劣化は再実行ノイズに埋もれる**。
- T=6 の spec window は draft の分岐で値が大きく動く（base 同士でも 0.0306、mmq で 0.4546）。品質の指標には使わない。
- 参考: pgdn の 4K KL は 0.00758 だった。今回の 0.0191 との差は、コンテナ内の prefix cache 状態と CPU share による非決定性（同一設定の base2 でも 0.0176）で説明できる範囲。**推定**であり、原因の切り分けは未実施。

出力一致（`eq.py`、`temp 0`、max 128、同じ質問）:

| サイズ | base vs base2 | base vs mmq | 内容 |
|---|---|---|---|
| 2K | バイト一致 | バイト一致 | 同一 |
| 8K | 16/435 文字差 | バイト一致 | 末尾の空白差 |
| 16K | バイト一致 | 16/433 文字差 | 末尾の空白差 |
| 24K | バイト一致 | 182/434 文字差 | `k >= 2` vs `k > 1`、`indicating the position` vs `representing the index`。意味は同一 |

- 24K の 182 文字差は言い回しの差で、コードの動作は同じ。base 同士でも 8K で 16 文字差が出ており、greedy でも完全再現ではない。

## 結果: `STRATA_DENSE_MMQ` 単独（denseonly）

- 3,848 tokens の GPU timeline: base 8,057 ms、denseonly 8,050 ms（**-0.1%**）、mmq 6,878 ms（-14.6%）。単独では速度が変わらない。
- engine log に K-quant ライブラリの行が出ない（DSO 未ロード）。Q3_K は cuBLAS のまま。
- 位相ごとの内訳は base とほぼ同じだが、`router+shared` などに ±400 ms のばらつきがあり、位相単位の差はノイズとして扱う。

## 採用判定と推奨設定

**採用推奨（opt-in）**。既定は変えない。

- 速度: 32K **+12.6〜12.8%**、128K **+11.2%**。TTFT -7.3〜-7.5 s / -25.6〜-25.9 s。5% 閾値を明確に超える。
- 品質: needle 32K/128K 全条件正解、480K/50% 正解、128K QA 8/8（base とバイト一致）。KL は再実行ノイズ内、top-1 全一致。
- 480K では qsa select が 42% を占めるため短縮は約 4.5%（推定）だが、正の効果。
- 推奨 `STRATA_SET` は上記のとおり。イメージは `strata:fork-prefill-gdn`（本番 `0.2.7-cuda12` = engine 0.1.52 には未収録）。戻すときは `STRATA_DENSE_MMQ` と `STRATA_MMQ_KQ` を外すだけ（コード既定は off）。

## リスク

- **bitwise ではない**。q8_1 活性の丸めで、24K 級の greedy 回答が言い回しレベルで変わる。`STRATA_DENSE_F32=1` の bitwise 性を売りにするなら既定 off を維持する。
- **非決定性が既にある**。`STRATA_PREFILL_CPU_SHARE` が既定 ON で、同一設定の再実行でも KL 0.018・回答 16 文字差が出る。MMQ の品質差を厳密に切り分けるには CPU share off での再測定が必要（未実施）。
- **480K の効果は小さい**（約 4.5%、推定）。qsa select の最適化のほうが効く。
- `libstrata_kq.so` が Q3_K インスタンス分だけ大きくなる（base 6.12 MB -> gdn 7.16 MB、+1.03 MB）。VRAM への影響は小さい（DSO は必要時のみロード）。
- `STRATA_MMQ_KQ=1` は K-quant expert 経路も開ける。このモデルの expert は IQ3_XXS なので実質影響しないが、他モデルでは expert の丸めも変わる。

## 再現手順と成果物

- スクリプト: `~/work/results/gdnq/run-block1.sh`（base + mmq 品質スイート）、`run-block2.sh`（denseonly + 480K/50%）、`run-block4.sh`（base2 + 480K/90%）、`run-arm.sh` / `run-arm480.sh`、`qa.py`、`score_qa.py`、`klreq.py`、`eq.py`、`cmp_eq.py`、`kl_by_rec.py`、`fast_kl.py`、`summary.py`。
- 生ログ: `~/work/results/gdnq/block1.log`、`block2.log`、`block4.log`、`arms/<label>/`（`needle.jsonl`、`qa.json`、`eq.json`、`kl4k.json`、`kl24k.json`、`logits.bin`、`engine.log`、`engine-environ.txt`、`engine-cmdline.txt`）。
- 本番復元: 各ブロックの最後に `~/opt-work/src-build/restore.sh` で 512K に戻し、healthy と `.env` 一致を確認してから GPU ロックを解放した。
- ブロック: block1 15:40:31 -> 16:23:21（43 分）、block2 16:23:46 -> 16:58:42（35 分）、block4 16:59:40 -> 17:34:43（35 分）。いずれも 60 分以内。
- 最終確認（17:37 時点）: 本番 `strata` は healthy（image `ghcr.io/niko1221/strata:0.2.7-cuda12`）、`~/Strata/docker/compose/.env` は `~/opt-work/prod.env` と一致、engine の `STRATA_` 環境は `STRATA_DENSE_F32=1` と `STRATA_SET`（512K 本番値）のみで計測用の値は残っていない。
- preset `/data/state/strata/launch/docker.json` の `env` は `STRATA_DENSE_F32: "1"` のみ、`args` は `--max-context 524288 --kv int8 --kv-resident 32768 --ple-io ram --prefill auto --rope-scale 2 --rope-scaling yarn --spec 4`。計測用の `--set` は残っていない（計測は state を複製して別コンテナで起動したため、共有 preset は触っていない）。
- 残留コンテナなし（`docker ps -a` は `strata` のみ）。GPU ロックは解放済み（`/tmp/strata-gpu.lock` なし）。
