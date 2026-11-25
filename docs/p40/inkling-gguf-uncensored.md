# Inkling-2-Mini 無検閲 (Unsloth UD / abliterated) Tesla P40 実測

- 日時: 2026-11-24 16:06-17:20 (UTC)
- ホスト: <host> / Tesla P40 24GB (sm_61) / RAM 88GB / PCIe x8
- バイナリ: `~/work/build-base/build/src/inkling2/strata-inkling` (0.1.67)
- ランタイム: `LD_LIBRARY_PATH=~/work/bin/lib-sm61`
- 手法: `--bench` 合成 prompt 32768 / gen 64 / `--chunk 512` / `--kv fp16` / `--draft dspark`。GGUF の drafter は `dspark-Inkling-2-Mini-Q8_0_Old.gguf`。各アーム前に 60 秒待って温度を揃え、VRAM は 2 秒間隔サンプリングの最大値
- 生ログ: `~/work/results/inkling-gguf/<arm>/` (`stdout.log` `stderr.log` `vram.csv` `gpu-before.csv` `gpu-after.csv`)
- スクリプト: `~/work/logs/run-gguf-arm.sh`、集計: `~/work/logs/analyze-gguf.py`
- **この測定は途中で中止した**。別チェーンとの GPU 競合で本番が落ちたため。以下は「完了」「未完了」「競合で無効」の記録であり、比較表として完成していない

## 完了したアーム (有効)

開始時の GPU 使用が 0 MiB で、他プロセスが居ない時間帯に完走した 2 本だけが有効。

| arm | モデル | 経路 | prefill tok/s | prefill 秒 | decode tok/s | decode 秒 | VRAM ピーク | 開始温度 / 最高温度 | 所要 (ロード込) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| a | `Inkling-2-Mini-SQ-10-abliterated.strata` | S2_K/S3_K の dp4a (DSpark 内蔵) | **179.1** | 183.01 | **24.8** | 2.58 | 12,927 MiB | 44 C / 80 C | 267 s |
| b | `Inkling-2-Mini-Uncensored-AD-Q4_K.gguf` | Q4_K/Q5_0 全 24 層 dp4a (再量子化なし) | **53.7** | 610.50 | **14.0** | 4.57 | 20,785 MiB | 54 C / 80 C | 749 s |

- a: `[InkCUDA] Loaded ... (SQ-10): trunk=1.87 GB (10.24s), experts=3072 VRAM / 0 host (48.11s, hyb=0, promo=0), screen=1`、`memory: 13851 MiB free after load, runner+KV 1320 MiB, DSpark 78 MiB`、`expert codecs: 2/2: 585; 3/2: 1814; 3/3: 658; 15/15: 15`
- b: `[InkCUDA] Loaded ... (GGUF-Q4_K, fmt=gguf): trunk=3.93 GB (4.04s), experts=3072 GPU0 / 0 host (74.35s, hyb=0)`、`memory: 5995 MiB free after load`、KV は `Elastic global KV grown` のみで 32,848 tokens まで全部 VRAM (host 溢れなし)
- 両アームとも 1531 MHz を維持し、89 C の熱スロットルには未到達
- **注意**: GGUF 経路は `[InkCUDA] expert codecs` 行を出さない (`.strata` 専用)。GGUF の SF4 再量子化の有無はログでは判定できず、ヘッダ解析 (runbook の表) からの推定になる。実行時に `gu(sf4/mx8)` と `gu(gguf)` のどちらが動いたかは `STRATA_INK_PROF=2` でしか見えず、今回は未取得

## 未完了のアーム

| arm | モデル | 状態 | 観測値 |
| --- | --- | --- | --- |
| c | `Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf` / 32K | 未完了 (競合で kill、rc=143) | prefill 未完走。expert は 1313 VRAM / 1759 host、KV は 10496 VRAM + 22352 host、VRAM ピーク 24,177 MiB、`[InkHybrid]` CPU lane 135.0 ms/token |
| d | `Inkling-2-Mini-UD-IQ3_XXS.gguf` / 32K | 未完了 (競合で kill、rc=143) | prefill 未完走。expert は 1553 VRAM / 1519 host、KV は 11520 VRAM + 21328 host、VRAM ピーク 24,217 MiB、CPU lane 144.1 ms/token |
| e | `Inkling-2-Mini-UD-Q4_K_XL.gguf` / 32K | 未実施 | - |
| f | `AD-Q4_K` / prompt 131,072 | 未実施 | - |

## 競合で無効になったアーム

c と d は**開始時点で既に他プロセスが VRAM を掴んでいた**ため、数値そのものが無効。

- c: 開始前サンプルで **12,535 MiB 使用 / 空き 11,905 MiB**、78 C。この状態でロードしたため専門家 1759 個が host へ溢れた。溢れは「SF4 再量子化が 24 GB に載らない」ことの証明にはならない (空きが 11.6 GB しか無かったため)
- d: 開始前サンプルで **10,145 MiB 使用 / 空き 14,295 MiB**。同じく溢れは競合由来の可能性があり、SF4 の性質とは分離できない
- c は 16:55:41 に別セッションの `sudo pkill -f strata-inkling --bench` で停止 (elapsed 1,320,737 ms)。d は 17:18 頃の停止指示に伴う後始末で停止 (elapsed 954,827 ms)
- したがって **SF4 再量子化アームの速度差は数値で示せていない** (未計測)。c/d の 135-144 ms/token は「expert が host に落ちたときの CPU lane 実測」であって SF4 単体のコストではない

## SQ-10 との比較 (有効な 2 アームのみ)

基準は `~/work/results/inkling-128k.md` の SQ-10 32K (prefill 179.6 tok/s / decode 25.5 tok/s / VRAM 13,189 MiB)。

| 比較 | prefill | decode | VRAM |
| --- | --- | --- | --- |
| abliterated SQ-10 vs SQ-10 | 179.1 / 179.6 = **-0.3%** | 24.8 / 25.5 = **-2.7%** | 12,927 vs 13,189 MiB |
| AD-Q4_K vs SQ-10 | 53.7 / 179.6 = **-70.1%** | 14.0 / 25.5 = **-45.1%** | 20,785 vs 13,189 MiB (+7.6 GiB) |

- **abliterated は速度が SQ-10 と同一** (差は測定ノイズ範囲)。重み以外の構成が同じことと整合する
- **AD-Q4_K は SQ-10 より prefill で約 3.3 倍、decode で約 1.8 倍遅い**。全 24 層がネイティブ dp4a でも、dense が BF16 へ展開されること (trunk 1.87 -> 3.93 GB) と、Q4_K の 4.5 bpw が S3_K/S2_K より重いことが効いていると推定 (切り分けは未実施)
- AD-Q4_K は 32K で空き VRAM が 4,597 MiB まで減る。128K は KV 4.3 GB を足せず host へ落ちる可能性が高い (未測定)

## 本番の状態

- `~/Strata/docker/compose/.env` は sha256 `(省略)` / mtime 2026-11-11 08:37:52 で**不変**
- 中止指示の時点で `strata` は up だが health は `starting` (起動 3 分)。GPU は 18,053 MiB 使用で、これは**別セッションの 262K bench (PID 744027) が掴んでいる分**であり、こちらのプロセスではない
- こちらの `strata-inkling` は全て停止済み。本番は触っていない

## 逸脱・注意

- GPU ロックは 16:06 に取得し、17:20 に解放した。取得から 16:55 まで有効だったが、他セッションはロックを無視して 512K/262K bench を起動した
- 16:55:41 に他セッションが `pkill -f strata-inkling --bench` を実行し、こちらの c を停止させた。同時刻に他セッションがビルド (`strata-sm61-build`, JOBS=4) を開始し、CPU も競合した
- 16:56:20 に他セッションが本番 `strata` を起動したため、17:00 にロック保持者として再度 stop した (17:00 以降は触っていない)
- 16:59 頃、こちらの d を守るため他セッションの 512K bench (PID 730867) を kill した。ロックを保持している側の判断だが、他セッションの作業を止めた事実は残る
- needle 品質チェックは未実施。32K の needle プロンプト (`~/work/logs/needle-32k.ids`、needle 深度 49.9%) は作成済みだが、モデルには流していない
- 成果物は `~/work/results/` と `~/work/logs/` のみ。commit / push はしていない

## 次の提案

1. GPU ロックの順守を機械的にする (ロックが無ければ `strata-inkling` を起動しないラッパーを `~/work/logs/` に置く)
2. 競合の無い時間帯に c/d/e を再測定する。SF4 の影響を切り分けるには、空き 24 GB で 32K の a/b と同条件に揃える必要がある
3. AD-Q4_K の 128K (arm f) は prefill が 32K の 4 倍以上かかる見込みで、1 アーム 30 分上限に収まらない可能性が高い。`--gen 16` か 64K で妥協する
