# Inkling-2-Mini prefill attention (sm_61) 検証手順

- 対象: ブランチ `perf/ink-attn-sm61` / コミット `ba6bfb870` (`engine/src/inkling2/cuda/ink_attn.cuh`, `engine/src/inkling2/cuda/ink_runner.cuh`)
- 変更: sm_61 で既存のタイル型 prefill attention を FP32 (SGEMM) で選ぶ。chunk 256 行ぶんの KV を 1 回の pack で共有するので、KV の再読みが 256 分の 1 になる
- 出典: `~/work/inkling-attn-plan.md` の 3.3 の写し + 測定時に見る数値の整理。計画の行番号は変更前のものなので、実装後は +20 程度ずれる

## 計画の検証方法 (写し)

計測は既存の計測手順に乗せる (`CONTEXT.md` のロック、本番停止、温度 60 秒待ち)。

1. **step 0 (前提の確認、各 3 分)**: 32K を `--kv fp16` と `--kv fp8` で 1 本ずつ。fp8 が 1.5 倍以上速ければ DRAM 説確定。差がなければ 1.5 のレイテンシ説を併記して先へ進む
2. **位相**: `STRATA_INK_PROF=2` で `[PROF PREFILL-M512 ... attn=...]` 行 (`ink_runner.cuh:1915, 1958, 3700-3704`、phase 3 = attention、`:3293-3403` が区間)
   - 合格: 128K の attn 合計が **914 s -> 300 s 以下**、かつ最終 chunk の attn が最初の chunk の 1.5 倍以内 (今は 8K 区間 37 s -> 128K 区間 139 s で 3.8 倍)
   - 参考: レポートの `STRATA_PREFILL_TIMING=1` は現行コードに見当たらない (未確認)。`STRATA_INK_PROF=2` を使う
3. **全体**: 32K 182.4 s -> **140 s 以下**、128K 1,386 s -> **700 s 以下** (目標 650 s)。`--bench` の tok/s で 32K 234 以上、128K 187 以上
4. **品質**: needle 24K/32K/128K 全通過、128K QA 8/8、`--bench` の greedy 出力が現行と 60/64 token 以上一致 (FP32 化で数値は良くなる方向。順序が変わるので bitwise 一致は期待しない)
5. **副作用**: 128K で `Host KV Offload` / `spilled` が出ないこと (VRAM +130 MiB)。decode 側は分岐に入らないので不変 (`:3310` は `mode == 0 && M >= 128`)

## 測定時に見る数値

一次指標は `STRATA_INK_PROF=2` の `attn=`。chunk ごとに 1 行出るので、次の 4 つを拾う。

| 見る値 | どこ | 現行 (計画の実測) | 合格線 |
| --- | --- | ---: | ---: |
| attn 合計 (128K) | `[PROF PREFILL-M512]` の `attn=` を全 chunk で足す | 914 s | 300 s 以下 |
| 最終 chunk / 最初の chunk | 上記の先頭行と末尾行の比 | 3.8 倍 (37 s -> 139 s) | 1.5 倍以内 |
| 全体時間 (128K) | 実行時間 or `--bench` | 1,386 s / 94.6 tok/s | 700 s / 187 tok/s 以上 |
| 全体時間 (32K) | 同上 | 182.4 s / 179.6 tok/s | 140 s / 234 tok/s 以上 |

読み方:

- `attn=` が 3 分の 1 前後まで落ち、最終 chunk が最初の chunk と同程度 -> タイル経路が効いている (KV 再読みが消えた)
- `attn=` が横ばい、または最終 chunk だけ伸びる -> タイル経路に入っていない疑い。まず stderr の `[InkCUDA] tiled prefill attention is off (...)` を探す
- 起動直後に `[InkCUDA] tiled prefill attention is off (workspace not available)` が出ていたら、その run は per-token 経路。作業領域は FP32 で約 156 MiB 必要 (空きが `need + 192 MiB` を切ると落ちる)

## 今回の実装で計画と変わった点

- 作業領域: 114 MiB -> **約 156 MiB** (計画は 240 MiB と見積もっていた)。内訳は q 6 + K/V 48 + S 48 + P 48 + O 6 MiB。FP32 化で増えるのは q / K / V / P の 4 本だけ (S と O は元から FP32)
- 安全弁: `STRATA_INK_PF_TILE=0` で従来の per-token 経路 (`k_ink_flash_gqa_gqa3`) に戻る。このとき作業領域の先行確保もしないので、VRAM は現行と同じ
- フォールバックのログは 1 行だけ (visit ごとに出さない)。理由は `workspace not available` か `STRATA_INK_PF_TILE=0` の 2 種類

## 追加で確認する回帰

- `STRATA_INK_PF_TILE=0` で 32K を 1 本。`attn=` が変更前と同水準 (誤差 ±2% 以内) なら、per-token 経路を壊していない
- sm_60 / sm_75+ は dtype 分岐が従来どおり (FP16) で、`attn=` も従来どおりになるはず。手元に該当カードが無い場合は未確認のまま残す
