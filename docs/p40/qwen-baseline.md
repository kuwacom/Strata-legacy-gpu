# Qwen3.8 IQ3_XXS 128K prefill / decode — fork-v9 (engine 0.1.57) と upstream 0.2.21 (engine 0.1.67) の実測

- 日時: 2026-11-24 11:04-11:59 (UTC)
- ホスト: <host> / Tesla P40 24GB (sm_61) / RAM 88GB / PCIe x8
- 画像: `strata:fork-v9` (現行本番, engine 0.1.57) / `strata:up-0.1.67` (engine 0.1.67)
- ハーネス: `~/work/bin/run-arm.sh` (TAG=baseline, CPUSET=36-47, TOKENS=128000, PORT=18080)
- 測定: `bench-api.py` の SSE (prefill = `--max-tokens 1` の TTFT、decode = `--tokens 64 --max-tokens 64`)、位相は `STRATA_PREFILL_TIMING=1`
- 本番 (`strata`, `~/Strata/docker/compose`) は各アームの前後に停止・復旧。5 アームすべて復旧は `prod_health=healthy` / `env_same=yes`
- 温度/クロック: 開始 55-62 C、終了 68-69 C、SM 1531 MHz 固定、`clocks_throttle_reasons.active` は全アーム `0x0`

要求 128,000 tokens に対し、エンジンログの実測は全アーム **107,460 tokens** (serve 行は 107,465)。トークン数が揃っているので位相比較は直接可能。

## 結果サマリ

| arm | block | 画像 (engine) | 有効だった本番スイッチ | TTFT s | prefill tok/s (timeline) | prefill tok/s (TTFT) | GPU timeline ms | decode tok/s (bench / engine) | healthy まで s | VRAM ピーク MiB |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ref | `ref` | fork-v9 (0.1.57) | 10/10 (仕様どおり) | 143.681 | 767.6 | 747.9 | 139,986 | 48.45 / 47.8 | 172 | 23,726 (定常値, ピーク未取得) |
| up-core | `up-core` | up-0.1.67 | 10/10 (仕様は 1/10 の意図) | 209.260 | 520.8 | 513.6 | 206,349 | 30.92 / 30.8 | 168 | 23,875 |
| up-sw | `up-sw` | up-0.1.67 | 11/11 (追加 3 のうち実質 1) | 209.866 | 519.3 | 512.1 | 206,935 | 30.45 / 30.4 | 172 | 23,875 |
| up-core-min | `up-core-min` (追加) | up-0.1.67 | 0/10 (仕様どおり) | 244.563 | 444.6 | 439.4 | 241,692 | 31.07 / 31.0 | 167 | 23,875 |
| up-sw-min | `up-sw-min` (追加) | up-0.1.67 | 3/3 (仕様どおり) | 207.651 | 524.8 | 517.5 | 204,749 | 28.15 / 28.1 | 168 | 23,875 |

- prefill tok/s (timeline) = 107,460 / GPU timeline、TTFT 版 = 107,465 / TTFT
- decode は 64 トークンの 1 回勝負でばらつく (±10% 程度)。upstream は 28.1-31.1、fork-v9 は 47.8-48.5
- VRAM ピークは 2 秒間隔の `nvidia-smi` サンプリング (upstream 4 アーム)。ref だけサンプラを回しておらず、engine log の「850 MiB of VRAM free with everything loaded」から 24,576-850 = 23,726 MiB (定常) を算出。実ピークは未取得

## fork-v9 の既知値との比較

既知値: 128K = 137,344 tokens / timeline 178,582 ms / 758 tok/s / TTFT 181.2 s

| 対象 | トークン基準 | 実測 | 既知値 | 差 |
| --- | --- | ---: | ---: | ---: |
| ref (今日) timeline | 107,460 | 767.6 tok/s | 769.1 tok/s (137,344 / 178.582) | -0.19% |
| ref (今日) TTFT | 107,465 | 747.9 tok/s | 758.0 tok/s (137,344 / 181.2) | -1.33% |
| ref (今日) の 137,344 換算 | - | 178,914 ms (推定) | 178,582 ms | +0.19% |

- **今日の ref は fork-v9 の既知値を 0.2% (timeline 基準) で再現した**。ハーネスと今日のマシン状態は既知値と同条件とみなせる
- 既知値の「758 tok/s」は TTFT 基準、「769 tok/s」は timeline 基準。混在しないよう本表は両方併記した
- トークン数が違う (107,460 vs 137,344) ため、以下は tok/s で正規化し、137,344 tokens 換算は「一定 tok/s と仮定した推定」として併記する

| arm | timeline 対 ref | tok/s 対 ref | 137,344 tokens 換算 (推定) | 既知 178,582 ms との差 |
| --- | ---: | ---: | ---: | ---: |
| ref | ±0 | 767.6 | 178,914 ms | +332 ms (+0.19%) |
| up-core | +66,363 ms (+47.41%) | 520.8 (-32.2%) | 263,742 ms | +85,160 ms (+47.7%) |
| up-sw | +66,949 ms (+47.83%) | 519.3 (-32.4%) | 264,483 ms | +85,901 ms (+48.1%) |
| up-core-min | +101,706 ms (+72.65%) | 444.6 (-42.1%) | 308,897 ms | +130,315 ms (+73.0%) |
| up-sw-min | +64,763 ms (+46.26%) | 524.8 (-31.6%) | 261,688 ms | +83,106 ms (+46.5%) |

**結論: 128K prefill は upstream 0.1.67 のほうが遅い。** 最良の upstream 構成 (up-sw-min) でも 524.8 tok/s で fork-v9 の 767.6 tok/s に対し **-31.6%**、仕様どおりの最小構成 (up-core-min) では 444.6 tok/s で **-42.1%**。128K 1 回あたり 65-102 秒余計にかかる。

decode も fork-v9 が速い (47.8-48.5 対 28.1-31.1 tok/s、約 -36%)。draft の受理率は両者ほぼ同じ (ref 46/48、up 47/49) なので、draft ではなく 1 step の素の速度差。

## 有効だった env (重要: 仕様どおりに走っていない)

`run-arm.sh` は `state/strata/launch/docker.json` の `env` を空にするが、**`state/strata/launch/docker.choices.json` の `extra` に残った本番 SET が起動時に再適用される**。このため `SET` を絞っても本番の 10 スイッチが効いたままになる (RUNBOOK の「choices は表示用、起動時には読まれない」と矛盾。0.2.21 の挙動と思われるが、機構の断定は未確認)。

確認方法と証拠:

- `arm-config-up-core.json` の `env` が本番 10 キーそのもの (up-sw は + SCORES_THREAD の 11 キー)
- up-core の engine log に `STRATA_PATTN_SM60=1: the prompt path's QSA attention on the sm_6x ...` の ack が出ている (SET に入れていない)
- up-core の engine log に `strata: STRATA_DENSE_MMQ has no effect in this engine (it has no MMQ kernels)` が出ている (同上)
- `state-up-core/.../docker.choices.json` の `extra` に 10 キーが残存

| arm | 仕様の SET | 実際に効いた env |
| --- | --- | --- |
| ref | 本番 SET (10) | 本番 SET (10)。差なし |
| up-core | 8 項目 (env は DENSE_F32 のみ) | 本番 SET (10) + DENSE_F32 |
| up-sw | up-core + PATTN_SM60, SELECT_SIMT, SCORES_THREAD | 本番 SET (10) + SCORES_THREAD。PATTN/SIMT は元から ON |
| up-core-min (追加) | up-core の 8 項目 + 本番 9 キーを明示 `=0` | 意図どおり (DENSE_F32 のみ 1) |
| up-sw-min (追加) | up-core-min + PATTN_SM60=1, SELECT_SIMT=1, SCORES_THREAD=1 | 意図どおり |

- 追加 2 アームは、仕様の up-core / up-sw が測れていないことを確認したため、`STRATA_X=0` の明示上書きで意図した env を再現したもの。up-core-min では engine の PATTN ack が消え、`DENSE_MMQ has no effect` の行も出ないことを確認済み
- したがって **仕様の up-core / up-sw の数値は up-core-min / up-sw-min**、**本番 env を引き継いだままの比較は up-core / up-sw** として読む

## 位相比較 (phaseDiff.py)

### ref と up-core (本番 env 同士、107,460 tokens どうし)

timeline +66,363 ms (+47.41%)

| phase | ref ms | up-core ms | 差 | % |
| --- | ---: | ---: | ---: | ---: |
| gemm gate/up | 20,259 | 58,760 | +38,501 | +190.0% |
| gdn | 20,751 | 37,189 | +16,438 | +79.2% |
| qsa proj | 9,816 | 15,677 | +5,861 | +59.7% |
| gdn out proj | 8,262 | 13,560 | +5,298 | +64.1% |
| hc read: mix | 1 | 3,547 | +3,546 | 新規 |
| hc read: down+silu | 9,897 | 12,874 | +2,977 | +30.1% |
| qsa attn | 17,803 | 19,983 | +2,180 | +12.2% |
| router+shared | 6,469 | 8,645 | +2,176 | +33.6% |
| hc write | 3,991 | 6,040 | +2,049 | +51.3% |
| gemm down | 10,719 | 0 | -10,719 | 位相名が消えた |
| dequant | 1,591 | 163 | -1,428 | -89.8% |
| gather | 934 | 277 | -657 | -70.3% |

位相名が版で変わるため、まとめて比べる:

| 群 | ref ms | up-core ms | 差 |
| --- | ---: | ---: | ---: |
| GEMM (gate/up + down) | 30,978 | 58,760 | +89.7% |
| GDN (gdn + conv+gates + recurrence + out proj) | 36,542 | 58,158 | +59.1% |
| QSA (proj + select + attn + indexer + indexer gemms + norm) | 33,299 | 41,277 | +24.0% |
| hc read (down+silu + up + inject + mix + hc read) | 20,452 | 27,886 | +36.4% |

- 最大の寄与は GEMM の +27.8 s。engine 自身が「この engine に MMQ カーネルは無い」と出しており、fork-v9 の dense MMQ (int8 dp4a) 経路が 0.1.67 に存在しないことと整合する (dequant が -1.4 s、gemm が +38.5 s という入れ替わりがその形)
- GDN が +21.6 s と大きい。スイッチ差では説明できず (本番 SET は GDN 系を含まない)、0.1.57 -> 0.1.67 のコード差とみられるが内訳は未確認

### up-core-min と up-sw-min (最小構成 + 3 スイッチ、107,460 tokens どうし)

timeline -36,943 ms (-15.29%)

| phase | up-core-min ms | up-sw-min ms | 差 | % |
| --- | ---: | ---: | ---: | ---: |
| qsa select | 30,661 | 4,922 | -25,739 | -83.9% |
| qsa attn | 30,844 | 19,973 | -10,871 | -35.3% |
| gemm gate/up | 58,777 | 58,755 | -22 | -0.0% |
| gdn | 36,432 | 36,347 | -85 | -0.2% |
| その他 | - | - | 合計 -226 | - |

差は qsa select と qsa attn にほぼ全部入る。他は動いていない。

### up-core と up-sw (仕様どおりの比較になっていない)

timeline +586 ms (+0.28%)。位相はすべて ±1% 以内 (最大は host grouping -13% = -124 ms)。再現性 ±0.25% と同程度で、**差なし**。

## up-sw の 3 スイッチは効いたか

| スイッチ | 効いたか | 根拠 |
| --- | --- | --- |
| `STRATA_PATTN_SM60=1` | 効く (ack あり)。ただし up-core / up-sw では元から ON だったため、up-sw の追加分としては効いていない | engine log の ack 行。ON/OFF の比較は up-core-min と up-sw-min の qsa attn (-10,871 ms, -35.3%) |
| `STRATA_SELECT_SIMT=1` | 効く。同じく up-core / up-sw では元から ON | qsa select -25,739 ms (-83.9%)。`qsa_select.hpp` の「sm_80 未満では SIMT=1 のときだけ FP32 タイルカーネル」と整合 |
| `STRATA_SCORES_THREAD=1` | 効かない | up-sw と up-core の差 +0.28% (ノイズ)。engine log に ack 行なし |

- 3 スイッチ全体 (up-core-min -> up-sw-min): **-36,943 ms (-15.3%)、tok/s +18.0%**。符号はプラス (速くなる)、大きさは timeline の 15%
- ただしこれは 3 個まとめた値。PATTN_SM60 と SELECT_SIMT の個別寄与は未取得 (1 個ずつ切ったアームが無い)
- 注意: upstream の `tune_knobs.py` は SELECT_SIMT を「gain is below the 3% margin」で never-pick にしているが、P40 の 128K prefill では -84% (select) と大きく出た。他機種の測定をそのまま P40 に当てはめると損をする

## 未取得・注意

- **未取得**: ref の VRAM ピーク (2 秒サンプリング)。定常値 23,726 MiB のみ
- **未取得**: upstream の 137,344 tokens での実測。128,000 要求で 107,460 tokens になったため、tok/s 正規化と線形換算で比較した
- **未取得**: PATTN_SM60 と SELECT_SIMT の個別寄与、SCORES_THREAD の単独寄与 (後者は「なし」と判断)
- **RAM 制約**: 全アームで expert arena (39.97 GiB 要求) が確保できず `expert arena read through the file cache` に落ちた。開始時の MemAvailable は約 12 GiB で、40 GB が hugepages (HugePages_Total 20,600) として予約済みのため。ref も同じ条件なので相対比較は成立するが、絶対値はこの機体の最良値ではない可能性がある
- **本番 .env**: 不変 (sha256 `(省略)`、5 アームすべて `env_same=yes`)
- **本番復旧**: 最終確認で `strata` は healthy、VRAM 23,645 MiB、GPU ロックなし
- **ハーネスの落とし穴**: `docker.choices.json` の `extra` を消さない限り `SET` を絞っても本番スイッチが効く。次回は `SET` に `STRATA_X=0` を明示するか、ハーネス側で `extra` も空にする必要がある

## 成果物

`~/work/results/baseline/`

- `block-<arm>.log` / `arm-<arm>.txt` / `prod-restore-<arm>.txt`
- `bench-<arm>.jsonl` / `engine-<arm>.log` / `docker-<arm>.log` / `docker-run-<arm>.log`
- `arm-config-<arm>.json` (実際に解決された args と env)
- `vram-<arm>.csv` (upstream 4 アームの 2 秒サンプリング)
- `state-<arm>/` / `pristine/` / `env-backup-<arm>`

arm = `ref`, `up-core`, `up-sw`, `up-core-min`, `up-sw-min`
