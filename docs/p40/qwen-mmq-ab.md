# Qwen3.8 IQ3_XXS 128K — MMQ 復活 (perf/mmq-revival 段 1) の A/B

- 日時: 2026-11-24 21:11-21:46 (UTC)
- ホスト: <host> / Tesla P40 24GB (sm_61) / RAM 88GB
- 画像: `strata:up-mmq` (engine `0.1.67-linux-x86_64-cuda-mmq`、`perf/mmq-revival` d2c77f1e5、BUILD.json の `source` は `perf/mmq-revival d2c77f1e5 (MMQ revival), sm_61`、component の `variant` は `mmq`、ビルドは `STRATA_MMQ=ON` / `STRATA_MMQ_KQUANTS=OFF` / `STRATA_EXPERIMENTAL_SM60=ON` / `CMAKE_CUDA_ARCHITECTURES=61-real`)
- 比較基準: `~/work/results/qwen-baseline.md` の ref (`strata:fork-v9`, engine 0.1.57) と up-sw-min (`strata:up-0.1.67`, engine 0.1.67 の最良)
- ハーネス: `~/work/bin/run-arm.sh` (TAG=mmqab, CPUSET=36-47, TOKENS=128000, PORT=18080)、位相は `STRATA_PREFILL_TIMING=1`
- 測定: 各アームの前に 60 秒待って温度を揃えた。本番 (`strata`) は各アームの前後に compose で停止・復旧し、3 アームすべて `prod_health=healthy` / `env_same=yes`
- 要求 128,000 tokens に対しエンジンログの実測は全アーム **107,460 tokens** (serve 行は 107,465)。同一プロンプトなので位相比較は直接可能

## STRATA_SET (upstream 最良アーム = baseline の up-sw-min と同一)

```
ple-io=ram prefill=32768 kv-resident=32768 draft-conf=0.7 STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 spec=4
STRATA_DENSE_MMQ=0 STRATA_MMQ_KQ=0 STRATA_HC_DOWN_BF16=0 STRATA_HC_INJ8=0 STRATA_HC_MIX16=0 STRATA_HC_MIXINJ=0
STRATA_MMQ_SQ=0 STRATA_PATTN_SM60=1 STRATA_SELECT_SIMT=1 STRATA_SCORES_THREAD=1
```

| アーム | block | 画像 (engine) | 追加スイッチ | 実効 env (arm-config の env) |
| --- | --- | --- | --- | --- |
| A | `armA` | up-mmq (0.1.67 mmq) | なし (`STRATA_PREFILL_MMQ` 未設定 = 既定 ON) | 上記 11 キーそのまま (`STRATA_PREFILL_MMQ` は env に載らない) |
| B | `armB` | up-mmq (0.1.67 mmq) | `STRATA_DENSE_MMQ=1` | 上記の DENSE_MMQ だけ 1 |
| Aoff (追加・対照) | `armAoff` | up-mmq (0.1.67 mmq) | `STRATA_PREFILL_MMQ=0` | 上記 + `STRATA_PREFILL_MMQ=0` |

- `docker.choices.json` の `extra` は消えないが、今回は上記 SET に本番 10 キーをすべて含めているので、`arm-config-*.json` の `env` は仕様どおり (本番値の漏れなし)。A / B の env に `STRATA_PREFILL_MMQ` は入っていない (= 未設定)
- Aoff は合格条件のアームではなく、`STRATA_PREFILL_MMQ` が実際に効いていることを示すための対照 (同じ画像・同じ SET でスイッチだけ 0)

## 結果サマリ

prefill tok/s (timeline) = 107,460 / GPU timeline、prefill tok/s (TTFT) = 107,465 / TTFT。

| arm | TTFT s | GPU timeline ms | prefill tok/s (timeline) | prefill tok/s (TTFT) | decode tok/s (bench / engine) | healthy まで s | VRAM 定常 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ref (fork-v9) | 143.681 | 139,986 | 767.6 | 747.9 | 48.45 / 47.8 | 172 | 23,726 |
| up-sw-min (0.1.67) | 207.651 | 204,749 | 524.8 | 517.5 | 28.15 / 28.1 | 168 | 23,726 |
| **A (up-mmq 既定)** | 193.931 | 190,411 | **564.4** | **554.1** | 30.88 / 30.8 | 182 | 23,654 |
| **B (+DENSE_MMQ=1)** | 183.531 | 180,206 | **596.3** | **585.5** | 29.65 / 29.6 | 172 | 23,654 |
| Aoff (MMQ=0, 対照) | 233.768 | 230,370 | 466.5 | 459.7 | 30.54 / 30.4 | 166 | 23,654 |

- VRAM は engine log の「N MiB of VRAM free with everything loaded」から 24,576 - 922 = 23,654 MiB (定常)。ピークは未取得 (2 秒サンプラを回していない)
- 温度/クロック: 開始 55-59 C、終了 69-70 C、SM 1531 MHz 固定、`clocks_throttle_reasons.active` は 3 アームすべて `0x0`

### 基準との差

| arm | timeline 対 ref | tok/s 対 ref | timeline 対 up-sw-min | tok/s 対 up-sw-min |
| --- | ---: | ---: | ---: | ---: |
| up-sw-min | +64,763 ms (+46.26%) | -31.6% | ±0 | ±0 |
| A | +50,425 ms (+36.02%) | **-26.5%** | -14,338 ms (-7.00%) | **+7.5%** |
| B | +40,220 ms (+28.73%) | **-22.3%** | -24,543 ms (-11.99%) | **+13.6%** |
| Aoff | +90,384 ms (+64.57%) | -39.2% | +25,621 ms (+12.51%) | -11.1% |

- up-sw-min と ref の差 64,763 ms のうち、A が 14,338 ms (22%)、B が 24,543 ms (38%) を戻した。**fork-v9 には届かない**
- decode は A/B/Aoff で 29.6-30.9 tok/s。3 アームとも fork-v9 の 47.8-48.5 には戻っていない (約 -36%)。Aoff は draft 受理 47/47 で上振れしており、±10% のばらつきを踏まえても「戻っていない」の判定は変わらない

## 位相別 (phaseDiff.py, 107,460 tokens どうし)

### A -> B (+DENSE_MMQ=1)

timeline -10,205 ms (-5.36%)

| phase | A ms | B ms | 差 | % |
| --- | ---: | ---: | ---: | ---: |
| gdn | 36,950 | 30,961 | -5,989 | -16.2% |
| qsa proj | 15,905 | 14,304 | -1,601 | -10.1% |
| gdn out proj | 14,425 | 13,079 | -1,346 | -9.3% |
| router+shared | 8,676 | 8,269 | -407 | -4.7% |
| gemm gate/up | 25,153 | 24,854 | -299 | -1.2% |
| gemm down | 15,083 | 14,916 | -167 | -1.1% |
| qsa attn | 19,927 | 19,940 | +13 | +0.1% |
| hc read: down+silu | 12,622 | 12,701 | +79 | +0.6% |

dense MMQ は expert ではなく dense 側 (gdn / qsa proj / gdn out proj) に効いている。

### ref (fork-v9) と B

timeline +40,220 ms (+28.73%)。残差の内訳:

| 群 | ref ms | B ms | 差 |
| --- | ---: | ---: | ---: |
| GDN (gdn + conv+gates + recurrence + out proj) | 36,542 | 51,455 | +14,913 |
| GEMM (gate/up + down) | 30,978 | 39,770 | +8,792 |
| hc read (down+silu + up + inject + mix + hc read) | 20,452 | 27,623 | +7,171 |
| QSA (proj + select + attn + indexer + indexer gemms + norm) | 33,299 | 39,852 | +6,553 |
| その他 (router+shared, hc write, embed+steps ほか) | - | - | +2,792 |

### gate-up と GDN (各アーム)

| phase | ref | up-sw-min | A | B | Aoff |
| --- | ---: | ---: | ---: | ---: | ---: |
| gemm gate/up | 20,259 | 58,755 | 25,153 | 24,854 | 46,235 |
| gemm down | 10,719 | 0 | 15,083 | 14,916 | 23,177 |
| dequant | 1,591 | 157 | 1,103 | 1,099 | 9,088 |
| gdn | 20,751 | 36,347 | 36,950 | 30,961 | 38,277 |
| gdn out proj | 8,262 | 13,450 | 14,425 | 13,079 | 14,633 |
| gdn conv+gates + recurrence | 7,529 | 7,413 | 7,420 | 7,415 | 7,411 |
| GDN 群 計 | 36,542 | 57,210 | 58,795 | 51,455 | 60,321 |
| GEMM 群 計 (gate/up + down) | 30,978 | 58,755 | 40,236 | 39,770 | 69,412 |

- up-sw-min は `gemm down` の位相行を出さない (0.1.67 の fused 経路では down が別の位相に畳まれる)。比較は GEMM 群 計で見る
- **expert の gate-up は A で 25,153 ms (up-sw-min の 58,755 ms から -57%)**。MMQ 段 1 の狙いどおり、dequant + FP16 GEMM が消えて int8 (Pascal は dp4a) になっている
- ただし fork-v9 の GEMM 群 30,978 ms には 8,792 ms 届かない (fork-v9 の `gemm gate/up` 20,259 + `gemm down` 10,719 に対し A/B は 24,854-25,153 + 14,916-15,083)

## MMQ が実際に有効か (起動ログでの確認)

| 確認項目 | 結果 | 根拠 |
| --- | --- | --- |
| どのエンジンが起動したか | `0.1.67-linux-x86_64-cuda-mmq` | 各アームの engine log の `engine started: /opt/strata/components/engine/0.1.67-linux-x86_64-cuda-mmq/strata` |
| そのビルドに MMQ が入っているか | 入っている (`STRATA_MMQ=ON`, `STRATA_MMQ_KQUANTS=OFF`, sm_61) | 画像内 `BUILD.json` (`source` は `perf/mmq-revival d2c77f1e5`)、`engine-component.json` の `variant: mmq`、`~/work/mmq/build/CMakeCache.txt` |
| 実行時に有効か (既定) | 有効 | `STRATA_PREFILL_MMQ` 未設定なので `prefill.cpp` の `mmq_plan()` は `env == nullptr` で on。P40 は cc 6.1 なので P100 (cc 6.0) の FP16 既定には当たらない |
| IQ3_XXS の tile があるか | ある | `mmq-config-pascal-dp4a.cuh` に IQ3_XXS の CASE があり、起動ログに `strata: prompt kernels: llama.cpp` の「no tile」行が出ていない (この行は tile が無い型でだけ出る) |
| スイッチが反映されているか | 反映されている | 同画像・同 SET で `STRATA_PREFILL_MMQ=0` (Aoff) にすると `dequant` 1,103 -> 9,088 ms、`gemm gate/up` 25,153 -> 46,235 ms、timeline 190,411 -> 230,370 ms (+21.0%)。MMQ を切ると dequant + FP16 GEMM に戻る |
| 明示の "MMQ on" ack 行 | **無い** | このビルドは有効時に何も出さない (出るのは「tile が無い」「DIRECT / GROUP_SORT 等のスイッチ」「MMQ 無しビルド」の行だけ)。有効性の判定は上記の挙動差と tile 行の不在で行った |

- `STRATA_DENSE_MMQ=1` の B では起動ログに `prefill gemm: dense MMQ on (STRATA_DENSE_MMQ=1)` が出る (A / Aoff では出ない)
- A / B とも「STRATA_DENSE_MMQ has no effect in this engine (it has no MMQ kernels)」のような「MMQ 無し」通知は出ていない

## 注意・未取得

- **up-mmq を MMQ=0 で使うと 0.1.67 より遅い** (Aoff 466.5 tok/s 対 up-sw-min 524.8、+12.5%)。0.1.67 は MMQ 無しビルドなので Strata 自前の fused int8 経路が既定で、up-mmq の MMQ=0 は dequant + FP16 GEMM に落ちるため。**MMQ を切るなら 0.1.67 のままのほうが速い**
- 未取得: VRAM ピーク (2 秒サンプリングなし)、A / B の 137,344 tokens での実測 (128,000 要求で 107,460 tokens になった。baseline と同じ条件なので相対比較は成立)
- 未取得: MMQ 段 1 の品質ゲート (`gemm_f32_parity` など) は本 A/B では回していない。今回の確認は速度とログのみ
- decode は 64 トークン 1 回勝負で ±10% 程度ばらつく。A 30.88 / B 29.65 / Aoff 30.54 の差は有意とみなさない
- RAM: 全アームで expert arena (39.97 GiB 要求) が確保できず `expert arena read through the file cache` に落ちた (開始時 MemAvailable 約 12 GiB、hugepages 予約で 40 GB)。baseline と同じ条件なので相対比較は成立するが、絶対値はこの機体の最良値ではない可能性がある
- 本番 `.env`: 不変 (sha256 `(省略)`、3 アームすべて `env_same=yes`)。本番は最終確認で healthy、GPU ロックなし
- `~/work/results/qwen-mmq-ab.md` は作っていない (本ファイルは下書き。共有台帳の更新はメインが行う)

## 成果物

`~/work/results/mmqab/`

- `block-arm{A,B,Aoff}.log` / `arm-arm{A,B,Aoff}.txt` / `prod-restore-arm{A,B,Aoff}.txt`
- `bench-arm{A,B,Aoff}.jsonl` (prefill 1 行 + decode 1 行)
- `engine-arm{A,B,Aoff}.log` (`strata prefill timing` の位相行)
- `arm-config-arm{A,B,Aoff}.json` (実際に解決された args と env)
- `docker-arm{A,B,Aoff}.log` / `docker-run-arm{A,B,Aoff}.log`
- `state-arm{A,B,Aoff}/` / `pristine/` / `env-backup-arm{A,B,Aoff}`

再計算: `python3 ~/work/bin/phaseDiff.py ~/work/results/mmqab/engine-armA.log ~/work/results/mmqab/engine-armB.log 107460 107460`
