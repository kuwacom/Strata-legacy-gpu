# prefill の位相プロファイルと chunk 長（fork-v3 + PATTN）

取得日: 2026-11-09。P40 1 枚、98 GiB RAM。
イメージ `strata:fork-v3`（engine 0.1.57）、512K、`spec=4`、`draft dflash2`。
本番相当の `STRATA_SET`（PATTN と DENSE_MMQ を含む）。

## 1. 位相プロファイル

条件: `STRATA_PREFILL_TIMING=1`、`BENCH_SEED=42`、prose、`max_tokens=1`、1 リクエストを 1 回。
生データ: `~/work/results/prof4/`（`state-arm/.../docker.engine.log` の該当 2 行）。

| | 24K | 128K |
|---|---:|---:|
| prompt トークン | 24,607 | 129,442 |
| GPU timeline | 37,954 ms | 205,401 ms |
| ms / 1k トークン | 1,542 | 1,587 |
| prefill | 626.8-627.4 tok/s | 622.8 tok/s |
| host staging | 228 ms | 1,206 ms |

128K の内訳（ms/1k トークン、括弧は GPU 時間の割合）:

| 位相 | ms/1k | % |
|---|---:|---:|
| gemm gate/up | 240 | 15.2 |
| gdn | 195 | 12.3 |
| qsa attn | 184 | 11.6 |
| gemm down | 125 | 7.9 |
| hc read: down+silu | 119 | 7.5 |
| qsa proj | 93 | 5.9 |
| gdn out proj | 83 | 5.2 |
| qsa select | 78 | 4.9 |
| router+shared | 75 | 4.7 |
| hc read: up | 69 | 4.3 |
| hc write | 64 | 4.0 |
| gdn recurrence | 59 | 3.7 |
| dequant | 46 | 2.9 |
| hc read: inject | 39 | 2.5 |
| hc read: mix | 33 | 2.1 |

- 24K と 128K で ms/1k がほぼ同じ。伸びるのは qsa attn（+5.6%）と qsa select（+12%）だけ
- HC 系の合計は 324 ms/1k（20.4%）、MoE 系は約 500 ms/1k（31%）
- host staging は 1.2 s / 205 s（0.6%）。prefill は GPU が実時間をほぼ占有する

### GEMM 単位の内訳（`STRATA_COLD_TRACE=1`、T=8192 の定常値）

| 呼び出し | 形 | ms |
|---|---|---:|
| hc_attn_down / hc_ffn_down | T=8192 N=320 K=10240 | 10.14 |
| hc_attn_up / hc_ffn_up | T=8192 N=10240 K=320 | 5.82 |
| hc_attn_inject / hc_ffn_inject | T=8192 N=4 K=10240 | 3.35 |
| attn_qkv | T=8192 N=10240 K=2560 | 21-31 |
| attn_gate | T=8192 N=6144 K=2560 | 14 |
| ssm_out | T=8192 N=2560 K=6144 | 15-18 |
| attn_q（QSA） | T=8192 N=12288 K=2560 | 24.6 |
| attn_output（QSA） | T=8192 N=2560 K=6144 | 20.0 |
| ffn_gate_shexp / ffn_up_shexp | T=8192 N=640 K=2560 | 2.5 / 2.7 |
| ffn_down_shexp | T=8192 N=2560 K=640 | 3.5（初回だけ 208） |
| ffn_gate_inp（router） | T=8192 N=512 K=2560 | 3.0 |

- 実寸は hidden D = 10240、hyper-connection の rank = 320、ストリーム数 = 4
- hc down は 5.3 TFLOPS、hc up は 9.1 TFLOPS。down が遅いのは K=10240 の活性変換を含むため
- `ffn_down_shexp` の初回 208 ms と `attn_gate` の初回 87 ms は cuBLAS のアルゴリズム選択コスト。
  定常値とは別物なので、1 本目の計測は捨てるか中央値で扱う
- 24K 未満のプロンプトでは、この初回コストが TTFT に効く

## 2. chunk 長

`--prefill auto` の既定は `prefill_auto_max = 8192`（`src/program/generate.cpp:815`）。
明示指定で 16384 / 24576 / 32768 を試した。`draft-conf` はランチャの連動を止めるため 0.7 を明示。

| chunk | 24K tok/s | 128K tok/s | 24K 差 | 128K 差 |
|---|---:|---:|---:|---:|
| auto (8192) | 626.7 | 622.4 | - | - |
| 16384 | 650.0 | 651.1 | +3.7% | +4.6% |
| 24576 | 648.4 | 659.8 | +3.5% | +6.0% |
| 32768 | 668.2 | 664.2 | +6.6% | +6.7% |

- A 同士（同一構成の再起動比較）のドリフトは -0.1% 以下
- chunk が大きいほど速い。重みの dequant と読み出しがチャンクごとに発生するため、チャンク数を
  減らすほど総仕事量が減る
- 24K プロンプトでは 32768 だと 1 チャンクで収まるので、24576（2 チャンク）より有利
- 32768 のときの prompt borrow は 8,372 スロット（13.53 GiB）。expert cache は 10,522 スロット

## 3. 32768 の副作用

`--prefill 32768` にすると、ランチャが VRAM を空けるために `--kv-resident` を 32768 から
4096 に落とす。decode の 4K/24K 実測では差が出なかった（下の最終検証）が、長文では
KV のストリーミングが増える可能性があるため、本番では `kv-resident=32768` を明示して
両立できることを確認した（expert cache 10,311 スロット、healthy）。

## 4. 最終構成（fork-v4）

```
STRATA_IMAGE=strata:fork-v4
STRATA_SET="ple-io=ram prefill=32768 kv-resident=32768 draft-conf=0.7 \
  STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 \
  STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 spec=4 \
  STRATA_PATTN_SM60=1 STRATA_SELECT_SIMT=1"
```

engine 起動フラグ: `--max-context 524288 --kv int8 --kv-resident 32768 --prefill 32768
--spec 4 --draft dflash2 --draft-conf 0.7`

A-B-A（A1 / B1 / A2、24K x2 + 128K x1、1 本目も含めて中央値）:

| 指標 | A1 | A2 | B1 | B1 vs A |
|---|---:|---:|---:|---:|
| prefill 24K (tok/s) | 628.3 | 625.1 | 671.0 | **+7.1%** |
| prefill 128K (tok/s) | 621.9 | 622.9 | 684.0 | **+9.9%** |
| TTFT 24K (s) | 39.3 | 39.5 | 36.8 | -6.4% |
| TTFT 128K (s) | 220.7 | 220.5 | 200.8 | -9.0% |
| decode 4K code (tok/s) | 47.2 | 47.1 | 49.5 | ノイズ |
| decode 4K prose | 40.6 | 41.8 | 42.5 | ノイズ |
| decode 24K code | 48.6 | 48.0 | 48.4 | ノイズ |
| decode 24K prose | 42.5 | 41.7 | 42.1 | ノイズ |

品質（B1）: needle 32K / 128K とも found（prefill 687.4 / 690.7 tok/s）、128K の 8 問 QA は 8/8。
品質チェックの詳細は `docs/context/` の既存手法と同じ（`needle.py` / `qa.py`）。

本番の 24K needle スモーク: TTFT 33.76 s、prefill 678.3 tok/s、found。

## 5. 未確認・注意

- 480K での再計測は未実施（chunk 32768 で 15 チャンクになる）
- `kv-resident=32768` と `prefill=32768` の同時指定は起動と 24K まで確認。480K の長文 decode は未確認
- chunk 長を変えると数値は bitwise でなくなる（和の順序と GEMM のタイルが変わる）。
  品質は needle と QA で確認した範囲
- `STRATA_SELECT_SIMT=1` 単独の寄与は 16384 の上で +1.7%（128K）。32768 の上では分離していない

