# Inkling-2-Mini SQ-10 / Tesla P40 512K コンテキスト構成計画

- 日時: 2026-11-24
- ホスト: <host> / Tesla P40 24GB (sm_61) / RAM 88GB / PCIe 3.0 x8
- コード: `~/work/build-base` (upstream/main d46f5f268, engine 0.1.67)
- バイナリ: `~/work/build-base/build/src/inkling2/strata-inkling`
- モデル: `<models>/inkling2/Inkling-2-Mini-SQ-10.strata` (10,471,526,400 B)

## 結論

512K を通すコマンドはこれ 1 つ。SQ-10、1 slot、KV は k8v4、DSpark は切る。

```sh
cd ~/work/build-base && \
LD_LIBRARY_PATH=~/work/bin/lib-sm61 CUDA_VISIBLE_DEVICES=0 \
./build/src/inkling2/strata-inkling --bench \
  --native <models>/inkling2/Inkling-2-Mini-SQ-10.strata --gpu 0 \
  --max-context 524288 --parallel 1 --chunk 512 \
  --kv k8v4 --kv-offload off --draft off \
  --prompt-len 524288 --gen 64
```

配信で使うときは `--bench ... --prompt-len ... --gen ...` を `--serve` に置き換える。

予想値（実測 8K ベースの差分モデル、誤差 ±0.3 GiB）:

| 項目 | 値 |
| --- | --- |
| VRAM ピーク（grow 時） | **19,022 MiB = 19.95 GB = 18.6 GiB**（残 5,417 MiB） |
| prefill 512K | **13,500 秒（3.75 h）**、幅 11,000-16,000 秒 |
| decode | **10 tok/s**、幅 9-11（DSpark on なら 12-16、未確認） |
| host KV | 0（全量 VRAM。`--kv-offload off` なので溢れたら 503） |

`--chunk` を省くと auto が 2048 を選び、workspace +1,256 MiB と SWA +94 MiB でピークは約 20.4 GiB になる。512K では `--chunk 512` を明示する。

## --kv の実装

| --kv | B/visit/pos | 内訳 (K + V + scales) | global 1 token | 量子化 (CUDA) |
| --- | ---: | --- | ---: | --- |
| fp16 | 4,096 | 2048 + 2048 + 0 | 32 KiB | IEEE half |
| fp8 | 2,112 | 1024 + 1024 + 32 + 32 | 16.5 KiB | E4M3, head amax/448 |
| k8v8 | 2,112 | 1024 + 1024 + 32 + 32 | 16.5 KiB | K/V とも int8 対称 amax/127 |
| k8v4 | 1,600 | 1024 + 512 + 32 + 32 | 12.5 KiB | K=int8, V=int4 対称 amax/7 |

- 保存 `ink_attn.cuh:155-185` `ink_kv_store_128()`、読み出し `:57-152` `ink_kv_load4()`
- K8V4 の V は CUDA では対称 int4。CPU 側 (`ink_batch.h:306-330`) は非対称 0..15 で実装が違う
- scale は head 単位 (8 個/token/visit) で fp32
- prefill も decode も `k_ink_flash_gqa_gqa3` / `k_ink_flash_gqa_splitk` の scalar 経路 (sm_61 は TC タイル経路の条件 `cur_cc >= 70 || cur_cc == 60` から外れる)
- **KV 路に DP4A は無い** (DP4A は expert MoE の `dp4a_i8`)
- **P40 では fp8 は k8v8 に完全に劣後** (同バイト数、int8 の方が精度が高く、fp8 だけソフト変換が入る)

## 512K の VRAM

固定分 (重み 10,588 + workspace 820 + sconv 3.7 + DSpark 初期 78 + transient) = 11,741 MiB。

| --kv | global | SWA | DSpark | 定常 | grow ピーク | GB | 残 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fp16 | 16,384 | 240 | 0 | 28,365 | 30,381 | 31.9 | -5,942 | 不可 |
| fp8 | 8,448 | 124 | 0 | 20,313 | 21,352 | 22.4 | 3,087 | 可 (k8v8 推奨) |
| k8v8 | 8,448 | 124 | 0 | 20,313 | 21,352 | 22.4 | 3,087 | 可 |
| **k8v4** | 6,400 | 94 | 0 | 18,235 | **19,022** | **20.0** | **5,417** | **推奨** |
| k8v4 + DSpark | 6,400 | 94 | 2,016 | 20,251 | 22,267 | 23.4 | 2,172 | 可 (ぎりぎり) |
| k8v8/fp8 + DSpark | 8,448 | 124 | 2,016 | 22,329 | 24,345 | 25.5 | 94 | 不可 |

- fp16 512K は auto host offload で VRAM 282,624 pos + host 241,744 pos (7.4 GiB) になり、**decode 0.8 tok/s**
- 256K なら fp16 が 21,165 MiB で 3.3 GiB 余裕

## prefill の見積り

128K in-flight 実測 (122,880 token を 1,112.6 s) から:

```
a = 4.206e-3 s/token      (線形項: 重み・MoE・SWA)
b = 7.891e-8 s/token/pos  (global attention の二次項)
T(n) = 4.206e-3 * n + 7.891e-8 * n^2 / 2
```

| n | T | 平均 | 末尾 chunk |
| ---: | ---: | ---: | ---: |
| 131,072 | 1,229 s (20.5 min) | 107 tok/s | 69 tok/s |
| 262,144 | 3,814 s (63.6 min) | 69 tok/s | 40 tok/s |
| 524,288 | **13,050 s (3.63 h)** | 40 tok/s | 22 tok/s |

k8v4 の補正は 0.85-1.25x -> **11,000-16,000 s (代表 13,500 s)**。KV grow のオーバーヘッドは 10 秒以下で無視できる。

## 品質: k8v4 はどこまで粗いか

| 量子化 | 最大誤差 | RMS | SNR |
| --- | --- | --- | ---: |
| int8 (K) | amax/254 | 0.007σ | ~43 dB |
| fp8 (K/V) | 6.25% | 3.6% | ~29 dB |
| int4 (V) | amax/14 | 0.13σ | **~18 dB** |

k8v4 は「K は fp8 より正確、V は fp8 より 11 dB 粗い」。512K で最初に壊れるのは sharp retrieval (数値・ID・固有名詞の完全コピー)。

512K を通す前の品質ゲート: needle 512K、512K QA 8 問、`--spec-check`、`--ring-check 97`、可能なら `--kld-eval` で k8v8 と比較。

## 未確認

- 8K より深い decode の実測がゼロ (128K アームは timeout)
- k8v4/k8v8 の prefill・decode・品質はすべて推定
- DSpark の 512K での接受率と、`ensure_ctx_cap` の 2 GiB transient
- 512K の prefill が serve 側のタイムアウトに耐えるか (3.6 h の TTFT は実用外。運用は 256K 以下が妥当)
- モデル自体の学習 ctx (`inkling_fit.py` の preset は 262,144 まで、`MAX_CONTEXT` は 1,048,576)

## 参照

- 実測: `~/work/results/inkling-baseline.md`、`~/work/results/inkling/a-sq10-128k-p131072-kvfp16-cap1200/`
- コード: `ink_runner.cuh:1966-1970,2117-2230,2460-2634`、`ink_attn.cuh:26-185,352-753`、`ink_cuda.cuh:337-394`、`ink_dspark.cuh:1127-1170,1400-1435`、`ink_cuda_backend.cu:182-303`、`ink_main.cpp:141,875-914`
