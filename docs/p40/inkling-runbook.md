# Inkling-2-Mini runbook (Tesla P40 24GB / sm_61 / 88GB RAM)

対象: `<host>`、upstream `Niko1221/Strata` main = 0.2.21 (`d46f5f268`)、`strata-inkling` 0.1.67 (`engine/src/inkling2/ink_main.cpp:30`)。
コードの行番号は `~/work/build-base` (upstream/main の worktree) のもの。`~/work/Strata-legacy-gpu` の作業ツリーは fork main のままで `engine/` が無いので、読むときは `build-base` 側を使う。

## 結論

- **推奨は SQ-15 (13.36 GiB)**。24 GiB に重みが全載りし、32K も 128K も global KV が VRAM に収まる (`inkling_fit.compute` の 24 GiB / sm=61 判定)。
- 品質優先の SQ-20 (18.02 GiB) も 32K / 64K は全載り。**128K は `--kv fp8` が必須**。FP16 だと global KV 4.29 GB が mapped pinned host に落ち、このホストは PCIe 3.0 **x8** なので decode が実用外まで落ちる。
- SQ-10 (9.75 GiB) は余裕があり、最初の起動確認に使う。
- sm_61 では WMMA / INT8 TC / BF16 TC / FP4 TC がすべて無効。MoE は dp4a の SIMT 経路になる。SQ-20 は全 expert が SF4/MX8 なので最も不利 (LUT デコード + FFMA)。**P40 では SQ-15 が速度・品質のバランスで最良と推定 (未測定)**。
- ローカルの `Inkling-2-Mini-SQ-15.strata` は 5,625,176,064 B しかなく、期待 14,348,316,672 B に対して**不完全**。SHA256SUMS.txt と `terminal/strata_control/inkling_family.py:75` のサイズで確認して再取得する。SQ-20 は未取得。

## 起動方法 (strata-inkling)

`strata-inkling` は単体バイナリで、モデルは `--native PATH` (別名 `--model PATH`) で渡す。`.strata` v1.1 / HF safetensors ディレクトリ / `.gguf` を受ける (`ink_main.cpp:870`, `:975`, `:1074`)。

| 引数 | 既定 | 意味 | 実装 |
| --- | --- | --- | --- |
| `--native PATH` / `--model PATH` | 必須 | モデルファイル / HF ディレクトリ | `ink_main.cpp:870` |
| `--max-context N` / `--ctx N` | 32768 | コンテキスト上限。KV は必要時に 8K 刻みで伸びる | `ink_main.cpp:875`, `ink_backend.h:18` |
| `--parallel N` / `--batch N` / `--slots N` | 1 (1..64) | 同時スロット数 | `ink_main.cpp:877`, `ink_batch.h:13` |
| `--chunk N` | 512 + 自動 (1-3 slot で最大 2048) | prefill チャンク。指定すると `chunk_auto=false` | `ink_main.cpp:879`, `ink_chunk.hpp:37-50` |
| `--kv auto, fp16, fp8, k8v8, k8v4` | auto (実質 fp16。parallel>=16 で fp8) | KV の型。global 1 visit あたり fp16=4096 B/pos、fp8=2112、k8v4=1600 | `ink_main.cpp:882-892`, `:1021-1024`, `ink_runner.cuh:1966-1970` |
| `--kv-offload auto, on, off` | auto | global KV を mapped pinned host へ。`on` は最初から全部 host | `ink_main.cpp:893`, `ink_runner.cuh:2161-2196` |
| `--vram-experts-gb F` | -1 (auto) | VRAM に置く expert の上限 GB | `ink_main.cpp:908`, `ink_backend.h:30` |
| `--ram-experts-gb F` | -1 (auto) | RAM に置く expert の上限 GB。**0 は「RAM 無し = GPU+SSD」** | `ink_main.cpp:910`, `ink_backend.h:31` |
| `--ssd-experts` | off | SSD 直ストリーミング | `ink_main.cpp:912` |
| `--vram-reserve-mib N` | -1 | 空き VRAM 予約 (>=0 で hybrid 有効) | `ink_main.cpp:914`, `ink_backend.h:27` |
| `--draft auto, dspark, off` | auto | DSpark の on/off (`off` 以外で `enable_dspark=true`) | `ink_main.cpp:895-897`, `ink_serve.h:488,507` |
| `--dspark PATH` | なし | 外部 drafter (mtp.safetensors / dspark-*.gguf) | `ink_main.cpp:906` |
| `--draft-beam N` / `--draft-cost R` / `--draft-adapt` / `--draft-plain X` | 4 / 4.2 / off / -1 | 投機の調整 | `ink_main.cpp:898-905`, `ink_backend.h:37-40` |
| `--gpu N` / `--gpus N` / `--gpus 0,1` / `--devices` | 0 / 1 | GPU 選択 (P40 1 枚なら `--gpu 0`) | `ink_main.cpp:918-944` |
| `--vision PATH` | なし | Vision シャード (0.64 GB)。画像要求まで VRAM 0 | `ink_main.cpp:872`, `ink_serve.h:531-539` |
| `--prompt-ids "1,2,.."` / `@file` | 既定 10 個 | 直接生成・品質ゲートの入力 | `ink_main.cpp:957-963` |
| `--prompt-len N` / `--gen N` / `--temp F` | 64 / 32 / 0.0 | bench と CLI 生成 | `ink_main.cpp:953-956,970` |
| `--bench` / `--selfcheck` / `--info` / `--serve` / `--kld-eval` / `--dump-ref` / `--spec-check` / `--ring-check` / `--batch-check` / `--export-tokenizer` | - | モード | `ink_main.cpp:851-869,964-969` |

既定値は `engine/src/inkling2/ink_backend.h:18-42` (max_ctx 32768, parallel 1, chunk 512 + auto, kv fp16 auto, kv_offload auto, dspark on, vram/ram experts auto)。
環境変数: `IK2_EP_SPLIT` / `IK2_GPUS` (`ink_main.cpp:992-1019`)、`STRATA_INK_Q8_0_DENSE` (`:1033-1036`) など。

### サーバとして立てる場合

**`strata-inkling --serve` 自体は HTTP サーバではない。** stdin/stdout の行プロトコル (`INFO` / `READY` / `GEN` / `BGEN` / `GENERATE` -> `PP` / `T` / `DONE` / `BDONE` / `ERR`) を話す (`ink_serve.h:1-3`, `:484-568`)。`ink_main.cpp` の引数解析に port も API key も無い。

OpenAI 互換 HTTP は `terminal/serve/server.py` (= `strata serve`) が被せる。server が `strata-inkling --serve` を子プロセスで起動する (`server.py:833`)。

| 引数 | 既定 | 意味 | 実装 |
| --- | --- | --- | --- |
| `strata serve --profile CFG.json` / `--config` | 必須 | run config | `terminal/strata_cli/cli.py:67-68` |
| `--port N` | 8095 | HTTP ポート | `server.py:9174` |
| `--api-key KEY` | 環境変数 `STRATA_API_KEY` | `/v1/*` に Bearer / x-api-key を要求 | `server.py:9202` |
| `--no-api-key` | - | キー無し (loopback 用) | `server.py:9205` |
| `--host ADDR` | 127.0.0.1 | listen アドレス | `server.py:9161` |
| `--parallel N` / `--max-queue` / `--queue-timeout` | config 準拠 | スロットと過負荷制御 | `cli.py:74-77` |

次の JSON を `~/work/inkling-sq15.json` に置く。run config のキーは CI と同じ (`/.github/ci/inkling/serve_chat.py:36-37`): `arch` `exe` `args` `cwd` `tokenizer` `model_name` `log` `host` `port`。`args` に `--serve` は入れない (server が足す)。例:

```json
{
  "arch": "inkling2_loop",
  "exe": "~/work/build-base/build/engine/src/inkling2/strata-inkling",
  "args": ["--native", "<models>/inkling2/Inkling-2-Mini-SQ-15.strata",
           "--max-context", "65536", "--parallel", "1", "--kv", "fp16", "--draft", "off"],
  "cwd": "~/work/build-base",
  "tokenizer": "~/work/inkling-tok",
  "model_name": "inkling-2-mini-sq-15",
  "log": "~/work/logs/inkling-sq15-serve.log",
  "host": "127.0.0.1",
  "port": 8095
}
```

```sh
read -rsp "API key: " STRATA_API_KEY; export STRATA_API_KEY
cd ~/work/build-base/terminal && python3 strata.py serve --profile ~/work/inkling-sq15.json --port 8095
```

`tokenizer` ディレクトリが無ければ server が `.strata` の Section 0 か隣の `tokenizer.json` + `chat_template.jinja` から自動生成する (`server.py:9083-9136`)。`<models>/inkling2/` には両方ある。

エンジンだけを直接叩く場合:

```sh
printf 'GEN 8 200002,200004,100,200010\nQUIT\n' | ./build/engine/src/inkling2/strata-inkling --serve --native <models>/inkling2/Inkling-2-Mini-SQ-10.strata --max-context 32768
```

## VRAM 見積り

`terminal/strata_control/inkling_fit.py` の `compute()` を **24 GiB / 88 GB / CUDA / sm=61 / DSpark on** で実行した値 (GB = 10^9)。カード予算は 24 GiB = 25.77 GB から safety 0.3 GB を引いた **25.47 GB** (`inkling_fit.py:305`)。

内訳の式 (`inkling_fit.py:204-223`):

```
必要 = dense + bodies
     + 1.30 (driver/context/workspace, 実測由来)
     + 0.85 (hybrid のときだけ)
     + 0.54 (8 slot 以上のときだけ)
     + DSpark (0.11 + ctx * 4096 B)
     + global KV (ctx * 32768 B/slot, FP16) x 1.6 (伸長ピーク)
     + SWA+sconv (0.172 GB/slot, FP16。chunk 512 なら +0.084 GB/slot)
```

dense / bodies は .strata の Section 1 / 4 (`inkling_fit.py:71-73`)。手元ヘッダで SQ-10 は sec1=1,871,628,224 B / sec4=8,599,769,088 B、SQ-15 は sec1=2,068,299,712 B / sec4=12,279,889,920 B を確認済み。SQ-20 は公表値のみ:

| build | dense (Section 1) | experts (Section 4) | 合計 |
| --- | ---: | ---: | ---: |
| SQ-10 | 1.87 GB | 8.60 GB | 9.75 GiB |
| SQ-15 | 2.07 GB | 12.28 GB | 13.36 GiB |
| SQ-20 | 3.67 GB | 15.68 GB | 18.02 GiB |

### 1 slot、DSpark on

| build | ctx | KV | 必要 VRAM | 予算 25.47 GB との差 | 判定 |
| --- | ---: | --- | ---: | ---: | --- |
| SQ-10 | 32K | FP16 | 13.90 | +11.57 | 全載り |
| SQ-10 | 128K | FP16 | 19.46 | +6.01 | 全載り |
| SQ-15 | 32K | FP16 | 17.78 | +7.69 | 全載り |
| SQ-15 | 128K | FP16 | 23.34 | +2.13 | 全載り (ぎりぎり) |
| SQ-20 | 32K | FP16 | 22.78 | +2.69 | 全載り |
| SQ-20 | 64K | FP16 | 24.64 | +0.83 | 全載り (ぎりぎり) |
| SQ-20 | 128K | FP16 | 28.34 | -2.87 | **KV 4.29 GB が host へ** |
| SQ-20 | 128K | FP8 | 24.82 | +0.65 | 全載り (ぎりぎり) |

FP8 のときの必要量: SQ-10 32K 12.96 / 128K 15.94、SQ-15 32K 16.84 / 128K 19.82、SQ-20 32K 21.84 / 64K 22.83。

### 複数 slot (32K)

| build | 4 slot FP16 | 4 slot FP8 |
| --- | ---: | ---: |
| SQ-10 | 19.57 | 15.80 |
| SQ-15 | 23.45 | 19.68 |
| SQ-20 | 28.45 -> KV 4.29 GB host | 24.68 (ぎりぎり全載り) |

8 slot / 32K は SQ-15 でもプランナーが global KV 8.59 GB を host へ回す (VRAM 側は dense 2.07 + bodies 12.28 + 予約 ~3.5 = 約 17.8 GB)。**このホストでは 8 slot 以上の decode は PCIe 律速になるので実用外**。

### 24GB で足りないとき、何がどれだけ溢れるか

- **SQ-20 / 128K / 1 slot / FP16**: 2.87 GB 超過。エンジンは experts 15.68 GB + dense 3.67 GB を VRAM に残し、**global KV 4.29 GB 全体**を pinned host に移す (VRAM 使用は約 21 GB)。重みは溢れない。
- **SQ-20 / 32K / 4 slot / FP16**: 28.45 GB -> 同じく global KV 4.29 GB が host。
- **SQ-20 / 128K / 4 slot / FP16**: 全部 VRAM なら 49.47 GB。実際は experts を VRAM に残して global KV 17.18 GB を host へ回す (VRAM 約 22.0 GB)。RAM は足りるが速度が死ぬ。
- **空き VRAM が 20 GiB しかない場合** (本番を止め忘れた等): SQ-20 32K でも global KV 1.07 GB が host に落ちる。16 GiB まで減ると experts 8.93 GB VRAM + 6.75 GB RAM に分割される (`compute()` の実値)。
- `--kv-offload off` にすると host へ逃がさず **max_ctx を切り下げる** (`ink_runner.cuh:2186-2195` の `Auto-sizing max_ctx`)。
- **expert を RAM に置く組み合わせ** (24 GB では不要だが比較用): SQ-20 に `--vram-experts-gb 12` を付けると experts 3.68 GB が RAM に溢れ、さらに tier (`STRATA_INK_TIER` 既定 on) が VRAM 側 12 GB のコピーも RAM に持つので **RAM 約 15.7 GB**。VRAM は 12 + dense 3.67 + 予約 ~2.2 = 約 17.9 GB で収まるが、tier の入れ替えと PCIe 読みで decode は大きく落ちる (未測定)。

host に落ちた global KV は 1 token ごとに 8 global visit 分を PCIe で読む。32 KiB/token/pos なので 32K で 1.07 GB/token、128K で 4.29 GB/token。PCIe 3.0 x8 (理論 7.9 GB/s、実効 6 GB/s 前後) では **32K で ~5 tok/s、128K で ~1.4 tok/s が上限** (概算。実測は未)。これが SQ-20 128K で `--kv fp8` を必須にする理由。

注意: エンジン自身のローダ予約はプランナーより厳しい。24 GB カードでは `base_reserve` 1600 MiB と `extra_reserve + 512 MiB` の大きい方 (~2.2 GiB) を experts 配置前に空けておく (`ink_cuda_backend.cu:222-225`, `ink_runner.cuh:396-401`, `:443-446`)。実際に選ばれた chunk と残 VRAM は起動ログの `[InkCUDA] memory:` 行で確認する。

## sm_61 で無効・低速になる経路

dp4a そのものは P40 でハードウェア動作する (`ink_cuda.cuh:384-394` の `__dp4a` は `__CUDA_ARCH__ == 610 || >= 750`)。問題は tensor core 系が全部落ちること。

- **WMMA FP16 TC MoE**: `k_ink_moe_wmma_f16` (`ink_moe_cuda.cuh:3169`) と `k_ink_moe_wmma_f16_bk64` (`:3406`) は `__CUDA_ARCH__ >= 700` の中身。さらに dispatch は `ink_cap_wmma_f16(major) = (major == 7)` (`ink_caps.hpp:17`, `ink_moe_cuda.cuh:2314-2321`) なので sm_61 では起動もしない。
- **INT8 tensor core MoE**: `ink_cap_int8_tc(major) = major >= 8` (`ink_caps.hpp:16`)。`k_ink_moe_sk_tc` / `k_ink_moe_q80_tc` / `k_ink_moe_fp16tc*` は `ink_moe_cuda.cuh:3679,3790,3832` の判定で sm_61 では選ばれない。代わりに S_K は `launch_ink_moe_gu_sk_mask` の dp4a 経路 (`:3800-3808`)、GGUF は `k_ink_moe_gu_gguf_dp4a` (`:3834-3859`)。
- **SF4/MX8 の TC 経路**: sm_120a (`ink_moe_cuda.cuh:1294`)、sm_89+ の hardware cvt (`ink_cuda.cuh:313,338,357`)、sm_80+ の FP16 TC (`ink_moe_cuda.cuh:2474,2613,2886`) はすべて無効。sm_61 は `k_ink_moe_gu_sf4_mx8` / `k_ink_moe_dn_sf4_mx8` (`:3768-3780`) の **LUT デコード + FFMA** になる。
- **BF16 tensor core / cuBLAS BF16**: sm_80+ のみ (`ink_runner.cuh:2739-2750`)。sm_61 は `gemm_bf16_compat` の else (`:2794-2822`) で、毎回 `k_ink_bf16_to_f32` (`ink_kernels.cuh:1785-1788`) により重みを F32 へ変換し `cublasSgemm`。重み帯域が 2 倍になり TC も無い。`f32_to_bf16` も <800 はソフトウェア丸め (`ink_cuda.cuh:270-279`)。
- **attention の sm_70/75 特化**: `.attn.*` の BF16->FP16 一括変換と FP16 TC は `sm_cc >= 70 && < 80` のみ (`ink_runner.cuh:345`)。sm_61 の attention GEMM は F32 経路。
- **prefill attention の TC タイル**: `ink_runner.cuh:3310` の条件は `mode == 0 && M >= 128 && (cur_cc >= 70 || cur_cc == 60)`。**sm_61 は除外**され、`k_ink_flash_gqa_gqa3` (`:3395-3400`) の per-token flash 経路になる。O(1) の 114 MiB KV タイル経路は使われない (workspace の確保だけはされる: `:2357-2361`)。
- **Q8_0 dense の int8 TC**: `ink_cuda_q8f_supported` は sm_80+ (`ink_kernels.cuh:1500`, `ink_main.cpp:1033-1036`)。sm_61 では GGUF の Q8_0 dense は BF16 のまま。
- **TF32**: router の `CUBLAS_COMPUTE_32F_FAST_TF32` は int8 TC があるカードのみ (`ink_moe_cuda.cuh:3658`)。sm_61 は素の FP32。
- **sm_61 で逆に有効な高速路**: Q8_ROW dense の prefill (M>16) は `k_ink_quant_x_row_i8` + cuBLAS INT8 (`ink_runner.cuh:2845,3011-3031`、`CUBLAS_GEMM_DEFAULT` なので TC 不使用)。DSpark の候補スクリーニングも `sm_cc == 61 || >= 75` で dp4a (`ink_dspark.cuh:1571,1611`、unembed screen 構築時のみ。約 0.46 GB VRAM、空き >2 GiB が条件: `ink_runner.cuh:692-710`)。
- CUDA Graph は 1024 スレッドまで可 (`volta_forever.hpp:97-103`)。`STRATA_INK_PROF` / `STRATA_INK_NO_GRAPH=1` で無効化される (`ink_cuda_backend.cu:305-319`)。

推定: SQ-10 / SQ-15 は expert が S2_K/S3_K 主体なので dp4a が効く。SQ-20 は全 expert が SF4/MX8 で tensor core が無いため、P40 では SQ-15 より遅くなる可能性が高い (未測定)。

## 測定

### prefill の位相内訳

**Qwen 側の `STRATA_PREFILL_TIMING` は Inkling では効かない。** この変数を読むのは `engine/src/prefill/prefill.cpp:4017` と `engine/sycl/src/prefill/prefill.cpp:2561` だけで、`engine/src/inkling2/` には参照が無い。

Inkling の相当物は `STRATA_INK_PROF`:

- `STRATA_INK_PROF=2`: prefill の chunk ごとに 1 行 + decode の位相。
- `STRATA_INK_PROF=1`: decode のみ (100 step ごと)。

```sh
cd ~/work/build-base
CUDA_VISIBLE_DEVICES=0 STRATA_INK_PROF=2 ./build/engine/src/inkling2/strata-inkling \
  --bench --native <models>/inkling2/Inkling-2-Mini-SQ-10.strata --gpu 0 \
  --max-context 32768 --parallel 1 --chunk 512 --kv fp16 --draft off \
  --prompt-len 8192 --gen 64 2> prof.log
grep -E "PREFILL-M|moe parts|DECODE-M1|^\[bench\]" prof.log
```

出力 (`ink_runner.cuh:1956-1964`, `:3700-3707`):

```
[PROF PREFILL-M512 / 1 iters] total=... ms/step | res_ctrl=... dense_qkv=... sconv_kv=... attn=... dense_wo=... moe=... unembed=...
[PROF moe parts / 1 iters] router+topk=... bin=... gu(sf4/mx8/sk)=... dn(sf4/mx8/sk)=... gu(gguf)=... dn(gguf)=...
[PROF DECODE-M1 / 100 iters] ...
```

注意: `STRATA_INK_PROF` を付けると CUDA Graph が切られ eager launch になる (`ink_cuda_backend.cu:311-319`)。**絶対値は遅くなる。位相の比率だけを見る**。総時間は PROF 無しで取る。

### decode の tok/s と TTFT

1. `--bench` の 1 行 (`ink_main.cpp:248-250`):

```
[bench] backend=cuda format=strata parallel=1 gpus=1 kv=fp16 prompt=8192 gen=64 | prefill=NNN.N ppts (NN.NNs) | decode=NN.N tps agg (NN.N tps/slot, NN.NNs)
```

TTFT は `prefill=(NN.NNs)` + 最初の 1 step。`--bench` は合成 prompt なので tokenizer は不要。

2. サーバ経由: エンジンが `PP <pos> <len> <prompt_ms> <ppts>` (`ink_serve.h:1127`) と `DONE <gen> <prompt_len> <prompt_ms> <dec_ms> ...` / `BDONE <slot> <gen> <reason> <dec_ms> read=<prompt_ms>` (`:583-600`) を出す。decode tok/s = `generated / (dec_ms/1000)`、エンジン側 TTFT = `prompt_ms/1000`。

3. `STRATA_INK_STEPTIME=1` (`ink_cuda_backend.cu:709-744`):

```
[InkCUDA STEPTIME] 100 one-token steps: host between steps X ms, step call wall Y ms, GPU span of the call Z ms
```

4. HTTP の TTFT (SSE の最初の delta まで):

```sh
cd ~/work/build-base && python3 - <<'PY'
import json, os, time, urllib.request
key = os.environ["STRATA_API_KEY"]
body = {"model": "inkling-2-mini-sq-15", "stream": True, "temperature": 0, "max_tokens": 32,
        "reasoning_effort": "none", "messages": [{"role": "user", "content": "List the first 10 primes."}]}
req = urllib.request.Request("http://127.0.0.1:8095/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
t = time.perf_counter()
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        if line.startswith(b"data: ") and b"delta" in line:
            print("TTFT %.3f s" % (time.perf_counter() - t)); break
PY
```

TTFT を測るときは thinking を切る (`reasoning_effort: "none"`、または `STRATA_INKLING_THINKING_BUDGET=0`)。既定は `low` で 2048 token の thinking が TTFT に入る (`docs/MODELS.md:311`, `docs/DETAILS.md:536`)。

### 品質・再現性ゲート

- `--selfcheck --native M --max-context 8192`: backend 初期化 + softcap 検証 (`ink_cuda_backend.cu:1141-1164`)。
- 5 回同じ生成で logits の sha256 が 1 種類: `--native M --draft off --prompt-ids "$(seq -s, 101 164)" --gen 16 --dump-ref out.bin` を 5 回 (CI の `b70_gate.sh:30-38` と同じ)。
- `--spec-check N --prompt-ids ...`: drafter が lossless か (`ink_main.cpp:599-823`)。
- `--ring-check 97 --prompt-ids @ids.txt`: SWA ring の batch/1-token 一致 (`:376-442`)。
- `--batch-check 8 --parallel 4 --prompt-ids ...`: slot 分離 (`:450-580`)。
- `--kld-eval ref.bin`: 参照 logits との KLD (`:256-323`)。

### P40 での測定アーム (推奨順)

各アームの前に `nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv` を見て 60 C 以下・1531 MHz を確認する。89 C で 1300 MHz に落ちる (`docs/OLDER_GPUS.md:265-266`)。

1. SQ-10 32K p1 chunk512 `--draft off` (基準)。
2. 同 `--draft dspark` (投機の効果)。
3. SQ-15 32K p1 chunk512 (本命)。
4. SQ-20 32K p1 chunk512、次に chunk1024/2048。
5. SQ-20 128K p1 `--kv fp8` と `--kv fp16` の比較 (host KV の有無が log で分かる)。
6. `STRATA_INK_PROF=2` で位相 (手順は上)。

## 既知の注意点

- **本番が GPU を 23.8 GiB 使っている**。起動前に GPU ロックを取り、`~/Strata/docker/compose` で stop する。`nvidia-smi` の free が 23,000 MiB 前後になることを確認する。
- **起動ログで必ず読む行**:
  - `[InkCUDA] Loaded ...: trunk=%.2f GB, experts=%zu VRAM / %zu host, hyb=%d, pf_buf=%.0fMB, sm120_tc=%d, screen=%d` (`ink_runner.cuh:713`)
  - `[InkCUDA] expert codecs (gate_up/down code: count, mean MB): ...` (`:392`)。SQ-20 なら 15/16、SQ-10 なら 2/3 主体。
  - `[InkCUDA] prefill chunk %d (auto: %.0f MiB free after the experts, ...)` (`ink_cuda_backend.cu:299`)
  - `[InkCUDA] Auto Host KV Offload ... = %d VRAM + %d mapped host tokens` または `Host KV Offload ON` (`ink_runner.cuh:2166,2182`)
  - `[InkCUDA] memory: %.0f MiB free after load, runner+KV took %.0f MiB (max_batch=%d), DSpark took %.0f MiB, %.0f MiB free now` (`ink_cuda_backend.cu:355`)
  - `[InkTier] ...` / `[InkHybrid] ...` (expert が VRAM に収まらないときだけ)
- **expert arena と RAM**: 全載りなら host arena は作られない (RAM は 2 GB 程度)。溢れた分だけ host arena を作り、Linux では空き RAM の 45% 未満なら pinned (`ink_pin_rule.hpp:44`)。`STRATA_INK_TIER` (既定 on) のときは **VRAM に置いた分のコピーも host に持つ**ので、`--vram-experts-gb` を絞ると RAM が「VRAM 分 + 溢れ分」に増える (`ink_runner.cuh:473-518`)。88 GB なら足りるが、`--ram-experts-gb 0` は「RAM を使わない = SSD 直」なので **設定しない** (`ink_backend.h:31`)。
- **KV の grow**: 起動時は global KV を 1 slot なら 8192 pos、2 slot 以上なら 4096 pos だけ VRAM に取る (`ink_runner.cuh:2169,2193`)。要求位置が超えると 8K 刻みで伸ばし (`ink_kv_grow.hpp:20-23`)、ピークは「純増 + 旧 visit 1 本分」(同 `:27-33`)。収まらなければ**残り全部**を一度に pinned host へ移す (`:40-44`)。縮小はしない。容量超過の要求は 503 で終わり、エンジンは生き残る (`ink_backend.h:77-80`)。
- **`--kv-offload on` は使わない**: global KV が最初から 100% host になる (`ink_runner.cuh:2161-2167`)。x8 リンクでは decode が数 tok/s になる。容量が足りないときは `--kv fp8` -> `--max-context` 縮小の順。
- **slot 数**: 1 slot あたり global KV 32 KiB/token (FP16) / 16 KiB (FP8)、SWA+sconv 0.172 GB (FP16) / 0.088 GB (FP8)。8 slot 以上でプランナーは workspace +0.54 GB (エンジンの予約は 960 -> 1280 MiB)、16 slot 以上で KV が自動 FP8 になり SWA ring も 1056 pos に固定 (`ink_main.cpp:1021-1024`, `ink_cuda_backend.cu:185-186`, `ink_runner.cuh:2095`)。P40 では SQ-15 なら 4 slot/32K まで VRAM 内、SQ-20 は 1 slot 基本で 4 slot は FP8 のぎりぎり。**8 slot 以上は host KV で実用外**。
- **chunk**: 指定なしだと「expert 配置後の空き」で 2048/1024/512 を選ぶ (`ink_chunk.hpp:43-50`)。選ばれた値は上のログに出る。最初の比較では `--chunk 512` に固定する。
- **P40 の熱**: ファン無し。89 C で `SW Thermal Slowdown` -> 1300 MHz。A/B の各アームは同じ温度から始める。
- **PCIe は現在 x8** (`nvidia-smi` の `pcie.link.width.current = 8`、max 16)。host KV / host expert の速度は x16 の半分。スロットかライザーの制約かは未確認。
- **モデルファイル**: `<models>/inkling2/SHA256SUMS.txt` と `inkling_family.py:74-76` のサイズ・SHA で確認する。SQ-15 は不完全 (5.63 GB / 14.35 GB)、SQ-20 は未取得。
- **サーバ側の設定** (`docs/DETAILS.md`): `inkling_thinking_budget` (既定: none/minimal 1024, low 2048, medium 4096, high 8192, xhigh/max 16384) と `STRATA_INKLING_THINKING_BUDGET` (`DETAILS.md:536`)、`STRATA_INKLING_EFFORT` (既定 low, `MODELS.md:311`)、`answer_room` (`DETAILS.md:537`)。共有 prefix は `STRATA_INKLING_SHARED_PREFIX=1` (`docs/RESEARCH_RUNS.md:43-54`)、厳密比較は `STRATA_INKLING_PROMPT_CKPT=0` (`server.py:6108-6111`)、`STRATA_INK_PREFILL_CAP=N` (`ink_serve.h:499-506`)。
- **起動コマンドの実パス**: バイナリが別のビルドディレクトリにある場合は `find ~/work -name strata-inkling -type f` で探す。

## ビルド (この runbook では未実施。別サブが並行中の可能性あり)

ビルドロックを取り、JOBS は 6 以下。ggml は `engine/third_party/strata-ggml` に同梱 (`engine/CMakeLists.txt:722`)。

```sh
sudo docker run --rm -v ~/work/build-base:/work/strata -v ~/work/build-base/build:/work/build -w /work/strata \
  nvidia/cuda:12.9.1-devel-ubuntu24.04 bash -lc "cmake -S engine -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_PORTABLE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=61-real -DSTRATA_BUILD_TESTS=OFF \
  && cmake --build /work/build --target strata-inkling -j 6"
```

生成物: `~/work/build-base/build/engine/src/inkling2/strata-inkling`。`-DSTRATA_EXPERIMENTAL_SM60=ON` は Qwen 側の Pascal 経路も有効にする (`docs/OLDER_GPUS.md:108-125`)。`61-real` を忘れると起動時に `no GPU code for Tesla P40` で止まる (`arch_probe.hpp:14-28`)。

## 最初に試す 1 コマンド

前提: 本番を止めて GPU ロックを取る。`cd ~/Strata/docker/compose && sudo docker compose -f docker-compose.nvidia.yml -f docker-compose.override.yml stop`。`nvidia-smi --query-gpu=memory.free` が 23,000 MiB 前後であること。

```sh
cd ~/work/build-base && CUDA_VISIBLE_DEVICES=0 ./build/engine/src/inkling2/strata-inkling --bench --native <models>/inkling2/Inkling-2-Mini-SQ-10.strata --gpu 0 --max-context 32768 --parallel 1 --chunk 512 --kv fp16 --draft off --prompt-len 512 --gen 64
```

期待する出力: `[bench] ... | prefill=... ppts (...) | decode=... tps agg (...)`、stderr に `[InkCUDA] Loaded ... experts=3072 VRAM / 0 host`、`[InkCUDA] memory: ... free now`、`[InkCUDA] prefill chunk ...`。
`--draft off` は DSpark の有無を切り分けるため。次に `--draft dspark` で比べる。

## 未確認

- P40 での実測 tok/s / TTFT は一切ない。参考値は RX 9070 XT 16GB の SQ-10 32K: prefill 2376 tok/s, decode 56.7 tok/s, TTFT 1.45 s (`docs/bench-results/2026-11-22-community-rx-9070-xt/README.md:80-84`)。
- SQ-15 の実ファイルが不完全なため、SQ-15 の実測は再取得後。
- SQ-20 のファイルは未取得。dense/bodies の値は `inkling_fit.py:71` の公表値で、手元ヘッダでは未検証。
- PCIe x8 がスロット由来かライザー由来か。
- sm_61 で `--kv k8v4` が品質・速度とも実用か (コード上は 1600 B/pos で動く)。
- `--parallel 2` 以上での VRAM 実測 (表はプランナー計算値)。
- ホストの実空き VRAM 24,576 MiB のうち driver が取る分 (現在は本番が 23,885 MiB 使用中で、空きは 555 MiB)。
- ビルドが本ツリーで通るか (この runbook では未実施)。
