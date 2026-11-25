# Inkling-2-Mini GGUF / abliterated .strata を Tesla P40 (sm_61) で動かす

検証日: 2026-11-25
対象: `<host>`、upstream/main `d46f5f268`、`strata-inkling` 0.1.67
コード: `~/work/build-base`
GPU: Tesla P40 24GB (sm_61)
この調査では GPU を使っていない。ビルド・ダウンロードもしていない。実行確認は CPU のみの `--info` と、取得済みファイルの GGUF/.strata ヘッダ解析だけ。

## 結論

- モデル指定は `--native PATH` (別名 `--model PATH`) だけ。`--gguf` は存在しない。`.gguf` は拡張子ではなく先頭 4 バイトの magic (`GGUF`) で判定される (`engine/src/inkling2/ink_gguf_loader.h:48-55,298-300`, `engine/src/inkling2/ink_manifest.h:468-472`)。
- `.gguf` は `.strata` へ変換されない。mmap から直接読み、メモリ上で canonical tensor を組み立てる。ディスクへの変換ファイル・キャッシュ・一時ファイルは無い (`engine/src/inkling2/ink_gguf_transcode.h:1-3`)。したがって「初回だけ変換」ではなく、起動のたびに同じ読込・再量子化コストを払う。
- P40 では tensor core 系 (INT8 TC / WMMA FP16 / BF16 TC / FP4 TC) がすべて無効。GGUF expert は `k_ink_moe_*_gguf_dp4a`、`.strata` の S_K expert は `launch_ink_moe_*_sk_mask` の dp4a、SF4/MX8 は LUT+FFMA に落ちる (`engine/src/inkling2/cuda/ink_moe_cuda.cuh:3678-3687,3725-3781,3784-3810,3812-3861`)。
- `dp4a_i8` は `__CUDA_ARCH__ == 610` でハードウェア `__dp4a` になる (`engine/src/inkling2/cuda/ink_cuda.cuh:382-394`)。P40 のビルドは `CMAKE_CUDA_ARCHITECTURES=61-real` なので、この分岐が有効。
- IQ3_XXS と Q4_K はどちらも DP4A を使える。ただし「レイヤ全体が GGUF_RAW のまま残る」条件は、gate/up と down の両方の型が `ink_gguf_has_gpu_kernel` に入ること (`engine/src/inkling2/cuda/ink_moe_cuda.cuh:410-417`)。片方でも外れると、そのレイヤは丸ごと SF4 へ再量子化される (`engine/src/inkling2/cuda/ink_runner.cuh:1037-1055,1455-1509`)。
- 実際のファイルは名前と中身が一致しない。UD-IQ3_XXS の expert は IQ2/IQ3 + NVFP4 の混合で、24 層中 DP4A に残るのは 4 層だけ。UD-Q4_K_XL には Q4_K が 1 個も無く、expert は NVFP4 20 層 + Q8_0 4 層。AD-IQ3_S は IQ3_S/IQ4_NL で 24 層すべて SF4 再量子化。AD-Q4_K だけが 24 層すべて Q4_K/Q5_0 の DP4A。
- P40 の無検閲なら **AD-Q4_K** が第一候補。速度・VRAM 優先で小さいなら **abliterated SQ-10 .strata** (expert の 99.5% が S2_K/S3_K で DP4A、DSpark 内蔵)。**AD-IQ3_S は P40 では最も不利** (全層 SF4 再量子化、DP4A 無し)。
- Unsloth/AtomicChat の `mmproj-*.gguf` は `strata-inkling` では読めない。テキストのみなら `--vision` を省略する。画像が必要なら `Inkling-2-Mini-Vision.strata` (636,526,592 B) を使う。
- 現行の `dspark-Inkling-2-Mini-Q8_0.gguf` (503,889,952 B) は llama.cpp mainline の dflash レイアウトで、このエンジンでは読めない。DSpark を使うには `dspark-Inkling-2-Mini-Q8_0_Old.gguf` (402,278,560 B) か `mtp.safetensors` (636,526,592 B) が必要。どちらも未取得。

## モデル指定と読み込み経路

### 引数

| 指定 | 既定 | 意味 | 実装 |
| --- | --- | --- | --- |
| `--native PATH` / `--model PATH` | 必須 | `.strata` / HF safetensors ディレクトリ / `.gguf` | `engine/src/inkling2/ink_main.cpp:870-871` |
| `--gguf PATH` | 無い | 不明なオプションとして終了する | `engine/src/inkling2/ink_main.cpp:985-987` |
| `--vision PATH` | 空 | VisionShard `.strata` または HF safetensors。GGUF は不可 | `engine/src/inkling2/ink_main.cpp:872-874`, `engine/src/inkling2/ink_vision.h:100-215` |
| `--dspark PATH` | 空 | 外部 drafter (`.gguf` / `mtp.safetensors`) | `engine/src/inkling2/ink_main.cpp:906-907` |
| `--draft auto\|dspark\|off` | `auto` | `off` 以外で DSpark を有効化 | `engine/src/inkling2/ink_main.cpp:895-897` |
| `--max-context N` | 32768 | KV の上限 | `engine/src/inkling2/ink_main.cpp:875-876`, `engine/src/inkling2/ink_backend.h:18` |
| `--parallel N` | 1 | 同時スロット数 | `engine/src/inkling2/ink_main.cpp:877-878` |
| `--kv auto\|fp16\|fp8\|k8v8\|k8v4` | auto (実質 fp16) | KV の型 | `engine/src/inkling2/ink_main.cpp:882-892` |
| `--gpu N` | 0 | 使う GPU | `engine/src/inkling2/ink_main.cpp:918-919` |

`--native` が無い場合のエラーは `engine/src/inkling2/ink_main.cpp:1073-1075`。`--info` は位置引数で PATH を取り、`--export-tokenizer PATH OUT_DIR` も `--native` を使わない。

### `.gguf` の読み込み経路

1. `InkModelFile::open` が safetensors -> GGUF -> `.strata` の順に判定する (`engine/src/inkling2/ink_manifest.h:454-476`)。GGUF 判定は `GgufAdapter::is_gguf_path` -> `has_gguf_magic` (`engine/src/inkling2/ink_gguf_loader.h:48-55,298-300`)。
2. `strata::GgufModel::open(path)` で mmap する (`engine/src/inkling2/ink_gguf_loader.h:330-338`)。ファイルは書き換えない。
3. GGUF metadata から `InkCfg` を作る (`engine/src/inkling2/ink_gguf_loader.h:340-428`)。`general.architecture` は `inkling2` または `inkling2_loop` のみ許可 (`:341-346`)。block_count が 24 (merged mainline) か 48 (Unsloth/PR #32187) かを判定する (`:399-407`)。
4. Section 1 相当の dense trunk をメモリ上で組み立てる (`engine/src/inkling2/ink_gguf_loader.h:431-746`)。dense は BF16 へ dequant される。Q8_0 dense だけは `keep_q8_0_dense()` が真のとき Q8_0 のまま残るが、その条件は `ink_cuda_q8f_supported` が真、つまり **sm_86 のみ** (`engine/src/inkling2/cuda/ink_cuda_backend.cu:1239-1250`, `engine/src/inkling2/ink_main.cpp:1030-1036`)。P40 では Q8_0 dense も BF16 になる。
5. expert を索引する (`engine/src/inkling2/ink_gguf_loader.h:870-957`)。NVFP4 が gate/up と down の両方なら bit-exact な SF4 へ repack (`:924-942`)。それ以外は `GGUF_RAW` (code_id 20) として元のブロックを保持する (`:938-941`)。
6. chat template は `tokenizer.chat_template` から常に読む (`engine/src/inkling2/ink_gguf_loader.h:977-979`)。tokenizer 全体は `keep_tokenizer=true` のときだけ展開する (`:972-976`)。

### `.strata` の読み込み経路

`StrataFileReader::open` が 4 KiB header の magic (`STRATA\0\1`)、profile、6 sections、4 KiB alignment を確認し、Section 0 の ZSTD manifest を展開する (`engine/src/artifact/strata_reader.cpp:33-111`)。`parse_manifest` が `config.text_config` と `trunk_tensors` を読み、`dspark_tensors` があれば DSpark 内蔵として扱う (`engine/src/inkling2/ink_manifest.h:217-337`)。CUDA 側は `engine/src/inkling2/cuda/ink_manifest.hpp:122-213` のパーサを `ink_runner.cuh:11` から使う。

### `.strata` へ変換するか / キャッシュ

変換しない。CUDA の GGUF 経路は、mmap した元ファイルから dense を BF16 へ、expert を `GGUF_RAW` のまま、または SF4 へ repack して RAM/VRAM に置くだけ。`.strata` を書き出すコードは無い。

- ディスク: 追加 0。元の `.gguf` 以外にファイルは増えない。
- キャッシュ先: 無い。`ink_gguf_transcode.h:1-3` に "Zero temp files on disk" と明記されている。CUDA 側も expert をコピーした後に `release_mapped_pages` で mapping を手放す (`engine/src/inkling2/ink_gguf_loader.h:864-868,1062-1064`)。
- 初回コスト: 「初回だけ」ではなく毎回。dense の dequant + trunk 組み立ては CPU のみの `--info` で実測 31-36 秒 (RSS 約 9.5 GB)。expert の SF4 repack と VRAM 転送は未測定 (GPU を使わない制約のため)。おおよそ 20-24 層 x 128 expert x 2.65M weight の dequant+quant で、数分かかると推定。
- 実ファイルサイズ: UD-IQ3_XXS 11,663,310,784 B / UD-Q4_K_XL 17,629,760,736 B / AD-IQ3_S 13,340,510,176 B / AD-Q4_K 16,525,559,776 B。

## 対応する量子化型

### ローダが dequant できる型

`ink_gguf_transcode.h` の `choose_sk_code_for_ggml` は「S8_K / S6_K / それ以外は S4_K」を選ぶだけで、ホワイトリストではない (`engine/src/inkling2/ink_gguf_transcode.h:86-113`)。実際の可否は `gguf_dequant_row_f32` -> `strata::quant::can_dequant` で決まる (`engine/src/inkling2/ink_gguf_loader.h:62-80`, `engine/src/quant/dequant.cpp:634-689`)。

対応する型 (F32/F16/BF16 は別扱い):
Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, MXFP4, NVFP4, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, IQ4_NL, IQ4_XS。

注意:
- Q8_1 (9) と Q8_K (15) は activation 用で、weight の dequant 対象ではない (`can_dequant` に含まれない)。`choose_sk_code_for_ggml` には名前が出るが、expert weight には現れない。
- Q1_0 (41) と TQ* (144-149) は `can_dequant` に無い。これらの型を weight に使う GGUF は dense で "unsupported tensor type"、expert で "unsupported ggml_type" になり読めない。今回の 4 ファイルには含まれない。
- `can_dequant` に含まれる型でも、CUDA のネイティブ MoE カーネルが無い型は SF4 へ再量子化される (次節)。

### CUDA がネイティブに読める expert 型

`ink_gguf_has_gpu_kernel` が返すのは次の ggml_type だけ (`engine/src/inkling2/cuda/ink_moe_cuda.cuh:410-417`):
0 F32, 1 F16, 2 Q4_0, 6 Q5_0, 8 Q8_0, 10 Q2_K, 11 Q3_K, 12 Q4_K, 13 Q5_K, 14 Q6_K, 18 IQ3_XXS, 20 IQ4_NL, 23 IQ4_XS, 30 BF16, 39 MXFP4。

これに含まれないが dequant できる型: 3 Q4_1, 7 Q5_1, 16 IQ2_XXS, 17 IQ2_XS, 19 IQ1_S, 21 IQ3_S, 22 IQ2_S, 29 IQ1_M, 40 NVFP4。これらはレイヤ単位で SF4 へ repack される (`engine/src/inkling2/cuda/ink_runner.cuh:1037-1055,1455-1509`)。NVFP4 は gate/up と down の両方が NVFP4 のときだけ bit-exact な SF4 repack になる (`engine/src/inkling2/cuda/ink_runner.cuh:926-942`, `engine/src/inkling2/ink_gguf_loader.h:88-116`)。

### 実ファイルの型 (ヘッダ解析)

GGUF の tensor info (name, dims, ggml_type) を読み、expert の gate_up/down を層ごとに集計した。重み本体は読んでいない。

| ファイル | エンジンが表示する build | dense の型 | expert の gu / dn (層) | P40 での expert 経路 |
| --- | --- | --- | --- | --- |
| UD-IQ3_XXS | `GGUF-IQ2_XXS` | Q8_0, Q4_K, Q5_K, Q6_K, F32, BF16 | 0-19: IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS + NVFP4、20-23: IQ3_XXS / IQ4_NL | 0-19 は SF4 再量子化 (lossy)、20-23 は DP4A |
| UD-Q4_K_XL | `GGUF-NVFP4` | Q8_0, F32, BF16 | 0-19: NVFP4 / NVFP4、20-23: Q8_0 / Q8_0 | 0-19 は bit-exact SF4、20-23 は DP4A |
| AD-IQ3_S | `GGUF-IQ3_S` | Q8_0, F32 | 0-23: IQ3_S / IQ4_NL | 24 層すべて SF4 再量子化 (lossy) |
| AD-Q4_K | `GGUF-Q4_K` | Q8_0, F32 | 0-23: Q4_K / Q5_0 | 24 層すべて DP4A (native) |

要点:
- **UD-Q4_K_XL に Q4_K は 1 個も入っていない。** ファイル名はレシピ名で、expert は NVFP4 (20 層) と Q8_0 (4 層) だけ。Q4_K/Q5_K/Q6_K は UD-IQ3_XXS の dense にだけ現れる。
- **UD-IQ3_XXS の expert は IQ3_XXS 主体ではない。** 0-19 層は IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS の gate/up と NVFP4 の down。この組み合わせは「両方ネイティブ」ではないので、20 層が SF4 へ再量子化される。DP4A に残るのは 20-23 層の 4 層だけ。
- **AD-IQ3_S は全 24 層が SF4 へ再量子化される。** IQ3_S (21) は dequant できるが CUDA ネイティブ kernel が無い。down の IQ4_NL はネイティブだが、gu が IQ3_S なのでレイヤ全体が repack 対象になる。
- **AD-Q4_K だけが全 24 層ネイティブ DP4A。** gate/up が Q4_K (12)、down が Q5_0 (6)。どちらも `ink_gguf_has_gpu_kernel` に入る。

補足: UD-IQ3_XXS の dense にある Q4_K / Q5_K / Q6_K は dequant されて BF16 になり、MoE カーネルは通らない。Q8_0 は dense では BF16、expert では DP4A。MXFP4 は今回の 4 ファイルには含まれない (NVFP4 は別の型)。

### VRAM の実効サイズ (推定)

dense は P40 では BF16 に展開され、SF4 repack は 4.5 bpw になる。ファイル内のバイト数ではなく、エンジンが VRAM に置く実効バイト数。

| ファイル | dense (BF16後) | expert (repack後) | 合計 | 32K FP16 の余裕 |
| --- | ---: | ---: | ---: | --- |
| UD-IQ3_XXS | 3.07 GiB | 12.36 GiB | 15.43 GiB | 約 4-6 GiB (推定) |
| UD-Q4_K_XL | 3.07 GiB | 14.71 GiB | 17.78 GiB | 約 1-3 GiB (かなり厳しい) |
| AD-IQ3_S | 3.01 GiB | 12.81 GiB | 15.83 GiB | 約 4-6 GiB (推定) |
| AD-Q4_K | 3.01 GiB | 13.76 GiB | 16.78 GiB | 約 2-4 GiB (推定) |
| SQ-10 abliterated (.strata) | 1.87 GB (Section 1) | 8.60 GB (Section 4) | 10.47 GB | 大きい |

「余裕」は 24 GiB = 25.77 GB から weights、KV、workspace、DSpark を引いた概算。実際の確保量は起動ログの `[InkCUDA] memory:` 行で確認する。UD-Q4_K_XL は 32K でもぎりぎりで、64K 以上は KV が host へ落ちる可能性が高い。

## sm_61 のカーネル分岐と DP4A

### 能力判定

`engine/include/strata/core/ink_caps.hpp:16-18`:
- `ink_cap_int8_tc(major) = major >= 8` -> sm_61 は false。
- `ink_cap_wmma_f16(major) = major == 7` -> sm_61 は false。
- `ink_cap_sm120` -> false。

`ink_moe_cuda.cuh:3678-3687` で `use_sm120_tc` / `use_fp16_tc` / `use_wmma_tc` がすべて false になる。`ink_has_int8_tc()` は `ink_moe_cuda.cuh:2305-2312`、`ink_has_wmma_f16()` は `:2314-2321`。

### 型別の落ち先

| expert の型 | P40 のカーネル | DP4A | 根拠 |
| --- | --- | --- | --- |
| Q4_K, Q5_K, Q6_K, Q8_0, Q2_K, Q3_K, Q4_0, Q5_0, IQ3_XXS, IQ4_NL, IQ4_XS | `k_ink_moe_gu_gguf_dp4a` / `k_ink_moe_dn_gguf_dp4a` -> `ink_gguf_row_dot_multi` | する (ハードウェア) | `ink_moe_cuda.cuh:864-966,3812-3861` |
| MXFP4, BF16, F16, F32 | 同じ GGUF カーネル内の float 分岐 | しない (FFMA) | `ink_moe_cuda.cuh:821-856` |
| IQ2_XXS/XS/S, IQ1_S/M, IQ3_S, NVFP4, Q4_1, Q5_1 | 一度 SF4 へ repack し `k_ink_moe_gu_sf4_mx8` / `k_ink_moe_dn_sf4_mx8` | しない (LUT デコード + FFMA) | `ink_runner.cuh:1037-1055,1455-1509`, `ink_moe_cuda.cuh:3725-3781` |
| S2_K/S3_K/S4_K/S5_K/S6_K/S8_K (.strata) | `launch_ink_moe_gu_sk_mask` / `launch_ink_moe_dn_sk_mask` | する | `ink_moe_cuda.cuh:3784-3810` |
| SF4/MX8 (.strata) | `k_ink_moe_gu_sf4_mx8` / `k_ink_moe_dn_sf4_mx8` | しない | `ink_moe_cuda.cuh:3725-3781` |
| Q8_0 の tensor core 版 | `k_ink_moe_q80_tc` は sm_61 では起動しない | - | `ink_moe_cuda.cuh:3832,985` |

### IQ3_XXS と Q4_K は P40 で DP4A を使えるか

使える。両方とも `ink_gguf_row_dot_multi` の中で `dp4a_i8` を呼ぶ。
- Q4_K: `ink_moe_cuda.cuh:528-557` (Q4_K 分岐、`dp4a_i8` は 551 行)
- IQ3_XXS: `ink_moe_cuda.cuh:658-690` (IQ3_XXS 分岐、`dp4a_i8` は 684 行)
- `dp4a_i8`: `ink_cuda.cuh:382-394`。`__CUDA_ARCH__ == 610` で `__dp4a((int)a,(int)b,c)`。

ただし、その型のレイヤが GGUF_RAW のまま残っていることが前提。`ink_runner.cuh:1037-1055` は「gu と dn の両方がネイティブ」でないと `need_sf4_repack=1` にする。UD-IQ3_XXS では 20-23 層だけが IQ3_XXS/IQ4_NL の両方ネイティブなので、DP4A は 4 層だけ。UD-Q4_K_XL では Q8_0/Q8_0 の 4 層だけ。AD-Q4_K では 24 層すべて Q4_K/Q5_0 なので、全層 DP4A。

### 速度の判断材料

- ネイティブ DP4A 同士なら、bpw が小さいほど decode の重み帯域が減る。IQ3_XXS (3.06 bpw) は Q4_K (4.5 bpw) より軽い。品質は Q4_K が上。P40 の decode は帯域律速なので、IQ3_XXS の方が速い可能性が高い (未測定)。
- SF4 は 4.5 bpw で DP4A ではない。LUT デコード + FFMA なので、同じバイト数なら DP4A より不利。UD-IQ3_XXS の 20 層、AD-IQ3_S の 24 層はこの経路。
- MXFP4 はネイティブだが float FMA で DP4A ではない。Q4_K より遅い可能性がある。
- Q8_0/Q6_K は DP4A だがバイト数が多く、decode は遅い。品質優先。
- 既知の P40 の DP4A 実測上限は 35.5 TFLOPS (CONTEXT.md)。本番 MMQ は 10-15 TFLOPS。GGUF の DP4A カーネルがこの上限にどこまで近いかは未測定。

## mmproj / vision

- `--vision` は既定で空。テキストのみなら省略でき、VRAM も使わない (`engine/src/inkling2/ink_backend.h:42`, `engine/src/inkling2/ink_serve.h:531-539`)。
- `strata-inkling` の vision は `VisionShard` profile の `.strata` か、HF safetensors の `model.visual.*` だけ。`vision_probe` は `StrataFileReader::open` で magic を確認するので、`mmproj-*.gguf` は "Invalid .strata magic" で失敗する (`engine/src/inkling2/ink_vision.h:100-215`)。`ink_gguf_loader.h:3` のコメントに mmproj の語はあるが、実際の mmproj 読込コードは無い。
- P40 で画像も使うなら `Inkling-2-Mini-Vision.strata` (636,526,592 B) を `--vision` に渡す。SC117 の abliterated 設定もこのファイルを使っている。
- Unsloth の `mmproj-BF16.gguf` / `mmproj-F16.gguf`、AtomicChat の `mmproj-Inkling-2-Mini-Uncensored-BF16/F16.gguf` は、このエンジンでは使えない。テキストベンチでは不要。

## abliterated .strata の互換性

対象: `<models>/inkling2/Inkling-2-Mini-SQ-10-abliterated.strata` (10,471,526,400 B)。

ヘッダと Section 0 manifest を解析した結果:
- magic `STRATA\0\1`、profile 0 (Tiered)、6 sections。Section の offset/length は SQ-10 と完全に同じ。
- `format=strata-v1.1`, `arch=inkling2_loop`, `build=SQ-10`。
- trunk_tensors 646 個。名前の集合は SQ-10 と完全一致 (差分 0)。
- dspark_tensors 54 個。名前の集合も SQ-10 と完全一致。
- text_config の値 (24 層、hidden 2304、inter 1152、24 heads、8 KV、head_dim 128、128 experts、top-8、vocab 201024/200058、loop_config) も同じ。
- chat_template 6868 B、tokenizer_json は空 (外部 tokenizer 前提)。

ローダの互換性:
- `StrataFileReader::open` は magic、num_sections、4 KiB alignment、ZSTD manifest だけを確認する。profile やモデル名のホワイトリストは無い (`engine/src/artifact/strata_reader.cpp:33-111`)。
- `parse_manifest` は `config.text_config` と `trunk_tensors` を要求する。すべて存在する (`engine/src/inkling2/ink_manifest.h:217-337`)。
- CUDA 側の `cuda/ink_manifest.hpp:122-213` も同じ前提で、`ink_runner.cuh:11` から使われる。`text_config` が無い場合は `config` 直下にフォールバックする。
- `--info` で実際に開けた: `arch=inkling2_loop format=strata build=SQ-10 layers=24 visits=48 hidden=2304 vocab=200058 experts=128 eos=200006 drafter=dspark images=1 image_pad=200054 parallel=64 spec_opts=1 dspark_f32_widen=1`。

expert の code_id 分布 (Section 3 の 3072 entry):
- S3_K/S3_K: 658
- S3_K/S2_K: 1814
- S2_K/S2_K: 585
- SF4/SF4: 15

つまり 3057/3072 = 99.5% が S2_K/S3_K で、P40 の dp4a 経路に乗る。SQ-10 と同一分布。

結論: **同じローダで読める。メタデータの互換性に問題は無い。** 異なるのは重み (と section hash) だけ。`--selfcheck --verify-hash` は実行していないので、Section の SHA-256 検証だけは未確認。

## DSpark drafter

### 指定方法

| モデル | 指定 | 実装 |
| --- | --- | --- |
| `.strata` (SQ-10/15/20, abliterated) | `--draft dspark` または `auto`。drafter は `dspark_tensors` に内蔵。`--dspark PATH` は `.strata` では使われない | `engine/src/inkling2/ink_manifest.h:312-327`, `engine/src/inkling2/ink_manifest.h:475-477` |
| `.gguf` + dspark GGUF | `--dspark /path/dspark-*.gguf --draft dspark` | `engine/src/inkling2/ink_main.cpp:906-907`, `engine/src/inkling2/ink_gguf_loader.h:806-845` |
| `.gguf` + mtp.safetensors | `--dspark /path/mtp.safetensors --draft dspark` | `engine/src/inkling2/ink_gguf_loader.h:773-805` |

`--dspark` を省略すると、モデルと同じディレクトリの `mtp.safetensors` または `*dspark*.gguf` を自動で探す (`engine/src/inkling2/ink_gguf_loader.h:748-771`)。最初に見つかった 1 個だけを使う。

`--draft off` は DSpark を無効化する (`engine/src/inkling2/ink_main.cpp:895-897`)。`--bench` の `use_dspark` は `cfg.bopts.enable_dspark` なので、`--draft off` を付けないと既定で有効になる (`engine/src/inkling2/ink_main.cpp:131-139,198`)。

### 現状の落とし穴

現在ダウンロードされている `dspark-Inkling-2-Mini-Q8_0.gguf` (503,889,952 B) は、2026-11-23 に Unsloth が差し替えた llama.cpp mainline の dflash レイアウト。`fuse_weight` と旧 `markov_head.hidden_out.weight` が無いため、`ink_dspark_layout_check` が拒否する (`engine/include/strata/core/ink_dspark_layout.hpp:3-8,19-38`)。`--info` でも `drafter=none` になり、次の警告が出る:

```
strata-inkling: the drafter file ... is in a layout this engine cannot read yet
(missing 'fuse_weight' (the llama.cpp mainline dflash layout: ctx_mix / markov_w2));
drafting is off for this start and the model starts normally.
```

読める drafter:
- Unsloth `dspark-Inkling-2-Mini-Q8_0_Old.gguf` 402,278,560 B (旧 padded-vocab レイアウト)。未取得。
- 元の `mtp.safetensors` 636,526,592 B (Thinking Machines)。未取得。

現状で GGUF を回すときは `--draft off` を付ける。さらに、自動発見で 503 MB を読ませないために `--dspark /dev/null` を付けると、`has_gguf_magic` が false になって drafter 無しとして扱われ、警告も出ない (CPU の `--info` で確認済み)。`/dev/null` は「drafter を指定しない」ための回避策で、モデル本体には影響しない。

`.strata` の abliterated は `dspark_tensors` を 54 個内蔵しているので、`--draft dspark` だけで動く (`--info` で `drafter=dspark` を確認)。

## tokenizer / chat template / config

- 3 つの GGUF (UD-IQ3_XXS, UD-Q4_K_XL, AD-IQ3_S) はすべて、`tokenizer.ggml.tokens` 200,058 個、`tokenizer.ggml.merges` 424,413 個、`tokenizer.chat_template` 6,868 B を埋め込んでいる (ヘッダ解析)。`tokenizer.ggml.model=gpt2`, `tokenizer.ggml.pre=inkling2`。
- chat template は `keep_tokenizer=false` でも `tokenizer.chat_template` から読まれる (`engine/src/inkling2/ink_gguf_loader.h:977-979`)。
- tokenizer 全体は `--export-tokenizer PATH OUT_DIR` で取り出せる。`chat_template.jinja` と `tokenizer.hf.json` を書く (`engine/src/inkling2/ink_main.cpp:33-58`)。GGUF の 200K tokens / 424K merges を再スキャンして HF tokenizer.json を合成する (`engine/src/inkling2/ink_gguf_loader.h:142-266`)。tokens が空なら空のまま true を返す (`:223`)。
- `config.json` は不要。アーキテクチャ設定は GGUF metadata から読む (`engine/src/inkling2/ink_gguf_loader.h:351-428`)。
- `strata-inkling` の直接生成と `--serve` は token ID をやり取りするので、tokenizer は不要 (`engine/src/inkling2/ink_serve.h:320-382`)。HTTP の `strata serve` は tokenizer ディレクトリを必要とする。
- 既に `<models>/inkling2/` と `<models>/inkling2/uncensored/` の両方に `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `special_tokens_map.json` がある。`strata serve` の `tokenizer` にはこのディレクトリを指定できる。abliterated は SC117 の `strata-tokenizer/` も使える。
- `--export-tokenizer` の出力名は `tokenizer.hf.json` なので、`strata serve` が `tokenizer.json` を期待する場合はリネームが必要 (未確認)。

## 実行コマンド

前提:
- GPU が空いていること。本番コンテナ `strata` は GPU を約 23.8 GB 使うので、本番を停止してから実行する。停止・復帰は GPU ロックの中で行う。
- バイナリは sm_61 ビルド: `~/work/build-base/build/src/inkling2/strata-inkling`。`CMakeCache.txt` は `CMAKE_CUDA_ARCHITECTURES=61-real`, `STRATA_ENABLE_CUDA=ON`, `STRATA_EXPERIMENTAL_SM60=ON`。
- `LD_LIBRARY_PATH=~/work/bin/lib-sm61` が必須。無いと `libcudart.so.12: cannot open shared object file` で落ちる。
- 下のコマンドは GPU をまだ実行していない。コード経路から構成したもので、起動・bench の実測は未確認。
- `--dspark /dev/null` は、現行の非対応 503 MB drafter を自動発見で読ませないための指定。DSpark を使わないので `--draft off` も付ける。

### UD-IQ3_XXS を P40 で動かす 1 コマンド

```bash
LD_LIBRARY_PATH=~/work/bin/lib-sm61 \
  ~/work/build-base/build/src/inkling2/strata-inkling \
  --bench --native <models>/inkling2/Inkling-2-Mini-UD-IQ3_XXS.gguf \
  --gpu 0 --max-context 32768 --parallel 1 --kv fp16 \
  --dspark /dev/null --draft off \
  --prompt-len 512 --gen 64
```

これは 20 層を SF4 へ再量子化する (lossy)。起動に数分かかる可能性がある。`--bench` の結果は `[bench] backend=cuda format=gguf ... prefill=... ppts ... decode=... tps agg ...` の 1 行 (`engine/src/inkling2/ink_main.cpp:248-250`)。

### 無検閲版 (AD-IQ3_S) を動かす 1 コマンド

```bash
LD_LIBRARY_PATH=~/work/bin/lib-sm61 \
  ~/work/build-base/build/src/inkling2/strata-inkling \
  --bench --native <models>/inkling2/uncensored/Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf \
  --gpu 0 --max-context 32768 --parallel 1 --kv fp16 \
  --dspark /dev/null --draft off \
  --prompt-len 512 --gen 64
```

これは 24 層すべてを SF4 へ再量子化する (lossy)。P40 では DP4A が 1 層も無いので、同じ 24 GB カードでは AD-Q4_K より遅いと推定する (未測定)。

### P40 の無検閲で推奨: AD-Q4_K

```bash
LD_LIBRARY_PATH=~/work/bin/lib-sm61 \
  ~/work/build-base/build/src/inkling2/strata-inkling \
  --bench --native <models>/inkling2/uncensored/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf \
  --gpu 0 --max-context 32768 --parallel 1 --kv fp16 \
  --dspark /dev/null --draft off \
  --prompt-len 512 --gen 64
```

24 層すべて Q4_K/Q5_0 のネイティブ DP4A。expert の再量子化なし (dense の Q8_0 は BF16 へ展開)。weights は 16.78 GiB で、32K はぎりぎりだが収まる見込み。

### 速度優先の無検閲: abliterated SQ-10 .strata

```bash
LD_LIBRARY_PATH=~/work/bin/lib-sm61 \
  ~/work/build-base/build/src/inkling2/strata-inkling \
  --bench --native <models>/inkling2/Inkling-2-Mini-SQ-10-abliterated.strata \
  --gpu 0 --max-context 32768 --parallel 1 --kv fp16 \
  --draft dspark \
  --prompt-len 512 --gen 64
```

expert の 99.5% が S2_K/S3_K で DP4A。DSpark 内蔵。10.47 GB で余裕がある。品質は SQ-10 相当で、AD-Q4_K より低い可能性がある (未測定)。

### 品質優先: UD-Q4_K_XL

```bash
LD_LIBRARY_PATH=~/work/bin/lib-sm61 \
  ~/work/build-base/build/src/inkling2/strata-inkling \
  --bench --native <models>/inkling2/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --gpu 0 --max-context 32768 --parallel 1 --kv fp16 \
  --dspark /dev/null --draft off \
  --prompt-len 512 --gen 64
```

expert は 20 層が bit-exact SF4、4 層が Q8_0 の DP4A。再量子化による品質劣化は無い (dense の Q8_0 -> BF16 を除く)。ただし weights 17.78 GiB で 24 GB ではかなり厳しい。OOM または KV の host 落ちが起きたら `--kv fp8` か `--max-context 16384` に下げる。

### 事前確認 (GPU を使わない)

```bash
LD_LIBRARY_PATH=~/work/bin/lib-sm61 \
  ~/work/build-base/build/src/inkling2/strata-inkling \
  --info <models>/inkling2/Inkling-2-Mini-UD-IQ3_XXS.gguf
```

`format=gguf build=GGUF-IQ2_XXS ... drafter=none` のように出れば、GGUF の metadata と dense trunk の組み立ては通っている。この `--info` は CPU だけで動く (実測 31-36 秒、RSS 約 9.5 GB)。

## リスクと未確認

- **GPU 実行は未確認。** この調査では GPU を使っていない。コマンドはコード経路から構成した。`init` の所要時間、SF4 repack の時間、VRAM の実測、`[bench]` の数値は未測定。
- **DSpark は現状の GGUF では使えない。** 503 MB の新レイアウトは拒否される。`_Old` (402 MB) か `mtp.safetensors` (636 MB) が必要で、どちらも未取得。
- **UD-IQ3_XXS と AD-IQ3_S は expert が lossy に再量子化される。** ファイル本来の量子化品質とは異なる。UD-Q4_K_XL と AD-Q4_K は expert の再量子化なし (どちらも dense の Q8_0 は BF16 へ展開)。
- **UD-Q4_K_XL は 24 GB で余裕が少ない。** 32K でも KV/workspace の確保次第で OOM または host spill になる。起動ログの `[InkCUDA] memory:` を確認する。
- **本番コンテナを止めずに実行しない。** 本番は GPU を約 23.8 GB 使う。`nvidia-smi` で空きを確認し、CONTEXT.md の GPU ロック手順に従う。
- **`strata serve` で GGUF を配信する経路は未試行。** `strata-inkling --serve` は token ID プロトコルなので tokenizer 不要。HTTP の `strata serve` は tokenizer ディレクトリが必要で、両モデル dir に既にある。
- **mmproj の拒否はコード上の判断。** 実際の `mmproj-*.gguf` はダウンロードしていない。`vision_probe` が GGUF を受けないことは `ink_vision.h:210-215` で確認した。
- **abliterated .strata の Section SHA-256 は未検証。** ファイルサイズと HF の LFS oid は一致しているが、`--selfcheck --verify-hash` は実行していない。
- **`--info` の RSS 9.5 GB は dense trunk の組み立てを含む。** 実際の CUDA 起動では expert の host arena も加わる。RAM 88 GB は足りる見込みだが、本番と同時に走らせない。

## 次の提案

1. GPU ロックを取って、上の 5 コマンド (UD-IQ3_XXS / AD-IQ3_S / AD-Q4_K / abliterated SQ-10 / UD-Q4_K_XL) を 1 回ずつ実行し、`[bench]` と `nvidia-smi` を記録する。まずは AD-Q4_K と abliterated SQ-10 の比較が P40 の実用判断に効く。
2. `UD-Q4_K_M` (16,610,544,608 B) / `UD-Q3_K_XL` (13,631,779,296 B) / `UD-IQ4_XS` (15,733,756,512 B) の GGUF ヘッダを同じ方法で調べ、全 24 層がネイティブ DP4A になるファイルを探す。今回の 4 ファイルはすべて SF4 層を含む。
3. `dspark-Inkling-2-Mini-Q8_0_Old.gguf` (402 MB) または `mtp.safetensors` (636 MB) を取得し、GGUF + DSpark の速度を測る。503 MB の新レイアウトは使わない。
4. 画像も使うなら `Inkling-2-Mini-Vision.strata` を `--vision` に渡して確認する。mmproj GGUF は使えない。
5. 品質ゲートは `--selfcheck --verify-hash`、`--kld-eval`、`--prompt-ids` の再現性で取る。P40 の実測はまだ無い。