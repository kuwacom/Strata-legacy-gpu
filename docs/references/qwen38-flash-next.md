# Qwen3.8-Flash-Next / GDN / QSA / hyper-connection の公開情報と Strata の対応

取得日: 2026-11-07。

## 1. モデルの公開情報

### 1.1 論文

- 出典: Zihan Qiu et al., "On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability", arXiv:2608.30320, 2026-08-31, https://arxiv.org/abs/2608.30320 （2026-11-07 取得）
- 要点（abstract より）:
  - 125B パラメータ、**6B activated per token**、加えて **51B の n-gram embedding table を accelerator の外に置く**。
  - token mixing は **Gated DeltaNet (GDN) と global attention の層単位ハイブリッド**で、**4 層に 1 層が full attention**。continued-pretraining 時にその full attention 層を **Qwen Sparse Attention (QSA)** に置換。QSA は「micro-block 粒度で context を score し、圧縮された軽量 indexer を使う」。
  - residual stream を **4 本に広げ、elementwise gate で読む**。これを **Gated Residual (GR)** と呼ぶ。
  - 397B-A17B の前世代に対して 14 benchmark 中 8 で上回り、残りも最大 2.6 点差。activated 1/3、training token 1/3、FLOPs 約 1/9。

### 1.2 NVIDIA NeMo AutoModel のモデルカード（実装仕様）

- 出典: NVIDIA, *Qwen3.8-Flash-Next — NeMo AutoModel*, https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/qwen/Qwen3.8-Flash-Next （2026-11-07 取得）
- 仕様:

| 項目 | 値 |
|---|---|
| decoder | **48 層: 36 GatedDeltaNet + 12 QSA**（3 GDN の後に 1 QSA を繰り返す） |
| experts | **512 routed experts, top-10 routing**, 各層に shared expert 1 個 |
| hyper-connection | **4 本の persistent stream（各 2,560 幅）**、attention 用と MoE 用に別々の learnable read/write |
| Engram | **51.2B パラメータの n-gram embedding table**（checkpoint 設定では `ple`）、decoder layer 2 に入る。320,001,536 行 × 160 幅、128 shard |
| context | 262,144 native |
| MTP head | checkpoint の 4B MTP head（NeMo は学習対象外） |
| QSA | index key を **4 個ずつ圧縮**し、**最大 512 causal group（= 2,048 token）**を選び、未完成の causal tail を足し、選ばれた token ID だけで main GQA を評価 |
| 最終 HC read | 4 本 × 2,560 を 1 本の 2,560 に collapse してから untied LM head |

### 1.3 Qwen 公式ブログ / HuggingFace

- 出典: Qwen, "Qwen3.8-Flash-Next: A New Architecture, Towards Ultimate Cost Efficiency", https://qwen.ai/blog?id=qwen3.8-flash-next （2026-11-07 取得）
- 出典: HuggingFace, https://huggingface.co/Qwen/Qwen3.8-Flash-Next （2026-11-07 取得）: "The Gated DeltaNet and Gated Attention pairing has been reworked into Gated DeltaNet and Qwen Sparse Attention (QSA). QSA operates at the micro-block level."

## 2. Strata 側の対応（実装の事実）

`docs/GDN_CONTRACT.md`, `docs/QWEN36.md`, `src/core/verify.cpp` より:

### 2.1 GDN（36 層）

- shapes: `S = 128`（head size）、`HK = 16` key head、`HV = 48` value head（VPK=3）、`C = 2*HK*S + HV*S = 10240` channel/token。
- 状態: `state[S][HV][S]` = 128×48×128 = **786,432 float = 3.15 MB / 層**、`history[C][3]` = 10240×3 float = 123 KB。
- decode の流れ: `q8+qkv/q-idx gemv` → `conv` → `ab` → `z` → `rec`（recurrence）→ `out-proj`。
- P40 実測（window あたり、36 層合計）: qkv gemv 4.24 ms、conv 0.20、ab 0.36、z 2.50、**rec 0.91**、out-proj 2.79。

**重要な帰結**: recurrence は 36 層 × 3.15 MB の state を read+write する = **約 226 MB / window**。0.91 ms で割ると **約 248 GB/s**。
つまり **GDN recurrence は既に帯域の上限に達しており、P40 ではこれ以上速くできない**。llama.cpp の #27980 が「qwen4exp が Pascal で 18 tok/s しか出ない、原因は ssm_scan/ssm_conv 疑い」としているのは、この state の帯域コストが理由だと説明できる（Strata は同じ構造を 45-50 tok/s で回している）。

→ **GDN recurrence の最適化は候補にしない**（既に floor）。`STRATA_GDN_CHUNKED` / `STRATA_GDN_TC` は sm_80+ 前提で、P40 では意味がない（93.7 KB / 122 KB の shared が必要で 48 KB の壁を越える）。

### 2.2 QSA（12 層）

- decode の流れ: `q8+kv-idx` → `k/v+norm-rope` → `kv+idx append` → `q+q-idx` → `scores+topk` → `kv-resolve` → `attention` → `gate` → `out-proj`。
- P40 実測（window あたり、12 層合計）: q+q-idx 1.83 ms、**attention 0.57 ms**、out-proj 0.98 ms、scores+topk 0.11 ms。
- decode の attention は window の 1.2% しかない。**prefill の attention は 1,860 ms / 16,005 ms（11.6%）** で大きい。
- QSA select は「512 group × 4 token = 2,048 token」を選ぶ（`qsa_select.cu`）。262K context でも attention の計算量は 2,048 token 分に固定される。

→ **prefill の attention が候補**（`qsa_pattn` の sm_61 適用）。decode の attention は対象外。

### 2.3 hyper-connection（GR）

- 4 本 × 2,560 = 10,240 次元の residual。`hc_lr = 320` の低ランク down/up で read する。
- 1 層に **2 回**（attn 用と ffn 用）read があり、それぞれ norm（4×2560 の RMS）+ down（320×10240）+ up（10240×320）+ inject。
- P40 実測: hc0 norm 0.65 / down 1.43 / up 1.30、hc1 norm 0.33 / down 1.40 / up 1.38（36 層合計）。QSA 層は 0.10/0.48/0.43 と 0.11/0.47/0.46（12 層合計）。
- 合計 **8.54 ms（window の 17.4%）**。重みは bf16 で 320×10240×2 = 6.55 MB × 4 回/層 = 26 MB/層 × 48 = 1.26 GB/window → 227 GB/s で 5.5 ms。実測 8.54 ms = **148 GB/s（65%）**。
- したがって **GR read には 2.5-3 ms の余地がある**（推定）。gfx1030 の persistent 版は read 1.35-2.10x で、まさにこの余地を突いている。

### 2.4 Engram / PLE（n-gram embedding）

- 51.2B の table を host から読む（`--ple-io direct/ram/mmap`）。P40 実測: prefill の `ple` は 449 ms / 16,005 ms（2.8%）、559-token prompt では 53 ms / 2,469（2.1%）。
- decode では `ple` は layer 2 で 1 回だけ（`ple_on` は l == 1）。P40 の window stage table には現れない（小さい）。
- 既存の最適化: `STRATA_PLE_BATCH`, `--ple-io`, `prefetch_rows`。新規の余地は小さい。

## 3. 外部の Pascal 実測（比較用）

- llama.cpp #27980（2026-08-29, https://github.com/ggml-org/llama.cpp/issues/27980 ）: 4× P40, Qwen3.8-Flash-Next UD-Q3_K_XL (83.8 GiB) で decode 18.0-18.1 tok/s、prefill 338-376 tok/s。
- Strata の P40 実測（1 枚、IQ3_S、`2026-10-12-p40-sm61-kernels`）: decode 35-38 tok/s（プロファイル時）、prefill 422-428 tok/s at 7K、256 tok/s at 565 token。
- つまり Strata は同じクラスの GPU で llama.cpp の約 2 倍の decode を出している。**Strata の GDN / expert 配置 / graph 化が効いている**。
- 依頼文の「decode 45-50 tok/s」はさらに新しいビルド（0.1.52 / 0.2.7 系）の数字で、上のプロファイル（0.1.43.1）より速い。**同一条件の比較ではない**点に注意。

## 4. モデル固有の候補への含意

| 対象 | P40 での判断 |
|---|---|
| GDN recurrence | 既に帯域 floor（248 GB/s）。触らない |
| GDN の dense 射影（qkv/z/out） | 帯域の 60-90%。GEMV カーネル側の改善余地（pascal-gemv.md） |
| QSA decode attention | 1.2%。優先度低 |
| QSA prefill attention | 11.6%。`qsa_pattn` の sm_61 適用が候補 |
| GR read | 17.4% で 65% の帯域。最大の単一候補（persistent / norm+down 融合） |
| Engram/PLE | 2-3%。優先度低 |
| shared expert | 6%。融合候補 |
| router | 4%。レイテンシ律速。融合候補 |
