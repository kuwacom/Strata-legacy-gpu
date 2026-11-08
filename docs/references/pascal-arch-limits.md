# Pascal (sm_61) の制約 — P40 で使える機能と使えない機能

取得日: 2026-11-07（すべての URL はこの日に取得）。

## 0. 前提の訂正（仕様）

依頼文では P40 を「24GB, HBM2 346GB/s」としているが、**Tesla P40 のメモリは GDDR5 で 346 GB/s**（HBM2 346 GB/s ではない。HBM2 は P100 の 732 GB/s）。
以下の表は公開仕様と NVIDIA の Pascal Tuning Guide に基づく。

| 項目 | P40 (GP102, sm_61) | 出典 |
|---|---|---|
| SM 数 | 30（128 FP32/SM） | Pascal Tuning Guide: "the similar GP102 design provides up to 30 SMs" |
| CUDA core | 3840 FP32 / 240 FP64 | 同上・TechPowerUp |
| FP32 | 12 TFLOPS（boost 時 11.8 実測相当） | NVIDIA P40 datasheet 系 |
| メモリ | 24 GB GDDR5, 384-bit, 346 GB/s | Microway / NVIDIA datasheet |
| L2 | 3 MB（GP102） | GP100 4 MB / GP104 2 MB は一次資料で確認（下記） |
| shared memory / SM | 96 KB（GP104/GP102 系） | Pascal Tuning Guide |
| shared memory / block | **最大 48 KB**（推奨 32 KB 以下） | Pascal Tuning Guide 1.4.5.1 |
| レジスタ | 64K 32-bit / SM、255 / thread | Pascal Tuning Guide 1.4.1.2 |
| warp / SM | 64、block / SM 32 | 同上 |
| FP16 演算 | **FP32 の 1/64**（GP104/GP102） | Pascal Tuning Guide 1.4.2.1, Wikipedia Pascal |
| INT8 dot | `__dp4a` / `__dp2a` あり（FP32 と同スループット） | Pascal Tuning Guide 1.4.2.2 |
| tensor core | なし | — |
| atomic | shared の 32-bit integer atomic が native、FP32 atomicAdd は denormal を 0 に flush | Pascal Tuning Guide 1.4.4 |

出典:
- NVIDIA, *Pascal Tuning Guide* (v13.5), https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html （2026-11-07）
- Wikipedia, *Pascal (microarchitecture)*, https://en.wikipedia.org/wiki/Pascal_(microarchitecture) （2026-11-07）: GP104/GP102 の L2 は 2 MB、GP100 は 4 MB、FP16 は GP104/106/107/108 で FP32 の 1/64
- Microway, *NVIDIA Tesla P40 GPU Accelerator (Pascal GP102) Up Close*, https://www.microway.com/hpc-tech-tips/nvidia-tesla-p40-gpu-accelerator-pascal-gp102-up-close/ （2026-11-07）: 24 GB GDDR5, 346 GB/s

> GP102 の L2 = 3 MB は複数のスペック表（TechPowerUp 等）の一致による。NVIDIA の一次資料で GP102 の L2 を明記したものを今回は取得できなかったため **未確認** と扱う。GP100=4 MB / GP104=2 MB は上記 Wikipedia の表で確認。

## 1. P40 で「使える」機能

| 機能 | sm_61 で使えるか | 条件・注意 |
|---|---|---|
| `__dp4a`（int8 dot） | **使える** | 4 つの int8 の積和を 1 命令で。Strata は `STRATA_DP4A` 経由で使用済み |
| shared memory 48 KB/block | 使える | 96 KB/SM あるので 2 block まで同時に載る。MMQ の tile 幅はここで頭打ち（llama.cpp #30235 で I>=192 がコンパイル不能） |
| LDG（read-only data cache） | 使える | GP104/GP102 は既定で global load を L2 のみにキャッシュ。`__restrict__ const` で LDG.CI になり L1 を使う。Strata の `STRATA_PDL_RESTRICT` は Pascal で `__restrict__` を残す（外すと decode が半減、pdl.hpp のコメント） |
| 共有メモリ atomic | 使える | 32-bit integer は native。FP32 atomicAdd は丸めが round-to-nearest-even、denormal flush |
| `cudaLaunchCooperativeKernel`（grid sync） | **使える** | CC 6.0 以上、Linux（MPS なし）または Windows。P40 は該当 |
| CUDA graph（capture / replay） | 使える | ただし pre-Ampere は repeat launch の CPU コストが **O(node 数)**（後述） |
| 複数 stream / stream priority | 使える | priority は hint。`cudaDeviceGetStreamPriorityRange` が 0..0 を返すカードでは無効（Strata は `STRATA_GRAPH_PRIO` でこの場合何もしない） |
| dynamic parallelism (CDP1) | 使える | sm_35+。ただし device 側 launch は数 µs のコストで、この用途では割に合わない |

## 2. P40 で「使えない」機能（明示）

| 機能 | 理由 | 出典 |
|---|---|---|
| **Programmatic Dependent Launch (PDL)** | **compute capability 9.0 以上が必須**。"Available starting with devices of compute capability 9.0" | CUDA Programming Guide 4.5, https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html （2026-11-07） |
| L2 persistence / `accessPolicyWindow` | **CC 8.0 以上**。"Devices of compute capability 8.0 and above have the capability to influence persistence of data in the L2 cache" | CUDA Programming Guide 4.15, https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html （2026-11-07） |
| `cp.async` / `memcpy_async` の非同期性 | Hopper/Ampere の cp.async は sm_80+。Pascal では `memcpy_async` は同期コピーに落ちる | CUDA Programming Guide 4.4.7（memcpy_async の記述）+ sm_80 以降の機能という一般知識 |
| tensor core（int8/FP16 mma） | Pascal に無い | Pascal Tuning Guide |
| FP16 高速演算 | GP102 は FP32 の 1/64。BF16 も同様に遅い（だから Strata は sm_61 で BF16 射影を FP32 で回す） | Pascal Tuning Guide 1.4.2.1 |
| Triton カーネル（Liger 等） | Triton の対応下限は sm_70（Volta）。Pascal は対象外 | Liger-Kernel README（"Our kernels inherit the full spectrum of hardware compatibility offered by Triton"）と Triton の対応表 |
| Marlin / W4A16 tensor-core GEMV | Ampere 以降の tensor core 前提 | Marlin (IST-DASLab) README |
| `cudaGraphInstantiateFlagUseNodePriority` の効果 | 優先度レンジが 0..0 のカードでは無変化 | `src/core/verify.cpp:1179` |
| CUDA 13 でのビルド | CUDA 13 は Pascal を切った。CUDA 12.9 + `-DSTRATA_EXPERIMENTAL_SM60=ON` | `docs/OLDER_GPUS.md:62-70` |

## 3. Pascal 固有の性能上の罠

1. **FP16 が 1/64**: BF16/FP16 の射影を tensor core なしで回すと FP32 の 64 分の 1。P40 で BF16 射影を FP32 にするのは正しい（`docs/OLDER_GPUS.md:26`）。
2. **global load は既定で L2 のみ**: L1 を使うには LDG（`const __restrict__`）。これを外すと P40 の decode が半減するという実測コメントが `include/strata/kernels/pdl.hpp:44-50` にある（#1469）。
3. **48 KB/block の壁**: shared に stage する tile の幅は 48 KB で決まる。llama.cpp の Pascal DP4A MMQ は `I=128` が上限（`I>=192` はコンパイル不能）。
4. **L2 3 MB はモデルに対して極小**: dense 重み 1 層で数十 MB。L2 に載るのは活性と codebook 程度。したがって「L2 に expert を置く」類の施策は無効。
5. **atomic は global で高コスト**: Hazy Research の megakernel でも B200 で「global atomic の処理に 50 µs」を観測。Pascal の atomic はさらに遅いので、grid barrier を atomic で組むなら block 数を絞る（Strata の GR persist は 80 block / 2 barrier）。
6. **熱**: P40 はファン無し。89 C で `SW Thermal Slowdown: Active`、1531→約 1300 MHz。A/B は同じ温度から始める（`docs/OLDER_GPUS.md:273-279`）。
