// src/prefill/dense_i8.cu - see include/strata/prefill/dense_i8.hpp.
//
// なぜ: P40 の dense 射影は llama.cpp の MMQ (I=64 行 x J=64 トークン、256 スレッド = 16 積/スレッド) で DP4A の
// 20% 強しか出ていない。理由は 2 つで、(1) 重みタイルの int8 展開は J タイル (64 トークン) でしか償却されない、
// (2) 1 スレッドの積が 4x4 しかなく、共有メモリの読み出しが 1 dp4a あたり 0.5 回 (8x8 のタイルなら 0.25 回) と
// 多い。ここでは重みを per-32 スケールの int8 に呼び出しごと 1 度だけ落とし、8 行 x 4 トークン/スレッドの素の
// DP4A GEMM で掛ける。展開のコストは「重み 14 MB 読み + int8 26 MB 書き」x 1 回で、GEMM 本体に比べれば無視できる。
//
// 数値: 重みは IQ4_XS などの 4 bit の値を int8 (127 段) に取り直すので、元のブロックの値そのものではない
// (量子化誤差は元の 4 bit の 3% 程度)。活性は MMQ と同じ「32 値ごとの最大絶対値 / 127」で丸める。積和は int32 で
// 厳密に取り、32 値ごとに fp32 のスケールを畳む (bitwise ではない。品質は needle/QA で見る)。
#include "strata/prefill/dense_i8.hpp"

#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#if !defined(__HIPCC__)

namespace strata::prefill::dense_i8 {
namespace {

// タイル: 1 ブロックが重み 128 行 x 64 トークン。256 スレッドが 8 行 x 4 トークンを持つ。
constexpr int kRows = 128;
constexpr int kTokens = 64;
constexpr int kThreads = 256;
constexpr int kChunk = 64;              // 共有メモリに 1 度に載せる K の値数
constexpr int kWords = kChunk / 4;      // 16 ワード = 1 行
constexpr int kStride = kWords + 1;     // 17 ワード (奇数): 行を 16 おきに持つスレッドの読みが bank で重ならない
constexpr int kGroups = kChunk / 32;    // 1 行に 2 つの per-32 スケール
constexpr int kPerRow = kWords / kGroups;   // 8 ワード = 32 値

// 活性 (と重み) の量子化: 1 warp が 32 値を 1 グループとして、最大絶対値 / 127 のスケールで int8 に丸める。
// float の活性を受け取らないのは、この経路の入力が FP16 だから (f16 -> f32 は厳密なので値は同じ)。
__global__ void __launch_bounds__(128) quantize_kernel(const __half* __restrict__ x, int64_t ld, int8_t* __restrict__ q,
                                                       float* __restrict__ d, int cols, int64_t rows) {
    const int lane = (int) threadIdx.x & 31;
    const int warp = (int) threadIdx.x >> 5;
    const int64_t row = blockIdx.y;
    const int g = (int) blockIdx.x * 4 + warp;   // グループ (32 値) の番号
    if (row >= rows || g * 32 >= cols) return;
    const int64_t base = row * ld + (int64_t) g * 32;
    const float v = __half2float(x[base + lane]);
    float m = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    // 全部 0 のグループはスケール 0 (q * 0 = 0)。1/0 を踏まないよう分ける
    const float s = m > 0.0f ? m / 127.0f : 0.0f;
    const float inv = m > 0.0f ? 127.0f / m : 0.0f;
    int qv = __float2int_rn(v * inv);
    qv = max(-127, min(127, qv));
    q[base + lane] = (int8_t) qv;
    if (lane == 0) d[row * (cols / 32) + g] = s;
}

__global__ void __launch_bounds__(kThreads) gemm_kernel(const int8_t* __restrict__ wq, const float* __restrict__ wd,
                                                        const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                        float* __restrict__ Y, int N, int T, int K, int ldy) {
    __shared__ int sw[kRows * kStride];
    __shared__ int sx[kTokens * kStride];
    __shared__ float sws[kRows * kGroups];
    __shared__ float sxs[kTokens * kGroups];
    const int tid = (int) threadIdx.x;
    // スレッド (rg, tg): 行 {rg + 16 i}、トークン {tg + 16 j}。行を 16 おきに持つので、行ストライドを奇数
    // ワードにしておくと 16 レーンが 16 個の別の bank を踏む (連続した 8 行を持つと 4-way になる)
    const int rg = tid & 15;
    const int tg = tid >> 4;
    const int row0 = (int) blockIdx.x * kRows;
    const int tok0 = (int) blockIdx.y * kTokens;
    const int ng = K / 32;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += kChunk) {
        // 重みタイルと活性タイルを共有メモリへ。行の中で 16 スレッドが 16 ワードを読むので、1 行 64 B が
        // まとまった転送になる
        for (int idx = tid; idx < kRows * kWords; idx += kThreads) {
            const int r = idx >> 4;
            const int c = idx & 15;
            const int row = row0 + r;
            sw[r * kStride + c] = row < N ? ((const int*) (wq + (int64_t) row * K + k0))[c] : 0;
        }
        for (int idx = tid; idx < kTokens * kWords; idx += kThreads) {
            const int r = idx >> 4;
            const int c = idx & 15;
            const int tok = tok0 + r;
            sx[r * kStride + c] = tok < T ? ((const int*) (xq + (int64_t) tok * K + k0))[c] : 0;
        }
        if (tid < kRows * kGroups) {
            const int r = tid / kGroups, g = tid % kGroups;
            const int row = row0 + r;
            sws[r * kGroups + g] = row < N ? wd[(int64_t) row * ng + k0 / 32 + g] : 0.0f;
        }
        if (tid < kTokens * kGroups) {
            const int r = tid / kGroups, g = tid % kGroups;
            const int tok = tok0 + r;
            sxs[r * kGroups + g] = tok < T ? xd[(int64_t) tok * ng + k0 / 32 + g] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int g = 0; g < kGroups; ++g) {
            int sum[8][4];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) sum[i][j] = 0;
            #pragma unroll
            for (int kk = 0; kk < kPerRow; ++kk) {
                const int off = g * kPerRow + kk;
                int wv[8], xv[4];
                #pragma unroll
                for (int i = 0; i < 8; ++i) wv[i] = sw[(rg + 16 * i) * kStride + off];
                #pragma unroll
                for (int j = 0; j < 4; ++j) xv[j] = sx[(tg + 16 * j) * kStride + off];
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) sum[i][j] = STRATA_DP4A(wv[i], xv[j], sum[i][j]);
            }
            // 32 値ぶんの厳密な int32 の和に、重みと活性の per-32 スケールを掛けて fp32 へ畳む
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float wi = sws[(rg + 16 * i) * kGroups + g];
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[i][j] = fmaf((float) sum[i][j], wi * sxs[(tg + 16 * j) * kGroups + g], acc[i][j]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + rg + 16 * i;
        if (row >= N) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int tok = tok0 + tg + 16 * j;
            if (tok < T) Y[(int64_t) tok * ldy + row] = acc[i][j];
        }
    }
}

// 展開と量子化の中間 (FP16) に使う上限。これを超える行列は行スライスで回す (エンジンの最大の dense 射影は
// attn_q の 12288 x 2560 = 63 MB なので 1 スライスで済む)
constexpr int64_t kF16Cap = 72ll << 20;
// 活性の量子化は 1024 トークンずつ (int8 2.6 MB + スケール 0.3 MB を K=2560 で)
constexpr int64_t kTokChunk = 1024;

void* g_wq = nullptr;
int64_t g_wq_cap = 0;
void* g_wd = nullptr;
int64_t g_wd_cap = 0;
void* g_w16 = nullptr;
int64_t g_w16_cap = 0;
void* g_xq = nullptr;
int64_t g_xq_cap = 0;
void* g_xd = nullptr;
int64_t g_xd_cap = 0;
bool g_failed = false;

/// バッファを必要バイト数まで grow する (前のカーネルがまだ読んでいる場合に備えて同期してから解放する)。
bool grow(void** p, int64_t* cap, int64_t bytes, cudaStream_t s) {
    if (*cap >= bytes) return true;
    if (*p != nullptr) {
        (void) cudaStreamSynchronize(s);
        (void) cudaFree(*p);
        *p = nullptr;
        *cap = 0;
    }
    if (cudaMalloc(p, (size_t) bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        *p = nullptr;
        return false;
    }
    *cap = bytes;
    return true;
}

}  // namespace

bool built() { return true; }

bool enabled() {
    // 毎回読む: 計測用のバイナリが同じプロセスで 2 経路を切り替える (エンジンでは射影 1 回につき 1 回)
    const char* e = std::getenv("STRATA_DENSE_I8");
    return e != nullptr && e[0] == '1';
}

bool fits(int ggml_type, int64_t N, int64_t K) {
    if (ggml_type < 0 || N <= 0 || K <= 0 || N > (1 << 20) || K > (1 << 16)) return false;
    if (K % 128 != 0) return false;           // 32 値スケールだけでなく 128 値単位の行で読む
    if (K * N > (int64_t) 1 << 26) return false;   // カーネル内は int の添字。重みの int8 像を 64 MB までにする
    // (2026-11-11, docs/prefill/dense-int8.md) この GEMM は形によらず一定の速さ (約 3.1 TMAC/s) で、本番の重み
    // N/K >= 4 の形 (attn_qkv 10240x2560, attn_q 12288x2560) でも llama.cpp の MMQ の 1/2.4 しか出ない。
    // 128K の実測は gdn +82.9%、qsa proj +65.2%、timeline +16.2% だったので、この門のまま既定オフで残す
    // (測った負けを消さないための記録)。MMQ が unpack を J=64 トークンで償却して 10-15 TFLOPS 出しており、
    // この経路の 6.1 TFLOPS (DP4A の実測上限 35 TFLOPS の 17%) では届かない
    if (N < 4 * K || N < 8192) return false;
    // 重みを FP16 に開くのは dequant_f16 なので、その対応表にある型だけを受ける
    return strata::kernels::dequant_bf16_supported(ggml_type);
}

bool run(const uint16_t* X, int ggml_type, const void* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
         void* stream) {
    if (!fits(ggml_type, N, K) || T <= 0 || X == nullptr || W == nullptr || Y == nullptr) return false;
    if (g_failed) return false;
    cudaStream_t s = (cudaStream_t) stream;
    if (ldy <= 0) ldy = N;

    // 1. 重みの int8 展開 (呼び出しごとに 1 回)。dequant_f16 で FP16 に開いてから per-32 で量子化する
    const int64_t wn = N * K;
    if (!grow(&g_wq, &g_wq_cap, wn, s) || !grow(&g_wd, &g_wd_cap, N * (K / 32) * 4, s)) { g_failed = true; return false; }
    const int64_t slice = std::max<int64_t>(1, kF16Cap / (K * 2));
    if (!grow(&g_w16, &g_w16_cap, std::min<int64_t>(N, slice) * K * 2, s)) { g_failed = true; return false; }
    for (int64_t r0 = 0; r0 < N; r0 += slice) {
        const int64_t rows = std::min<int64_t>(slice, N - r0);
        strata::kernels::dequant_f16(ggml_type, W, r0, rows, K, (uint16_t*) g_w16, stream);
        const dim3 grid((unsigned) ((K / 32 + 3) / 4), (unsigned) rows);
        quantize_kernel<<<grid, 128, 0, s>>>((const __half*) g_w16, K, (int8_t*) g_wq + r0 * K,
                                            (float*) g_wd + r0 * (K / 32), (int) K, rows);
        if (cudaGetLastError() != cudaSuccess) { g_failed = true; return false; }
    }

    // 2. 活性を量子化しながら GEMM。1 起動のブロック数が 30 SM を十分に埋めるよう 1024 トークンずつ
    const int64_t tc = std::min<int64_t>(T, kTokChunk);
    if (!grow(&g_xq, &g_xq_cap, tc * K, s) || !grow(&g_xd, &g_xd_cap, tc * (K / 32) * 4, s)) { g_failed = true; return false; }
    for (int64_t t0 = 0; t0 < T; t0 += tc) {
        const int64_t rows = std::min<int64_t>(tc, T - t0);
        const dim3 qgrid((unsigned) ((K / 32 + 3) / 4), (unsigned) rows);
        quantize_kernel<<<qgrid, 128, 0, s>>>((const __half*) (X + t0 * K), K, (int8_t*) g_xq, (float*) g_xd, (int) K, rows);
        if (cudaGetLastError() != cudaSuccess) { g_failed = true; return false; }
        const dim3 grid((unsigned) ((N + kRows - 1) / kRows), (unsigned) ((rows + kTokens - 1) / kTokens));
        gemm_kernel<<<grid, kThreads, 0, s>>>((const int8_t*) g_wq, (const float*) g_wd, (const int8_t*) g_xq,
                                             (const float*) g_xd, Y + t0 * ldy, (int) N, (int) rows, (int) K, (int) ldy);
        if (cudaGetLastError() != cudaSuccess) { g_failed = true; return false; }
    }
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "prefill gemm: the dense projections' weights expand to int8 once per call and multiply "
                             "in a plain DP4A GEMM (STRATA_DENSE_I8=1)\n");
    }
    return true;
}

}  // namespace strata::prefill::dense_i8

#else  // __HIPCC__

namespace strata::prefill::dense_i8 {
bool built() { return false; }
bool enabled() { return false; }
bool fits(int, int64_t, int64_t) { return false; }
bool run(const uint16_t*, int, const void*, float*, int64_t, int64_t, int64_t, int64_t, void*) { return false; }
}  // namespace strata::prefill::dense_i8

#endif
