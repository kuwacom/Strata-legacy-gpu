// src/prefill/dense_i8_bench.cu - the dense projections by the int8-expanded DP4A GEMM (STRATA_DENSE_I8=1) against the
// MMQ path the engine runs today and against the exact FP16 product of the same dequantized weights: speed (T=4096)
// and error (T=64).  `--peak` instead measures what this card's DP4A can do at all (no memory traffic), which is the
// ceiling every int8 dense projection shares.  GPU, synthetic weights, no model; exit 77 without a CUDA device.
#include "strata/prefill/dense_i8.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
struct QType { int type, elems, bytes, dat; const char* name; };
struct Shape { int qt; int64_t N, K; const char* what; };
// the dense projections of Qwen3.8-Flash-Next (attn_q 12288 x 2560, attn_qkv 10240, attn_gate 6144, ssm_out /
// attn_output 2560 x 6144, the shared expert 640, attn_k / attn_v 512), in the types the pack uses
const QType kTypes[] = {{23, 256, 136, 0, "IQ4_XS"}, {11, 256, 110, 108, "Q3_K"}, {12, 256, 144, 0, "Q4_K"},
                        {42, 64, 18, 0, "Q2_0"}};
const Shape kShapes[] = {{0, 10240, 2560, "attn_qkv"}, {1, 10240, 2560, "attn_qkv"}, {3, 10240, 2560, "attn_qkv"},
                          {2, 10240, 2560, "attn_qkv"}, {0, 6144, 2560, "attn_gate"}, {1, 6144, 2560, "attn_gate"},
                          {0, 12288, 2560, "attn_q"},   {1, 12288, 2560, "attn_q"},   {0, 2560, 6144, "ssm_out"},
                          {1, 2560, 6144, "ssm_out"},   {0, 512, 2560, "attn_k"},    {0, 640, 2560, "shexp_gate"},
                          {0, 2560, 640, "shexp_down"}};
__global__ void fill_f16(__half* p, int64_t n, unsigned seed, float scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = (unsigned) i * 2654435761u + seed;
    x ^= x >> 16; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    p[i] = __float2half(((float) (x & 0xffff) / 32768.0f - 1.0f) * scale);
}
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

// ---- the card's DP4A ceiling, measured with no memory traffic (the same shape of loop as the GEMM's inner body) ----
// A: 16 independent dp4a chains, nothing else - the instruction's own throughput.
__global__ void __launch_bounds__(256) peak_a(int iters, int* sink) {
    const int a = (int) threadIdx.x * 3 + 1, b = (int) blockIdx.x * 5 + 7;
    int acc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = a ^ i;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] = STRATA_DP4A(a, b, acc[i]);
    }
    int s = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) s += acc[i];
    if (s == 0x7ffffffe) *sink = s;
}
// B: the GEMM's fold: 32 dp4a per 32 values, then one int32 -> float * scale add (the per-32 scale of the weight and
// the activation).  8 rows x 4 tokens per thread per 32 values, as in dense_i8.cu.
__global__ void __launch_bounds__(256) peak_b(int iters, int* sink) {
    const int a = (int) threadIdx.x * 3 + 1, b = (int) blockIdx.x * 5 + 7;
    float facc[8][4];
    int sum[8][4];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) { facc[i][j] = 0.0f; sum[i][j] = i + j; }
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) sum[i][j] = STRATA_DP4A(a + i, b + j, sum[i][j]);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) facc[i][j] = fmaf((float) sum[i][j], 1.5e-5f, facc[i][j]);
        }
    }
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) s += facc[i][j];
    if (s == 1.0f) *sink = (int) s;
}
// C: A plus the shared-memory reads of the same body (12 words per 32 dp4a: 8 weight words, 4 activation words).
__global__ void __launch_bounds__(256) peak_c(int iters, int* sink) {
    __shared__ int sm[256 * 17];
    for (int i = (int) threadIdx.x; i < 256 * 17; i += 256) sm[i] = i * 7;
    __syncthreads();
    const int rg = (int) threadIdx.x & 15, tg = (int) threadIdx.x >> 4;
    int acc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            int sum[8][4];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) sum[i][j] = 0;
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                int wv[8], xv[4];
                #pragma unroll
                for (int i = 0; i < 8; ++i) wv[i] = sm[(rg + 16 * i) * 17 + g * 8 + kk];
                #pragma unroll
                for (int j = 0; j < 4; ++j) xv[j] = sm[(tg + 16 * j) * 17 + g * 8 + kk];
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) sum[i][j] = STRATA_DP4A(wv[i], xv[j], sum[i][j]);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[(i * 4 + j) & 15] += sum[i][j];
        }
    }
    int s = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) s += acc[i];
    if (s == 0x7ffffffe) *sink = s;
}

int peak(int blocks, int iters) {
    int* sink = nullptr;
    if (!ok(cudaMalloc((void**) &sink, 4), "sink")) return 1;
    cudaStream_t st = nullptr;
    cudaStreamCreate(&st);
    // dp4a lane-ops per thread and iteration: A 16, B and C 2 groups x 8 kk x 32 pairs = 512
    const int64_t dp_per_thread[3] = {16, 512, 512};
    std::printf("peak: %d blocks x 256 threads, %d iterations\n", blocks, iters);
    const char* names[3] = {"A dp4a only", "B dp4a + per-32 fold", "C dp4a + fold + shared reads"};
    for (int k = 0; k < 3; ++k) {
        cudaEvent_t a = nullptr, b = nullptr;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a, st);
        if (k == 0) peak_a<<<blocks, 256, 0, st>>>(iters, sink);
        else if (k == 1) peak_b<<<blocks, 256, 0, st>>>(iters, sink);
        else peak_c<<<blocks, 256, 0, st>>>(iters, sink);
        cudaEventRecord(b, st);
        if (!ok(cudaStreamSynchronize(st), "peak")) return 1;
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        const double macs = (double) blocks * 256.0 * (double) iters * (double) dp_per_thread[k] * 4.0;
        std::printf("  %-32s %8.2f ms  %6.2f TMAC/s  %6.2f TFLOPS  (%.0f%% of 47 TOPS, %.0f%% of 11.7)\n", names[k], ms,
                    macs / ms * 1e-9, 2.0 * macs / ms * 1e-9, 2.0 * macs / ms * 1e-12 / 47.0 * 100.0,
                    2.0 * macs / ms * 1e-12 / 11.7 * 100.0);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        if (cudaGetLastError() != cudaSuccess) { std::printf("  launch failed\n"); return 1; }
    }
    cudaFree(sink);
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) { std::printf("no CUDA device (skipped)\n"); return 77; }
    if (!strata::prefill::dense_i8::built()) { std::printf("no dense_i8 in this build (skipped)\n"); return 77; }
    if (argc > 1 && std::strcmp(argv[1], "--peak") == 0) return peak(60, 20000);
    cudaStream_t st = nullptr;
    if (!ok(cudaStreamCreate(&st), "stream")) return 1;
    const int64_t scratch_elems = 32ll << 20;   // GEMM_SCRATCH in src/prefill/prefill.cpp
    const int64_t T_speed = quick ? 1024 : 4096, T_err = 64;
    std::mt19937 rng(20261111u);
    std::printf("dense_i8_bench: T %lld for speed, %lld for error; MMQ=1 for the baseline\n", (long long) T_speed,
                (long long) T_err);
    std::printf("%-11s %-6s %6s %5s %6s %10s %8s %10s %8s %10s %8s\n", "what", "type", "N", "K", "T", "mmq ms", "TFLOPS",
                "i8 ms", "TFLOPS", "speedup", "maxrel");
    double tot_ratio = 0.0;
    int n_ratio = 0;
    for (const Shape& s : kShapes) {
        const QType& q = kTypes[s.qt];
        if (s.K % q.elems != 0) continue;
        const size_t blocks = (size_t) (s.N * s.K / q.elems), bytes = blocks * (size_t) q.bytes;
        std::vector<uint8_t> h(bytes);
        for (auto& b : h) b = (uint8_t) (rng() & 0xff);
        for (size_t k = 0; k < blocks; ++k) {
            const uint16_t v = sane_half(rng);
            std::memcpy(&h[k * (size_t) q.bytes + (size_t) q.dat], &v, 2);
        }
        void* W = nullptr;
        if (!ok(cudaMalloc(&W, bytes), "W") || !ok(cudaMemcpy(W, h.data(), bytes, cudaMemcpyHostToDevice), "W copy")) return 1;
        uint16_t* W16 = nullptr;
        if (!ok(cudaMalloc((void**) &W16, (size_t) s.N * s.K * 2), "W16")) return 1;
        strata::kernels::dequant_f16(q.type, W, 0, s.N, s.K, W16, st);
        uint16_t* X = nullptr;
        if (!ok(cudaMalloc((void**) &X, (size_t) T_speed * s.K * 2), "X")) return 1;
        fill_f16<<<(unsigned) ((T_speed * s.K + 255) / 256), 256, 0, st>>>((__half*) X, T_speed * s.K, 17u, 2.0f);
        float* Y[3] = {nullptr, nullptr, nullptr};
        for (int i = 0; i < 3; ++i)
            if (!ok(cudaMalloc((void**) &Y[i], (size_t) T_speed * s.N * 4), "Y")) return 1;
        if (!ok(cudaStreamSynchronize(st), "x")) return 1;

        strata::prefill::Gemm gemm;
        std::string err;
        if (!gemm.init(st, scratch_elems, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }

        const auto time_ms = [&](const char* i8, int reps, float* dst) -> double {
            setenv("STRATA_DENSE_I8", i8, 1);
            gemm.native(X, q.type, W, dst, T_speed, s.N, s.K, s.N, 0.0f);
            if (!ok(cudaStreamSynchronize(st), "warm")) return -1.0;
            cudaEvent_t a = nullptr, b = nullptr;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            cudaEventRecord(a, st);
            for (int r = 0; r < reps; ++r) gemm.native(X, q.type, W, dst, T_speed, s.N, s.K, s.N, 0.0f);
            cudaEventRecord(b, st);
            if (!ok(cudaStreamSynchronize(st), "run")) return -1.0;
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, a, b);
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return (double) ms / reps;
        };
        const double flops = 2.0 * (double) s.N * (double) s.K * (double) T_speed;
        const double ms_mmq = time_ms("0", 3, Y[0]);
        const double ms_i8 = time_ms("1", 3, Y[1]);
        setenv("STRATA_DENSE_I8", "0", 1);

        // error against the exact FP16 product of the same dequantized weight (the engine's non-MMQ path)
        std::vector<float> a1((size_t) T_err * s.N), rf((size_t) T_err * s.N);
        setenv("STRATA_DENSE_I8", "1", 1);
        gemm.native(X, q.type, W, Y[1], T_err, s.N, s.K, s.N, 0.0f);
        cudaMemcpy(a1.data(), Y[1], a1.size() * 4, cudaMemcpyDeviceToHost);
        gemm.f16(X, W16, Y[2], T_err, s.N, s.K, s.N, 0.0f);
        cudaMemcpy(rf.data(), Y[2], rf.size() * 4, cudaMemcpyDeviceToHost);
        setenv("STRATA_DENSE_I8", "0", 1);
        double num = 0.0, den = 0.0, mx = 0.0;
        for (size_t i = 0; i < rf.size(); ++i) {
            num += (double) (a1[i] - rf[i]) * (a1[i] - rf[i]);
            den += (double) rf[i] * rf[i];
            const double d = std::fabs((double) a1[i] - rf[i]) / (std::fabs((double) rf[i]) + 1e-6);
            if (d > mx) mx = d;
        }
        const double rms = den > 0.0 ? std::sqrt(num / den) : 0.0;
        std::printf("%-11s %-6s %6lld %5lld %6lld %10.2f %8.2f %10.2f %8.2f %9.2fx %7.2e (rms %.4f)\n", s.what, q.name,
                    (long long) s.N, (long long) s.K, (long long) T_speed, ms_mmq, flops / ms_mmq * 1e-9, ms_i8,
                    flops / ms_i8 * 1e-9, ms_mmq / ms_i8, mx, rms);
        if (ms_i8 > 0.0 && ms_mmq > 0.0) { tot_ratio += ms_mmq / ms_i8; ++n_ratio; }
        cudaFree(W);
        cudaFree(W16);
        cudaFree(X);
        for (int i = 0; i < 3; ++i) cudaFree(Y[i]);
    }
    std::printf("dense_i8_bench: mean speedup %.2fx over %d shapes\n", n_ratio ? tot_ratio / n_ratio : 0.0, n_ratio);
    return 0;
}
