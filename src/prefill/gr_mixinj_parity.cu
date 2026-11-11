// src/prefill/gr_mixinj_parity.cu - the hyper-connection read's fused inject + mix (gr_mix_r16_inj, STRATA_HC_MIXINJ=1)
// against the pair it replaces (gr_mix_r16 + Gemm::bf16_narrow on N = HC = 4), on synthetic inputs of the model's shape
// (N = 2560, 4 streams, K = 10,240).  `mixed` and its BF16 / FP16 images must be the pair's bits; `inj` sums k in
// another order than the narrow kernel, so it is checked within FP32 rounding - and both are measured against an FP64
// product of the same BF16 inputs, since a faster sum order must not be the less accurate one.
//   gr_mixinj_parity            the checks (T = 1 .. 32768, the chunk sizes' boundaries)
//   gr_mixinj_parity --bench    and the timings of the pair and the fused kernel at T = 2048 / 8192 / 32768
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int N = 2560, HC = 4, D = N * HC;

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    if (!ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc")) std::exit(2);
    return p;
}

// 決定的な擬似乱数でデバイス側を埋める (ホストに T x 10,240 x 4 B を置くと 1 形状で 2 GB になる)
__global__ void fill_f32(float* p, size_t n, uint32_t seed, float scale) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t s = (uint32_t) i * 2654435761u ^ seed;
        s ^= s >> 15;
        s *= 2246822519u;
        s ^= s >> 13;
        p[i] = scale * ((float) (s & 0x00ffffffu) / 8388608.0f - 1.0f);
    }
}
__global__ void fill_bf16(uint16_t* p, size_t n, uint32_t seed, float scale) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t s = (uint32_t) i * 2654435761u ^ seed;
        s ^= s >> 15;
        s *= 2246822519u;
        s ^= s >> 13;
        const float v = scale * ((float) (s & 0x00ffffffu) / 8388608.0f - 1.0f);
        uint32_t u;
        std::memcpy(&u, &v, 4);
        u += 0x7fffu + ((u >> 16) & 1u);
        p[i] = (uint16_t) (u >> 16);
    }
}

struct Inputs {
    int64_t T;
    uint16_t *d_xn16, *d_wi;
    float *d_gated;
    Inputs(int64_t t, uint32_t seed) : T(t) {
        d_xn16 = dalloc<uint16_t>((size_t) T * D);
        d_wi = dalloc<uint16_t>((size_t) HC * D);
        d_gated = dalloc<float>((size_t) T * D);
        // xn16: 正規化済みの活性 (O(1))、gated: up の出力 (sigmoid の前、数単位)、wi: 小さい重み
        fill_bf16<<<512, 256>>>(d_xn16, (size_t) T * D, seed, 1.5f);
        fill_f32<<<512, 256>>>(d_gated, (size_t) T * D, seed ^ 0x9e3779b9u, 3.0f);
        fill_bf16<<<64, 256>>>(d_wi, (size_t) HC * D, seed ^ 0x7f4a7c15u, 0.05f);
        cudaDeviceSynchronize();   // 埋めた値がテストのストリームから見えるようにする
    }
    ~Inputs() { cudaFree(d_xn16); cudaFree(d_wi); cudaFree(d_gated); }
};

struct Outputs {
    int64_t T;
    float *mixed, *inj;
    uint16_t *bf, *h;
    explicit Outputs(int64_t t) : T(t) {
        mixed = dalloc<float>((size_t) T * N);
        inj = dalloc<float>((size_t) T * HC);
        bf = dalloc<uint16_t>((size_t) T * N);
        h = dalloc<uint16_t>((size_t) T * N);
    }
    ~Outputs() { cudaFree(mixed); cudaFree(inj); cudaFree(bf); cudaFree(h); }
};

// sampled tokens' `inj` against an FP64 product of the same BF16 inputs: max |error| / max |value|
double inj_vs_exact(const Inputs& in, const float* d_inj, const std::vector<int64_t>& rows, double& mag) {
    std::vector<uint16_t> wx((size_t) HC * D);
    cudaMemcpy(wx.data(), in.d_wi, wx.size() * 2, cudaMemcpyDeviceToHost);
    double worst = 0.0;
    mag = 0.0;
    for (const int64_t t : rows) {
        std::vector<uint16_t> x((size_t) D);
        cudaMemcpy(x.data(), in.d_xn16 + (size_t) t * D, (size_t) D * 2, cudaMemcpyDeviceToHost);
        float got[HC];
        cudaMemcpy(got, d_inj + (size_t) t * HC, HC * 4, cudaMemcpyDeviceToHost);
        for (int j = 0; j < HC; ++j) {
            double s = 0.0;
            for (int k = 0; k < D; ++k) s += (double) from_bf16(x[(size_t) k]) * (double) from_bf16(wx[(size_t) j * D + k]);
            worst = std::max(worst, std::fabs((double) got[j] - s));
            mag = std::max(mag, std::fabs(s));
        }
    }
    return worst;
}

int check(strata::prefill::Gemm& gm, cudaStream_t st, int64_t T, uint32_t seed) {
    Inputs in(T, seed);
    Outputs ref(T), got(T);
    strata::prefill::gr_mix_r16(in.d_xn16, D, in.d_gated, ref.mixed, ref.bf, T, st, ref.h);
    const bool pair_inj = gm.bf16_narrow(in.d_xn16, in.d_wi, ref.inj, T, HC, D, 0);
    if (!strata::prefill::gr_mix_r16_inj(in.d_xn16, D, in.d_gated, in.d_wi, got.mixed, got.bf, got.h, nullptr, got.inj, T, st)) {
        std::printf("FAIL T=%lld: gr_mix_r16_inj refused\n", (long long) T);
        return 1;
    }
    if (!ck(cudaStreamSynchronize(st), "run")) return 1;
    const size_t n = (size_t) T * N;
    std::vector<float> a(n), b(n);
    std::vector<uint16_t> abf(n), bbf(n), ah(n), bh(n);
    cudaMemcpy(a.data(), ref.mixed, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), got.mixed, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(abf.data(), ref.bf, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(bbf.data(), got.bf, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(ah.data(), ref.h, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(bh.data(), got.h, n * 2, cudaMemcpyDeviceToHost);
    size_t mixed_bits = 0, bf_diff = 0, h_diff = 0, nonfinite = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(b[i])) ++nonfinite;
        mixed_bits += std::memcmp(&a[i], &b[i], 4) != 0;
        bf_diff += abf[i] != bbf[i];
        h_diff += ah[i] != bh[i];
    }
    // inject: ペア側 (narrow) との差、および両方の FP64 からのずれ
    double dd = 0.0, mag = 0.0;
    if (pair_inj) {
        std::vector<float> ia((size_t) T * HC), ib((size_t) T * HC);
        cudaMemcpy(ia.data(), ref.inj, ia.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(ib.data(), got.inj, ib.size() * 4, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < ia.size(); ++i) {
            dd = std::max(dd, (double) std::fabs(ia[i] - ib[i]));
            mag = std::max(mag, (double) std::fabs(ia[i]));
        }
    }
    std::mt19937 g(seed ^ 0x5a5a5a5au);
    std::vector<int64_t> rows;
    for (int i = 0; i < 4; ++i) rows.push_back(T <= 4 ? (int64_t) i % T : (int64_t) (g() % (uint32_t) T));
    double im = 0.0;
    const double ea = pair_inj ? inj_vs_exact(in, ref.inj, rows, im) : 0.0;
    double gm2 = 0.0;
    const double eb = inj_vs_exact(in, got.inj, rows, gm2);
    const double rel = dd / std::max(mag, 1e-30);
    const bool bad = nonfinite != 0 || mixed_bits != 0 || bf_diff != 0 || h_diff != 0 ||
                     (pair_inj && (rel > 1e-4 || eb > 2.0 * ea + 1e-6 * im));
    std::printf("%s T=%-6lld mixed %s, images %s; inject%s vs narrow %.2e rel; vs FP64 (sampled) narrow %.2e, "
                "fused %.2e\n",
                bad ? "FAIL" : "ok  ", (long long) T, mixed_bits == 0 ? "bitwise" : "differ",
                bf_diff == 0 && h_diff == 0 ? "bitwise" : "differ",
                pair_inj ? "" : " (narrow not taken below 64 tokens)", rel, ea / std::max(im, 1e-30),
                eb / std::max(gm2, 1e-30));
    return bad ? 1 : 0;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void bench(strata::prefill::Gemm& gm, cudaStream_t st, int64_t T) {
    Inputs in(T, 11);
    Outputs o(T);
    cudaEvent_t e0, e1, em;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&em);
    std::vector<float> tp, tf;
    for (int i = 0; i < 15; ++i) {
        const bool fused_first = i & 1;   // 先に走らせる側を入れ替える
        for (int k = 0; k < 2; ++k) {
            if ((k == 0) == fused_first) {
                cudaEventRecord(e0, st);
                strata::prefill::gr_mix_r16_inj(in.d_xn16, D, in.d_gated, in.d_wi, o.mixed, o.bf, o.h, nullptr, o.inj, T, st);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                if (i >= 3) tf.push_back(ms);
            } else {
                cudaEventRecord(e0, st);
                strata::prefill::gr_mix_r16(in.d_xn16, D, in.d_gated, o.mixed, o.bf, T, st, o.h);
                cudaEventRecord(em, st);
                gm.bf16_narrow(in.d_xn16, in.d_wi, o.inj, T, HC, D, 0);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                float ms, msi;
                cudaEventElapsedTime(&ms, e0, e1);
                cudaEventElapsedTime(&msi, em, e1);
                if (i >= 3) { tp.push_back(ms); }
                if (i == 14) std::printf("      (the pair: mix + inject %.3f ms of %.3f)\n", msi, ms);
            }
        }
    }
    // ペアは xn16 を 2 回読み (20,480 B ずつ)、融合は 1 回。出力は同じ
    const double bytes_pair = (double) T * (20480.0 + 40960.0 + 20480.0 + 20480.0);
    const double bytes_fused = (double) T * (20480.0 + 40960.0 + 20480.0);
    std::printf("  T=%-6lld pair %.3f ms (%.0f GB/s) | fused %.3f ms (%.0f GB/s): %.2fx\n", (long long) T, median(tp),
                bytes_pair / median(tp) / 1e6, median(tf), bytes_fused / median(tf) / 1e6, median(tp) / median(tf));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(em);
}

}  // namespace

int main(int argc, char** argv) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        std::printf("gr_mixinj_parity: no CUDA device, skipped\n");
        return 77;
    }
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    cudaStream_t st = nullptr;
    if (!ck(cudaStreamCreate(&st), "stream")) return 1;
    strata::prefill::Gemm gm;
    std::string err;
    if (!gm.init(st, 0, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }
    int fails = 0;
    uint32_t seed = 202;
    for (const int64_t T : {1, 2, 7, 31, 63, 64, 65, 127, 128, 129, 600, 1000, 2048, 4099, 8192, 32768})
        fails += check(gm, st, T, seed++);
    if (do_bench)
        for (const int64_t T : {2048, 8192, 32768}) bench(gm, st, T);
    cudaStreamDestroy(st);
    std::printf("gr_mixinj_parity: %d failures\n", fails);
    return fails == 0 ? 0 : 1;
}
