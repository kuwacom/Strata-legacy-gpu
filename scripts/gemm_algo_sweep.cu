// gemm_algo_sweep.cu - P40 (sm_61): cublasSgemm (default) against cublasGemmEx with CUDA_R_32F operands and each
// fixed algorithm CUBLAS_GEMM_ALGO0..23, on the dense prompt projections' shapes.  Also one concatenated qkv+gate
// GEMM (N = 10240 + 6144) against the two separate calls.
//   nvcc -O3 -arch=sm_61 gemm_algo_sweep.cu -o gemm_algo_sweep -lcublas && ./gemm_algo_sweep
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void fill_f32(float* p, long n, unsigned seed, float scale) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = (unsigned) i * 2654435761u + seed;
    x ^= x >> 16; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    p[i] = ((float) (x & 0xffff) / 32768.0f - 1.0f) * scale;
}
static double ms(cudaEvent_t a, cudaEvent_t b) { float t = 0; cudaEventElapsedTime(&t, a, b); return t; }
#define CK(x) do { auto s_ = (x); if (s_ != 0) { std::fprintf(stderr, "FAIL %s = %d line %d\n", #x, (int) s_, __LINE__); std::exit(1); } } while (0)

struct Shape { const char* name; long N, K; };

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const long T = argc > 1 ? std::atol(argv[1]) : 8192;
    const Shape shapes[] = {{"gdn qkv", 10240, 2560}, {"gdn gate", 6144, 2560}, {"gdn out", 2560, 6144}};
    cublasHandle_t h; CK(cublasCreate(&h));
    cudaStream_t st; CK(cudaStreamCreate(&st)); CK(cublasSetStream(h, st));
    void* ws; CK(cudaMalloc(&ws, 32u << 20)); CK(cublasSetWorkspace(h, ws, 32u << 20));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const float one = 1.0f, zero = 0.0f;

    for (const Shape& s : shapes) {
        const long N = s.N, K = s.K;
        float *W, *X, *Y; CK(cudaMalloc(&W, N * K * 4)); CK(cudaMalloc(&X, T * K * 4)); CK(cudaMalloc(&Y, T * N * 4));
        fill_f32<<<(unsigned) ((N * K + 255) / 256), 256, 0, st>>>(W, N * K, 1u, 0.05f);
        fill_f32<<<(unsigned) ((T * K + 255) / 256), 256, 0, st>>>(X, T * K, 7u, 2.0f);
        const double fl = 2.0 * N * K * T / 1e12;
        double base = 1e30;
        for (int r = 0; r < 4; ++r) {
            cudaEventRecord(e0, st);
            CK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W, (int) K, X, (int) K, &zero, Y, (int) N));
            cudaEventRecord(e1, st); CK(cudaEventSynchronize(e1));
            if (r) base = fmin(base, ms(e0, e1));
        }
        std::printf("SHAPE %-9s N %5ld K %5ld T %ld | sgemm default %8.3f ms %5.2f TFLOPS\n", s.name, N, K, T, base, fl / (base / 1e3));
        for (int a = 0; a <= 23; ++a) {
            double best = 1e30;
            cublasStatus_t stt = CUBLAS_STATUS_SUCCESS;
            for (int r = 0; r < 3; ++r) {
                cudaEventRecord(e0, st);
                stt = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W, CUDA_R_32F, (int) K, X,
                                   CUDA_R_32F, (int) K, &zero, Y, CUDA_R_32F, (int) N, CUBLAS_COMPUTE_32F,
                                   (cublasGemmAlgo_t) a);
                cudaEventRecord(e1, st); CK(cudaEventSynchronize(e1));
                if (stt != CUBLAS_STATUS_SUCCESS) break;
                if (r) best = fmin(best, ms(e0, e1));
            }
            if (stt != CUBLAS_STATUS_SUCCESS) {
                std::printf("  ALGO %2d: status %d\n", a, (int) stt);
                continue;
            }
            std::printf("  ALGO %2d: %8.3f ms %5.2f TFLOPS  %+6.1f%%\n", a, best, fl / (best / 1e3), 100.0 * (best / base - 1.0));
        }
        cudaFree(W); cudaFree(X); cudaFree(Y);
    }

    // concatenated qkv+gate: one GEMM N = 10240 + 6144 against the two
    {
        const long N1 = 10240, N2 = 6144, N = N1 + N2, K = 2560;
        float *W, *X, *Y; CK(cudaMalloc(&W, N * K * 4)); CK(cudaMalloc(&X, T * K * 4)); CK(cudaMalloc(&Y, T * N * 4));
        fill_f32<<<(unsigned) ((N * K + 255) / 256), 256, 0, st>>>(W, N * K, 1u, 0.05f);
        fill_f32<<<(unsigned) ((T * K + 255) / 256), 256, 0, st>>>(X, T * K, 7u, 2.0f);
        double two = 1e30, one_ = 1e30;
        for (int r = 0; r < 4; ++r) {
            cudaEventRecord(e0, st);
            CK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) N1, (int) T, (int) K, &one, W, (int) K, X, (int) K, &zero, Y, (int) N));
            CK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) N2, (int) T, (int) K, &one, W + N1 * K, (int) K, X, (int) K, &zero, Y + N1, (int) N));
            cudaEventRecord(e1, st); CK(cudaEventSynchronize(e1));
            if (r) two = fmin(two, ms(e0, e1));
            cudaEventRecord(e0, st);
            CK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W, (int) K, X, (int) K, &zero, Y, (int) N));
            cudaEventRecord(e1, st); CK(cudaEventSynchronize(e1));
            if (r) one_ = fmin(one_, ms(e0, e1));
        }
        const double fl = 2.0 * N * K * T / 1e12;
        std::printf("CONCAT qkv+gate N %ld T %ld | two calls %8.3f ms %5.2f TFLOPS | one call %8.3f ms %5.2f TFLOPS %+6.1f%%\n",
                    N, T, two, fl / (two / 1e3), one_, fl / (one_ / 1e3), 100.0 * (one_ / two - 1.0));
        cudaFree(W); cudaFree(X); cudaFree(Y);
    }
    return 0;
}
