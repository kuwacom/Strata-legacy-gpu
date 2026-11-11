// include/strata/prefill/dense_i8.hpp - the dense projections with their weights expanded to int8 once per call and
// multiplied by a plain DP4A GEMM (opt-in: STRATA_DENSE_I8=1).  See src/prefill/dense_i8.cu for why.
#pragma once

#include <cstdint>

namespace strata::prefill::dense_i8 {

/// This build has the path (CUDA; the kernels are DP4A, compute capability 6.1 and newer).
bool built();
/// STRATA_DENSE_I8=1 (read at every call so a parity/bench binary can flip it).
bool enabled();
/// The expand and the GEMM cover this matrix: a ggml type `dequant_f16` reads, K a multiple of 128 values (the
/// weight and activation int8 planes carry one fp32 scale per 32 values), and no overflow of the int32 indices the
/// kernels use.
bool fits(int ggml_type, int64_t N, int64_t K);

/// Y[t, i] = sum_k dequant(W[i, k]) * X[t, k] for the [N, K] native-block weight `W` of `ggml_type` and the FP16
/// activations `X` (K halves per row), into `Y` (`ldy` floats per row; ldy = N when 0).  The weight is expanded to
/// int8 (one fp32 scale per 32 values) once per call, the activations 1024 rows at a time.  False: nothing was
/// launched or the expansion hit an allocation/type the path does not take - the caller runs its usual product.
bool run(const uint16_t* X, int ggml_type, const void* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
         void* stream);

}  // namespace strata::prefill::dense_i8
