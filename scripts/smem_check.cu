// shared memory of the DP4A MMQ configs at J=64 on sm_61 (cc 610), the check behind the I=128 choice
#include "mmq.cuh"

#include <cstdio>

// stubs for the host symbols ggml-cuda.cu / the driver would provide (unused here: no CUDA call is made)
int ggml_cuda_get_device() { return 0; }
void ggml_cuda_error(const char *, const char *, const char *, int, const char *) { __builtin_trap(); }
CUresult cuGetErrorString(CUresult, const char **pStr) { if (pStr) *pStr = ""; return CUDA_SUCCESS; }

int main() {
    const int cc = 610;
    const size_t smpbo = 49152;   // prop.sharedMemPerBlockOptin, sm_61
    const ggml_type ts[] = {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS,
                            GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_0, GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0};
    for (ggml_type t : ts) {
        for (int J : {48, 56, 64}) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, false, cc);
            if (c.type == GGML_TYPE_COUNT) { std::printf("%-14s J=%d: no config\n", ggml_type_name(t), J); continue; }
            const size_t nb = mmq_get_nbytes_shared(c, cc);
            std::printf("%-14s J=%2d: I=%3d  shared %6zu B  %s\n", ggml_type_name(t), J, c.I, nb,
                        nb <= smpbo ? "fits" : "OVER");
        }
    }
    return 0;
}
