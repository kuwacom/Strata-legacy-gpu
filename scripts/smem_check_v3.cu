// v3: shared memory of the DP4A MMQ configs on sm_61 for the I tiles the v3 overlay writes, and the J that
// mul_mat_q_switch_J picks for a given ncols_opt (the same loop as mmq.cuh, smpbo = 49,152 B)
#include "mmq.cuh"

#include <cstdio>

int ggml_cuda_get_device() { return 0; }
void ggml_cuda_error(const char *, const char *, const char *, int, const char *) { __builtin_trap(); }
CUresult cuGetErrorString(CUresult, const char **pStr) { if (pStr) *pStr = ""; return CUDA_SUCCESS; }

static ggml_cuda_mmq_config make(ggml_type t, int I, int J) {
    return ggml_cuda_mmq_config(t, 256, 2, I, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false);
}

// the pinned configs' I per (type, J): what the overlay is compared against
static int pinned_I(ggml_type t, int J) {
    const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, false, 610);
    return c.type == GGML_TYPE_COUNT ? -1 : c.I;
}

int main() {
    const int cc = 610;
    const size_t smpbo = 49152;
    const ggml_type ts[] = {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS,
                            GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_0, GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0};
    std::printf("== shared memory at J=48 / J=64 (I = 64, 96, 128), smpbo %zu B\n", smpbo);
    for (ggml_type t : ts) {
        for (int J : {48, 64}) {
            std::printf("%-10s J=%2d pinned I=%2d:", ggml_type_name(t), J, pinned_I(t, J));
            for (int I : {64, 96, 128}) {
                const size_t nb = mmq_get_nbytes_shared(make(t, I, J), cc);
                std::printf("  I=%3d %6zu B%s", I, nb, nb <= smpbo ? "" : " OVER");
            }
            std::printf("\n");
        }
    }

    std::printf("\n== mul_mat_q_switch_J's pick (smallest J with the fewest tiles, configs over smpbo skipped)\n");
    const int ncols[] = {64, 80, 96, 100, 128, 200, 469, 600, 1200};
    for (ggml_type t : {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q2_0}) {
        std::printf("%-10s:", ggml_type_name(t));
        for (int n : ncols) {
            int J_best = 0, best = 1 << 30;
            for (int J = 8; J <= 128 && best > 1; J += 8) {
                const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, false, cc);
                if (c.type == GGML_TYPE_COUNT) continue;
                if (mmq_get_nbytes_shared(c, cc) > smpbo) continue;
                const int nt = (n + c.J - 1) / c.J;
                if (nt < best) { J_best = J; best = nt; }
            }
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J_best, false, cc);
            std::printf("  n=%4d -> J=%2d I=%3d", n, J_best, c.I);
        }
        std::printf("\n");
    }
    return 0;
}
