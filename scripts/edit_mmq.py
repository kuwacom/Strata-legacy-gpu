import io, sys, pathlib
root = pathlib.Path.home() / "work/wt/pgdn"

def edit(path, old, new, count=1):
    p = root / path
    s = p.read_text()
    n = s.count(old)
    if n != count:
        print(f"FAIL {path}: found {n} of {count}\n---\n{old}\n---")
        sys.exit(1)
    p.write_text(s.replace(old, new))
    print(f"ok {path} ({count})")

# 1. moe_mmq.cu: the two HIP-only kernels are the dense-MMQ path's, now on CUDA too
edit("src/prefill/moe_mmq.cu",
"""#if defined(__HIPCC__)   // #820: only the HIP dense-MMQ path (Gemm::native_mmq) uses these two
__global__ void f16_to_f32_kernel""",
"""// #820: only the HIP dense-MMQ path (Gemm::native_mmq) uses these two; the same path on CUDA (STRATA_DENSE_MMQ=1) too
__global__ void f16_to_f32_kernel""")

edit("src/prefill/moe_mmq.cu",
"""#if defined(__HIPCC__)
void f16_to_f32(const uint16_t* x, float* y, int64_t n, void* stream) {""",
"""void f16_to_f32(const uint16_t* x, float* y, int64_t n, void* stream) {""")

edit("src/prefill/moe_mmq.cu",
"""    ck(cudaGetLastError(), "set_bounds");
}
#endif""",
"""    ck(cudaGetLastError(), "set_bounds");
}""")

# 2. Q3_K is an MMQ type on CUDA (the dense GDN projections)
edit("src/prefill/moe_mmq.cu",
"""        // Unsloth's UD-Q4/Q5/Q6_K_XL experts; on HIP also the dense GGUF projections (STRATA_DENSE_MMQ)
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q5_1: case GGML_TYPE_Q6_K:
            return kq_ok();""",
"""        // Unsloth's UD-Q4/Q5/Q6_K_XL experts; on HIP also the dense GGUF projections (STRATA_DENSE_MMQ)
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q5_1: case GGML_TYPE_Q6_K:
#if !defined(GGML_USE_HIP)
        case GGML_TYPE_Q3_K:   // the dense GDN projections' Q3_K weights (STRATA_DENSE_MMQ on CUDA)
#endif
            return kq_ok();""")

# 3. the K-quant library dispatches Q3_K
edit("src/prefill/mmq_kq_lib.cu",
"""// (Q4_K, Q5_K, Q5_1, Q6_K: the experts of Unsloth's UD-Q4/Q5/Q6_K_XL files).  MODEL-UPDATE Phase 1: the engine loads""",
"""// (Q4_K, Q5_K, Q5_1, Q6_K: the experts of Unsloth's UD-Q4/Q5/Q6_K_XL files; Q3_K: the dense projections' STRATA_DENSE_MMQ).  MODEL-UPDATE Phase 1: the engine loads""")

edit("src/prefill/mmq_kq_lib.cu",
"""        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, a, s); break;""",
"""        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, a, s); break;
        case GGML_TYPE_Q3_K: mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, a, s); break;   // the dense projections (STRATA_DENSE_MMQ)""")

# 4. the Q3_K MMQ instance joins the K-quant library
edit("CMakeLists.txt",
"""    set(_strata_kq_types q4_k q5_k q5_1 q6_k)""",
"""    set(_strata_kq_types q3_k q4_k q5_k q5_1 q6_k)   # q3_k: the dense GDN projections' STRATA_DENSE_MMQ""")

# 5. gemm.cu: the dense MMQ path is no longer HIP-only (opt-in at run time, STRATA_DENSE_MMQ=1)
p = root / "src/prefill/gemm.cu"
s = p.read_text()
old = "#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)"
n = s.count(old)
if n != 4:
    print(f"FAIL gemm.cu: found {n} of 4")
    sys.exit(1)
p.write_text(s.replace(old, "#if defined(STRATA_PREFILL_MMQ)"))
print("ok src/prefill/gemm.cu (4)")
print("ALL EDITS OK")
