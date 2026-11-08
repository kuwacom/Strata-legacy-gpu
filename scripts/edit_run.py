import pathlib, sys
p = pathlib.Path.home() / "work/wt/pgdn/src/prefill/moe_mmq.cu"
s = p.read_text()
old = """#if defined(STRATA_MMQ_KQUANTS) && defined(STRATA_KQ_DSO)
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_Q5_1: {"""
new = """#if defined(STRATA_MMQ_KQUANTS) && defined(STRATA_KQ_DSO)
        // Q3_K: the dense GDN projections' type, in the library next to the engine (STRATA_DENSE_MMQ on CUDA)
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_Q5_1:
#if !defined(GGML_USE_HIP)
        case GGML_TYPE_Q3_K:
#endif
        {"""
n = s.count(old)
if n != 1:
    print("FAIL count", n)
    sys.exit(1)
p.write_text(s.replace(old, new))
print("ok")
