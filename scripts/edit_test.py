import pathlib, sys
root = pathlib.Path.home() / "work/wt/pgdn"
p = root / "tests/cuda/prefill_mmq_kquant_test.cpp"
s = p.read_text()
old = """                product(ctx, s, ("Q8_0 down" + tag).c_str(), GGML_TYPE_Q8_0, 2560, 640, counts, trial + 13);
                ++trial;"""
new = """                product(ctx, s, ("Q8_0 down" + tag).c_str(), GGML_TYPE_Q8_0, 2560, 640, counts, trial + 13);
                // Q3_K: the dense GDN projections' type (STRATA_DENSE_MMQ on CUDA), at their gate/up shape
                product(ctx, s, ("Q3_K gate/up" + tag).c_str(), GGML_TYPE_Q3_K, 1280, 2560, counts, trial + 23);
                ++trial;"""
if s.count(old) != 1:
    print("FAIL insert Q3_K cases"); sys.exit(1)
s = s.replace(old, new)
old2 = """            }
        }
        ck(cudaStreamDestroy(s), "destroy");"""
new2 = """            }
            // the dense qkv shape itself (10240 rows, 2560 columns): two experts of one row each
            product(ctx, s, "Q3_K qkv dense", GGML_TYPE_Q3_K, 10240, 2560, {1, 1}, 41);
        }
        ck(cudaStreamDestroy(s), "destroy");"""
if s.count(old2) != 1:
    print("FAIL insert dense case"); sys.exit(1)
s = s.replace(old2, new2)
p.write_text(s)
print("ok tests/cuda/prefill_mmq_kquant_test.cpp")

# build script target list
q = root / "scripts/build-pgdn.sh"
t = q.read_text()
t = t.replace("prefill_mmq_tab_test", "prefill_mmq_kquant_test")
q.write_text(t)
print("ok scripts/build-pgdn.sh")
