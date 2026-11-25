// engine/src/inkling2/cuda/ink_moe_sk_occ.cuh - S_K MoE (gu/dn) の occupancy / 重み先行ロード実験用バリアント
//
// STRATA_INK_SK_OCC で切り替える (既定 0 = 従来カーネル k_ink_moe_*_sk_fmt のまま):
//   0 : 従来 (launch_bounds(128,2)、s ループは unroll 1、先行ロード無し)
//   1 : 実験 A。本体は従来と同一で __launch_bounds__(128,4)。REG<=128 を強制して 4 CTA/SM を狙う
//   2 : 実験 B。__launch_bounds__(128,3) で 1 sub-block 先の重み 16 B をレジスタへ先行ロードする。
//       REG が 128 を超えるので A とは排他 (4 CTA/SM は諦め、現行の 3 CTA/SM を維持したまま MLP だけ上げる)
//
// 演算は従来カーネルと同じ式・同じ順序で、結果は bit 一致する。従来カーネルと既存 env の意味は変えない。
// 注: このヘッダは ink_moe_cuda.cuh の namespace strata::ink の内側で include される
#pragma once

// S_K 実験バリアントの選択 (0 = 従来)。getenv は 1 回だけ読む
inline int ink_sk_occ_mode() {
    static const int mode = [] { const char* e = std::getenv("STRATA_INK_SK_OCC"); return e ? std::atoi(e) : 0; }();
    return mode;
}

// 1 sub-block 分の重み payload 16 B (lane g の 2-bit 部) を先行ロードする。
// ld32_u は 2 バイト境界保証のため 16-bit x2 のロードになるが、sk_unpack64_g_to_u32 と同じ命令列になる
__device__ __forceinline__ void sk_pre16(const uint8_t* bg, int g, uint32_t (&q)[4]) {
    const uint8_t* p = bg + 6 + g * 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) q[i] = ld32_u(p + i * 4);
}

// 先行ロード済みの q[4] から w[16] を作る。式と順序は sk_unpack64_g_to_u32 の S2_K / S3_K と同一 (bit 一致のため)
template <int FMT>
__device__ __forceinline__ void sk_unpack64_pre_g(const uint32_t (&q)[4], const uint8_t* hp, uint32_t (&w)[16]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t qq = q[i];
        const uint32_t b0 = qq & 0xFFu, b1 = (qq >> 8) & 0xFFu, b2 = (qq >> 16) & 0xFFu, b3 = qq >> 24;
        if constexpr (FMT == FMT_S2_K) {
            w[4*i + 0] = (b0 & 3u) | (((b0 >> 2) & 3u) << 8) | (((b0 >> 4) & 3u) << 16) | (((b0 >> 6) & 3u) << 24);
            w[4*i + 1] = (b1 & 3u) | (((b1 >> 2) & 3u) << 8) | (((b1 >> 4) & 3u) << 16) | (((b1 >> 6) & 3u) << 24);
            w[4*i + 2] = (b2 & 3u) | (((b2 >> 2) & 3u) << 8) | (((b2 >> 4) & 3u) << 16) | (((b2 >> 6) & 3u) << 24);
            w[4*i + 3] = (b3 & 3u) | (((b3 >> 2) & 3u) << 8) | (((b3 >> 4) & 3u) << 16) | (((b3 >> 6) & 3u) << 24);
        } else {   // FMT_S3_K
            const uint16_t h = ld16_u(hp + i * 2);
            const uint32_t h0 = h & 0xFFu, h1 = h >> 8;
            w[4*i + 0] = (b0 & 3u) | (((b0 >> 2) & 3u) << 8) | (((b0 >> 4) & 3u) << 16) | (((b0 >> 6) & 3u) << 24)
                       | ((h0 & 1u) << 2) | (((h0 >> 1) & 1u) << 10) | (((h0 >> 2) & 1u) << 18) | (((h0 >> 3) & 1u) << 26);
            w[4*i + 1] = (b1 & 3u) | (((b1 >> 2) & 3u) << 8) | (((b1 >> 4) & 3u) << 16) | (((b1 >> 6) & 3u) << 24)
                       | (((h0 >> 4) & 1u) << 2) | (((h0 >> 5) & 1u) << 10) | (((h0 >> 6) & 1u) << 18) | (((h0 >> 7) & 1u) << 26);
            w[4*i + 2] = (b2 & 3u) | (((b2 >> 2) & 3u) << 8) | (((b2 >> 4) & 3u) << 16) | (((b2 >> 6) & 3u) << 24)
                       | ((h1 & 1u) << 2) | (((h1 >> 1) & 1u) << 10) | (((h1 >> 2) & 1u) << 18) | (((h1 >> 3) & 1u) << 26);
            w[4*i + 3] = (b3 & 3u) | (((b3 >> 2) & 3u) << 8) | (((b3 >> 4) & 3u) << 16) | (((b3 >> 6) & 3u) << 24)
                       | (((h1 >> 4) & 1u) << 2) | (((h1 >> 5) & 1u) << 10) | (((h1 >> 6) & 1u) << 18) | (((h1 >> 7) & 1u) << 26);
        }
    }
}

// 実験 B 用 gu。1 sub-block 先の重みを先にレジスタへ置き、その load を今の sub-block の dp4a と重ねる。
// S2_K / S3_K 以外と n_e==1 は従来実装へ委譲する (同じ関数なので bit 一致)
template <int FMT, int MAX_M>
__device__ __forceinline__ void sk_gu_fused_4lane_pf(
    const uint8_t* gr, const uint8_t* ur,
    const InkQ8_1_G64* const (&xp)[MAX_M],
    int n_e, int g,
    float (&ag)[MAX_M], float (&au)[MAX_M]) {
    if constexpr (FMT != FMT_S2_K && FMT != FMT_S3_K) {
        sk_gu_fused_4lane<FMT, MAX_M>(gr, ur, xp, n_e, g, ag, au);
    } else {
        if (n_e <= 1) { sk_gu_fused_4lane<FMT, MAX_M>(gr, ur, xp, n_e, g, ag, au); return; }
        constexpr int SB = ink_sb_bytes(FMT);
        constexpr float HALF = ink_sk_half(FMT);
        uint32_t qg[4], qu[4];
        sk_pre16(gr, g, qg);
        sk_pre16(ur, g, qu);
#pragma unroll 1
        for (int s = 0; s < 9; ++s) {
            const uint8_t* bg = gr + s * SB;
            const uint8_t* bu = ur + s * SB;
            uint32_t cg[4], cu[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { cg[i] = qg[i]; cu[i] = qu[i]; }
            // 次の sub-block の load をここで発行する。cg/cu は既に控えてあるので依存が無く、dp4a と重なる
            if (s + 1 < 9) {
                sk_pre16(gr + (s + 1) * SB, g, qg);
                sk_pre16(ur + (s + 1) * SB, g, qu);
            }
            uint32_t w[16];
            const float eg = f16_to_f32(ld16_u(bg)) * ((float)bg[2 + g] * (1.0f / 255.0f));
            sk_unpack64_pre_g<FMT>(cg, bg + 6 + 64 + g * 8, w);
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G64& xb = xp[m][s * 4 + g];
                    const int32_t ig = dp64_i8_smem(w, xb);
                    ag[m] = fmaf(eg, xb.d * (float)ig - HALF * xb.s, ag[m]);
                }
            }
            const float eu = f16_to_f32(ld16_u(bu)) * ((float)bu[2 + g] * (1.0f / 255.0f));
            sk_unpack64_pre_g<FMT>(cu, bu + 6 + 64 + g * 8, w);
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G64& xb = xp[m][s * 4 + g];
                    const int32_t iu = dp64_i8_smem(w, xb);
                    au[m] = fmaf(eu, xb.d * (float)iu - HALF * xb.s, au[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
                ag[m] += __shfl_xor_sync(0xFFFFFFFFu, ag[m], 2); ag[m] += __shfl_xor_sync(0xFFFFFFFFu, ag[m], 1);
                au[m] += __shfl_xor_sync(0xFFFFFFFFu, au[m], 2); au[m] += __shfl_xor_sync(0xFFFFFFFFu, au[m], 1);
            }
        }
    }
}

// 実験 B 用 dn。sk_gu_fused_4lane_pf と同じく 1 sub-block 先行ロードする
template <int FMT, int MAX_M>
__device__ __forceinline__ void sk_dn_paired2_4lane_pf(
    const uint8_t* pr,
    const InkQ8_1_G64* const (&hp)[MAX_M],
    int n_e, int g,
    float (&a0)[MAX_M], float (&a1)[MAX_M]) {
    if constexpr (FMT != FMT_S2_K && FMT != FMT_S3_K) {
        sk_dn_paired2_4lane<FMT, MAX_M>(pr, hp, n_e, g, a0, a1);
    } else {
        if (n_e <= 1) { sk_dn_paired2_4lane<FMT, MAX_M>(pr, hp, n_e, g, a0, a1); return; }
        constexpr int SB = ink_sb_bytes(FMT);
        constexpr float HALF = ink_sk_half(FMT);
        uint32_t q0[4];
        sk_pre16(pr, g, q0);
#pragma unroll 1
        for (int s = 0; s < 9; ++s) {
            const uint8_t* blk = pr + s * SB;
            uint32_t c0[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) c0[i] = q0[i];
            if (s + 1 < 9) sk_pre16(pr + (s + 1) * SB, g, q0);
            const float eff = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
            const int sub_g = s * 4 + g;
            const bool is_r1 = (sub_g >= 18);
            const int xg = is_r1 ? (sub_g - 18) : sub_g;
            uint32_t w[16];
            sk_unpack64_pre_g<FMT>(c0, blk + 6 + 64 + g * 8, w);
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G64& xb = hp[m][xg];
                    const int32_t isum = dp64_i8_smem(w, xb);
                    const float term = eff * (xb.d * (float)isum - HALF * xb.s);
                    if (is_r1) a1[m] += term; else a0[m] += term;
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
                a0[m] += __shfl_xor_sync(0xFFFFFFFFu, a0[m], 2); a0[m] += __shfl_xor_sync(0xFFFFFFFFu, a0[m], 1);
                a1[m] += __shfl_xor_sync(0xFFFFFFFFu, a1[m], 2); a1[m] += __shfl_xor_sync(0xFFFFFFFFu, a1[m], 1);
            }
        }
    }
}

// 実験 A/B 用 gu カーネル。本体は k_ink_moe_gu_sk_fmt と同一で、launch_bounds と重みロードの先行化だけが違う
template <int FMT, int MAX_M, int OCC>
__global__ void __launch_bounds__(128, (OCC == 1 ? 4 : 3)) k_ink_moe_gu_sk_fmt_occ(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const InkQ8_1_G64* __restrict__ x_sk,
    float* __restrict__ h_f32) {
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || ed.gu_code != FMT) return;
    const int row0 = tb.off[e];
    const int n_total = tb.cnt[e];

    const int tid  = threadIdx.x;
    const int sub  = tid >> 2;
    const int g    = tid & 3;
    const int feat = blockIdx.x * 32 + sub;

    const uint8_t* body = sec4_base + ed.body_off;
    const uint8_t* gu_base = body + kInkInter * 2;
    constexpr size_t row_b = (size_t)(kInkHidden / 256) * ink_sb_bytes(FMT);
    const uint8_t* gr = gu_base + (size_t)(2 * feat) * row_b;
    const uint8_t* ur = gr + row_b;
    const float inv_sq = (feat < kInkInter) ? (1.0f / f16_to_f32(ld16_u(body + feat * 2))) : 0.0f;

    constexpr int SM_M = (MAX_M <= 4) ? MAX_M : 1;
    __shared__ alignas(16) InkQ8_1_G64 s_xq[SM_M][kInkHidden / 64];

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        float ag[MAX_M] = {}, au[MAX_M] = {};
        const InkQ8_1_G64* xp[MAX_M];
        if constexpr (MAX_M <= 4) {
            constexpr int N4 = (int)((kInkHidden / 64) * sizeof(InkQ8_1_G64) / 16);
            for (int m = 0; m < n_e; ++m) {
                const uint4* src = reinterpret_cast<const uint4*>(x_sk + (size_t)tb.tok_idx[row0 + off_m + m] * (kInkHidden / 64));
                uint4* dst = reinterpret_cast<uint4*>(s_xq[m]);
                for (int i = tid; i < N4; i += 128) dst[i] = src[i];
            }
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) xp[m] = s_xq[m];
            __syncthreads();
        } else {
#pragma unroll
            for (int m = 0; m < MAX_M; ++m)
                xp[m] = (m < n_e) ? (x_sk + (size_t)tb.tok_idx[row0 + off_m + m] * (kInkHidden / 64)) : nullptr;
        }
        if (feat < kInkInter) {
            if constexpr (OCC == 2) sk_gu_fused_4lane_pf<FMT, MAX_M>(gr, ur, xp, n_e, g, ag, au);
            else                    sk_gu_fused_4lane<FMT, MAX_M>(gr, ur, xp, n_e, g, ag, au);
            if (g == 0) {
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < n_e) h_f32[(size_t)(row0 + off_m + m) * kInkInter + feat] = situ_f32(ag[m], au[m]) * inv_sq;
                }
            }
        }
        if constexpr (MAX_M <= 4) { if (off_m + MAX_M < n_total) __syncthreads(); }
    }
}

// 実験 A/B 用 dn カーネル。本体は k_ink_moe_dn_sk_fmt と同一
template <int FMT, int MAX_M, int OCC>
__global__ void __launch_bounds__(128, (OCC == 1 ? 4 : 3)) k_ink_moe_dn_sk_fmt_occ(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const InkQ8_1_G64* __restrict__ h_sk,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair) {
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || ed.dn_code != FMT) return;
    const int row0 = tb.off[e];
    const int n_total = tb.cnt[e];

    const int tid    = threadIdx.x;
    const int sub    = tid >> 2;
    const int g      = tid & 3;
    const int pair_r = blockIdx.x * 32 + sub;

    const uint8_t* body = sec4_base + ed.body_off;
    constexpr size_t pair_b = 9ull * ink_sb_bytes(FMT);
    const uint8_t* pr = body + ed.dn_off + (size_t)pair_r * pair_b;

    constexpr int SM_M = (MAX_M <= 4) ? MAX_M : 1;
    __shared__ alignas(16) InkQ8_1_G64 s_hq[SM_M][kInkInter / 64];

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        float a0[MAX_M] = {}, a1[MAX_M] = {};
        const InkQ8_1_G64* hp[MAX_M];
        if constexpr (MAX_M <= 4) {
            constexpr int N4 = (int)((kInkInter / 64) * sizeof(InkQ8_1_G64) / 16);
            for (int m = 0; m < n_e; ++m) {
                const uint4* src = reinterpret_cast<const uint4*>(h_sk + (size_t)(row0 + off_m + m) * (kInkInter / 64));
                uint4* dst = reinterpret_cast<uint4*>(s_hq[m]);
                for (int i = tid; i < N4; i += 128) dst[i] = src[i];
            }
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) hp[m] = s_hq[m];
            __syncthreads();
        } else {
#pragma unroll
            for (int m = 0; m < MAX_M; ++m)
                hp[m] = (m < n_e) ? (h_sk + (size_t)(row0 + off_m + m) * (kInkInter / 64)) : nullptr;
        }
        if (pair_r < kInkHidden / 2) {
            if constexpr (OCC == 2) sk_dn_paired2_4lane_pf<FMT, MAX_M>(pr, hp, n_e, g, a0, a1);
            else                    sk_dn_paired2_4lane<FMT, MAX_M>(pr, hp, n_e, g, a0, a1);
            if (g == 0) {
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < n_e) {
                        const int pid = tb.pairs[row0 + off_m + m];
                        const float rw = topk_w[pid];
                        y_pair[(size_t)pid * kInkHidden + 2 * pair_r]     = a0[m] * rw;
                        y_pair[(size_t)pid * kInkHidden + 2 * pair_r + 1] = a1[m] * rw;
                    }
                }
            }
        }
        if constexpr (MAX_M <= 4) { if (off_m + MAX_M < n_total) __syncthreads(); }
    }
}

// STRATA_INK_SK_OCC=1|2 のときだけ従来ランチャの代わりに起動する。起動しなければ false を返し従来経路へ戻る
#define INK_SK_OCC_LAUNCH_GU(FMTV)                                                                     \
    if (mask & (1u << FMTV)) {                                                                          \
        if (mode == 1) k_ink_moe_gu_sk_fmt_occ<FMTV, MAX_M, 1><<<grid, 128, 0, st>>>(sec4_base, experts, tb, x_sk, h_f32); \
        else           k_ink_moe_gu_sk_fmt_occ<FMTV, MAX_M, 2><<<grid, 128, 0, st>>>(sec4_base, experts, tb, x_sk, h_f32); \
    }
#define INK_SK_OCC_LAUNCH_DN(FMTV)                                                                     \
    if (mask & (1u << FMTV)) {                                                                          \
        if (mode == 1) k_ink_moe_dn_sk_fmt_occ<FMTV, MAX_M, 1><<<grid, 128, 0, st>>>(sec4_base, experts, tb, h_sk, topk_w, y_pair); \
        else           k_ink_moe_dn_sk_fmt_occ<FMTV, MAX_M, 2><<<grid, 128, 0, st>>>(sec4_base, experts, tb, h_sk, topk_w, y_pair); \
    }

template <int MAX_M>
inline bool launch_ink_moe_gu_sk_occ(
    dim3 grid, cudaStream_t st, uint32_t mask,
    const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
    const InkQ8_1_G64* x_sk, float* h_f32) {
    const int mode = ink_sk_occ_mode();
    // mask==0 は汎用カーネル k_ink_moe_gu_sk (format 非特化) の担当なので従来に任せる
    if ((mode != 1 && mode != 2) || mask == 0) return false;
    INK_SK_OCC_LAUNCH_GU(FMT_S1_5)
    INK_SK_OCC_LAUNCH_GU(FMT_S2_K)
    INK_SK_OCC_LAUNCH_GU(FMT_S3_K)
    INK_SK_OCC_LAUNCH_GU(FMT_S4_K)
    INK_SK_OCC_LAUNCH_GU(FMT_S5_K)
    INK_SK_OCC_LAUNCH_GU(FMT_S6_K)
    INK_SK_OCC_LAUNCH_GU(FMT_S8_K)
    return true;
}

template <int MAX_M>
inline bool launch_ink_moe_dn_sk_occ(
    dim3 grid, cudaStream_t st, uint32_t mask,
    const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
    const InkQ8_1_G64* h_sk, const float* topk_w, float* y_pair) {
    const int mode = ink_sk_occ_mode();
    if ((mode != 1 && mode != 2) || mask == 0) return false;
    INK_SK_OCC_LAUNCH_DN(FMT_S1_5)
    INK_SK_OCC_LAUNCH_DN(FMT_S2_K)
    INK_SK_OCC_LAUNCH_DN(FMT_S3_K)
    INK_SK_OCC_LAUNCH_DN(FMT_S4_K)
    INK_SK_OCC_LAUNCH_DN(FMT_S5_K)
    INK_SK_OCC_LAUNCH_DN(FMT_S6_K)
    INK_SK_OCC_LAUNCH_DN(FMT_S8_K)
    return true;
}

#undef INK_SK_OCC_LAUNCH_GU
#undef INK_SK_OCC_LAUNCH_DN
