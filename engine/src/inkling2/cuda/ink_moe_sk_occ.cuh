// engine/src/inkling2/cuda/ink_moe_sk_occ.cuh - S_K MoE (gu/dn) の occupancy 版
//
// STRATA_INK_SK_OCC で切り替える (既定 1 = この版、0 = 従来カーネル k_ink_moe_*_sk_fmt のまま):
//   従来は MAX_M==16 で __launch_bounds__(128,2)、REG 160/158/134/132 で 3 CTA/SM (12 warp)
//   こちらは __launch_bounds__(128,4) で REG を 128 以下に強制し 4 CTA/SM (16 warp) にする
//   演算は従来カーネルと同じ式・同じ順序で、結果は bit 一致する (同一プロセス内 20/20 visit で diff 0)
//
// 実測 (SQ-10 / P40、32K prefill): gu(sk) 609.21 -> 567.08 / dn(sk) 358.13 -> 308.64 ms/step、
// prefill 242.3 -> 254.2 ppts (+4.9%)。128K prefill +3.57%、decode +2.01% (8K)
// 重み 1 sub-block 先行ロード版 (旧 OCC=2) は MLP が増えないまま 4.4% 遅かったため削除した
//
// SQ-10 の S_K は S2_K / S3_K だけなので実体はこの 2 形式のみ。他の形式と mask==0 は従来カーネルへ委譲する
// 注: このヘッダは ink_moe_cuda.cuh の namespace strata::ink の内側で include される
#pragma once

// occupancy 版を使うか (既定 1)。getenv は 1 回だけ読む。0 以外は 1 とみなす
inline int ink_sk_occ_mode() {
    static const int mode = [] { const char* e = std::getenv("STRATA_INK_SK_OCC"); return e ? (std::atoi(e) != 0 ? 1 : 0) : 1; }();
    return mode;
}

// gu 用。本体は k_ink_moe_gu_sk_fmt と同一で、違うのは __launch_bounds__ だけ
template <int FMT, int MAX_M>
__global__ void __launch_bounds__(128, 4) k_ink_moe_gu_sk_fmt_occ(
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
            sk_gu_fused_4lane<FMT, MAX_M>(gr, ur, xp, n_e, g, ag, au);
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

// dn 用。本体は k_ink_moe_dn_sk_fmt と同一
template <int FMT, int MAX_M>
__global__ void __launch_bounds__(128, 4) k_ink_moe_dn_sk_fmt_occ(
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
            sk_dn_paired2_4lane<FMT, MAX_M>(pr, hp, n_e, g, a0, a1);
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

// S2_K / S3_K を occupancy 版で起動し、起動した bit を mask から落とす。
// 残りの bit が 0 なら true を返して呼び出し側の従来経路を止める。残れば false で従来経路に任せる
// (mask を参照で受け取るのは、担当外の形式を呼び出し側の if 連鎖にそのまま流すため)
template <int MAX_M>
inline bool launch_ink_moe_gu_sk_occ(
    dim3 grid, cudaStream_t st, uint32_t& mask,
    const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
    const InkQ8_1_G64* x_sk, float* h_f32) {
    // mask==0 は汎用カーネル k_ink_moe_gu_sk (format 非特化) の担当なので従来に任せる
    if (mask == 0 || ink_sk_occ_mode() == 0) return false;
    uint32_t handled = 0;
    if (mask & (1u << FMT_S2_K)) {
        k_ink_moe_gu_sk_fmt_occ<FMT_S2_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, x_sk, h_f32);
        handled |= 1u << FMT_S2_K;
    }
    if (mask & (1u << FMT_S3_K)) {
        k_ink_moe_gu_sk_fmt_occ<FMT_S3_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, x_sk, h_f32);
        handled |= 1u << FMT_S3_K;
    }
    mask &= ~handled;
    return mask == 0;
}

template <int MAX_M>
inline bool launch_ink_moe_dn_sk_occ(
    dim3 grid, cudaStream_t st, uint32_t& mask,
    const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
    const InkQ8_1_G64* h_sk, const float* topk_w, float* y_pair) {
    if (mask == 0 || ink_sk_occ_mode() == 0) return false;
    uint32_t handled = 0;
    if (mask & (1u << FMT_S2_K)) {
        k_ink_moe_dn_sk_fmt_occ<FMT_S2_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, h_sk, topk_w, y_pair);
        handled |= 1u << FMT_S2_K;
    }
    if (mask & (1u << FMT_S3_K)) {
        k_ink_moe_dn_sk_fmt_occ<FMT_S3_K, MAX_M><<<grid, 128, 0, st>>>(sec4_base, experts, tb, h_sk, topk_w, y_pair);
        handled |= 1u << FMT_S3_K;
    }
    mask &= ~handled;
    return mask == 0;
}
