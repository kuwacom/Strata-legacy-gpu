#!/usr/bin/env python3
# cpu-lookahead.diff: CPU pool の 1 層先読み（最小実装）
import io, sys

HPP = "/home/manager/work/wt/lookahead/include/strata/core/expert_source.hpp"
CPP = "/home/manager/work/wt/lookahead/src/core/expert_source.cpp"

def patch(path, old, new, count=1):
    s = io.open(path, encoding="utf-8").read()
    n = s.count(old)
    if n != count:
        sys.exit("anchor %r found %d times in %s (expected %d)" % (old[:60], n, path, count))
    s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="").write(s)

# ---------- header: public accessor + private snapshot ----------
patch(HPP,
"""    /// STRATA_LOOKAHEAD_STATS=1 (measurement only): remember each prediction (top k, k+4 and k+10 per token) and let the
    /// dispatch score it against the experts the next layer really routes outside the GPU cache.
    void enable_stats(int64_t n_layers);""",
"""    /// 149-pool: the experts the most recent submit predicted for `layer`, copied into `out` (at most `cap`).
    /// Returns how many; 0 when the prediction was for another layer or has not finished.  The CPU pool's read-ahead
    /// uses this: the same prediction the file tier warms, so no second router dot is paid.  A prediction only warms
    /// caches - it never changes which experts are computed or how.
    int64_t predicted_for(int64_t layer, int64_t* out, int64_t cap);
    /// STRATA_LOOKAHEAD_STATS=1 (measurement only): remember each prediction (top k, k+4 and k+10 per token) and let the
    /// dispatch score it against the experts the next layer really routes outside the GPU cache.
    void enable_stats(int64_t n_layers);""")

patch(HPP,
"""    std::vector<float> x_;
    std::atomic<int64_t> predicted_{0}, skipped_{0};""",
"""    std::vector<float> x_;
    std::atomic<int64_t> predicted_{0}, skipped_{0};
    /// 149-pool: the next layer's prediction, kept for the CPU pool's read-ahead (under `mu_`)
    std::vector<int64_t> last_want_;
    int64_t last_layer_ = -1;""")

# ---------- cpp: budget + read-ahead helper, inserted before run() ----------
patch(CPP,
"""void RouterLookahead::run() {""",
"""// 149-pool: the read-ahead's byte budget for one layer (STRATA_POOL_PREFETCH_MB, default 8 MiB - about six experts on
// the IQ3_XXS pack, where a decode layer routes one to three of them outside the VRAM cache).
static uint64_t pool_prefetch_budget() {
    static const uint64_t mb = [] {
        const char* v = std::getenv("STRATA_POOL_PREFETCH_MB");
        const long n = v != nullptr ? std::atol(v) : 8;
        return (uint64_t) (n > 0 ? n : 8);
    }();
    return mb << 20;
}

// 149-pool: the CPU pool's read-ahead.  The pool's workers read a layer's expert blobs only after that layer's router
// has rung, and the GPU then waits for them at the end of the layer's FFN (the profile's `waitCPU`).  The prediction the
// file tier already pays for names the next layer's CPU experts a layer early, so this touches their blobs now: the
// DRAM read moves under the current layer's GPU work instead of onto the pool's own window.  It reads and discards -
// no expert is computed, no weight is changed, and the output is the same bits with the read-ahead on or off.
static void pool_warm_ahead(const ExpertSource* src, int64_t layer, const int64_t* experts, int64_t n, uint64_t budget) {
    const FileExpertSource* fs = dynamic_cast<const FileExpertSource*>(src);
    if (fs == nullptr || layer < 0 || n <= 0) return;
    const uint64_t bytes = strata::kernels::cpu::expert_layout().blob_bytes(layer);
    if (bytes == 0) return;
    volatile uint8_t sink = 0;
    uint64_t used = 0;
    for (int64_t j = 0; j < n && used + bytes <= budget; ++j) {
        const uint8_t* b = fs->resident_blob(layer, experts[j]);   // the RAM copy only: a file-tier read is warm's job
        if (b == nullptr) continue;
        // one line at a time: a sequential read of the blob lands in this core's L2 (4 MiB) and the shared L3, where
        // the pool's workers then hit instead of going to DRAM.  The sink defeats the compiler's dead-store removal.
        for (uint64_t o = 0; o < bytes; o += 64) sink ^= b[o];
        used += bytes;
    }
    (void) sink;
}

void RouterLookahead::run() {""")

# ---------- cpp: snapshot + the read-ahead call ----------
patch(CPP,
"""        predicted_.fetch_add((int64_t) want.size(), std::memory_order_relaxed);""",
"""        predicted_.fetch_add((int64_t) want.size(), std::memory_order_relaxed);
        if (dj == 0) {   // 149-pool: the next layer's list, for predicted_for and the read-ahead below
            {
                std::lock_guard<std::mutex> lk(mu_);
                last_layer_ = layer;
                last_want_.assign(want.begin(), want.end());
            }
            static const bool pool_pf = [] { const char* v = std::getenv("STRATA_POOL_PREFETCH"); return v != nullptr && v[0] != '0'; }();
            if (pool_pf && !want.empty())
                pool_warm_ahead(src_, layer, want.data(), (int64_t) want.size(), pool_prefetch_budget());
        }""")

# ---------- cpp: the accessor ----------
patch(CPP,
"""void RouterLookahead::enable_stats(int64_t n_layers) {""",
"""int64_t RouterLookahead::predicted_for(int64_t layer, int64_t* out, int64_t cap) {
    if (out == nullptr || cap <= 0) return 0;
    std::lock_guard<std::mutex> lk(mu_);
    if (last_layer_ != layer) return 0;
    const int64_t n = std::min<int64_t>((int64_t) last_want_.size(), cap);
    for (int64_t j = 0; j < n; ++j) out[j] = last_want_[(size_t) j];
    return n;
}

void RouterLookahead::enable_stats(int64_t n_layers) {""")
print("patch applied")
