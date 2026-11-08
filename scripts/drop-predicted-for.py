#!/usr/bin/env python3
# drop the unused accessor + snapshot: the read-ahead lives inside RouterLookahead::run(), so nothing else reads them
import io, sys
HPP = "/home/manager/work/wt/lookahead/include/strata/core/expert_source.hpp"
CPP = "/home/manager/work/wt/lookahead/src/core/expert_source.cpp"

def patch(path, old, new):
    s = io.open(path, encoding="utf-8").read()
    if s.count(old) != 1:
        sys.exit("anchor %r found %d times in %s" % (old[:60], s.count(old), path))
    io.open(path, "w", encoding="utf-8", newline="").write(s.replace(old, new))

patch(HPP,
"""    /// 149-pool: the experts the most recent submit predicted for `layer`, copied into `out` (at most `cap`).
    /// Returns how many; 0 when the prediction was for another layer or has not finished.  The CPU pool's read-ahead
    /// uses this: the same prediction the file tier warms, so no second router dot is paid.  A prediction only warms
    /// caches - it never changes which experts are computed or how.
    int64_t predicted_for(int64_t layer, int64_t* out, int64_t cap);
""", "")

patch(HPP,
"""    /// 149-pool: the next layer's prediction, kept for the CPU pool's read-ahead (under `mu_`)
    std::vector<int64_t> last_want_;
    int64_t last_layer_ = -1;
""", "")

patch(CPP,
"""int64_t RouterLookahead::predicted_for(int64_t layer, int64_t* out, int64_t cap) {
    if (out == nullptr || cap <= 0) return 0;
    std::lock_guard<std::mutex> lk(mu_);
    if (last_layer_ != layer) return 0;
    const int64_t n = std::min<int64_t>((int64_t) last_want_.size(), cap);
    for (int64_t j = 0; j < n; ++j) out[j] = last_want_[(size_t) j];
    return n;
}

""", "")

patch(CPP,
"""        if (dj == 0) {   // 149-pool: the next layer's list, for predicted_for and the read-ahead below
            {
                std::lock_guard<std::mutex> lk(mu_);
                last_layer_ = layer;
                last_want_.assign(want.begin(), want.end());
            }
            static const bool pool_pf = [] { const char* v = std::getenv("STRATA_POOL_PREFETCH"); return v != nullptr && v[0] != '0'; }();
            if (pool_pf && !want.empty())
                pool_warm_ahead(src_, layer, want.data(), (int64_t) want.size(), pool_prefetch_budget());
        }""",
"""        // 149-pool: the next layer's CPU experts are read ahead now, under this layer's GPU work (see pool_warm_ahead)
        if (dj == 0) {
            static const bool pool_pf = [] { const char* v = std::getenv("STRATA_POOL_PREFETCH"); return v != nullptr && v[0] != '0'; }();
            if (pool_pf && !want.empty())
                pool_warm_ahead(src_, layer, want.data(), (int64_t) want.size(), pool_prefetch_budget());
        }""")
print("dropped")
