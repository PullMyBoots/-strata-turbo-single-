#pragma once

#include <cstdint>

namespace strata::turbo {

struct GdnGroup {
    float* state;
    const float* h;
    const float* gate;
    const float* beta;
    const float* z;
    float* y;
    const int32_t* keep;
    int tokens;
};

struct GdnGroups { GdnGroup group[8]; };

// Sessions occupy independent blocks in the same launch. Every per-column
// recurrence, reduction and normalization retains the upstream operation order.
void gdn_groups_run(const GdnGroups&, int groups, int channels, int h_k, int h_v,
                    const float* gamma, float eps, void* stream);
void gdn_groups_commit(const GdnGroups&, int groups, int channels, int h_k,
                       int h_v, void* stream);

} // namespace strata::turbo
