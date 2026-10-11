#pragma once
#include <cstdint>

namespace strata::kernels {
// Per-step records are never reused inside a window. The final ring store
// publishes payload visibility to the CPU; no inference math is changed.
void resident_dispatch_publish(const float* x, const int32_t* ids, const float* weights,
                               int64_t n, int64_t k, float* x_out, int32_t* ids_out,
                               float* weights_out, uint32_t* seq, const uint32_t* skip,
                               uint32_t ring, int32_t* trace, void* stream);
}
