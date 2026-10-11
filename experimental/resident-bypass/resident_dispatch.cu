#include "resident_dispatch.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
__global__ void publish_kernel(const float* __restrict__ x,
                               const int32_t* __restrict__ ids,
                               const float* __restrict__ weights,
                               int64_t n, int64_t k, float* __restrict__ x_out,
                               int32_t* __restrict__ ids_out,
                               float* __restrict__ weights_out,
                               uint32_t* seq, const uint32_t* skip,
                               uint32_t ring, int32_t* trace) {
    const bool resident = *skip == ring;
    if (resident) {
        // The shared legacy route staging may already belong to a later
        // layer when the host wakes up. Keep every bypassed route stable.
        for (int64_t i = threadIdx.x; i < k; i += blockDim.x) trace[2 + i] = ids[i];
    } else {
        // The unchanged miss path blocks the stream at its plan/CPU flags,
        // so the ordinary activation staging cannot be overwritten yet.
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) x_out[i] = x[i];
        for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
            ids_out[i] = ids[i];
            weights_out[i] = weights[i];
        }
    }
    // Every writer must fence its mapped stores before thread 0 publishes
    // the completion ring. A single lane's fence cannot order other lanes.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        trace[1] = resident ? 1 : 0;
        __threadfence_system();
        *(volatile uint32_t*) seq = ring;
        *(volatile uint32_t*) trace = ring;
    }
}
}

void resident_dispatch_publish(const float* x, const int32_t* ids, const float* weights,
                               int64_t n, int64_t k, float* x_out, int32_t* ids_out,
                               float* weights_out, uint32_t* seq, const uint32_t* skip,
                               uint32_t ring, int32_t* trace, void* stream) {
    // Same block size as the existing doorbell. Selection is based solely
    // on actual route residency, never on a device name or timing threshold.
    publish_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(x, ids, weights, n, k,
        x_out, ids_out, weights_out, seq, skip, ring, trace);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "resident_dispatch_publish: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}
}
