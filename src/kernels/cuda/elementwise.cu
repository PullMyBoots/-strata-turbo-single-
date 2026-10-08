// src/kernels/cuda/elementwise.cu - P2.S5's glue kernels.  See the header for why each exists.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/dp4a.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

__global__ void embedding_gather_kernel(const uint8_t* __restrict__ codes,
                                       const float* __restrict__ scales,
                                       const float* __restrict__ offsets, int64_t n,
                                       int code_bits, int code_bias, int group_elems,
                                       float* __restrict__ out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (codes[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = __fmul_rn((float) (code + code_bias), scales[group]);
    out[i] = __fadd_rn(product, offsets ? offsets[group] : 0.0f);
}

/// `ggml_compute_softplus_f32`: `log1p(exp(x))`, with the large-x branch that avoids overflow.
///
/// The branch is not cosmetic. `exp(89)` overflows f32 and `exp(20)` is already 4.85e8 where `log1p` loses
/// relative precision; above 20 the function is `x` to within f32 anyway.
__device__ __forceinline__ float softplus_dev(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

__global__ void gdn_gate_kernel(const float* __restrict__ alpha, const float* __restrict__ dt,
                                const float* __restrict__ ssm_a, float* __restrict__ gate, int64_t h_v) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h_v) return;
    const int64_t t = i / h_v;      // `n_tokens` is the leading dim; the real call has one token
    const int64_t h = i % h_v;
    gate[i] = softplus_dev(alpha[i] + dt[h]) * ssm_a[h];
    (void) t;
}

__global__ void scale_kernel(float* __restrict__ x, int64_t n, float s) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// `dst[i] += src[i]`.  See the header: this is what lets R4's GPU half and CPU half run at the same
// time and still add up to one `parts` buffer.
__global__ void add_kernel(float* __restrict__ dst, const float* __restrict__ src, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f16_from_f32(x[i]);
}

__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = bf16_from_f32(x[i]);
}

__global__ void silu_kernel(float* __restrict__ x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // DOUBLE then cast, matching `ref/gdn.py`'s numpy: its arrays are f32 but `np.exp` on an f32 array is
    // computed to f32 precision by a different algorithm than `expf`, and the reference is the oracle.  The
    // difference is in the last bits and this is one line.
    const double v = (double) x[i];
    x[i] = (float) (v / (1.0 + exp(-v)));
}

/// A bounded launch: a zero-length grid is illegal, and `n == 0` is a real call (an empty selection).
inline unsigned grid_for(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

/// One WARP per row, reduced through shuffles.  The rows here are short and few - (24, 256), (2, 256),
/// (4, 128) - so a block-per-row tree would spend its time in `__syncthreads` for 8 warps of 32, and the whole
/// call is 30 rows.  A warp reduction with NO shared memory and NO barrier is the shape that fits.
__global__ void rms_norm_weighted_kernel(float* __restrict__ x, const float* __restrict__ w, int64_t rows,
                                         int64_t cols, float eps) {

    const int lane = threadIdx.x & 31;
    const int64_t row = ((int64_t) blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    // **THE ROW GUARD IS NOT DECORATION, AND ITS ABSENCE WAS THE QSA BUG.**  The launcher rounds the grid up to
    // whole 4-warp blocks, so a call with `rows` = 2 - the QSA k-norm, the only non-multiple-of-4 row count in
    // the engine - runs EIGHT warps: rows 0 and 1 are the data, and rows 2 and 3 write 2*cols floats PAST the
    // end of `b.kcur`, which in the arena is exactly where `b.vcur` begins.  `b.vcur` was therefore silently
    // replaced by two rows of `rms_norm(...) * attn_k_norm`: a normalized vector with l2 ~20.6 against V's
    // ~5.7, which the attention then attended to.  `rope_neox_kernel` has had this guard all along.
    if (row >= rows) return;
    float* r = x + row * cols;
    float acc = 0.0f;
    for (int64_t c = lane; c < cols; c += 32) acc += r[c] * r[c];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    // The MEAN, not the sum: `ref/qsa.py::rms_norm` divides by `np.mean(np.square(x))`.  Broadcasting the
    // reciprocal from lane 0 keeps all 32 lanes on the same value - computing `rsqrt` per lane would be the
    // same number but a needless 32-way divergence in the last bit.
    float inv = 0.0f;
    if (lane == 0) inv = rsqrtf(acc / (float) cols + eps);
    inv = __shfl_sync(0xFFFFFFFFu, inv, 0);
    for (int64_t c = lane; c < cols; c += 32) r[c] = (w ? r[c] * w[c] : r[c]) * inv;
}

void sync_if_needed(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

bool check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace

void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets,
                      int64_t n, int code_bits, int code_bias, int group_elems,
                      float* out, void* stream) {
    if (n <= 0) return;
    embedding_gather_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(
        codes, scales, offsets, n, code_bits, code_bias, group_elems, out);
    check_launch("embedding_gather");
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v, void* stream) {
    if (n_tokens <= 0 || h_v <= 0) return;
    const int64_t n = n_tokens * h_v;
    gdn_gate_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(alpha, dt, ssm_a, gate, h_v);
    check_launch("gdn_gate");
    sync_if_needed(stream, "gdn_gate");
}

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    scale_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(x, n, s);
    check_launch("scale_inplace");
    sync_if_needed(stream, "scale_inplace");
}

void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    add_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(dst, src, n);
    check_launch("add_inplace");
    sync_if_needed(stream, "add_inplace");
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f16_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(x, y, n);
    check_launch("f32_to_f16_bulk");
    sync_if_needed(stream, "f32_to_f16_bulk");
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_bf16_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(x, y, n);
    check_launch("f32_to_bf16_bulk");
    sync_if_needed(stream, "f32_to_bf16_bulk");
}

void silu_inplace(float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    silu_kernel<<<grid_for(n), THREADS, 0, (cudaStream_t) stream>>>(x, n);
    check_launch("silu_inplace");
    sync_if_needed(stream, "silu_inplace");
}

/// THE DOORBELL.  One thread, one INCREMENT - the cost is the launch, and inside a graph that is paid once.
///
/// **IT INCREMENTS THE MEMORY, AND IT DOES NOT TAKE THE VALUE AS AN ARGUMENT.**  The first version did -
/// doorbell_ring(d_seq, *(h_seq) + 1u) - and that is a CLONED LITERAL: the expression is evaluated on the HOST
/// at CAPTURE time, so the graph rings 1 on every replay and a host waiting for a change waits forever.  It
/// would have looked like a working doorbell on the first token, which is the worst way for it to be wrong.
/// **THE FENCE IS NOT DECORATION, AND ITS ABSENCE WAS COSTING ~10 ms PER TOKEN ON THE HOST SIDE.**
///
/// The ring PUBLISHES three buffers the host is about to read - `h_x_f`, `h_ids`, `h_weights` - so the increment
/// must be ordered after their writes, or the host can observe the ring and then read a payload that has not
/// landed.  `__threadfence_system()` is what orders them, and it covers the HOST as well as the device, which is
/// the whole point: the reader is the CPU.
///
/// Without it the host loop was compensating with a driver call.  `session_loop` polled with `cudaEventQuery`
/// on EVERY spin iteration, because round 195 had measured that a memory-only spin never saw the datum flip -
/// and that measurement was right about the symptom and wrong about the cause.  The cause is this missing fence:
/// the write was not ordered into host-visible memory, so no amount of reading it would show it, and the driver
/// call was flushing the whole pipeline enough to make it appear.  A 10-22 us driver call per iteration is a
/// very expensive substitute for one fence instruction.
///
/// **THE STORE IS VOLATILE**, like `doorbell_publish_kernel`'s.  On RDNA4 (gfx1201) a plain store to mapped pinned
/// memory stays in the GPU's L2 until the stream is synchronized - the host never saw the ring (tests/hip/handoff:
/// 0 of 100 rings seen without a sync; volatile, a system-scope atomic store or a fence after the store: 100 of 100).
__global__ void doorbell_ring_kernel(uint32_t* seq) {
    __threadfence_system();
    *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
}

__global__ void doorbell_wait_kernel(const volatile uint32_t* flag, const volatile uint32_t* seq) {
    const uint32_t want = *seq;
    while (*flag != want) strata_spin_pause();
    __threadfence_system();
}

void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    if (d_flag == nullptr || d_seq == nullptr) return;
    doorbell_wait_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(d_flag, d_seq);
    check_launch("doorbell_wait");
}

__global__ void copy_from_mapped_kernel(float4* __restrict__ dst, const volatile float4* src, int64_t n4) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x) {
        const float4 v = const_cast<const float4*>(src)[i];
        dst[i] = v;
    }
}

// the CPU rows of a verify window, skipping the rows the GPU plan computes itself (the
// pool writes +0.0 into those, so this writes +0.0 too): block = row, the plan's hit rows `dst[0, *count)`.
__global__ void copy_rows_from_mapped_kernel(float4* __restrict__ dst, const volatile float4* src, int64_t row4,
                                             const int32_t* __restrict__ hit_rows, const int32_t* __restrict__ count) {
    const int row = blockIdx.x;
    __shared__ int hit;
    if (threadIdx.x == 0) {
        int h = 0;
        const int c = *count;
        for (int i = 0; i < c; ++i) h |= hit_rows[i] == row;
        hit = h;
    }
    __syncthreads();
    float4* d = dst + (int64_t) row * row4;
    if (hit) {
        for (int64_t i = threadIdx.x; i < row4; i += blockDim.x) d[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        const volatile float4* sr = src + (int64_t) row * row4;
        for (int64_t i = threadIdx.x; i < row4; i += blockDim.x) d[i] = const_cast<const float4*>(sr)[i];
    }
}
namespace {
__global__ void scatter_rows_kernel(const float4* __restrict__ src, float4* dst, const int32_t* __restrict__ rows,
                                    int64_t w4, const int32_t* count = nullptr) {
    const int64_t r = blockIdx.x;
    if (count != nullptr && r >= *count) return;
    const float4* s = src + r * w4;
    float4* d = dst + (int64_t) rows[r] * w4;
    for (int64_t i = threadIdx.x; i < w4; i += blockDim.x) d[i] = s[i];
}
}  // namespace
void scatter_rows_f32(const float* src, float* dst, const int32_t* rows, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "scatter_rows_f32: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    scatter_rows_kernel<<<(unsigned) n, 128, 0, (cudaStream_t) stream>>>((const float4*) src, (float4*) dst, rows, width / 4);
}
void scatter_rows_f32_counted(const float* src, float* dst, const int32_t* rows, const int32_t* count,
                              int64_t cap, int64_t width, void* stream) {
    if (cap <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "scatter_rows_f32_counted: width/alignment invalid\n");
        std::exit(1);
    }
    scatter_rows_kernel<<<(unsigned) cap, 128, 0, (cudaStream_t) stream>>>((const float4*) src, (float4*) dst, rows,
                                                                       width / 4, count);
    check_launch("scatter_rows_f32_counted");
}
void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows,
                           const int32_t* count, void* stream) {
    if (rows <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_rows_from_mapped: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    copy_rows_from_mapped_kernel<<<(unsigned) rows, 128, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src,
                                                                                  width / 4, hit_rows, count);
}
void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    if ((n & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_from_mapped: n must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    const int64_t n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    copy_from_mapped_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src, n4);
    check_launch("copy_from_mapped");
}

__global__ void doorbell_publish_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids,
                                        const float* __restrict__ w, int n, int k, float* x_out, int32_t* ids_out,
                                        float* w_out, uint32_t* seq) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) x_out[i] = x[i];
    if ((int) threadIdx.x < k) { ids_out[threadIdx.x] = ids[threadIdx.x]; w_out[threadIdx.x] = w[threadIdx.x]; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
    }
}

void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    doorbell_publish_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(x, ids, weights, (int) n, (int) k, x_out, ids_out,
                                                                    weights_out, d_seq);
    check_launch("doorbell_publish");
}

__global__ void copy_i32_from_mapped_kernel(int32_t* __restrict__ dst, const volatile int32_t* src, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    copy_i32_from_mapped_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(dst, (const volatile int32_t*) src, (int) n);
    check_launch("copy_i32_from_mapped");
}

void doorbell_ring(uint32_t* d_seq, void* stream) {
    if (d_seq == nullptr) return;
    doorbell_ring_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(d_seq);
    check_launch("doorbell_ring");
    sync_if_needed(stream, "doorbell_ring");
}

void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    // 4 warps per block, so a row count that is not a multiple of 4 wastes at most 3 warps rather than
    // launching a block per row for a 2-row call.
    const unsigned warps_per_block = 4;
    const unsigned grid = (unsigned) ((rows + warps_per_block - 1) / warps_per_block);
    rms_norm_weighted_kernel<<<grid, warps_per_block * 32, 0, (cudaStream_t) stream>>>(x, w, rows, cols, eps);
    check_launch("rms_norm_weighted");
    sync_if_needed(stream, "rms_norm_weighted");
}

__global__ void mtp_hidden_correct_kernel(float* hidden, float* snapshot, const float* state,
                                          int width, int depth, int depths, float gain) {
    const int i = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const float* stats = state + (size_t)depths * width + depth * 4;
    const float x = hidden[i];
    snapshot[i] = x;
    // Require evidence on held-out next-round observations before applying
    // a mean correction. Even an unhelpful correction cannot change output:
    // the unmodified target model verifies every proposed token.
    if (stats[0] >= 8 && stats[2] < stats[1] * 0.99f)
        hidden[i] = fmaf(gain, state[(size_t)depth * width + i], x);
}

__global__ void mtp_hidden_learn_kernel(const float* teacher, const float* snapshot, float* state,
                                        int width, int depths, float gain, float rate) {
    const int depth = blockIdx.x, tid = threadIdx.x;
    float raw = 0, corrected = 0;
    float* bias = state + (size_t)depth * width;
    for (int i = tid; i < width; i += blockDim.x) {
        const float t = teacher[(size_t)depth * width + i], s = snapshot[(size_t)depth * width + i];
        const float old = bias[i];
        if (!isfinite(t) || !isfinite(s) || !isfinite(old)) continue;
        const float diff = t - s;
        const float after = t - fmaf(gain, old, s);
        raw += diff * diff;
        corrected += after * after;
        // Prevent a rare outlier from dominating the request's draft bias.
        const float bounded = fminf(1.0f, fmaxf(-1.0f, diff));
        bias[i] = fmaf(rate, bounded - old, old);
    }
    __shared__ float red_raw[8], red_corrected[8];
    for (int o = 16; o > 0; o >>= 1) {
        raw += __shfl_xor_sync(0xffffffffu, raw, o);
        corrected += __shfl_xor_sync(0xffffffffu, corrected, o);
    }
    if ((tid & 31) == 0) { red_raw[tid >> 5] = raw; red_corrected[tid >> 5] = corrected; }
    __syncthreads();
    if (tid == 0) {
        raw = 0; corrected = 0;
        for (int w = 0; w < 8; ++w) { raw += red_raw[w]; corrected += red_corrected[w]; }
        float* stats = state + (size_t)depths * width + depth * 4;
        stats[0] += 1;
        stats[1] = stats[0] <= 1 ? raw / width : fmaf(0.1f, raw / width - stats[1], stats[1]);
        stats[2] = stats[0] <= 1 ? corrected / width : fmaf(0.1f, corrected / width - stats[2], stats[2]);
    }
}

void mtp_hidden_correct(float* hidden, float* snapshot, const float* state,
                        int width, int depth, int depths, float gain, void* stream) {
    if (width <= 0 || depth < 0 || depth >= depths) return;
    mtp_hidden_correct_kernel<<<(width + 255) / 256, 256, 0, (cudaStream_t)stream>>>(hidden, snapshot, state, width, depth, depths, gain);
    check_launch("mtp_hidden_correct");
}
void mtp_hidden_learn(const float* teacher, const float* snapshots, float* state,
                      int width, int rows, int depths, float gain, float rate, void* stream) {
    if (width <= 0 || rows <= 0 || rows > depths) return;
    mtp_hidden_learn_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(teacher, snapshots, state, width, depths, gain, rate);
    check_launch("mtp_hidden_learn");
}

__device__ __forceinline__ float mtp_affine_adjust(float x, const float* state, size_t p, size_t plane, float gain) {
    const float mx = state[p], my = state[plane + p];
    const float variance = fmaxf(0.0f, state[2 * plane + p]), covariance = state[3 * plane + p];
    // Identity-centered ridge prior and bounded extrapolation. Calibration
    // changes draft proposals only; the target head is never corrected.
    const float slope = fminf(2.0f, fmaxf(0.25f, (covariance + 0.02f) / (variance + 0.02f)));
    const float bias = fminf(1.0f, fmaxf(-1.0f, my - slope * mx));
    return fmaf(gain, fmaf(slope, x, bias) - x, x);
}

__global__ void mtp_hidden_affine_correct_kernel(float* hidden, float* snapshot, const float* state,
                                                 int width, int depth, int depths, float gain) {
    const int i = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const size_t plane = (size_t)depths * width;
    const float* stats = state + 4 * plane + depth * 4;
    const float x = hidden[i];
    snapshot[i] = x;
    if (stats[0] >= 8 && stats[2] < stats[1] * 0.99f)
        hidden[i] = mtp_affine_adjust(x, state, (size_t)depth * width + i, plane, gain);
}

__global__ void mtp_hidden_affine_learn_kernel(const float* teacher, const float* snapshot, float* state,
                                               int width, int depths, float gain, float rate) {
    const int depth = blockIdx.x, tid = threadIdx.x;
    const size_t plane = (size_t)depths * width;
    float* stats = state + 4 * plane + depth * 4;
    const bool first = stats[0] < 1;
    float raw = 0, corrected = 0;
    for (int i = tid; i < width; i += blockDim.x) {
        const size_t p = (size_t)depth * width + i;
        const float x = snapshot[p], y = teacher[p];
        if (!isfinite(x) || !isfinite(y)) continue;
        const float after = y - mtp_affine_adjust(x, state, p, plane, gain);
        const float diff = y - x;
        raw += diff * diff; corrected += after * after;
        if (first) {
            state[p] = x; state[plane + p] = y;
            state[2 * plane + p] = 0; state[3 * plane + p] = 0;
        } else {
            const float dx = x - state[p], dy = y - state[plane + p];
            state[p] = fmaf(rate, dx, state[p]);
            state[plane + p] = fmaf(rate, dy, state[plane + p]);
            state[2 * plane + p] = (1.0f - rate) * fmaf(rate, dx * dx, state[2 * plane + p]);
            state[3 * plane + p] = (1.0f - rate) * fmaf(rate, dx * dy, state[3 * plane + p]);
        }
    }
    __shared__ float red_raw[8], red_corrected[8];
    for (int o = 16; o > 0; o >>= 1) {
        raw += __shfl_xor_sync(0xffffffffu, raw, o);
        corrected += __shfl_xor_sync(0xffffffffu, corrected, o);
    }
    if ((tid & 31) == 0) { red_raw[tid >> 5] = raw; red_corrected[tid >> 5] = corrected; }
    __syncthreads();
    if (tid == 0) {
        raw = 0; corrected = 0;
        for (int w = 0; w < 8; ++w) { raw += red_raw[w]; corrected += red_corrected[w]; }
        stats[0] += 1;
        stats[1] = first ? raw / width : fmaf(0.1f, raw / width - stats[1], stats[1]);
        stats[2] = first ? corrected / width : fmaf(0.1f, corrected / width - stats[2], stats[2]);
    }
}

void mtp_hidden_affine_correct(float* hidden, float* snapshot, const float* state,
                               int width, int depth, int depths, float gain, void* stream) {
    if (width <= 0 || depth < 0 || depth >= depths) return;
    mtp_hidden_affine_correct_kernel<<<(width + 255) / 256, 256, 0, (cudaStream_t)stream>>>(hidden, snapshot, state, width, depth, depths, gain);
    check_launch("mtp_hidden_affine_correct");
}
void mtp_hidden_affine_learn(const float* teacher, const float* snapshots, float* state,
                             int width, int rows, int depths, float gain, float rate, void* stream) {
    if (width <= 0 || rows <= 0 || rows > depths) return;
    mtp_hidden_affine_learn_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(teacher, snapshots, state, width, depths, gain, rate);
    check_launch("mtp_hidden_affine_learn");
}

}  // namespace strata::kernels

namespace strata::kernels { namespace {
__global__ void mtp_token_bias_apply_kernel(float* logits,const float* bias,const int32_t* vocab,int count){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)logits[i]+=bias[vocab?vocab[i]:i];
}
__global__ void mtp_token_bias_learn_kernel(float* bias,const int32_t* labels,int rows,int vocab,float gain){
 // One thread orders updates when different depths refer to the same token.
 if(threadIdx.x)return;
 for(int j=0;j<rows;++j){const int q=labels[j],t=labels[8+j];if(q<0||q>=vocab||t<0||t>=vocab)continue;
  if(q==t){bias[q]*=.95f;continue;}
  const float step=gain/(1.f+.125f*j);
  bias[q]=fmaxf(-1.5f,bias[q]-step);bias[t]=fminf(1.5f,bias[t]+step);
 }
}
}
void mtp_token_bias_apply(float* logits,const float* bias,const int32_t* vocab,int count,void* stream){
 mtp_token_bias_apply_kernel<<<(count+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(logits,bias,vocab,count);
}
void mtp_token_bias_learn(float* bias,const int32_t* labels,int rows,int vocab,float gain,void* stream){
 mtp_token_bias_learn_kernel<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(bias,labels,rows,vocab,gain);
}
} // namespace strata::kernels

namespace strata::kernels { namespace {
__global__ void mtp_token_feature_score_kernel(const float* query,const float* weights,float* scores,
                                               const int32_t* used,int width,int capacity){
    const int slot=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(slot>=capacity||slot>=*used)return;
    float dot=0,norm=0;
    for(int i=lane;i<width;i+=32){const float x=query[i];dot=fmaf(weights[(size_t)slot*width+i],x,dot);norm=fmaf(x,x,norm);}
    for(int d=16;d;d>>=1){dot+=__shfl_down_sync(0xffffffff,dot,d);norm+=__shfl_down_sync(0xffffffff,norm,d);}
    if(!lane)scores[slot]=fminf(1.5f,fmaxf(-1.5f,dot*rsqrtf(fmaxf(norm/width,1e-12f))));
}
__global__ void mtp_token_feature_add_kernel(float* logits,const float* scores,const int32_t* slots,
                                            const int32_t* vocab,int count){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const int slot=slots[vocab?vocab[i]:i];if(slot>=0)logits[i]+=scores[slot];
}
__global__ void mtp_token_feature_learn_kernel(float* weights,const float* snapshots,int32_t* slots,int32_t* used,
                                               const int32_t* labels,int rows,int width,int vocab,float gain){
    __shared__ float sums[8],scale;
    const int tid=threadIdx.x;
    // Ordered rows make updates deterministic even when labels repeat.
    for(int j=0;j<rows;++j){
        const int q=labels[j],t=labels[8+j],qs=labels[16+j],ts=labels[24+j];
        if(q<0||q>=vocab||t<0||t>=vocab||qs<0||ts<0)continue;
        if(!tid){slots[q]=qs;slots[t]=ts;}
        if(q==t){
            for(int i=tid;i<width;i+=blockDim.x)weights[(size_t)qs*width+i]*=.995f;
            __syncthreads();continue;
        }
        const float* x=snapshots+(size_t)j*width;float norm=0;
        for(int i=tid;i<width;i+=blockDim.x)norm=fmaf(x[i],x[i],norm);
        for(int d=16;d;d>>=1)norm+=__shfl_down_sync(0xffffffff,norm,d);
        if(!(tid&31))sums[tid/32]=norm;__syncthreads();
        if(!tid){norm=0;for(int k=0;k<8;++k)norm+=sums[k];scale=(gain/(1.f+.125f*j))*rsqrtf(fmaxf(norm/width,1e-12f))/width;}
        __syncthreads();
        for(int i=tid;i<width;i+=blockDim.x){
            const float d=x[i]*scale;
            weights[(size_t)qs*width+i]=fmaxf(-.25f,fminf(.25f,weights[(size_t)qs*width+i]-d));
            weights[(size_t)ts*width+i]=fmaxf(-.25f,fminf(.25f,weights[(size_t)ts*width+i]+d));
        }
        __syncthreads();
    }
    if(!tid)*used=labels[32];
}
}
void mtp_token_feature_apply(float* logits,const float* query,const float* weights,float* scores,
    const int32_t* slots,const int32_t* used,const int32_t* vocab,int count,int width,int capacity,void* stream){
    mtp_token_feature_score_kernel<<<(capacity+7)/8,256,0,static_cast<cudaStream_t>(stream)>>>(query,weights,scores,used,width,capacity);
    mtp_token_feature_add_kernel<<<(count+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(logits,scores,slots,vocab,count);
}
void mtp_token_feature_learn(float* weights,const float* snapshots,int32_t* slots,int32_t* used,
    const int32_t* labels,int rows,int width,int vocab,float gain,void* stream){
    mtp_token_feature_learn_kernel<<<1,256,0,static_cast<cudaStream_t>(stream)>>>(weights,snapshots,slots,used,labels,rows,width,vocab,gain);
}
} // namespace strata::kernels

namespace strata::kernels { namespace {
__device__ void peer_head_wait(uint32_t* state,int index,int error){
    const uint32_t epoch=*reinterpret_cast<const volatile uint32_t*>(state);
    volatile uint32_t* ready=state+index;const unsigned long long start=clock64();
    while(*ready!=epoch){if(clock64()-start>12000000000ULL){*reinterpret_cast<volatile uint32_t*>(state+96)=error;break;}}
}
__global__ void peer_head_publish_query_kernel(const uint32_t* input,volatile uint32_t* mapped,int words,uint32_t* state){
    for(int i=threadIdx.x;i<words;i+=blockDim.x)mapped[i]=input[i];
    __threadfence_system();__syncthreads();
    if(!threadIdx.x){*reinterpret_cast<volatile uint32_t*>(state+32)=*reinterpret_cast<const volatile uint32_t*>(state);__threadfence_system();}
}
__global__ void peer_head_receive_query_kernel(const volatile uint32_t* mapped,uint32_t* output,int words,uint32_t* state){
    if(!threadIdx.x)peer_head_wait(state,32,1);__syncthreads();
    for(int i=threadIdx.x;i<words;i+=blockDim.x)output[i]=mapped[i];
}
__global__ void peer_head_publish_result_kernel(const int32_t* ids,const float* scores,volatile int32_t* mapped_ids,
                                               volatile float* mapped_scores,int rows,uint32_t* state){
    const int i=threadIdx.x;if(i<rows){mapped_ids[i]=ids[i];mapped_scores[i]=scores[i];}
    __threadfence_system();__syncthreads();
    if(!i){*reinterpret_cast<volatile uint32_t*>(state+64)=*reinterpret_cast<const volatile uint32_t*>(state);__threadfence_system();}
}
__global__ void peer_head_merge_result_kernel(const volatile int32_t* peer_ids,const volatile float* peer_scores,
                                             const int32_t* tail_ids,const float* tail_scores,int32_t* output,int rows,int split,uint32_t* state){
    if(!threadIdx.x)peer_head_wait(state,64,2);__syncthreads();
    const int i=threadIdx.x;if(i>=rows)return;const float a=peer_scores[i],b=tail_scores[i];
    const bool tail_wins=b>-__int_as_float(0x7f800000)&&(isnan(a)||b>a);
    output[i]=tail_wins?split+tail_ids[i]:peer_ids[i];
}
}
void peer_head_publish_query(const void* input,void* mapped,int words,uint32_t* state,void* stream){
    peer_head_publish_query_kernel<<<1,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const uint32_t*>(input),static_cast<volatile uint32_t*>(mapped),words,state);
}
void peer_head_receive_query(const void* mapped,void* output,int words,uint32_t* state,void* stream){
    peer_head_receive_query_kernel<<<1,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const volatile uint32_t*>(mapped),static_cast<uint32_t*>(output),words,state);
}
void peer_head_publish_result(const int32_t* ids,const float* scores,int32_t* mapped_ids,float* mapped_scores,int rows,uint32_t* state,void* stream){
    peer_head_publish_result_kernel<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(ids,scores,mapped_ids,mapped_scores,rows,state);
}
void peer_head_merge_result(const int32_t* mapped_ids,const float* mapped_scores,const int32_t* tail_ids,const float* tail_scores,int32_t* output,int rows,int split,uint32_t* state,void* stream){
    peer_head_merge_result_kernel<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(mapped_ids,mapped_scores,tail_ids,tail_scores,output,rows,split,state);
}
} // namespace strata::kernels
