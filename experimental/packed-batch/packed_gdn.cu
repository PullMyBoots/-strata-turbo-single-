// Arithmetic transcribed from pinned Strata verify_kernels.cu (MIT),
// fb58e0dbc8399662c0e47c76578c6e878b14f6cf. Only independent-session dispatch
// is added; target state precision and per-column reductions are unchanged.
#include "packed_gdn.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
namespace strata::turbo {
namespace {
constexpr int S=128, RG=4, RPG=S/RG;
__global__ void __launch_bounds__(S*RG) groups_run_kernel(GdnGroups groups,
    int C, int h_k, int h_v, const float* gamma, float eps) {
    const GdnGroup a=groups.group[blockIdx.y];
    float* __restrict__ state=a.state;
    const float* __restrict__ hbuf=a.h;
    const float* __restrict__ gate=a.gate;
    const float* __restrict__ beta=a.beta;
    const float* __restrict__ z=a.z;
    float* __restrict__ y=a.y;
    const int T=a.tokens;
    const int32_t* n_keep=nullptr;
    constexpr bool ALL_OUT=true;
    constexpr int t_out_begin=0;

    __shared__ float sk[2][S], sq[2][S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    const int qh = head % h_k;
    const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    const int n = ALL_OUT ? T : (n_keep ? *n_keep : T);
    if (n > 0) {
        if (rg == 0) sk[0][col] = hbuf[qk + qh * S + col];
        else if (rg == 1 && (ALL_OUT || 0 >= t_out_begin)) sq[0][col] = hbuf[qh * S + col];
    }
    const float gam = (rg == 0) ? gamma[col] : 0.0f;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = ALL_OUT ? __ldg(&base[r * row_stride]) : base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const int cur = t & 1, nxt = (t + 1) & 1;
        const float* ht = hbuf + (size_t) t * C;
        const bool need_out = ALL_OUT || (t >= t_out_begin);
        __syncthreads();
        if (t + 1 < n) {
            const float* ht_next = ht + C;
            const bool need_next = ALL_OUT || (t + 1 >= t_out_begin);
            if (rg == 0) sk[nxt][col] = ht_next[qk + qh * S + col];
            else if (rg == 1 && need_next) sq[nxt][col] = ht_next[qh * S + col];
        }
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[cur][rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        if (!need_out) {
#pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = fmaf(g, s[r], sk[cur][rg * RPG + r] * delta);
            continue;
        }
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[cur][rg * RPG + r] * delta);
            o = fmaf(s[r], sq[cur][rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        // every thread takes the shuffles, as the single-token kernel does: inside the `rg == 0` branch hipcc on
        // gfx1151 gave another rounding for 3% of the outputs (gdn_parity section 5 on Aurora)
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if (rg == 0 && (col & 31) == 0) wsum[col >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            const float yv = oc * scale * gam * (1.0f / (1.0f + __expf(-zz)));
            y[(size_t) t * value_dim + head * S + col] = yv;

        }
    }
    if (!ALL_OUT && n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

__global__ void __launch_bounds__(32*RG) groups_commit_kernel(GdnGroups groups,
    int C, int h_k, int h_v) {
    const GdnGroup a=groups.group[blockIdx.z];
    float* __restrict__ state=a.state;
    const float* __restrict__ hbuf=a.h;
    const float* __restrict__ gate=a.gate;
    const float* __restrict__ beta=a.beta;
    const int32_t* __restrict__ n_keep=a.keep;
    const int n_max=a.tokens;
    constexpr bool DBG=false;

    const int n = *n_keep;
    if (n <= 0) return;
    if (DBG && n > n_max) {
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
            printf("strata DBG: gdn_step_commit: n_keep %d is past the window's %d rows: commit skipped (#937)\n", n, n_max);
        return;
    }
    __shared__ float sk[S];
    __shared__ float red[RG][32];
    const int head = blockIdx.x;
    const int col_local = threadIdx.x;
    const int col = blockIdx.y * 32 + col_local;
    const int rg = threadIdx.y;
    const int tid = rg * 32 + col_local;    // 0 .. 127
    const int qh = head % h_k;
    const int qk = S * h_k;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();
        sk[tid] = ht[qk + qh * S + tid];
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col_local] = kv;
        __syncthreads();
        const float kv_col = red[0][col_local] + red[1][col_local] + red[2][col_local] + red[3][col_local];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

void check() {
    auto e=cudaGetLastError();
    if(e!=cudaSuccess){std::fprintf(stderr,"packed GDN: %s\n",cudaGetErrorString(e));std::abort();}
}
} // namespace
void gdn_groups_run(const GdnGroups& groups,int n,int C,int h_k,int h_v,
                    const float* gamma,float eps,void* stream) {
    groups_run_kernel<<<dim3(h_v,n),dim3(S,RG),0,(cudaStream_t)stream>>>(groups,C,h_k,h_v,gamma,eps);
    check();
}
void gdn_groups_commit(const GdnGroups& groups,int n,int C,int h_k,int h_v,void* stream) {
    groups_commit_kernel<<<dim3(h_v,4,n),dim3(32,RG),0,(cudaStream_t)stream>>>(groups,C,h_k,h_v);
    check();
}
} // namespace strata::turbo
