// Exact decode top-k: compare against both the original GPU kernel and an independent
// CPU sort, including ties, signed zero, infinities, NaNs, and partial tail blocks.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace k = strata::kernels;
static void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); }
}
static uint32_t key(float x) {
    const float v = x + 0.f;
    if (std::isnan(v)) return 0;
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}
static bool check(int64_t ctx, int nq, int mode) {
    constexpr int64_t stride = 262144 / 4 + 2;
    const auto shape = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, shape);
    std::vector<int32_t> steps(nq * k::kStepCount, 0);
    std::vector<float> scores(nq * stride, -999.f);
    std::mt19937 rng(271828);
    std::uniform_real_distribution<float> dist(0.f, 200.f);
    const float special[] = {0.f, -0.f, -1.f, 1.f, std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    for (int q = 0; q < nq; ++q) {
        auto* st = steps.data() + q * k::kStepCount;
        const int64_t n = ctx - nq + q + 1;
        st[k::kStepPos] = n - 1; st[k::kStepNKv] = n; st[k::kStepNBid] = n / 4;
        st[k::kStepWidth] = k::qsa_selection_width(n, shape);
        for (int b = 0; b <= n / 4; ++b)
            scores[q * stride + b] = mode == 1 ? 0.f : mode == 2 ? float(b % 7) :
                                      mode == 3 ? special[b % 7] : dist(rng);
        if (n % 4 && mode != 3) scores[q * stride + n / 4] += 1e9f;
    }
    float* ds; int32_t *dt, *dr, *dn; void* scratch;
    ck(cudaMalloc(&ds, scores.size() * sizeof(float)));
    ck(cudaMalloc(&dt, steps.size() * sizeof(int32_t)));
    ck(cudaMalloc(&dr, nq * cap * sizeof(int32_t)));
    ck(cudaMalloc(&dn, nq * cap * sizeof(int32_t)));
    ck(cudaMalloc(&scratch, k::qsa_topk_parallel_scratch_bytes(nq)));
    ck(cudaMemcpy(ds, scores.data(), scores.size() * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dt, steps.data(), steps.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    auto ref = [&] { k::qsa_block_topk_ref(ds, dt, nq, stride, cap, shape, dr, nullptr); };
    auto candidate = [&] { k::qsa_block_topk_parallel(ds, dt, nq, stride, cap, shape, dn, scratch, nullptr); };
    ref(); candidate(); ck(cudaDeviceSynchronize());
    std::vector<int32_t> r(nq * cap), a(nq * cap);
    ck(cudaMemcpy(r.data(), dr, r.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(a.data(), dn, a.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int q = 0; q < nq; ++q) {
        const auto* st = steps.data() + q * k::kStepCount;
        const int n = st[k::kStepNKv], width = st[k::kStepWidth];
        std::vector<int> blocks((n + 3) / 4);
        std::iota(blocks.begin(), blocks.end(), 0);
        std::sort(blocks.begin(), blocks.end(), [&](int x, int y) {
            const auto kx = key(scores[q * stride + x]), ky = key(scores[q * stride + y]);
            return kx != ky ? kx > ky : x < y;
        });
        std::vector<int32_t> cpu;
        for (int b : blocks) {
            for (int c = 0; c < 4 && b * 4 + c < n && (int)cpu.size() < width; ++c) cpu.push_back(b * 4 + c);
            if ((int)cpu.size() == width) break;
        }
        std::sort(cpu.begin(), cpu.end());
        if (!std::equal(cpu.begin(), cpu.end(), a.begin() + q * cap) ||
            !std::equal(cpu.begin(), cpu.end(), r.begin() + q * cap)) ok = false;
    }
    if (!ok) std::fprintf(stderr, "FAIL ctx=%lld nq=%d mode=%d\n", (long long)ctx, nq, mode);
    if (mode == 0 && ctx >= 131072) {
        cudaEvent_t start, stop; ck(cudaEventCreate(&start)); ck(cudaEventCreate(&stop));
        auto time = [&](auto run) {
            ck(cudaEventRecord(start));
            for (int i = 0; i < 50; ++i) run();
            ck(cudaEventRecord(stop)); ck(cudaEventSynchronize(stop));
            float ms; ck(cudaEventElapsedTime(&ms, start, stop)); return ms / 50;
        };
        const float old = time(ref), now = time(candidate);
        std::printf("ctx=%lld nq=%d topk %.3f -> %.3f ms (%.2fx), exact=%s\n",
                    (long long)ctx, nq, old, now, old / now, ok ? "yes" : "NO");
        ck(cudaEventDestroy(start)); ck(cudaEventDestroy(stop));
    }
    ck(cudaFree(ds)); ck(cudaFree(dt)); ck(cudaFree(dr)); ck(cudaFree(dn)); ck(cudaFree(scratch));
    return ok;
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    for (int nq : {1, 4, 8})
        for (int64_t ctx : {1111, 8192, 131072, 131075, 256013, 262144})
            for (int mode = 0; mode < 4; ++mode) if (!check(ctx, nq, mode)) return 1;
    std::puts("PASS: 72 exact top-k cases match original GPU and independent CPU sort");
    return 0;
}
