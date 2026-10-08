// src/core/mtp.cpp - see include/strata/core/mtp.hpp.
#include "strata/core/mtp.hpp"
#include "strata/core/draft_head_router.hpp"
#include "strata/core/draft_residual_forecast.hpp"
#include "strata/core/shared_weights.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/spec/adaptive_lookahead.hpp"
#include "strata/core/on_device.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
constexpr int GGML_Q8_0 = 8;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped | (peer_portable() ? cudaHostAllocPortable : 0)) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

// 64-bit seek/tell on a `FILE*`: `fseek`/`ftell` take a 32-bit `long` on Windows and would wrap past 2 GiB.
#if defined(_WIN32)
#define STRATA_FILE_SEEK64(f, o, w) _fseeki64((f), (long long) (o), (w))
#define STRATA_FILE_TELL64(f) _ftelli64(f)
#else
#define STRATA_FILE_SEEK64(f, o, w) fseeko((f), (off_t) (o), (w))
#define STRATA_FILE_TELL64(f) ftello(f)
#endif

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    // Loader fix (0.1.15+loaderfix.2): `ifstream::read` reaches the disk as 4095-byte reads on MSVC - the same
    // split `load_experts_ranges` had - so the drafter's 111 MiB dense blob paid 4 KiB per operation on a cold
    // start.  `fread` passes a request bigger than the stream buffer straight to `ReadFile()`.
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    // A `FILE*` has no destructor and the short-read path below returns early: the guard closes on every path.
    struct Closer {
        FILE* f;
        ~Closer() { if (f != nullptr) std::fclose(f); }
    } closer{f};
    if (STRATA_FILE_SEEK64(f, 0, SEEK_END) != 0) return false;
    const long long n = (long long) STRATA_FILE_TELL64(f);
    if (n < 0 || STRATA_FILE_SEEK64(f, 0, SEEK_SET) != 0) return false;
    out.resize((size_t) n);
    return n == 0 || std::fread(out.data(), 1, (size_t) n, f) == (size_t) n;
}

}  // namespace

MtpDrafter::~MtpDrafter() {
    if (cs_) cudaStreamSynchronize(cs_);
    if (head_trace_) std::fclose(head_trace_);
    delete head_router_;
    delete residual_forecast_;
    if(feature_trace_)std::fclose(feature_trace_);
    if(feature_snapshots_)cudaFree(feature_snapshots_);
    if(residual_trace_)std::fclose(residual_trace_);
    if(residual_snapshots_)cudaFree(residual_snapshots_);
    qsa_state_release(st_);
    for (auto& e : prefill_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : prefill_dev_exec_) if (e) cudaGraphExecDestroy(e);
    if (pf_dev_) cudaFree(pf_dev_);
    for (auto& e : round_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : step_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : round_exec_c_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : step_exec_c_) if (e) cudaGraphExecDestroy(e);
    if (cparams_) cudaFree(cparams_);
    if (cring_) cudaFree(cring_);
    if (dinv_) cudaFree(dinv_);
    if (cscratch_) cudaFree(cscratch_);
    if (h_cparams_) cudaFreeHost(h_cparams_);
    if (h_chist_) cudaFreeHost(h_chist_);
    if (cs_) cudaStreamDestroy(cs_);
    if (dense_) shared_weight_free(dense_);
    if (experts_) shared_weight_free(experts_);
    if (state_arena_) cudaFree(state_arena_);
    if (arena_) cudaFree(arena_);
    if (head_logits_) cudaFree(head_logits_);
    if (hidden_calibration_) cudaFree(hidden_calibration_);
    if(token_bias_)cudaFree(token_bias_);
    if(token_feature_weights_)cudaFree(token_feature_weights_);
    if(token_feature_scores_)cudaFree(token_feature_scores_);
    if(token_feature_snapshots_)cudaFree(token_feature_snapshots_);
    if(label_snapshots_)cudaFree(label_snapshots_);
    if(label_trace_)std::fclose(label_trace_);
    if(token_feature_map_)cudaFree(token_feature_map_);
    if(token_feature_count_)cudaFree(token_feature_count_);
    if(h_token_labels_)cudaFreeHost(h_token_labels_);
    if (hidden_snapshots_) cudaFree(hidden_snapshots_);
    if (dhead_) shared_weight_free(dhead_);
    if (dvocab_) shared_weight_free(dvocab_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_row_, h_out_, h_prob_};
    for (void* h : hosts) if (h) cudaFreeHost(h);
}

const float* MtpDrafter::f32(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "f32") return (const float*) (dense_ + t.off);
    return nullptr;
}
const uint16_t* MtpDrafter::bf16(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "bf16") return (const uint16_t*) (dense_ + t.off);
    return nullptr;
}
const void* MtpDrafter::q8(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "q8_0") return dense_ + t.off;
    return nullptr;
}

bool MtpDrafter::load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
                      int64_t window) {
    cudaGetDevice(&device_);   // a layer split's last stage on another GPU: the drafter lives there
    g_ = &g;
    ss_ = &ss;
    max_t_ = max_t;
    rt_dir_ = rt_dir;
    if (max_t < 1 || max_t > strata::kernels::kVerifyMaxT) { err = "mtp: max_t out of range"; return false; }
    // Loader fix (0.1.15+loaderfix.2): the two reads below are the whole “drafter files” cost; reporting
    // them apart from the rest of the stage is what makes the next regression visible.
    const auto t_files = std::chrono::steady_clock::now();
    // ---- the index and the dense weights
    {
        std::ifstream idx(rt_dir + "/dense.txt");
        if (!idx) { err = "mtp: cannot open " + rt_dir + "/dense.txt (run tools/mtp_rt.py)"; return false; }
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            Tensor t;
            is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
            if (!is) { err = "mtp: malformed dense.txt line: " + line; return false; }
            tensors_.push_back(t);
        }
        std::vector<uint8_t> blob;
        if (!read_file(rt_dir + "/dense.bin", blob)) { err = "mtp: cannot read dense.bin"; return false; }
        for(const auto& t:tensors_) {
            if(t.rows<=0 || t.cols<=0 || t.rows>INT32_MAX || t.cols>INT32_MAX || t.off>blob.size() || t.bytes>blob.size()-t.off) {
                err="mtp: dense tensor bounds mismatch: "+t.name;return false;
            }
            const uint64_t elements=(uint64_t)t.rows*t.cols;
            const uint64_t expected=t.kind=="f32"?elements*4:t.kind=="bf16"?elements*2:
                t.kind=="q8_0" && t.cols%32==0?elements/32*34:0;
            if(expected==0 || expected!=t.bytes) {err="mtp: dense tensor format/size mismatch: "+t.name;return false;}
        }
        const cudaError_t alloc = shared_weight_upload((void**) &dense_, blob.data(), blob.size(), "mtp-dense");
        if (alloc != cudaSuccess) {
            size_t free_bytes = 0, total_bytes = 0;
            const cudaError_t info = cudaMemGetInfo(&free_bytes, &total_bytes);
            err = "mtp: dense weights allocation failed (" + std::string(cudaGetErrorString(alloc)) +
                  "), requested " + std::to_string(blob.size() >> 20) + " MiB, CUDA0 free " +
                  (info == cudaSuccess ? std::to_string(free_bytes >> 20) + " MiB" : "unknown");
            return false;
        }
        vram_ += blob.size();
    }
    // ---- the 512 routed experts, one blob each
    {
        expert_blob_bytes_ = strata::kernels::cpu::BLOB;
        const std::string format_path = rt_dir + "/experts-native.txt";
        // Optional, strict metadata. An absent file retains the original Q2_0S2 runtime. A malformed or
        // mismatched file must fail before upload instead of interpreting native bytes as S2 weights.
        if (std::filesystem::exists(format_path)) {
            std::ifstream format(format_path);
            std::string tag, extra;
            int version = 0, gu = -1, down = -1;
            int64_t h = 0, ff = 0, ne = 0;
            if (!(format >> tag >> version >> gu >> down >> h >> ff >> ne) ||
                tag != "strata-mtp-native" || version != 1 || (format >> extra) ||
                h != g.n_embd || ff != g.n_ff || ne != g.n_expert ||
                !strata::kernels::native_expert_supported(gu, down, h, ff)) {
                err = "mtp: invalid or incompatible experts-native.txt"; return false;
            }
            expert_layout_ = strata::kernels::native_expert_layout(gu, down, h, ff);
            expert_blob_bytes_ = expert_layout_.bytes;
            native_experts_ = true;
        }
        const uint64_t bytes = (uint64_t) g.n_expert * expert_blob_bytes_;
        // Loader fix (0.1.15+loaderfix.2): each 64 MiB read below reached the disk as ~16k 4095-byte reads under
        // MSVC's `basic_filebuf::xsgetn`, which is what made 675 MiB of drafter experts take minutes.
        FILE* f = std::fopen((rt_dir + "/experts.bin").c_str(), "rb");
        if (f == nullptr) { err = "mtp: cannot open experts.bin"; return false; }
        struct Closer {
            FILE* f;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } closer{f};
        if (STRATA_FILE_SEEK64(f, 0, SEEK_END) != 0 || STRATA_FILE_TELL64(f) != (long long) bytes ||
            STRATA_FILE_SEEK64(f, 0, SEEK_SET) != 0) {
            err = "mtp: experts.bin size does not match its format and geometry"; return false;
        }
        if (shared_weight_alloc((void**) &experts_, bytes, "mtp-experts") != cudaSuccess) { err = "mtp: the 512 experts do not fit in VRAM"; return false; }
        std::vector<uint8_t> chunk(64u << 20);
        for (uint64_t off = 0; off < bytes;) {
            const uint64_t n = std::min<uint64_t>(chunk.size(), bytes - off);
            if (std::fread(chunk.data(), 1, (size_t) n, f) != (size_t) n) { err = "mtp: experts.bin is truncated"; return false; }
            if (!shared_weights_reader() && cudaMemcpy(experts_ + off, chunk.data(), n, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "mtp: experts upload failed"; return false;
            }
            off += n;
        }
        if (shared_weight_publish(experts_, bytes, "mtp-experts") != cudaSuccess) { err="mtp: cannot publish shared experts";return false; }
        vram_ += bytes;
    }
    const char* required[] = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                              "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
                              "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"};
    for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); ++i) {
        const char* n = required[i];
        if (!q8(n) && !(i < 6 && bf16(n))) {
            err = std::string("mtp: ") + n + (i < 6 ? " is missing (q8_0 or bf16)" : " is missing (q8_0)");
            return false;
        }
    }

    // ---- the layer's own K/V (dense attention: no indexer state is read)
    const strata::kernels::QsaShapes s = shapes_of(g);
    const int64_t max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
    // KV streaming: the drafter only reads its last `window` cells, so with streaming on its K/V is a ring of the
    // window (plus the cells a round writes ahead of its queries) over a host copy, refilled on a resume. The host copy
    // is pinned after the expert arena has pinned what it could: if it does not fit, the K/V stays whole in VRAM.
    if(const char* v=std::getenv("STRATA_MTP_SPARSE");v&&std::atoi(v)==1)sparse_=true;
    if(sparse_ && (!q8("self_attn.indexer.index_qk_proj.weight") ||
                  !f32("self_attn.indexer.k_layernorm.weight") ||
                  !f32("self_attn.indexer.q_layernorm.weight"))) {
        err="mtp: sparse attention needs the auxiliary indexer tensors";return false;
    }
    int64_t ring = !sparse_ && (window > 0 && window < max_cells) ? window + 4 * (int64_t) max_t + 64 : 0;
    // K8V4 never applies to the drafter: its own attention paths (below, and verify.cpp) handle whole formats
    // only, whatever ring shape it takes (0, a window, or the -1 fully-resident fallback).
    const bool kv_hybrid_was = qsa_kv_hybrid();
    const bool kv_int8_was = qsa_kv_int8();
    qsa_set_kv_hybrid(false);
    if (kv_hybrid_was) qsa_set_kv_int8(true);   // the drafter under --kv k8v4: plain INT8
    uint64_t sb = qsa_state_bytes(g, max_cells, false, ring);
    if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
    if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[ss.qsa_primary()], ring) == 0) {
        if (st_.kv_mode == 0) { err = "mtp: state init failed"; return false; }
        std::fprintf(stderr, "strata mtp: no pinned RAM left for the draft layer's K/V copy; keeping it in VRAM\n");
        cudaGetLastError();
        qsa_state_release(st_);
        cudaFree(state_arena_);
        st_ = QsaState{};
        ring = -1;   // fully resident
        sb = qsa_state_bytes(g, max_cells, false, ring);
        if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
        if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[ss.qsa_primary()], ring) == 0) { err = "mtp: state init failed"; return false; }
    }
    qsa_set_kv_int8(kv_int8_was);
    qsa_set_kv_hybrid(kv_hybrid_was);
    st_.auxiliary_indexer = sparse_;
    qsa_state_zero(st_, g, nullptr);
    cudaDeviceSynchronize();
    vram_ += sb;

    // ---- buffers
    window_ = !sparse_ && (window > 0 && window < max_cells) ? window : 0;
    cap_ = sparse_ ? strata::kernels::qsa_selection_width(max_cells,s) : (((window_ > 0 ? window_ : max_cells) + 63) / 64) * 64;
    sparse_max_blocks_=max_cells/s.idx_block+1;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);
    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t R2 = 2 * T;   // step/pos rows: T catch-up rows + up to T-2 chain steps
    bool ok = mapped(T * 4 + 64, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(R2 * 4 * 4 + 64, (void**) &h_step_, (void**) &m_step_) &&
              mapped(R2 * NH * 4 + 64, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped(64, (void**) &h_row_, (void**) &m_row_) &&
              mapped(T * 4 + 64, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 64, (void**) &h_prob_, (void**) &m_prob_);
    if (!ok) { err = "mtp: mapped staging failed"; return false; }
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(R2 * 4); pos_ = b.take<int32_t>(R2 * NH); row_ = b.take<int32_t>(4);
        ident_ = b.take<int32_t>(T * (uint64_t) cap_);
        Rin_ = b.take<float>(T * HC * N); R_ = b.take<float>(T * HC * N);
        emb_ = b.take<float>(T * N); en_ = b.take<float>(T * N); e2_ = b.take<float>(T * N);
        hn_ = b.take<float>(T * HC * N); h2_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); bo_ = b.take<float>(T * N);
        xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes((int) (NH * HD), 8));
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>((uint64_t) attn_scratch_floats_);   // the full layer runs one row at a time
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); y_ = b.take<float>(T * N);
        sample_ = b.take<float>(T * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        grp_ptr_ = b.take<unsigned long long>(T * K); grp_start_ = b.take<int32_t>(T * K + 1);
        grp_counts_ = b.take<int32_t>(4);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * (native_experts_ ? 36 : 34)); hit_xs_ = b.take<float>(T * (N / 32));
        hit_scratch_ = b.take<uint8_t>(native_experts_ ?
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff) :
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff));
        sh_scratch_ = (float*) b.take<uint8_t>(strata::kernels::shared_expert_scratch_bytes(g.n_ff));
        x_bf16_ = b.take<uint16_t>(N);
        out_ids_ = b.take<int32_t>(T + 4);
        probs_ = b.take<float>(T + 4);
        dummy_inj_ = b.take<float>(HC);
        if(sparse_){
            index_qk_=b.take<float>(T*(uint64_t)(s.idx_n_head+1)*s.idx_dim);
            index_scores_=b.take<float>(T*(uint64_t)sparse_max_blocks_);
            index_tail_checkpoint_=b.take<float>((s.idx_block-1)*s.idx_dim);
            index_dead_checkpoint_=b.take<float>(s.idx_dim);
            index_pos_checkpoint_=b.take<int32_t>(1);
            index_topk_scratch_=b.take<uint8_t>(strata::kernels::qsa_topk_parallel_scratch_bytes(1));
        }
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) { err = "mtp: buffers do not fit"; return false; }
    cudaMemset(arena_, 0, count.used);
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    vram_ += count.used;
    {
        std::vector<int32_t> id((size_t) (T * (uint64_t) cap_));
        for (uint64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < cap_; ++i) id[(size_t) (t * (uint64_t) cap_ + (uint64_t) i)] = (int32_t) i;
        cudaMemcpy(ident_, id.data(), id.size() * 4, cudaMemcpyHostToDevice);
    }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) { err = "mtp: stream"; return false; }
    if(sparse_)std::fprintf(stderr,"strata mtp: draft-only sparse attention over full context, %lld selected cells, KV mode%d; target unchanged\n",(long long)cap_,st_.kv_mode);
    if (const char* flag = std::getenv("STRATA_MTP_HIDDEN_CAL"); flag && (std::atoi(flag) == 1 || std::atoi(flag) == 2)) {
        hidden_affine_ = std::atoi(flag) == 2;
        const size_t cells = (size_t)max_t_ * g.n_embd;
        const size_t state_cells = cells * (hidden_affine_ ? 4 : 1) + max_t_ * 4;
        if (cudaMalloc((void**)&hidden_calibration_, state_cells * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**)&hidden_snapshots_, cells * sizeof(float)) != cudaSuccess) {
            err = "mtp: hidden calibration buffers do not fit"; return false;
        }
        vram_ += (state_cells + cells) * sizeof(float);
        if (const char* v = std::getenv("STRATA_MTP_HIDDEN_GAIN")) hidden_gain_ = std::clamp((float)std::atof(v), 0.0f, 1.0f);
        if (const char* v = std::getenv("STRATA_MTP_HIDDEN_RATE")) hidden_rate_ = std::clamp((float)std::atof(v), 0.001f, 0.5f);
        if (!reset_hidden_calibration(err)) return false;
        std::fprintf(stderr, "strata mtp: request-local hidden %s correction gain=%.3f rate=%.3f, target verification unchanged\n", hidden_affine_ ? "affine" : "bias", hidden_gain_, hidden_rate_);
    }
    const double files_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_files).count();
    std::fprintf(stderr, "strata mtp: draft layer loaded, %.0f MiB of VRAM (experts %.0f, dense %.0f), files read in %.2f s (%.0f MiB/s)\n",
                 (double) vram_ / 1048576.0, (double) g.n_expert * expert_blob_bytes_ / 1048576.0,
                 (double) tensors_.back().off / 1048576.0, files_s,
                 files_s > 0 ? ((double) g.n_expert * expert_blob_bytes_ + (double) tensors_.back().off) /
                                   1048576.0 / files_s : 0.0);
    if (native_experts_)
        std::fprintf(stderr, "strata mtp: auxiliary native experts gu_type=%d down_type=%d, target weights unchanged\n",
                     expert_layout_.gu_type, expert_layout_.d_type);
    return true;
}

uint64_t MtpDrafter::bind_bytes(uint64_t head_row_bytes, int64_t n_vocab) const {
    uint64_t bytes = head_logits_ ? 0 : (uint64_t) max_t_ * (uint64_t) n_vocab * sizeof(float);
    if (dhead_ == nullptr) {
        if (FILE* f = std::fopen((rt_dir_ + "/draft_vocab.bin").c_str(), "rb")) {
            std::fseek(f, 0, SEEK_END);
            const long size = std::ftell(f);
            std::fclose(f);
            if (size >= 4 && size % 4 == 0) bytes += (uint64_t) (size / 4) * head_row_bytes + (uint64_t) size;
        }
    }
    // coupled draft sampling (STRATA_SPEC_COUPLED=1 only): an upper bound - the id -> subset map, the penalty ring,
    // the split scratch (64 lists of 64 entries at most) and the parameters
    if(const char* v=std::getenv("STRATA_MTP_TOKEN_BIAS");v&&v[0]=='1')bytes+=(uint64_t)n_vocab*4+65536;
    if(const char* v=std::getenv("STRATA_MTP_TOKEN_FEATURE");v&&v[0]=='1')
        bytes+=(uint64_t)kTokenFeatureSlots*g_->n_embd*4+(uint64_t)max_t_*g_->n_embd*4+(uint64_t)n_vocab*4+65536;
    if (coupled_draft_env() && cparams_ == nullptr)
        bytes += (uint64_t) n_vocab * 4 + (uint64_t) (kCoupledHistCap + max_t_) * 4 + 64ull * 64ull * 8ull + 4096;
    if(const char* path=std::getenv("STRATA_MTP_HEAD_ROUTER");path&&path[0])
        bytes+=DraftHeadRouter::reserve_bytes(path,head_row_bytes);
    if(const char* path=std::getenv("STRATA_MTP_RESIDUAL_FORECAST");path&&path[0])
        bytes+=DraftResidualForecast::reserve_bytes(path);
    return bytes;
}

bool MtpDrafter::setup_coupled(std::string& err) {
    const int64_t nv = dhead_ != nullptr ? n_dvocab_ : n_vocab_;
    const size_t scratch = strata::kernels::coupled_draft_scratch_bytes((int) nv);
    if (scratch == 0) {
        std::fprintf(stderr, "strata mtp: STRATA_SPEC_COUPLED: %lld draft logits are too wide for the coupled sampler; "
                             "argmax drafts\n", (long long) nv);
        return true;
    }
    const size_t ring = (size_t) (kCoupledHistCap + max_t_) * sizeof(int32_t);
    if (cudaMalloc((void**) &cparams_, sizeof(strata::kernels::SamplerParams)) != cudaSuccess ||
        cudaMalloc((void**) &cring_, ring) != cudaSuccess || cudaMalloc(&cscratch_, scratch) != cudaSuccess ||
        !mapped(sizeof(strata::kernels::SamplerParams), (void**) &h_cparams_, (void**) &m_cparams_) ||
        !mapped((size_t) kCoupledHistCap * sizeof(int32_t), (void**) &h_chist_, (void**) &m_chist_)) {
        err = "mtp: the coupled draft sampler's buffers do not fit";
        return false;
    }
    cudaMemset(cring_, 0xff, ring);   // -1: no token
    std::fill(h_chist_, h_chist_ + kCoupledHistCap, -1);
    vram_ += sizeof(strata::kernels::SamplerParams) + ring + scratch;
    if (dhead_ != nullptr) {   // token id -> subset index (-1: not in the draft head), for the penalties
        std::vector<int32_t> sub((size_t) n_dvocab_), inv((size_t) n_vocab_, -1);
        cudaMemcpy(sub.data(), dvocab_, sub.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < sub.size(); ++i)
            if (sub[i] >= 0 && sub[i] < n_vocab_ && inv[(size_t) sub[i]] < 0) inv[(size_t) sub[i]] = (int32_t) i;
        if (cudaMalloc((void**) &dinv_, inv.size() * sizeof(int32_t)) != cudaSuccess) {
            err = "mtp: the coupled draft sampler's token map does not fit";
            return false;
        }
        cudaMemcpy(dinv_, inv.data(), inv.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        vram_ += inv.size() * sizeof(int32_t);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("mtp: coupled setup: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    coupled_ok_ = true;
    std::fprintf(stderr, "strata mtp: coupled draft sampling on (STRATA_SPEC_COUPLED): sampled requests draft with the "
                         "target's chain and Philox draw over %lld tokens\n", (long long) nv);
    return true;
}

void MtpDrafter::set_draft_sampling(const strata::kernels::SamplerParams& sp) {
    coupled_active_ = coupled_ok_ && !sp.greedy && sp.temperature > 0.0f;
    if (!coupled_active_) return;
    *h_cparams_ = sp;   // read by the next round graph (after the previous one has synced)
    std::fill(h_chist_, h_chist_ + kCoupledHistCap, -1);
}

void MtpDrafter::set_draft_history(const int32_t* tail, int64_t n_tail, int32_t next) {
    if (!coupled_active_) return;
    const int h = coupled_hist_len(h_cparams_->penalty_last_n, kCoupledHistCap);
    coupled_hist_base(tail, n_tail, next, h, h_chist_ + (kCoupledHistCap - h));
}

namespace {
// #474: what to change when the draft head's token subset does not fit the VRAM left - a smaller subset that setup
// ships (data/draft_vocab_*.bin; their token counts below), and how to pick it.  The start used to stop at "the draft
// head does not fit" with no hint.  Text only, after the failure: nothing changes for a start that fits.
void draft_head_hint(int64_t n_tokens, int64_t row_bytes) {
    size_t free_b = 0, total_b = 0;
    const bool have_free = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
    auto mib = [&](int64_t n) { return (double) (n * row_bytes) / 1048576.0; };
    std::string free_s;
    if (have_free) {
        char b[64];
        std::snprintf(b, sizeof(b), " and %.0f MiB is free", (double) free_b / 1048576.0);
        free_s = b;
    }
    std::fprintf(stderr, "strata mtp: the draft head over %lld tokens needs %.0f MiB of VRAM%s.\n",
                 (long long) n_tokens, mib(n_tokens), free_s.c_str());
    struct Subset { const char* name; int64_t tokens; const char* what; };
    static const Subset smaller[] = {{"cyrillic", 58963, "English, code and the Cyrillic script"},
                                     {"en", 40525, "English and code"}};
    std::string opts;
    for (const Subset& s : smaller)
        if (s.tokens < n_tokens) {
            char b[160];
            std::snprintf(b, sizeof(b), "%s--draft-vocab %s (%s, ~%.0f MiB)", opts.empty() ? "" : " or ", s.name,
                          s.what, mib(s.tokens));
            opts += b;
        }
    if (!opts.empty())
        std::fprintf(stderr, "strata mtp: hint: a smaller draft vocabulary needs less VRAM: %s. Start once with it - "
                             "START-HERE.bat --draft-vocab en (Windows) or ./setup.sh --draft-vocab en - and the model "
                             "keeps it (\"draft_vocab\" in its strata-*.json config); or a smaller --context in "
                             "setup.\n",
                     opts.c_str());
    else
        std::fprintf(stderr, "strata mtp: hint: this is already the smallest shipped draft vocabulary: a smaller "
                             "--context (or closing what else uses the GPU) leaves it room.\n");
}
}  // namespace

bool MtpDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err) {
    const OnDevice on_device(device_);
    wt_ = &wt;
    head_ = head;
    window_R_ = window_R;
    const WeightRef* wo = wt.find("output.weight");
    if (!wo) { err = "mtp: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;
    if (head == nullptr || !head->loaded()) { err = "mtp: the draft layer needs the native head (--native)"; return false; }
    if (head_logits_ == nullptr &&
        cudaMalloc((void**) &head_logits_, (size_t) max_t_ * (size_t) n_vocab_ * sizeof(float)) != cudaSuccess) {
        err = "mtp: the draft logits do not fit";
        return false;
    }
    // the draft head's token subset, when tools/draft_vocab.py wrote one
    if (dhead_ == nullptr) {
        std::vector<uint8_t> raw;
        if (read_file(rt_dir_ + "/draft_vocab.bin", raw) && raw.size() >= 4 && raw.size() % 4 == 0) {
            n_dvocab_ = (int64_t) (raw.size() / 4);
            const int64_t row_bytes = (int64_t) head->row_bytes();   // a vocabulary row of the native head
            if (shared_weight_upload((void**) &dvocab_, raw.data(), raw.size(), "mtp-vocab") != cudaSuccess ||
                shared_weight_alloc((void**) &dhead_, (size_t) (n_dvocab_ * row_bytes), "mtp-head") != cudaSuccess) {
                err = "mtp: the draft head does not fit";
                draft_head_hint(n_dvocab_, row_bytes);
                return false;
            }
            if (!shared_weights_reader()) strata::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, dvocab_, n_dvocab_, dhead_, nullptr);
            cudaDeviceSynchronize();
            if(shared_weight_publish(dhead_,(size_t)(n_dvocab_*row_bytes),"mtp-head")!=cudaSuccess){err="mtp: cannot publish shared head";return false;}
            vram_ += (uint64_t) (n_dvocab_ * row_bytes) + raw.size();
            std::fprintf(stderr, "strata mtp: draft head over %lld tokens (%.1f MiB)\n", (long long) n_dvocab_,
                         (double) (n_dvocab_ * row_bytes) / 1048576.0);
        }
    }
    if(const char* v=std::getenv("STRATA_MTP_TOKEN_BIAS");v&&v[0]=='1'&&!token_bias_){
        if(cudaMalloc((void**)&token_bias_,(size_t)n_vocab_*4)!=cudaSuccess||
           !mapped(256,(void**)&h_token_labels_,(void**)&m_token_labels_)){
            err="mtp: token calibration allocation";return false;
        }
        vram_+=(uint64_t)n_vocab_*4;
        if(cudaMemsetAsync(token_bias_,0,(size_t)n_vocab_*4,cs_)!=cudaSuccess){err="mtp: token calibration initialization";return false;}
        std::fprintf(stderr,"strata mtp: experimental request-local token correction, target verifier unchanged\n");
    }
    if(const char* v=std::getenv("STRATA_MTP_TOKEN_FEATURE");v&&v[0]=='1'&&!token_feature_weights_){
        const size_t cells=(size_t)kTokenFeatureSlots*g_->n_embd;
        if(cudaMalloc((void**)&token_feature_weights_,cells*4)!=cudaSuccess||
           cudaMalloc((void**)&token_feature_scores_,kTokenFeatureSlots*4)!=cudaSuccess||
           cudaMalloc((void**)&token_feature_snapshots_,(size_t)max_t_*g_->n_embd*4)!=cudaSuccess||
           cudaMalloc((void**)&token_feature_map_,(size_t)n_vocab_*4)!=cudaSuccess||
           cudaMalloc((void**)&token_feature_count_,4)!=cudaSuccess||
           (!h_token_labels_&&!mapped(256,(void**)&h_token_labels_,(void**)&m_token_labels_))){
            err="mtp: contextual token calibration allocation";return false;
        }
        vram_+=(cells+(size_t)max_t_*g_->n_embd+(size_t)n_vocab_+kTokenFeatureSlots+1)*4;
        token_feature_slots_.assign((size_t)n_vocab_,-1);
        if(!reset_hidden_calibration(err))return false;
        std::fprintf(stderr,"strata mtp: contextual request-local token calibration; all emissions use original target verification\n");
    }
    if(const char* path=std::getenv("STRATA_MTP_LABEL_TRACE");path&&path[0]&&!label_trace_){
        if(cudaMalloc((void**)&label_snapshots_,(size_t)max_t_*g_->n_embd*4)!=cudaSuccess){err="mtp: native label snapshots";return false;}
        const std::string dir(path);
        label_trace_=std::fopen((dir+"/pairs.bin").c_str(),"wb");
        if(!label_trace_){err="mtp: native label trace file";return false;}
        std::setvbuf(label_trace_,nullptr,_IONBF,0);
        std::ofstream meta(dir+"/meta.json");
        meta<<"{\"cols\":"<<g_->n_embd<<",\"record_bytes\":"<<(32+4*g_->n_embd)<<",\"header\":[\"request\",\"depth\",\"sample\",\"width\",\"proposal_global_id\",\"target_global_id\",\"head_type\",\"reserved\"],\"teacher\":\"actual unchanged target argmax with verified input history\"}\n";
        if(!meta){err="mtp: native label trace metadata";return false;}
        vram_+=(uint64_t)max_t_*g_->n_embd*4;
    }
    if (coupled_draft_env() && cparams_ == nullptr && !setup_coupled(err)) return false;
    if(const char* path=std::getenv("STRATA_MTP_HEAD_ROUTER");path&&path[0]&&!head_router_){
        auto* router=new DraftHeadRouter;
        if(!router->load(path,dhead_?dhead_:head_->weights(),dvocab_,head_->type(),(int)(dhead_?n_dvocab_:n_vocab_),(int)g_->n_embd,head_->row_bytes(),err)){delete router;return false;}
        head_router_=router;vram_+=DraftHeadRouter::reserve_bytes(path,head_->row_bytes());
    }


    if(const char* path=std::getenv("STRATA_MTP_RESIDUAL_FORECAST");path&&path[0]&&!residual_forecast_){
        auto forecast=std::make_unique<DraftResidualForecast>();
        if(!forecast->load(path,(int)g_->hc,(int)g_->n_embd,f32("pre_fc_norm_hidden.weight"),err))return false;
        residual_forecast_=forecast.release();vram_+=DraftResidualForecast::reserve_bytes(path);
    }
    if(const char* path=std::getenv("STRATA_MTP_RESIDUAL_TRACE");path&&path[0]&&!residual_trace_){
        std::filesystem::create_directories(path);
        const size_t H=(size_t)g_->hc*g_->n_embd;
        if(cudaMalloc((void**)&residual_snapshots_,(size_t)max_t_*H*4)!=cudaSuccess){err="mtp residual trace: snapshots";return false;}
        residual_trace_=std::fopen((std::string(path)+"/pairs.bin").c_str(),"wb");
        if(!residual_trace_){err="mtp residual trace: file";return false;}
        std::setvbuf(residual_trace_,nullptr,_IONBF,0);
        std::ofstream meta(std::string(path)+"/meta.json");meta<<"{\"cols\":"<<H<<",\"record_bytes\":"<<(16+H*8)<<",\"fields\":[\"request\",\"depth\",\"sequence\",\"width\"]}\n";
        std::fprintf(stderr,"strata mtp: draft residual teacher tracing enabled; performance includes tracing overhead\n");
    }
    if(const char* path=std::getenv("STRATA_MTP_FEATURE_TRACE");path&&path[0]&&!feature_trace_){
        std::filesystem::create_directories(path);
        if(cudaMalloc((void**)&feature_snapshots_,(size_t)max_t_*g_->n_embd*4)!=cudaSuccess){err="mtp feature trace: snapshots";return false;}
        feature_trace_=std::fopen((std::string(path)+"/pairs.bin").c_str(),"wb");
        if(!feature_trace_){err="mtp feature trace: file";return false;}
        std::setvbuf(feature_trace_,nullptr,_IONBF,0);
        std::ofstream meta(std::string(path)+"/meta.json");meta<<"{\"cols\":"<<g_->n_embd<<",\"record_bytes\":"<<(16+g_->n_embd*8)<<",\"fields\":[\"request\",\"depth\",\"sequence\",\"width\"]}\n";
        std::fprintf(stderr,"strata mtp: teacher feature tracing enabled; timings include tracing overhead\n");
    }
    if (const char* path=std::getenv("STRATA_MTP_HEAD_TRACE"); path && path[0] && !head_trace_) {
        std::filesystem::create_directories(path);
        const auto prefix=std::string(path)+"/";
        const int64_t nv=dhead_?n_dvocab_:n_vocab_;
        const size_t bytes=(size_t)nv*head_->row_bytes();
        std::vector<uint8_t> payload(bytes);
        if(cudaMemcpy(payload.data(),dhead_?dhead_:head_->weights(),bytes,cudaMemcpyDeviceToHost)!=cudaSuccess){err="mtp head trace: weight copy";return false;}
        std::ofstream weights(prefix+"head.bin",std::ios::binary);weights.write((char*)payload.data(),payload.size());
        std::vector<int32_t> ids((size_t)nv);
        if(dvocab_)cudaMemcpy(ids.data(),dvocab_,ids.size()*4,cudaMemcpyDeviceToHost);
        else for(int64_t i=0;i<nv;++i)ids[i]=(int32_t)i;
        std::ofstream vocab(prefix+"ids.bin",std::ios::binary);vocab.write((char*)ids.data(),ids.size()*4);
        std::ofstream meta(prefix+"meta.json");meta<<"{\"type\":"<<head_->type()<<",\"rows\":"<<nv<<",\"cols\":"<<g_->n_embd<<",\"row_bytes\":"<<head_->row_bytes()<<",\"trace_record_bytes\":"<<(16+g_->n_embd/32*36)<<"}\n";
        head_trace_=std::fopen((prefix+"inputs.bin").c_str(),"wb");
        if(!weights||!vocab||!meta||!head_trace_){err="mtp head trace: files";return false;}
        std::setvbuf(head_trace_,nullptr,_IONBF,0);
        std::fprintf(stderr,"strata mtp: head input tracing enabled; these measurements include tracing overhead\n");
    }
    return true;
}
void MtpDrafter::trace_head_input(int depth,int64_t cell,int32_t token){
    if(!head_trace_||head_trace_count_>=8192)return;
    const size_t bytes=(size_t)g_->n_embd/32*36;
    std::vector<uint8_t> record(16+bytes);
    const int32_t header[4]={depth,(int32_t)cell,token,(int32_t)bytes};std::memcpy(record.data(),header,16);
    if(cudaMemcpy(record.data()+16,xq_,bytes,cudaMemcpyDeviceToHost)!=cudaSuccess)return;
    if(std::fwrite(record.data(),1,record.size(),head_trace_)==record.size())++head_trace_count_;
}

// The layer for T rows.  full = false stops after the K/V append (the prompt only needs the cache).
bool MtpDrafter::record_forward(int T, int step_row0, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    // step_row0 >= 0: the full layer on step rows [step_row0, +T); step_row0 < 0: K/V only on rows [-1 - step_row0, +T)
    const bool full = step_row0 >= 0;
    const int row0 = full ? step_row0 : -1 - step_row0;
    if (full && T != 1) { err = "mtp: the full layer runs one row at a time (its attention scratch is sized for one)"; return false; }
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const int32_t* step = step_ + row0 * 4;
    const int32_t* pos = pos_ + row0 * NH;
    try {
        // Optional original-precision auxiliary projections. The default Q8
        // runtime retains its original operations and shared quantizations.
        auto projection = [&](const char* name,const float* input,const uint8_t* quant,float* output,int ni,int no,int rows) {
            bool dims=false;
            for(const auto& t:tensors_)if(t.name==name && t.rows==no && t.cols==ni)dims=true;
            if(!dims)throw std::runtime_error(std::string("mtp: projection dimensions mismatch: ")+name);
            if(const uint16_t* weight=bf16(name))
                bf16_gemv_fp32_mmvf_multi(input,ni,weight,output,no,ni,no,rows,cs);
            else {
                const void* quantized_weight=q8(name);
                if(!quantized_weight)throw std::runtime_error(std::string("mtp: missing projection ")+name);
                native_mmvq(GGML_Q8_0,quantized_weight,quant,output,ni,no,rows,cs);
            }
        };
        // ---- the two input branches
        const WeightRef* we = wt_->find("token_embd.weight");
        if (!we) { err = "mtp: token_embd.weight is missing"; return false; }
        if (const NativeEmbed* ne = native_embed()) {   // plan v0.3 P6: the GGUF-form table
            ne->gather_dev(tok_, T, emb_, cs);
        } else {
            const auto* codes = (const uint8_t*) we->data;
            const auto* scales = (const float*) (codes + we->codes_bytes);
            const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
            embedding_gather_dev(codes, scales, offsets, tok_, T, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                 (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), emb_, cs);
        }
        native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, cs);
        if(q8("fc_embedding.weight"))native_quantize_q8_1(en_, xq_, (int) N, T, cs);
        projection("fc_embedding.weight",en_,xq_,e2_,(int)N,(int)N,T);
        // Root and accepted-history prefill use the exact target residual. Only
        // a subsequent auxiliary proposal consumes an approximate residual.
        if(full && coupled_j_>0 && residual_forecast_ &&
           !residual_forecast_->run(Rin_,cs,err))return false;
        static const bool stream_hidden=[] {const char* v=std::getenv("STRATA_MTP_HNORM_STREAM");return v&&std::atoi(v)==1;}();
        if(stream_hidden)native_qsa_rms_norm_grouped(Rin_,f32("pre_fc_norm_hidden.weight"),hn_,(int)N,(int)HC,T*(int)HC,EPS,cs);
        else native_qsa_rms_norm_weighted(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) (HC * N), T, EPS, cs);
        for (int c0 = 0; c0 < T * HC; c0 += 8) {
            const int nc = (int) std::min<int64_t>(8, T * HC - c0);
            if(q8("fc_hidden.weight"))native_quantize_q8_1(hn_ + (size_t) c0 * N, xq_, (int) N, nc, cs);
            projection("fc_hidden.weight",hn_+(size_t)c0*N,xq_,h2_+(size_t)c0*N,(int)N,(int)N,nc);
        }
        add_streams_broadcast(h2_, e2_, R_, N, (int) HC, T, cs);
        // ---- the attention hyper-connection
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                fa[t].w_norm = f32("attn_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("attn_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("attn_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("attn_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        // ---- attention: K/V into the layer's own cache, then (full) dense attention over every cell
        auto norm_rope = [&](float* data, const float* gamma, int rows, int cols, const int32_t* p) {
            native_qsa_rms_norm_weighted(data, gamma, data, cols, rows, EPS, cs);
            if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), p, cs);
            else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st_.cos_tab, st_.sin_tab, p, cs);
        };
        if(sparse_ || q8("self_attn.k_proj.weight") || q8("self_attn.v_proj.weight") || (full && q8("self_attn.q_proj.weight")))
            native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
        if(sparse_)projection("self_attn.indexer.index_qk_proj.weight",mixed_,xq_,index_qk_,
                              (int)N,(int)((s.idx_n_head+1)*s.idx_dim),T);
        projection("self_attn.k_proj.weight",mixed_,xq_,kcur_,(int)N,(int)(NKV*HD),T);
        projection("self_attn.v_proj.weight",mixed_,xq_,vcur_,(int)N,(int)(NKV*HD),T);
        for (int t = 0; t < T; ++t) {
            norm_rope(kcur_ + t * NKV * HD, f32("self_attn.k_norm.weight"), (int) NKV, (int) HD, pos + t * NH);
            if (st_.kv_rot) {   // rotated K and V (kv_q4.hpp): Q4_0, and INT8 with STRATA_KV_ROT=1
                fwht256_inplace_cuda(kcur_ + t * NKV * HD, NKV, cs);
                fwht256_inplace_cuda(vcur_ + t * NKV * HD, NKV, cs);
            }
            // stored in the state's own format (#293 appended rotated INT8 K/V as Q4_0, into pools INT8 never has)
            if (st_.kv_q4)
                kv_append_q4_step(st_.k_q4, st_.v_q4, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                  vcur_ + t * NKV * HD, s, cs, &st_.host);
            else if (st_.kv_int8)
                kv_append_q8_step(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step + t * 4,
                                  kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st_.host);
            else
                kv_append_step(st_.k_pool, st_.v_pool, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                               vcur_ + t * NKV * HD, s, cs, &st_.host);
            if(sparse_){
                const QsaIndexerBuffers ib{st_.idx_tail,st_.idx_dead,st_.idx_pooled,st_.idx_block_pos};
                const float* raw=index_qk_+(size_t)t*(s.idx_n_head+1)*s.idx_dim+s.idx_n_head*s.idx_dim;
                native_qsa_indexer_append(raw,step+t*4,0,f32("self_attn.indexer.k_layernorm.weight"),
                                          EPS,ib,s,st_.max_cells,rope_scaling(),cs);
            }
        }
        if (!full) return true;
        projection("self_attn.q_proj.weight",mixed_,xq_,qfull_,(int)N,(int)(NH*2*HD),T);
        for (int t = 0; t < T; ++t) {
            float* qc = qcur_ + t * NH * HD;
            if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4, (size_t) HD * 4,
                                  (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                err = "mtp: q split failed";
                return false;
            }
            norm_rope(qc, f32("self_attn.q_norm.weight"), (int) NH, (int) HD, pos + t * NH);
            if (st_.kv_rot) fwht256_inplace_cuda(qc, NH, cs);
        }
        const QsaAttnPools pools = qsa_attn_pools(st_);
        if(sparse_){
            norm_rope(index_qk_,f32("self_attn.indexer.q_layernorm.weight"),
                      (int)s.idx_n_head,(int)s.idx_dim,pos);
            qsa_block_scores(st_.idx_pooled,st_.idx_dead,index_qk_,step,1,sparse_max_blocks_,s,index_scores_,cs);
            qsa_block_topk_parallel(index_scores_,step,1,sparse_max_blocks_,cap_,s,ident_,index_topk_scratch_,cs);
            qsa_kv_resolve(st_,g,ident_,step,1,cap_,cs);
        }else if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
        qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs);
        if (st_.kv_rot) fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
        for (int t = 0; t < T; ++t)
            native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD, (int) NH, (int) HD, cs);
        if(q8("self_attn.o_proj.weight"))native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
        projection("self_attn.o_proj.weight",attn32_,xq_,bo_,(int)(NH*HD),(int)N,T);
        // ---- the MLP hyper-connection (the attention write folded in)
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = bo_ + t * N; fa[t].inj_prev = inj_ + t * HC;
                fa[t].w_norm = f32("mlp_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("mlp_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("mlp_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("mlp_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj2_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        // ---- MoE: router, the 512 resident experts, the shared expert, the combine, the write
        for (int t = 0; t < T; ++t) {
            bf16_gemv_fp32_mmvf(mixed_ + t * N, bf16("mlp.gate.weight"), logits_ + t * g.n_expert, (int) N, (int) g.n_expert, cs);
            if (native_router_enabled()) native_router_top10(logits_ + t * g.n_expert, ids_ + t * K, w_ + t * K, cs);
            else router_top10(logits_ + t * g.n_expert, 1, (int) g.n_expert, (int) K, ids_ + t * K, w_ + t * K, cs);
        }
        moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) expert_blob_bytes_, grp_ptr_,
                           grp_start_, grp_counts_, hit_dst_, hit_slot_, cs);
        if (native_experts_) {
            native_quantize_q8_1(mixed_, hit_xq_, (int) N, T, cs);
            native_expert_grouped(expert_layout_, grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_,
                                  (int64_t) T * K, (int64_t) T * K, hit_xq_, hit_scratch_, parts_, cs);
        } else {
            quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
            moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K, (int64_t) T * K, hit_xq_,
                           hit_xs_, hit_scratch_, parts_, cs);
        }
        NativeSharedWeights nsw;
        nsw.gate_type = GGML_Q8_0; nsw.gate_data = q8("mlp.shared_expert.gate_proj.weight");
        nsw.up_type = GGML_Q8_0; nsw.up_data = q8("mlp.shared_expert.up_proj.weight");
        nsw.down_type = GGML_Q8_0; nsw.down_data = q8("mlp.shared_expert.down_proj.weight");
        nsw.q8_1 = xq_;
        const SForm none{};
        for (int t = 0; t < T; ++t) {
            f32_to_bf16_bulk(mixed_ + t * N, x_bf16_, N, cs);
            shared_expert(nullptr, nullptr, x_bf16_, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr, none,
                          nullptr, nullptr, nullptr, bf16("mlp.shared_expert_gate.weight"), sh_scratch_, shared_ + t * N,
                          N, g.n_ff, 32, cs, mixed_ + t * N, &nsw);
            if (native_moe_combine_enabled())
                native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            else
                moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            gr_write(R_ + (size_t) t * HC * N, y_ + t * N, inj2_ + t * HC, gs, R_ + (size_t) t * HC * N, cs);
        }
        // ---- the final mixer and the main model's head
        if(residual_snapshots_ && cudaMemcpyAsync(residual_snapshots_+(size_t)coupled_j_*HC*N,
            R_,(size_t)HC*N*4,cudaMemcpyDeviceToDevice,cs)!=cudaSuccess){err="mtp residual trace: snapshot";return false;}
        static const bool fuse_mixer=[] {const char* v=std::getenv("STRATA_MTP_FUSED_MIXER");return v&&std::atoi(v)==1;}();
        if(fuse_mixer){
            // Auxiliary predictions only: target mixer, weights and acceptance
            // stay unchanged. Reuse the completed layer's HC workspace.
            FusedGrArgs fa[kFusedGrMaxT];
            for(int t=0;t<T;++t){
                fa[t].R=R_+(size_t)t*HC*N;fa[t].R_out=R_+(size_t)t*HC*N;
                fa[t].w_norm=f32("hyper_connection_mixer.hc_norm.weight");
                fa[t].w_down=bf16("hyper_connection_mixer.input_mix_weight_down.weight");
                fa[t].w_up=bf16("hyper_connection_mixer.input_mix_weight_up.weight");
                fa[t].eps=EPS;fa[t].lo=lo_+t*g.hc_lr;fa[t].rs=rs_+t*HC;
                fa[t].mixed=sample_+t*N;
            }
            fused_gr_read_multi(fa,T,xn_,cs);
        }else for (int t = 0; t < T; ++t)
            gr_read(R_ + (size_t) t * HC * N, f32("hyper_connection_mixer.hc_norm.weight"),
                    bf16("hyper_connection_mixer.input_mix_weight_down.weight"),
                    bf16("hyper_connection_mixer.input_mix_weight_up.weight"), nullptr, EPS, gs, ss.block.gr,
                    sample_ + t * N, dummy_inj_, cs);
        if (hidden_calibration_) {
            if (hidden_affine_)
                mtp_hidden_affine_correct(sample_, hidden_snapshots_ + (size_t)coupled_j_ * N, hidden_calibration_,
                                          (int)N, coupled_j_, max_t_, hidden_gain_, cs);
            else mtp_hidden_correct(sample_, hidden_snapshots_ + (size_t)coupled_j_ * N, hidden_calibration_,
                                    (int)N, coupled_j_, max_t_, hidden_gain_, cs);
        }
        if(label_snapshots_&&!coupled_rec_&&cudaMemcpyAsync(label_snapshots_+(size_t)coupled_j_*N,sample_,N*4,cudaMemcpyDeviceToDevice,cs)!=cudaSuccess){err="mtp: native label snapshot";return false;}
        if(feature_snapshots_&&cudaMemcpyAsync(feature_snapshots_+(size_t)coupled_j_*N,sample_,N*4,cudaMemcpyDeviceToDevice,cs)!=cudaSuccess){err="mtp feature trace: snapshot";return false;}
        native_quantize_q8_1(sample_, xq_, (int) N, T, cs);
        const bool sub = dhead_ != nullptr;
        const bool routed=head_router_&&!coupled_rec_;
        const int64_t nv=routed?head_router_->size():(sub?n_dvocab_:n_vocab_);
        const int32_t* output_vocab=routed?head_router_->vocab():(sub?dvocab_:nullptr);
        if(routed){if(!head_router_->run(xq_,head_logits_,cs,err))return false;}
        else native_mmvq(head_->type(), sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, T, cs);
        if(token_feature_weights_&&!coupled_rec_){
            if(cudaMemcpyAsync(token_feature_snapshots_+(size_t)coupled_j_*N,sample_,N*4,cudaMemcpyDeviceToDevice,cs)!=cudaSuccess){err="mtp: contextual token snapshot";return false;}
            mtp_token_feature_apply(head_logits_,sample_,token_feature_weights_,token_feature_scores_,token_feature_map_,token_feature_count_,output_vocab,(int)nv,(int)N,kTokenFeatureSlots,cs);
        }else if(token_bias_&&!coupled_rec_)mtp_token_bias_apply(head_logits_,token_bias_,output_vocab,(int)nv,cs);
        if (coupled_rec_) {
            // coupled draft sampling: the target's chain and Philox draw for the row that will verify this draft
            // (counter = this cell + 1, from its step record), penalties over the ring; T == 1 (full layer)
            coupled_draft_sample(head_logits_, (int) nv, sub ? dvocab_ : nullptr, sub ? dinv_ : nullptr, (int) n_vocab_,
                                 cparams_, cring_, kCoupledHistCap, coupled_j_, step, cscratch_, out_ids_, probs_, cs);
            return true;
        }
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) nv, nullptr, 0, sp, out_ids_, cs);
        row_top_prob(head_logits_, T, (int) nv, out_ids_, probs_, cs);
        if (output_vocab) map_ids(out_ids_, output_vocab, T, cs);
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

namespace {
bool finish_capture(cudaStream_t cs, bool ok, cudaGraphExec_t& exec, const char* what, std::string& err) {
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("mtp: ") + what + " capture: " + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    // an explicit upload: the first launch's implicit one blocked behind a device-side spin (verify.cpp)
    cudaGraphUpload(exec, cs);
    cudaStreamSynchronize(cs);
    return true;
}
}  // namespace

bool MtpDrafter::prepare_graphs(std::string& err) {
    const OnDevice on_device(device_);
    for(int t=1;t<=max_t_;++t)if(!capture_round(t,false,err))return false;
    for(int j=1;j<max_t_-1;++j)if(!capture_step(j,false,err))return false;
    return true;
}

bool MtpDrafter::reset_hidden_calibration(std::string& err) {
    if(label_trace_)++label_request_;
    if(feature_trace_)++feature_trace_request_;
    if(residual_trace_)++residual_trace_request_;
    const OnDevice on_device(device_);
    if(sparse_ && !index_checkpoint(true,cs_,err))return false;
    if(token_bias_&&cudaMemsetAsync(token_bias_,0,(size_t)n_vocab_*4,cs_)!=cudaSuccess){err="mtp: token calibration reset";return false;}
    if(token_feature_weights_){
        token_feature_used_=0;std::fill(token_feature_slots_.begin(),token_feature_slots_.end(),-1);
        if(cudaMemsetAsync(token_feature_weights_,0,(size_t)kTokenFeatureSlots*g_->n_embd*4,cs_)!=cudaSuccess||
           cudaMemsetAsync(token_feature_map_,0xff,(size_t)n_vocab_*4,cs_)!=cudaSuccess||
           cudaMemsetAsync(token_feature_count_,0,4,cs_)!=cudaSuccess){err="mtp: contextual token reset";return false;}
    }
    if (!hidden_calibration_) return true;
    const size_t bytes = ((size_t)max_t_ * g_->n_embd * (hidden_affine_ ? 4 : 1) + max_t_ * 4) * sizeof(float);
    if (cudaMemsetAsync(hidden_calibration_, 0, bytes, cs_) != cudaSuccess) {
        err = "mtp: hidden calibration reset failed"; return false;
    }
    return true;
}

bool MtpDrafter::index_checkpoint(bool save,cudaStream_t cs,std::string& err){
    if(!sparse_)return true;
    const auto s=shapes_of(*g_);
    auto copy=[&](void* checkpoint,void* state,size_t bytes){
        return cudaMemcpyAsync(save?checkpoint:state,save?state:checkpoint,bytes,cudaMemcpyDeviceToDevice,cs)==cudaSuccess;
    };
    if(!copy(index_tail_checkpoint_,st_.idx_tail,(s.idx_block-1)*s.idx_dim*sizeof(float)) ||
       !copy(index_dead_checkpoint_,st_.idx_dead,s.idx_dim*sizeof(float)) ||
       !copy(index_pos_checkpoint_,st_.idx_block_pos,sizeof(int32_t))){
        err="mtp: auxiliary indexer checkpoint copy failed";return false;
    }
    return true;
}

bool MtpDrafter::settle_verified(std::string& err){
    if(!sparse_)return true;
    const OnDevice on_device(device_);
    return index_checkpoint(false,cs_,err) && idle(err);
}

bool MtpDrafter::finish_verified(int T,const int32_t* tokens,int64_t p,int a,std::string& err){
    if(!sparse_)return true;
    if(T<1 || T>max_t_ || a<0 || a>=T){err="mtp: final verified prefix out of range";return false;}
    const OnDevice on_device(device_);
    if(!index_checkpoint(false,cs_,err))return false;
    return prefill(window_R_,tokens,a+1,p,err);
}

void MtpDrafter::set_token_bias_gain(float gain){token_bias_gain_=std::isfinite(gain)?std::clamp(gain,0.f,2.f):0.f;}
bool MtpDrafter::learn_tokens(const int32_t* proposals,const int32_t* teacher,int rows,std::string& err){
    if(label_trace_&&!coupled_active_&&label_count_<131072){
        if(!proposals||!teacher||rows<1||rows>=max_t_){err="mtp: native label teacher range";return false;}
        const OnDevice on(device_);const size_t H=(size_t)g_->n_embd;
        std::vector<float> snapshots((size_t)rows*H);
        if(cudaMemcpy(snapshots.data(),label_snapshots_,snapshots.size()*4,cudaMemcpyDeviceToHost)!=cudaSuccess){err="mtp: native label snapshot download";return false;}
        std::vector<unsigned char> record(32+H*4);
        for(int j=0;j<rows&&label_count_<131072;++j){
            const int32_t header[8]={label_request_,j,label_count_,(int32_t)H,proposals[j],teacher[j],head_->type(),0};
            std::memcpy(record.data(),header,32);std::memcpy(record.data()+32,snapshots.data()+(size_t)j*H,H*4);
            if(std::fwrite(record.data(),1,record.size(),label_trace_)!=record.size()){err="mtp: native label trace write";return false;}
            ++label_count_;
        }
    }
    if((!token_bias_&&!token_feature_weights_)||token_bias_gain_<=0||coupled_active_)return true;
    if(!proposals||!teacher||rows<1||rows>=max_t_){err="mtp: token teacher range";return false;}
    const OnDevice on_device(device_);
    for(int j=0;j<rows;++j){
        h_token_labels_[j]=proposals[j];h_token_labels_[8+j]=teacher[j];
        if(token_feature_weights_){
            auto slot=[&](int32_t token){
                if(token<0||token>=n_vocab_)return -1;
                auto& index=token_feature_slots_[(size_t)token];
                if(index<0&&token_feature_used_<kTokenFeatureSlots)index=token_feature_used_++;
                return index;
            };
            h_token_labels_[16+j]=slot(proposals[j]);h_token_labels_[24+j]=slot(teacher[j]);
        }
    }
    if(token_feature_weights_)h_token_labels_[32]=token_feature_used_;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if(token_feature_weights_)
        strata::kernels::mtp_token_feature_learn(token_feature_weights_,token_feature_snapshots_,token_feature_map_,token_feature_count_,m_token_labels_,rows,(int)g_->n_embd,(int)n_vocab_,token_bias_gain_,cs_);
    else strata::kernels::mtp_token_bias_learn(token_bias_,m_token_labels_,rows,(int)n_vocab_,token_bias_gain_,cs_);
    if(cudaGetLastError()!=cudaSuccess){err="mtp: token teacher update";return false;}return true;
}

bool MtpDrafter::learn_hidden(const float* teacher, int rows, std::string& err, const float* residual_teacher) {
    if(residual_trace_ && rows>0 && residual_trace_count_<16384){
        if(!residual_teacher || rows>max_t_){err="mtp residual trace: invalid teacher rows";return false;}
        const OnDevice on_device(device_);const size_t H=(size_t)g_->hc*g_->n_embd;
        std::vector<float> predicted((size_t)rows*H),truth((size_t)rows*H);
        if(cudaMemcpy(predicted.data(),residual_snapshots_,predicted.size()*4,cudaMemcpyDeviceToHost)!=cudaSuccess ||
           cudaMemcpy(truth.data(),residual_teacher,truth.size()*4,cudaMemcpyDeviceToHost)!=cudaSuccess){err="mtp residual trace: download";return false;}
        std::vector<uint8_t> record(16+H*8);
        for(int j=0;j<rows && residual_trace_count_<16384;++j){
            const int32_t header[4]={residual_trace_request_,j,residual_trace_count_,(int32_t)H};std::memcpy(record.data(),header,16);
            std::memcpy(record.data()+16,predicted.data()+(size_t)j*H,H*4);
            std::memcpy(record.data()+16+H*4,truth.data()+(size_t)j*H,H*4);
            if(std::fwrite(record.data(),1,record.size(),residual_trace_)!=record.size()){err="mtp residual trace: write";return false;}
            ++residual_trace_count_;
        }
    }
    if(feature_trace_&&rows>0&&feature_trace_count_<16384){
        if(!teacher||rows>max_t_){err="mtp feature trace: invalid teacher rows";return false;}
        const OnDevice on_device(device_);const size_t n=(size_t)rows*g_->n_embd;
        std::vector<float> predicted(n),truth(n);
        if(cudaMemcpy(predicted.data(),feature_snapshots_,n*4,cudaMemcpyDeviceToHost)!=cudaSuccess||cudaMemcpy(truth.data(),teacher,n*4,cudaMemcpyDeviceToHost)!=cudaSuccess){err="mtp feature trace: download";return false;}
        std::vector<uint8_t> record(16+(size_t)g_->n_embd*8);
        for(int j=0;j<rows&&feature_trace_count_<16384;++j){
            const int32_t header[4]={feature_trace_request_,j,feature_trace_count_,(int32_t)g_->n_embd};std::memcpy(record.data(),header,16);
            std::memcpy(record.data()+16,predicted.data()+(size_t)j*g_->n_embd,(size_t)g_->n_embd*4);
            std::memcpy(record.data()+16+g_->n_embd*4,truth.data()+(size_t)j*g_->n_embd,(size_t)g_->n_embd*4);
            if(std::fwrite(record.data(),1,record.size(),feature_trace_)!=record.size()){err="mtp feature trace: write";return false;}++feature_trace_count_;
        }
    }
    if (!hidden_calibration_ || rows <= 0) return true;
    if (!teacher || rows > max_t_) { err = "mtp: invalid hidden teacher rows"; return false; }
    const OnDevice on_device(device_);
    if (hidden_affine_)
        strata::kernels::mtp_hidden_affine_learn(teacher, hidden_snapshots_, hidden_calibration_, (int)g_->n_embd,
                                                rows, max_t_, hidden_gain_, hidden_rate_, cs_);
    else strata::kernels::mtp_hidden_learn(teacher, hidden_snapshots_, hidden_calibration_, (int)g_->n_embd,
                                          rows, max_t_, hidden_gain_, hidden_rate_, cs_);
    return true;
}

std::string MtpDrafter::hidden_calibration_report() const {
    if (!hidden_calibration_) return {};
    const OnDevice on_device(device_);
    std::vector<float> stats((size_t)max_t_ * 4);
    const float* source = hidden_calibration_ + (size_t)max_t_ * g_->n_embd * (hidden_affine_ ? 4 : 1);
    if (cudaMemcpyAsync(stats.data(),source,stats.size()*sizeof(float),cudaMemcpyDeviceToHost,cs_) != cudaSuccess ||
        cudaStreamSynchronize(cs_) != cudaSuccess) return "hidden calibration diagnostic copy failed";
    std::ostringstream out;
    out << (hidden_affine_ ? "affine" : "bias") << " teacher count/raw MSE/corrected MSE:";
    for (int j=0;j<max_t_-1;++j) out << " " << j << ":" << stats[j*4] << "/" << stats[j*4+1] << "/" << stats[j*4+2];
    return out.str();
}

bool MtpDrafter::capture_prefill(int T, std::string& err) {
    if (prefill_exec_[T]) return true;
    using namespace strata::kernels;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) T * g_->n_head, cs_);
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T)
    return finish_capture(cs_, ok, prefill_exec_[T], "prefill", err);
}

bool MtpDrafter::capture_prefill_dev(int T, std::string& err) {
    if (prefill_dev_exec_[T]) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T); tok_/step_/pos_ filled before launch
    return finish_capture(cs_, ok, prefill_dev_exec_[T], "prefill (device inputs)", err);
}

bool MtpDrafter::capture_round(int T, bool coupled, std::string& err) {
    cudaGraphExec_t& exec = coupled ? round_exec_c_[T] : round_exec_[T];
    if (exec) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    bool ok = true;
    // coupled: the request's chain and the penalty history's base, for this round's drafts
    if (coupled) coupled_draft_stage(m_cparams_, m_chist_, cparams_, cring_, kCoupledHistCap, cs_);
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) 2 * T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) 2 * T * g_->n_head, cs_);
    copy_i32_from_mapped(row_, m_row_, 2, cs_);
    copy_from_mapped(Rin_, window_R_, (int64_t) T * HCN, cs_);
    // the catch-up: K/V for the window's T cells, then the full layer for row a only (its cell's K/V is written
    // again, identically), staged by the host in step row 2*max_t - 1; the draft chain is one graph per step
    // (`capture_step`) so the host can stop it when a draft is unlikely
    const int ra = 2 * max_t_ - 1;
    // In the prefix path T is the accepted prefix, including row a. The full
    // forward below appends row a's K/V itself; rejected rows are never needed.
    static const bool prefix_env = [] { const char* v = std::getenv("STRATA_MTP_PREFIX_CATCHUP"); return v && std::atoi(v) == 1; }();
    const bool prefix=sparse_ || prefix_env;
    if(sparse_)ok=index_checkpoint(false,cs_,err);
    if(ok)ok = !prefix || T > 1 ? record_forward(prefix ? T - 1 : T, -1, cs_, err) : true;
    if (ok) mtp_select(Rin_, HCN, tok_, row_, Rin_, tok_, nullptr, 0, cs_);
    if (ok) {
        copy_i32_from_mapped(step_ + ra * 4, m_step_ + ra * 4, 4, cs_);
        copy_i32_from_mapped(pos_ + ra * g_->n_head, m_pos_ + ra * g_->n_head, g_->n_head, cs_);
        coupled_rec_ = coupled;
        coupled_j_ = 0;
        ok = record_forward(1, ra, cs_, err);
        coupled_rec_ = false;
    }
    if(ok && sparse_)ok=index_checkpoint(true,cs_,err);
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, 0, cs_, probs_, m_prob_);
    return finish_capture(cs_, ok, exec, coupled ? "round (coupled)" : "round", err);
}

// Chain step j (1..max_t-2): one row at the cell staged in step row `max_t + j - 1`, from the previous step's
// residual and token (left in Rin_[0] / tok_[0] by mtp_select); draft j and its probability to the mapped outputs.
bool MtpDrafter::capture_step(int j, bool coupled, std::string& err) {
    cudaGraphExec_t& exec = coupled ? step_exec_c_[j] : step_exec_[j];
    if (exec) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    const int row = max_t_ + j - 1;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    copy_i32_from_mapped(step_ + row * 4, m_step_ + row * 4, 4, cs_);
    copy_i32_from_mapped(pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head, cs_);
    coupled_rec_ = coupled;
    coupled_j_ = j;
    bool ok = record_forward(1, row, cs_, err);
    coupled_rec_ = false;
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, j, cs_, probs_, m_prob_);
    return finish_capture(cs_, ok, exec, coupled ? "step (coupled)" : "step", err);
}

void MtpDrafter::kv_restore(int64_t upto) {
    const OnDevice on_device(device_);
    if (st_.kv_mode != 2 || upto <= 0) return;
    // the ring's blocks below `upto`, from the host copy: a checkpoint resume may have left later cells in them
    const strata::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t b1 = (upto + s.page_size - 1) / s.page_size, b0 = std::max<int64_t>(0, b1 - st_.n_slots);
    strata::kernels::kv_ring_restore(qsa_attn_pools(st_), st_.host, qsa_kv_format(st_), b0, b1, st_.n_slots, s, cs_);
    cudaStreamSynchronize(cs_);
}

bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    const int64_t HCN = g_->hc * g_->n_embd;
    // cells the window can never reach again need no K/V
    const int64_t first_needed = (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0;
    // E-4: every group's token / step / position records uploaded at once; each group is then device copies and a
    // graph on the one stream, with a single sync at the end (a group of <= max_t rows used to be staged in mapped
    // memory and synced before the next: ~5,500 host round trips on a 32K prompt).  The same work in the same
    // order.  STRATA_MTP_PREFILL_SYNC=1 keeps the old loop, =0 forces E-4.
    // HIP defaults to the old loop.  This pass runs whenever E-9 (Prefill::draft_kv) declines, which it does for the
    // drafter's ring (KV streaming, --kv-resident), and on gfx1201 / ROCm 7.2.4 the E-4 queue (a graph launch and four
    // device copies per group, ~2,000 groups per 8192-token chunk, no sync) sometimes never completes: the prompt hangs
    // in hipGraphLaunch or the final sync until the watchdog ends the engine.  R9700, full IQ3_XXS, --kv-resident
    // 32768, mixed load (chats, 32K and 7K prompts, deep follow-ups): E-4 hung in the first round on 2 of 2 tries,
    // the old loop ran 30 of 30 rounds clean, prompt speed unchanged.
    static const bool per_group_sync = [] {
        if (const char* v = std::getenv("STRATA_MTP_PREFILL_SYNC")) return std::atoi(v) != 0;
#if defined(STRATA_USE_HIP)
        return true;
#else
        return false;
#endif
    }();
    const int64_t NHp = g_->n_head, per_row = 1 + 4 + NHp;
    if (!per_group_sync && n > 0) {
        if (pf_cap_ < n * per_row) {
            if (pf_dev_) cudaFree(pf_dev_);
            pf_dev_ = nullptr;
            pf_cap_ = 0;
            if (cudaMalloc((void**) &pf_dev_, (size_t) (n * per_row) * sizeof(int32_t)) != cudaSuccess) {
                err = "mtp prefill: the input records do not fit";
                return false;
            }
            pf_cap_ = n * per_row;
        }
        std::vector<int32_t> rec((size_t) (n * per_row));
        int32_t* tk = rec.data();
        int32_t* stp = tk + n;
        int32_t* ps = stp + 4 * n;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t cell = cell0 + i;
            tk[i] = next_tokens[i];
            stp[i * 4 + 0] = (int32_t) cell;
            stp[i * 4 + 1] = (int32_t) (cell + 1);
            stp[i * 4 + 2] = (int32_t) ((cell + 1) / 4);
            stp[i * 4 + 3] = (int32_t) (sparse_ ? std::min<int64_t>(cell+1,cap_) : cell+1);
            for (int64_t h = 0; h < NHp; ++h) ps[i * NHp + h] = (int32_t) cell;
        }
        if (cudaMemcpyAsync(pf_dev_, rec.data(), rec.size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "mtp prefill: the input records upload failed";
            return false;
        }
        const int32_t* d_tk = pf_dev_;
        const int32_t* d_stp = d_tk + n;
        const int32_t* d_ps = d_stp + 4 * n;
        for (int64_t c = 0; c < n; c += max_t_) {
            const int T = (int) std::min<int64_t>(max_t_, n - c);
            if (cell0 + c + T <= first_needed) continue;
            if (!capture_prefill_dev(T, err)) return false;
            if (cudaMemcpyAsync(tok_, d_tk + c, (size_t) T * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(step_, d_stp + c * 4, (size_t) T * 16, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(pos_, d_ps + c * NHp, (size_t) (T * NHp) * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(Rin_, R_rows + (size_t) c * HCN, (size_t) T * HCN * sizeof(float), cudaMemcpyDeviceToDevice,
                                cs_) != cudaSuccess ||
                cudaGraphLaunch(prefill_dev_exec_[T], cs_) != cudaSuccess) {
                err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        if(!index_checkpoint(true,cs_,err))return false;
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        ms_prefill += ms_since(t0);
        return true;
    }
    for (int64_t c = 0; c < n; c += max_t_) {
        const int T = (int) std::min<int64_t>(max_t_, n - c);
        if (cell0 + c + T <= first_needed) continue;
        if (!capture_prefill(T, err)) return false;
        for (int t = 0; t < T; ++t) {
            const int64_t cell = cell0 + c + t;
            h_tok_[t] = next_tokens[c + t];
            h_step_[t * 4 + 0] = (int32_t) cell;
            h_step_[t * 4 + 1] = (int32_t) (cell + 1);
            h_step_[t * 4 + 2] = (int32_t) ((cell + 1) / 4);
            h_step_[t * 4 + 3] = (int32_t) (sparse_ ? std::min<int64_t>(cell+1,cap_) : cell+1);
            for (int64_t h = 0; h < g_->n_head; ++h) h_pos_[t * g_->n_head + h] = (int32_t) cell;
        }
        if (cudaMemcpyAsync(Rin_, R_rows + (size_t) c * HCN, (size_t) T * HCN * sizeof(float), cudaMemcpyDeviceToDevice,
                            cs_) != cudaSuccess ||
            cudaGraphLaunch(prefill_exec_[T], cs_) != cudaSuccess ||
            cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    if(!index_checkpoint(true,cs_,err) || !idle(err))return false;
    ms_prefill += ms_since(t0);
    return true;
}

bool MtpDrafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                       float* probs, float min_p, int* n_drafts) {
    const OnDevice on_device(device_);
    if (T < 1 || T > max_t_ || a < 0 || a >= T) { err = "mtp: draft arguments out of range"; return false; }
    const bool cp = coupled_active_;   // coupled draft sampling for this request: its own graphs
    static const bool prefix_env = [] { const char* v = std::getenv("STRATA_MTP_PREFIX_CATCHUP"); return v && std::atoi(v) == 1; }();
    const bool prefix=sparse_ || prefix_env;
    const int catch_t = prefix ? a + 1 : T;
    if (!capture_round(catch_t, cp, err)) return false;
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (sparse_ ? std::min<int64_t>(cell+1,cap_) : cell+1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, coupled_draft_cell(p, a, 0));   // p + a: draft 0's cell
    h_row_[0] = a;
    h_row_[1] = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (cudaGraphLaunch(cp ? round_exec_c_[catch_t] : round_exec_[catch_t], cs_) != cudaSuccess ||
        cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    drafts[0] = ((volatile int32_t*) h_out_)[0];
    if(proposal_hook_ && !proposal_hook_(drafts[0],err))return false;
    trace_head_input(0,coupled_draft_cell(p,a,0),drafts[0]);
    float pj = ((volatile float*) h_prob_)[0];
    if (probs) probs[0] = pj;
    std::array<float,8> chain_prob{};
    chain_prob[0]=pj;
    int n = 1;
    // the chain continues while the last draft is likely enough to be verified
    for (int j = 1; j < std::min(max_t_ - 1, max_drafts_) &&
         (lookahead_ ? lookahead_->continue_draft(chain_prob.data(),n,std::min(max_t_-1,max_drafts_)) : pj >= min_p); ++j) {
        if (!capture_step(j, cp, err)) return false;
        put(max_t_ + j - 1, coupled_draft_cell(p, a, j));   // p + a + j; coupled: drawn at counter cell + 1
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (cudaGraphLaunch(cp ? step_exec_c_[j] : step_exec_[j], cs_) != cudaSuccess ||
            cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        drafts[j] = ((volatile int32_t*) h_out_)[j];
        if(proposal_hook_ && !proposal_hook_(drafts[j],err))return false;
        trace_head_input(j,coupled_draft_cell(p,a,j),drafts[j]);
        pj = ((volatile float*) h_prob_)[j];
        chain_prob[j]=pj;
        if (probs) probs[j] = pj;
        ++n;
    }
    for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
    if (n_drafts) *n_drafts = n;
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool MtpDrafter::draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                             float* probs, float min_p, int* n_drafts) {
    const OnDevice on_device(device_);
    // row 0 is the real pair; rows 1.. repeat it and only write cells the next round overwrites
    const int64_t HCN = g_->hc * g_->n_embd;
    for (int t = 0; t < T; ++t)
        if (cudaMemcpy((void*) (window_R_ + (size_t) t * HCN), R_row, (size_t) HCN * sizeof(float),
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            err = "mtp: staging the first residual failed";
            return false;
        }
    std::vector<int32_t> toks((size_t) T, token);
    return draft(T, toks.data(), cell, 0, drafts, err, probs, min_p, n_drafts);
}

}  // namespace strata::core
