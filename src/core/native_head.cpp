#include "strata/core/native_head.hpp"
#include "strata/core/shared_weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/elementwise.hpp"
#include <atomic>

#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>

namespace strata::core {

struct NativePeerHead {
    int primary=-1, other=-1, rows=0, tail=0, width=0, type=-1;
    void* weights=nullptr;
    void* query=nullptr;
    void* query_host=nullptr;
    float* logits=nullptr;
    float* logits_host=nullptr;
    float* tail_logits=nullptr;
    bool compact=false, graph=false;
    uint32_t* graph_state=nullptr;
    cudaGraphExec_t graph_exec[9]={};
    uint32_t epoch=0;
    int *peer_ids=nullptr, *tail_ids=nullptr;
    float *peer_scores=nullptr, *tail_scores=nullptr;
    int *peer_ids_host=nullptr, *tail_ids_host=nullptr, *merged_host=nullptr;
    float *peer_scores_host=nullptr, *tail_scores_host=nullptr;
    cudaStream_t stream=nullptr;
    ~NativePeerHead() {
        int saved=-1;cudaGetDevice(&saved);
        if(other>=0) {
            cudaSetDevice(other);
            if(stream)cudaStreamSynchronize(stream);
            for(auto& exec:graph_exec)if(exec)cudaGraphExecDestroy(exec);
            if(weights)cudaFree(weights);
            if(query)cudaFree(query);
            if(logits)cudaFree(logits);
            if(peer_ids)cudaFree(peer_ids);
            if(peer_scores)cudaFree(peer_scores);
            if(stream)cudaStreamDestroy(stream);
        }
        if(primary>=0) {
            cudaSetDevice(primary);
            if(tail_logits)cudaFree(tail_logits);
            if(tail_ids)cudaFree(tail_ids);
            if(tail_scores)cudaFree(tail_scores);
        }
        if(graph_state)cudaFreeHost(graph_state);
        if(query_host)cudaFreeHost(query_host);
        if(logits_host)cudaFreeHost(logits_host);
        if(peer_ids_host)cudaFreeHost(peer_ids_host);
        if(tail_ids_host)cudaFreeHost(tail_ids_host);
        if(merged_host)cudaFreeHost(merged_host);
        if(peer_scores_host)cudaFreeHost(peer_scores_host);
        if(tail_scores_host)cudaFreeHost(tail_scores_host);
        if(saved>=0)cudaSetDevice(saved);
    }
};

NativeHead::~NativeHead() {
    delete peer_;
    if (scratch_) cudaFree(scratch_);
    if (weights_) shared_weight_free(weights_);
}

bool NativeHead::load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, std::string& err) {
    if (loaded()) { err = "native head is already loaded"; return false; }
    if (n_in <= 0 || n_out <= 0 || n_in > INT_MAX || n_out > INT_MAX || n_in % 256) {
        err = "native head requires positive int32 dimensions and whole 256-value rows";
        return false;
    }
    try {
        // The architecture is the metadata shard's; output.weight comes from whichever shard holds it (shard 2
        // of Unsloth's UD-Q4_K_XL, whose shard 1 holds no tensor).  GgufModel refuses a duplicate across shards.
        const strata::GgufModel model(shards);
        err = strata::check_architecture(model.meta());
        if (!err.empty()) return false;
        size_t at = 0;
        const strata::TensorInfo* tensor = model.find("output.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!tensor || !strata::kernels::native_mmvq_supported((int) tensor->type) || tensor->shape.size() != 2 ||
            tensor->shape[0] != (uint64_t) n_in || tensor->shape[1] != (uint64_t) n_out) {
            err = "native head: expected a natively supported output.weight with the canonical head dimensions";
            return false;
        }
        const uint64_t bytes = strata::kernels::native_mmvq_weight_bytes((int) tensor->type, (int) n_in, (int) n_out);
        const uint64_t payload = gguf.file_size() - gguf.data_start();
        if (tensor->offset > payload || bytes > payload - tensor->offset) {
            err = "native head: truncated output.weight payload";
            return false;
        }
        void* weights = nullptr;
        void* scratch = nullptr;
        cudaError_t status = shared_weight_upload(&weights, gguf.tensor_data(*tensor), bytes, "output-head");
        if (status == cudaSuccess)
            status = cudaMalloc(&scratch, strata::kernels::native_q8_1_bytes((int) n_in, 1));
        if (status != cudaSuccess) {
            if (scratch) cudaFree(scratch);
            if (weights) shared_weight_free(weights);
            err = std::string("native head upload: ") + cudaGetErrorString(status);
            return false;
        }
        weights_ = weights;
        scratch_ = scratch;
        bytes_ = bytes;
        n_in_ = (int) n_in;
        n_out_ = (int) n_out;
        type_ = (int) tensor->type;
        const char* parallel=std::getenv("STRATA_HEAD_PARALLEL");
        if(parallel && parallel[0]=='1') {
            int dev=-1,nd=0;cudaGetDevice(&dev);
            if(cudaGetDeviceCount(&nd)!=cudaSuccess) {err="native peer head: device query failed";return false;}
            // In the supported two-device layer split the head belongs to the last stage (device 1).
            // A single-device deployment keeps its original head and memory budget.
            if(nd==2 && dev==1) {
                auto p=std::make_unique<NativePeerHead>();
                const char* compact=std::getenv("STRATA_HEAD_PARALLEL_COMPACT");
                const char* graph=std::getenv("STRATA_HEAD_PARALLEL_GRAPH");
                p->graph=graph && graph[0]=='1';
                p->compact=p->graph || (compact && compact[0]=='1');
                p->primary=dev;p->other=0;p->width=n_in_;p->type=type_;
                p->rows=(n_out_/512)*256;p->tail=n_out_-p->rows;
                if(p->rows<=0) {err="native peer head: vocabulary is too small";return false;}
                const size_t qbytes=strata::kernels::native_q8_1_bytes(n_in_,8);
                const size_t outbytes=(size_t)8*p->rows*4;
                cudaError_t e=cudaMalloc((void**)&p->tail_logits,(size_t)8*p->tail*4);
                if(e==cudaSuccess && p->compact)e=cudaMalloc((void**)&p->tail_ids,8*4);
                if(e==cudaSuccess && p->compact)e=cudaMalloc((void**)&p->tail_scores,8*4);
                if(e==cudaSuccess && p->compact)e=cudaHostAlloc((void**)&p->peer_ids_host,8*4,cudaHostAllocPortable|cudaHostAllocMapped);
                if(e==cudaSuccess && p->compact)e=cudaHostAlloc((void**)&p->tail_ids_host,8*4,cudaHostAllocPortable|cudaHostAllocMapped);
                if(e==cudaSuccess && p->compact)e=cudaHostAlloc((void**)&p->merged_host,8*4,cudaHostAllocPortable|cudaHostAllocMapped);
                if(e==cudaSuccess && p->compact)e=cudaHostAlloc((void**)&p->peer_scores_host,8*4,cudaHostAllocPortable|cudaHostAllocMapped);
                if(e==cudaSuccess && p->compact)e=cudaHostAlloc((void**)&p->tail_scores_host,8*4,cudaHostAllocPortable|cudaHostAllocMapped);
                if(e==cudaSuccess)e=cudaHostAlloc(&p->query_host,qbytes,cudaHostAllocPortable|cudaHostAllocMapped);
                if(e==cudaSuccess)e=cudaHostAlloc((void**)&p->logits_host,outbytes,cudaHostAllocPortable|cudaHostAllocMapped);
                if(e==cudaSuccess && p->graph)e=cudaHostAlloc((void**)&p->graph_state,512,cudaHostAllocPortable|cudaHostAllocMapped);
                if(e==cudaSuccess && p->graph)std::memset(p->graph_state,0,512);
                if(e==cudaSuccess)e=cudaSetDevice(p->other);
                if(e==cudaSuccess)e=cudaStreamCreateWithFlags(&p->stream,cudaStreamNonBlocking);
                if(e==cudaSuccess)e=cudaMalloc(&p->weights,(size_t)p->rows*row_bytes());
                if(e==cudaSuccess)e=cudaMalloc(&p->query,qbytes);
                if(e==cudaSuccess)e=cudaMalloc((void**)&p->logits,outbytes);
                if(e==cudaSuccess && p->compact)e=cudaMalloc((void**)&p->peer_ids,8*4);
                if(e==cudaSuccess && p->compact)e=cudaMalloc((void**)&p->peer_scores,8*4);
                if(e==cudaSuccess)e=cudaMemcpy(p->weights,gguf.tensor_data(*tensor),(size_t)p->rows*row_bytes(),cudaMemcpyHostToDevice);
                const cudaError_t restored=cudaSetDevice(dev);
                if(e!=cudaSuccess || restored!=cudaSuccess) {
                    err=std::string("native peer head setup: ")+cudaGetErrorString(e!=cudaSuccess?e:restored);
                    return false;
                }
                std::fprintf(stderr,"strata native peer head: exact vocabulary rows %d/%d on CUDA0/CUDA1; original logits and sampler\n",p->rows,p->tail);
                peer_=p.release();
            }
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native head: ") + error.what();
        return false;
    }
}

bool NativeHead::run(const float* mixed, float* logits, void* stream, std::string& err) const {
    if (!loaded() || !mixed || !logits || !stream) {
        err = "native head requires loaded weights, device buffers and an explicit stream";
        return false;
    }
    try {
        if (type_ == 13) {
            strata::kernels::native_q5_k_f32(weights_, mixed, scratch_, logits, n_in_, n_out_, 1, stream);
        } else {
            strata::kernels::native_quantize_q8_1(mixed, scratch_, n_in_, 1, stream);
            strata::kernels::native_mmvq(type_, weights_, scratch_, logits, n_in_, n_out_, 1, stream);
        }
    } catch (const std::exception& error) {
        err = std::string("native head launch: ") + error.what();
        return false;
    }
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        err = std::string("native head launch: ") + cudaGetErrorString(status);
        return false;
    }
    return true;
}

bool NativeHead::project_parallel(const void* x_q8,float* logits,int T,void* stream,std::string& err) const {
    if(!peer_ || !x_q8 || !logits || !stream || T<1 || T>8) {
        err="native peer head: invalid projection arguments";return false;
    }
    auto& p=*peer_;const auto cs=(cudaStream_t)stream;
    auto checked=[](cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
    try {
        checked(cudaSetDevice(p.primary));
        const size_t qbytes=strata::kernels::native_q8_1_bytes(n_in_,T);
        checked(cudaMemcpyAsync(p.query_host,x_q8,qbytes,cudaMemcpyDeviceToHost,cs));
        checked(cudaStreamSynchronize(cs));
        checked(cudaSetDevice(p.other));
        checked(cudaMemcpyAsync(p.query,p.query_host,qbytes,cudaMemcpyHostToDevice,p.stream));
        strata::kernels::native_mmvq(type_,p.weights,p.query,p.logits,n_in_,p.rows,T,p.stream);
        checked(cudaMemcpyAsync(p.logits_host,p.logits,(size_t)T*p.rows*4,cudaMemcpyDeviceToHost,p.stream));
        checked(cudaSetDevice(p.primary));
        strata::kernels::native_mmvq(type_,(const uint8_t*)weights_+(size_t)p.rows*row_bytes(),x_q8,
                                    p.tail_logits,n_in_,p.tail,T,cs);
        checked(cudaMemcpy2DAsync(logits+p.rows,(size_t)n_out_*4,p.tail_logits,(size_t)p.tail*4,
                                  (size_t)p.tail*4,T,cudaMemcpyDeviceToDevice,cs));
        checked(cudaSetDevice(p.other));
        checked(cudaStreamSynchronize(p.stream));
        checked(cudaSetDevice(p.primary));
        checked(cudaMemcpy2DAsync(logits,(size_t)n_out_*4,p.logits_host,(size_t)p.rows*4,
                                  (size_t)p.rows*4,T,cudaMemcpyHostToDevice,cs));
        return true;
    } catch(const std::exception& e) {
        cudaSetDevice(p.primary);err=std::string("native peer head projection: ")+e.what();return false;
    }
}

bool NativeHead::parallel_compact() const { return peer_ && peer_->compact; }

bool NativeHead::parallel_graph() const { return peer_ && peer_->graph; }

bool NativeHead::record_parallel_graph(const void* x_q8,int32_t* out,int T,void* stream,std::string& err) const {
    if(!parallel_graph()||!x_q8||!out||!stream||T<1||T>8){err="graph peer head: record arguments";return false;}
    auto& p=*peer_;using namespace strata::kernels;
    try {
        peer_head_publish_query(x_q8,p.query_host,(int)native_q8_1_bytes(n_in_,T)/4,p.graph_state,stream);
        native_mmvq(type_,(const uint8_t*)weights_+(size_t)p.rows*row_bytes(),x_q8,p.tail_logits,n_in_,p.tail,T,stream);
        SamplerParams sp;sp.greedy=true;sp.temperature=0;
        sample_tokens(p.tail_logits,T,p.tail,nullptr,0,sp,p.tail_ids,stream);
        selected_token_scores(p.tail_logits,p.tail_ids,p.tail_scores,T,p.tail,stream);
        peer_head_merge_result(p.peer_ids_host,p.peer_scores_host,p.tail_ids,p.tail_scores,out,T,p.rows,p.graph_state,stream);
        return cudaPeekAtLastError()==cudaSuccess;
    }catch(const std::exception& e){err=std::string("graph peer head record: ")+e.what();return false;}
}

bool NativeHead::launch_parallel_graph(int T,std::string& err) const {
    if(!parallel_graph()||T<1||T>8){err="graph peer head: launch arguments";return false;}
    auto& p=*peer_;using namespace strata::kernels;
    auto check=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
    try {
        check(cudaSetDevice(p.other));
        if(!p.graph_exec[T]){
            check(cudaStreamBeginCapture(p.stream,cudaStreamCaptureModeThreadLocal));
            peer_head_receive_query(p.query_host,p.query,(int)native_q8_1_bytes(n_in_,T)/4,p.graph_state,p.stream);
            native_mmvq(type_,p.weights,p.query,p.logits,n_in_,p.rows,T,p.stream);
            SamplerParams sp;sp.greedy=true;sp.temperature=0;
            sample_tokens(p.logits,T,p.rows,nullptr,0,sp,p.peer_ids,p.stream);
            selected_token_scores(p.logits,p.peer_ids,p.peer_scores,T,p.rows,p.stream);
            peer_head_publish_result(p.peer_ids,p.peer_scores,p.peer_ids_host,p.peer_scores_host,T,p.graph_state,p.stream);
            cudaGraph_t graph=nullptr;check(cudaStreamEndCapture(p.stream,&graph));
            const auto e=cudaGraphInstantiate(&p.graph_exec[T],graph,0);cudaGraphDestroy(graph);check(e);
            check(cudaGraphUpload(p.graph_exec[T],p.stream));check(cudaStreamSynchronize(p.stream));
        }
        if(++p.epoch==0)++p.epoch;
        p.graph_state[96]=0;*reinterpret_cast<volatile uint32_t*>(p.graph_state)=p.epoch;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        check(cudaGraphLaunch(p.graph_exec[T],p.stream));
        check(cudaSetDevice(p.primary));return true;
    }catch(const std::exception& e){cudaSetDevice(p.primary);err=std::string("graph peer head launch: ")+e.what();return false;}
}

bool NativeHead::check_parallel_graph(std::string& err) const {
    if(!parallel_graph())return true;
    auto& p=*peer_;const uint32_t error=*reinterpret_cast<const volatile uint32_t*>(p.graph_state+96);
    const uint32_t done=*reinterpret_cast<const volatile uint32_t*>(p.graph_state+64);
    if(error||done!=p.epoch){err="graph peer head: mapped transport timeout or stale result";return false;}
    return true;
}


bool NativeHead::project_parallel_greedy(const void* x_q8,int32_t* out,int T,void* stream,std::string& err) const {
    if(!parallel_compact() || !x_q8 || !out || !stream || T<1 || T>8) {
        err="native compact peer head: invalid projection arguments";return false;
    }
    auto& p=*peer_;const auto cs=(cudaStream_t)stream;
    auto checked=[](cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
    try {
        checked(cudaSetDevice(p.primary));
        const size_t qbytes=strata::kernels::native_q8_1_bytes(n_in_,T);
        checked(cudaMemcpyAsync(p.query_host,x_q8,qbytes,cudaMemcpyDeviceToHost,cs));
        checked(cudaStreamSynchronize(cs));
        strata::kernels::SamplerParams sp;sp.greedy=true;sp.temperature=0;
        checked(cudaSetDevice(p.other));
        checked(cudaMemcpyAsync(p.query,p.query_host,qbytes,cudaMemcpyHostToDevice,p.stream));
        strata::kernels::native_mmvq(type_,p.weights,p.query,p.logits,n_in_,p.rows,T,p.stream);
        strata::kernels::sample_tokens(p.logits,T,p.rows,nullptr,0,sp,p.peer_ids,p.stream);
        strata::kernels::selected_token_scores(p.logits,p.peer_ids,p.peer_scores,T,p.rows,p.stream);
        checked(cudaMemcpyAsync(p.peer_ids_host,p.peer_ids,T*4,cudaMemcpyDeviceToHost,p.stream));
        checked(cudaMemcpyAsync(p.peer_scores_host,p.peer_scores,T*4,cudaMemcpyDeviceToHost,p.stream));
        checked(cudaSetDevice(p.primary));
        strata::kernels::native_mmvq(type_,(const uint8_t*)weights_+(size_t)p.rows*row_bytes(),x_q8,
                                    p.tail_logits,n_in_,p.tail,T,cs);
        strata::kernels::sample_tokens(p.tail_logits,T,p.tail,nullptr,0,sp,p.tail_ids,cs);
        strata::kernels::selected_token_scores(p.tail_logits,p.tail_ids,p.tail_scores,T,p.tail,cs);
        checked(cudaMemcpyAsync(p.tail_ids_host,p.tail_ids,T*4,cudaMemcpyDeviceToHost,cs));
        checked(cudaMemcpyAsync(p.tail_scores_host,p.tail_scores,T*4,cudaMemcpyDeviceToHost,cs));
        checked(cudaStreamSynchronize(cs));
        checked(cudaSetDevice(p.other));checked(cudaStreamSynchronize(p.stream));
        checked(cudaSetDevice(p.primary));
        for(int t=0;t<T;++t) {
            const float a=p.peer_scores_host[t],b=p.tail_scores_host[t];
            // The original sampler ignores NaNs / -inf and picks id 0 if no candidate wins.
            const bool tail_wins=b>-std::numeric_limits<float>::infinity() && (std::isnan(a) || b>a);
            p.merged_host[t]=tail_wins?p.rows+p.tail_ids_host[t]:p.peer_ids_host[t];
        }
        checked(cudaMemcpyAsync(out,p.merged_host,T*4,cudaMemcpyHostToDevice,cs));
        return true;
    } catch(const std::exception& e) {
        cudaSetDevice(p.primary);err=std::string("native compact peer head: ")+e.what();return false;
    }
}

bool NativeHead::materialize_parallel(float* logits,int T,void* stream,std::string& err) const {
    if(!peer_ || !logits || !stream || T<1 || T>8) {err="native peer head: invalid materialization";return false;}
    auto& p=*peer_;const auto cs=(cudaStream_t)stream;
    auto checked=[](cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
    try {
        checked(cudaSetDevice(p.other));
        checked(cudaMemcpyAsync(p.logits_host,p.logits,(size_t)T*p.rows*4,cudaMemcpyDeviceToHost,p.stream));
        checked(cudaSetDevice(p.primary));
        checked(cudaMemcpy2DAsync(logits+p.rows,(size_t)n_out_*4,p.tail_logits,(size_t)p.tail*4,
                                  (size_t)p.tail*4,T,cudaMemcpyDeviceToDevice,cs));
        checked(cudaSetDevice(p.other));checked(cudaStreamSynchronize(p.stream));
        checked(cudaSetDevice(p.primary));
        checked(cudaMemcpy2DAsync(logits,(size_t)n_out_*4,p.logits_host,(size_t)p.rows*4,
                                  (size_t)p.rows*4,T,cudaMemcpyHostToDevice,cs));
        checked(cudaStreamSynchronize(cs));return true;
    } catch(const std::exception& e) {
        cudaSetDevice(p.primary);err=std::string("native peer head materialization: ")+e.what();return false;
    }
}

// ================================ plan v0.3 P6: THE NATIVE EMBEDDING ================================

namespace {
const NativeEmbed* g_embed = nullptr;
}
void set_native_embed(const NativeEmbed* e) { g_embed = e; }
const NativeEmbed* native_embed() { return g_embed; }

NativeEmbed::~NativeEmbed() {
    if (host_) cudaFreeHost(host_);
    else if (dev_) cudaFree(const_cast<void*>(dev_));   // the VRAM fallback below
}

bool NativeEmbed::load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab, std::string& err) {
    try {
        const strata::GgufModel model(shards);
        // --embd-gguf's one-tensor file (tools/embd_bf16_pack.py) says "strata-embd": only its tensor is checked
        const strata::MetaValue* arch = model.meta().get("general.architecture");
        err = arch != nullptr && arch->s == "strata-embd" ? std::string() : strata::check_architecture(model.meta());
        if (!err.empty()) { err = "native embedding: " + err; return false; }
        size_t at = 0;
        const strata::TensorInfo* t = model.find("token_embd.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) n_embd || t->shape[1] != (uint64_t) n_vocab ||
            !strata::kernels::embed_type_supported((int) t->type) || n_embd % 256) {
            err = "native embedding: token_embd.weight is absent, of another shape, or of a type without a GPU "
                  "dequantizer";
            return false;
        }
        row_ = strata::kernels::iq_row_bytes((int) t->type, n_embd);
        bytes_ = (uint64_t) row_ * (uint64_t) n_vocab;
        // the table is copied out of the mapping below: a truncated shard must be an error, not a read past EOF
        if (!model.in_bounds(*t, at) || strata::tensor_payload_bytes(*t) != bytes_) {
            err = "native embedding: token_embd.weight's payload is truncated or not " + std::to_string(bytes_) +
                  " B (" + gguf.path() + ")";
            bytes_ = 0;
            return false;
        }
        if (cudaHostAlloc(&host_, bytes_, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            // Under WSL2 the driver's pinned/mapped host budget (~1 GiB) can be spent by the GPU contexts
            // themselves (three cards). The table is only gathered from, so keep it in the current device's VRAM
            // instead: it costs its size there and reads faster than over PCIe.
            cudaGetLastError();
            host_ = nullptr;
            void* d = nullptr;
            if (cudaMalloc(&d, bytes_) != cudaSuccess ||
                cudaMemcpy(d, gguf.tensor_data(*t), bytes_, cudaMemcpyHostToDevice) != cudaSuccess) {
                if (d) cudaFree(d);
                cudaGetLastError();
                err = "native embedding: cannot pin " + std::to_string(bytes_ >> 20) + " MiB, nor place it in VRAM";
                return false;
            }
            std::fprintf(stderr, "strata: native embedding: cannot pin %llu MiB, kept in VRAM instead\n",
                         (unsigned long long) (bytes_ >> 20));
            dev_ = d;
        } else {
            std::memcpy(host_, gguf.tensor_data(*t), bytes_);
            void* d = nullptr;
#if defined(STRATA_USE_HIP)
            // #325: the Windows HIP stack can refuse the device alias of a mapped allocation (and, when it gives
            // one, it is the host address itself - unified addressing; kernels read it correctly there, a
            // device-to-device copy into it does not land: tests/hip/mapped_alias). The table is only gathered from,
            // so without an alias it goes into VRAM like the unpinnable case above, instead of failing the start.
            if (cudaHostGetDevicePointer(&d, host_, 0) != cudaSuccess || d == nullptr) {
                cudaGetLastError();
                d = nullptr;
                if (cudaMalloc(&d, bytes_) != cudaSuccess ||
                    cudaMemcpy(d, gguf.tensor_data(*t), bytes_, cudaMemcpyHostToDevice) != cudaSuccess) {
                    if (d) cudaFree(d);
                    cudaGetLastError();
                    err = "native embedding: no device alias for the mapped table, and no VRAM to copy it into";
                    return false;
                }
                cudaFreeHost(host_);   // the destructor frees dev_ when host_ is null
                host_ = nullptr;
                std::fprintf(stderr, "strata: native embedding: no device alias for the mapped table, kept in VRAM\n");
            }
#else
            if (cudaHostGetDevicePointer(&d, host_, 0) != cudaSuccess) {
                err = "native embedding: no device alias for the mapped table";
                return false;
            }
#endif
            dev_ = d;
        }
        type_ = (int) t->type;
        n_embd_ = n_embd;
        n_vocab_ = n_vocab;
        return true;
    } catch (const std::exception& e) {
        err = std::string("native embedding: ") + e.what();
        return false;
    }
}

void NativeEmbed::gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const {
    strata::kernels::iq_embed_rows(type_, dev_, row_, tokens, n_tok, n_embd_, out, stream);
}

void NativeEmbed::gather_one(int64_t token, float* out, void* stream) const {
    strata::kernels::iq_dequant_f32(type_, (const uint8_t*) dev_ + (size_t) token * row_, n_embd_, out, stream);
}

}  // namespace strata::core
