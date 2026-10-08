#include "strata/core/peer_decode.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/peer_decode.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <stdexcept>

namespace strata::core {
namespace {
bool mapped(size_t bytes,void** h,void** m){
    if(cudaHostAlloc(h,bytes,cudaHostAllocPortable|cudaHostAllocMapped)!=cudaSuccess)return false;
    std::memset(*h,0,bytes);return cudaHostGetDevicePointer(m,*h,0)==cudaSuccess;
}
template<class T> bool alloc(T*& p,size_t count){return cudaMalloc((void**)&p,count*sizeof(T))==cudaSuccess&&cudaMemset(p,0,count*sizeof(T))==cudaSuccess;}
}
void PeerDecode::abort(){
    if(h_flags_){*(volatile uint32_t*)(h_flags_+layers_*32)=1;std::atomic_thread_fence(std::memory_order_seq_cst);}
}
bool PeerDecode::release(int timeout_ms){
    abort();if(!stream_)return true;const OnDevice on(peer_);
    const auto start=std::chrono::steady_clock::now();cudaError_t e;
    while((e=cudaStreamQuery(stream_))==cudaErrorNotReady){
        if(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()>timeout_ms)return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return e==cudaSuccess;
}
PeerDecode::~PeerDecode(){
    abort();
    {const OnDevice on(peer_);if(stream_)cudaStreamSynchronize(stream_);
        for(auto& g:graphs_)if(g)cudaGraphExecDestroy(g);
        if(stream_)cudaStreamDestroy(stream_);
        for(void* p:{(void*)peer_table_,(void*)q8_,(void*)scratch_,(void*)parts_,(void*)ids_,(void*)epoch_,(void*)plan_})if(p)cudaFree(p);}
    {const OnDevice on(primary_);if(primary_table_)cudaFree(primary_table_);
        for(void* p:{h_q8_,(void*)h_ids_,(void*)h_parts_,(void*)h_flags_})if(p)cudaFreeHost(p);}
}
bool PeerDecode::init(PeerExperts& peer,const VerifyHits& primary,const ModelGeometry& g,int max_t,int k,std::string& err){
    cudaGetDevice(&primary_);peer_=peer.device();layers_=(int)g.n_layers;experts_=(int)g.n_expert;
    h_=(int)g.n_embd;ff_=(int)g.n_ff;mt_=max_t;k_=k;cap_=mt_*k_;q8_bytes_=mt_*(h_/32)*36;
    if(!primary.static_residency||peer_<0||peer_==primary_||mt_<1||mt_>8||cap_>128||h_%32||!strata::kernels::cpu::expert_layout().native){
        err="peer decode requires two distinct GPUs, static native expert residency and at most 8 rows";return false;}
    std::vector<int32_t> res((size_t)layers_*experts_);
    if(cudaMemcpy(res.data(),primary.d_res,res.size()*4,cudaMemcpyDeviceToHost)!=cudaSuccess){err="peer decode: residency download";return false;}
    std::vector<unsigned long long> a(res.size()),b(res.size());size_t na=0,nb=0;
    for(int l=0;l<layers_;++l)for(int e=0;e<experts_;++e){const size_t i=(size_t)l*experts_+e;const int slot=res[i];
        if(slot>=0&&slot<primary.n_slots){a[i]=(unsigned long long)(primary.cache_base+(primary.slot_off?primary.slot_off[slot]:(uint64_t)slot*primary.blob));++na;}
        else if(peer.has(l,e)){b[i]=(unsigned long long)peer.slot_ptr(l,e);++nb;}
        else{err="peer decode: nonresident expert at layer "+std::to_string(l)+", expert "+std::to_string(e);return false;}}
    for(const auto& f:strata::kernels::cpu::expert_layout().fmt)layouts_.push_back(strata::kernels::native_expert_layout(f.gu_type,f.d_type,f.n_embd,f.n_ff));
    if(layouts_.size()!=(size_t)layers_){err="peer decode: native layout count";return false;}
    if(!alloc(primary_table_,a.size())||cudaMemcpy(primary_table_,a.data(),a.size()*8,cudaMemcpyHostToDevice)!=cudaSuccess||
       !mapped((size_t)layers_*q8_bytes_,&h_q8_,&m_q8_)||
       !mapped((size_t)layers_*cap_*4,(void**)&h_ids_,(void**)&m_ids_)||
       !mapped((size_t)layers_*cap_*h_*4,(void**)&h_parts_,(void**)&m_parts_)||
       !mapped((size_t)(layers_*32+32)*4,(void**)&h_flags_,(void**)&m_flags_)){
        err="peer decode: portable handoff allocation";return false;}
    const int po=(4+cap_+1+2*cap_+1)&~1;plan_i32_=po+2*cap_+16;
    {const OnDevice on(peer_);
        if(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking)!=cudaSuccess||!alloc(peer_table_,b.size())||
           cudaMemcpy(peer_table_,b.data(),b.size()*8,cudaMemcpyHostToDevice)!=cudaSuccess||!alloc(q8_,q8_bytes_)||
           !alloc(ids_,cap_)||!alloc(epoch_,1)||!alloc(plan_,plan_i32_)||!alloc(parts_,(size_t)cap_*h_)||
           !alloc(scratch_,strata::kernels::native_expert_scratch_bytes(cap_,ff_))){
            err="peer decode: worker allocation";return false;}}
    std::fprintf(stderr,"strata peer decode: CUDA%d + CUDA%d, %zu + %zu immutable native experts; GPU-side dispatch, %.1f MiB mapped results\n",
        primary_,peer_,na,nb,(double)layers_*cap_*h_*4/1048576.0);return true;
}
bool PeerDecode::prepare(int t,std::string& err){
    if(t<1||t>mt_){err="peer decode: invalid width";return false;}if(graphs_[t])return true;
    using namespace strata::kernels;const OnDevice on(peer_);
    if(cudaStreamBeginCapture(stream_,cudaStreamCaptureModeThreadLocal)!=cudaSuccess){err="peer decode: begin capture";return false;}
    std::string fail;
    try{const int po=(4+cap_+1+2*cap_+1)&~1;auto* ptr=(unsigned long long*)(plan_+po);
        auto* start=plan_+4;auto* dst=start+cap_+1;auto* tok=dst+cap_;
        for(int l=0;l<layers_;++l){
            peer_await_input(q8_,ids_,epoch_,(uint8_t*)m_q8_+(size_t)l*q8_bytes_,m_ids_+(size_t)l*cap_,
                m_flags_+l*32,m_flags_+l*32+16,t*(h_/32)*36,t*k_,m_flags_+layers_*32,stream_);
            peer_partial_plan(ids_,t*k_,k_,peer_table_+(size_t)l*experts_,experts_,plan_,cap_,stream_);
            native_expert_grouped(layouts_[l],ptr,start,plan_,dst,tok,t*k_,t*k_,q8_,scratch_,parts_,stream_);
            peer_scatter_output(m_parts_+(size_t)l*cap_*h_,parts_,ids_,peer_table_+(size_t)l*experts_,h_,t*k_,stream_);
            peer_publish_output(m_flags_+l*32+16,epoch_,m_flags_+layers_*32,stream_);
        }}catch(const std::exception& e){fail=e.what();}
    cudaGraph_t graph=nullptr;auto ce=cudaStreamEndCapture(stream_,&graph);
    if(!fail.empty()||ce!=cudaSuccess){if(graph)cudaGraphDestroy(graph);err="peer decode capture: "+(fail.empty()?std::string(cudaGetErrorString(ce)):fail);return false;}
    auto ie=cudaGraphInstantiate(&graphs_[t],graph,0);cudaGraphDestroy(graph);
    if(ie!=cudaSuccess){err="peer decode instantiate: "+std::string(cudaGetErrorString(ie));return false;}
    auto ue=cudaGraphUpload(graphs_[t],stream_);auto se=cudaStreamSynchronize(stream_);
    if(ue!=cudaSuccess||se!=cudaSuccess){err="peer decode upload: "+std::string(cudaGetErrorString(ue!=cudaSuccess?ue:se));return false;}
    return true;
}
bool PeerDecode::launch(int t,std::string& err){
    if(!prepare(t,err))return false;
    const OnDevice on(peer_);auto e=cudaGraphLaunch(graphs_[t],stream_);
    if(e!=cudaSuccess){err="peer decode launch: "+std::string(cudaGetErrorString(e));return false;}return true;
}
bool PeerDecode::wait(std::string& err){
    const OnDevice on(peer_);auto e=cudaStreamSynchronize(stream_);
    if(e!=cudaSuccess){err="peer decode completion: "+std::string(cudaGetErrorString(e));return false;}return true;
}
void PeerDecode::publish(int l,int t,const void* q8,const int32_t* ids,int32_t* plan,void* stream){
    using namespace strata::kernels;
    peer_publish_input((uint8_t*)m_q8_+(size_t)l*q8_bytes_,m_ids_+(size_t)l*cap_,m_flags_+l*32,q8,ids,
        t*(h_/32)*36,t*k_,m_flags_+layers_*32,stream);
    peer_partial_plan(ids,t*k_,k_,primary_table_+(size_t)l*experts_,experts_,plan,cap_,stream);
}
void PeerDecode::combine(int l,int t,const float* parts,const int32_t* ids,const float* weights,
                          const float* shared,float* out,void* stream){
    using namespace strata::kernels;
    peer_await_output(m_flags_+l*32,m_flags_+l*32+16,m_flags_+layers_*32,stream);
    peer_combine(parts,m_parts_+(size_t)l*cap_*h_,ids,primary_table_+(size_t)l*experts_,weights,shared,out,h_,k_,t,stream);
}
}
