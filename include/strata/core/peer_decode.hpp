#pragma once
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {
class PeerExperts;struct VerifyHits;struct ModelGeometry;
// One captured GPU-side worker per verify width. The resident expert partition
// is immutable for this object's lifetime; both devices together must hold ALL experts.
class PeerDecode {
public:
    PeerDecode()=default;~PeerDecode();
    PeerDecode(const PeerDecode&)=delete;PeerDecode& operator=(const PeerDecode&)=delete;
    bool init(PeerExperts& peer,const VerifyHits& primary,const ModelGeometry& g,int max_t,int k,std::string& err);
    bool prepare(int t,std::string& err);
    bool launch(int t,std::string& err);
    bool wait(std::string& err);
    bool release(int timeout_ms);
    void abort();
    void publish(int layer,int t,const void* q8,const int32_t* ids,int32_t* primary_plan,void* stream);
    void combine(int layer,int t,const float* primary_parts,const int32_t* ids,const float* weights,
                 const float* shared,float* out,void* stream);
private:
    int primary_=-1,peer_=-1,layers_=0,experts_=0,h_=0,ff_=0,mt_=0,k_=0,cap_=0,q8_bytes_=0;
    cudaStream_t stream_=nullptr;cudaGraphExec_t graphs_[9]={};
    unsigned long long *primary_table_=nullptr,*peer_table_=nullptr;
    void *h_q8_=nullptr,*m_q8_=nullptr;int32_t *h_ids_=nullptr,*m_ids_=nullptr;
    float *h_parts_=nullptr,*m_parts_=nullptr;
    uint32_t *h_flags_=nullptr,*m_flags_=nullptr,*epoch_=nullptr;
    uint8_t *q8_=nullptr,*scratch_=nullptr;float* parts_=nullptr;
    int32_t *ids_=nullptr,*plan_=nullptr;int plan_i32_=0;
    std::vector<strata::kernels::NativeExpertLayout> layouts_;
};
}
