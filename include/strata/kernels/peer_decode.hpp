#pragma once
#include <cstdint>
namespace strata::kernels {
// GPU-to-GPU expert work via portable mapped memory. No host dispatch per layer.
// Writer threads fence their own stores before a ready/done flag is published.
void peer_publish_input(void* host_q8,int32_t* host_ids,uint32_t* ready,const void* q8,const int32_t* ids,
                        int q8_bytes,int n_entries,const uint32_t* abort,void* stream);
void peer_await_input(void* q8,int32_t* ids,uint32_t* epoch,const void* host_q8,const int32_t* host_ids,
                      const uint32_t* ready,const uint32_t* done,int q8_bytes,int n_entries,
                      const uint32_t* abort,void* stream);
void peer_partial_plan(const int32_t* ids,int n_entries,int k,const unsigned long long* table,int n_expert,
                       int32_t* plan,int cap,void* stream);
void peer_scatter_output(float* host,const float* out,const int32_t* ids,const unsigned long long* table,
                         int n_embd,int n_entries,void* stream);
void peer_publish_output(uint32_t* done,const uint32_t* epoch,const uint32_t* abort,void* stream);
void peer_await_output(const uint32_t* ready,const uint32_t* done,const uint32_t* abort,void* stream);
void peer_combine(const float* own,const float* host_peer,const int32_t* ids,const unsigned long long* table,
                  const float* weights,const float* shared,float* out,int n_embd,int k,int n_tok,void* stream);
}
