#include "strata/kernels/peer_decode.hpp"
#include "strata/kernels/dp4a.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace strata::kernels {
namespace {
__global__ void publish_input_kernel(volatile int* hx,volatile int* hi,volatile uint32_t* ready,
                                     const int* x,const int* ids,int nx,int ni,const volatile uint32_t* abort) {
    if(*abort)return;
    for(int i=threadIdx.x;i<nx;i+=blockDim.x)hx[i]=x[i];
    for(int i=threadIdx.x;i<ni;i+=blockDim.x)hi[i]=ids[i];
    __threadfence_system();__syncthreads();
    if(threadIdx.x==0){*ready=*ready+1;__threadfence_system();}
}
__global__ void await_input_kernel(int* x,int* ids,uint32_t* epoch,const volatile int* hx,
                                   const volatile int* hi,const volatile uint32_t* ready,
                                   const volatile uint32_t* done,int nx,int ni,const volatile uint32_t* abort) {
    __shared__ uint32_t e;
    if(threadIdx.x==0){uint32_t v;do{v=*ready;strata_spin_pause();}while(v<=*done&&!*abort);e=v;}
    __syncthreads();if(*abort)return;__threadfence_system();
    for(int i=threadIdx.x;i<nx;i+=blockDim.x)x[i]=hx[i];
    for(int i=threadIdx.x;i<ni;i+=blockDim.x)ids[i]=hi[i];
    if(threadIdx.x==0)*epoch=e;
}
__global__ void partial_plan_kernel(const int32_t* ids,int n,int k,const unsigned long long* table,int ne,
                                    int32_t* pl,int cap) {
    __shared__ int expert[128],first[128],wc[4],owned[128];
    const int i=threadIdx.x,lane=i&31,warp=i>>5,e=i<n?ids[i]:-1;
    const unsigned long long addr=e>=0&&e<ne?table[e]:0;
    expert[i]=e;owned[i]=i<n&&addr!=0;__syncthreads();
    int at=i;if(owned[i])for(int j=0;j<i;++j)if(expert[j]==e){at=j;break;}
    first[i]=owned[i]?at:0x7fffffff;
    const unsigned mask=__ballot_sync(0xffffffffu,owned[i]&&at==i);
    if(lane==0)wc[warp]=__popc(mask);__syncthreads();
    int group=__popc(mask&((1u<<lane)-1));for(int w=0;w<warp;++w)group+=wc[w];
    int offset=0,prior=0;if(owned[i])for(int j=0;j<n;++j){offset+=first[j]<at;prior+=j<i&&expert[j]==e;}
    int32_t* st=pl+4;int32_t* dst=st+cap+1;int32_t* tok=dst+cap;
    const int po=(4+cap+1+2*cap+1)&~1;auto* ptr=(unsigned long long*)(pl+po);
    if(owned[i]){dst[offset+prior]=i;tok[offset+prior]=i/k;if(at==i){st[group]=offset;ptr[group]=addr;}}
    __syncthreads();if(i==0){int entries=0;for(int j=0;j<n;++j)entries+=owned[j];
        const int groups=wc[0]+wc[1]+wc[2]+wc[3];st[groups]=entries;pl[0]=groups;pl[1]=entries;pl[2]=0;}
}
__global__ void scatter_kernel(volatile float* host,const float* out,const int32_t* ids,
                               const unsigned long long* table,int h) {
    const int entry=blockIdx.y;if(!table[ids[entry]])return;
    for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<h;j+=gridDim.x*blockDim.x)host[entry*h+j]=out[entry*h+j];
    __threadfence_system();
}
__global__ void publish_output_kernel(volatile uint32_t* done,const uint32_t* epoch,const volatile uint32_t* abort){
    if(*abort)return;__threadfence_system();*done=*epoch;__threadfence_system();
}
__global__ void await_output_kernel(const volatile uint32_t* ready,const volatile uint32_t* done,
                                    const volatile uint32_t* abort){
    const uint32_t e=*ready;while(*done<e&&!*abort)strata_spin_pause();__threadfence_system();
}
__global__ void combine_kernel(const float* own,const volatile float* peer,const int32_t* ids,
                               const unsigned long long* table,const float* weights,const float* shared,
                               float* out,int h,int k){
    const int tk=blockIdx.y,col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=h)return;const int base=tk*k;
    float part=table[ids[base]]?own[base*h+col]:peer[base*h+col];float sum=__fadd_rn(0.f,part)*weights[base];
    for(int j=1;j<k;++j){part=table[ids[base+j]]?own[(base+j)*h+col]:peer[(base+j)*h+col];
        sum+=__fadd_rn(0.f,part)*weights[base+j];}
    out[tk*h+col]=sum+(shared?shared[tk*h+col]:0.f);
}
void check(){const auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
}
void peer_publish_input(void* hx,int32_t* hi,uint32_t* ready,const void* q8,const int32_t* ids,
                        int bytes,int n,const uint32_t* abort,void* stream){
    publish_input_kernel<<<1,256,0,(cudaStream_t)stream>>>((int*)hx,hi,ready,(const int*)q8,ids,bytes/4,n,abort);check();
}
void peer_await_input(void* q8,int32_t* ids,uint32_t* epoch,const void* hx,const int32_t* hi,
                      const uint32_t* ready,const uint32_t* done,int bytes,int n,const uint32_t* abort,void* stream){
    await_input_kernel<<<1,256,0,(cudaStream_t)stream>>>((int*)q8,ids,epoch,(const int*)hx,hi,ready,done,bytes/4,n,abort);check();
}
void peer_partial_plan(const int32_t* ids,int n,int k,const unsigned long long* table,int ne,int32_t* pl,int cap,void* stream){
    if(n>128||cap<n)throw std::invalid_argument("peer plan: entry capacity");
    partial_plan_kernel<<<1,128,0,(cudaStream_t)stream>>>(ids,n,k,table,ne,pl,cap);check();
}
void peer_scatter_output(float* host,const float* out,const int32_t* ids,const unsigned long long* table,int h,int n,void* stream){
    scatter_kernel<<<dim3(4,n),256,0,(cudaStream_t)stream>>>(host,out,ids,table,h);check();
}
void peer_publish_output(uint32_t* done,const uint32_t* epoch,const uint32_t* abort,void* stream){
    publish_output_kernel<<<1,1,0,(cudaStream_t)stream>>>(done,epoch,abort);check();
}
void peer_await_output(const uint32_t* ready,const uint32_t* done,const uint32_t* abort,void* stream){
    await_output_kernel<<<1,1,0,(cudaStream_t)stream>>>(ready,done,abort);check();
}
void peer_combine(const float* own,const float* host,const int32_t* ids,const unsigned long long* table,
                  const float* weights,const float* shared,float* out,int h,int k,int n,void* stream){
    combine_kernel<<<dim3((h+255)/256,n),256,0,(cudaStream_t)stream>>>(own,host,ids,table,weights,shared,out,h,k);check();
}
}
