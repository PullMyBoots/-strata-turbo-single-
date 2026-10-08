#include "strata/core/draft_head_router.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/device/device_radix_sort.cuh>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace strata::core {
namespace {
namespace k=strata::kernels;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<uint8_t> read(const std::string& path,size_t bytes){std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=(std::streamoff)bytes)throw std::runtime_error("draft router: asset size mismatch: "+path);f.seekg(0);std::vector<uint8_t> x(bytes);if(!f.read((char*)x.data(),bytes))throw std::runtime_error("draft router: asset read: "+path);return x;}
uint64_t fnv(const void* data,size_t bytes){uint64_t h=14695981039346656037ull;const auto* p=(const uint8_t*)data;for(size_t i=0;i<bytes;++i)h=(h^p[i])*1099511628211ull;return h;}
struct Profile{int r=0,k=0,n=0,h=0,type=0;uint64_t row=0,head_hash=0,vocab_hash=0;};
Profile profile(const std::string& path){Profile p;std::string magic;std::ifstream f(path+"/router.txt");if(!(f>>magic>>p.r>>p.k>>p.n>>p.h>>p.type>>p.row>>std::hex>>p.head_hash>>p.vocab_hash)||magic!="SRH1"||p.r<32||p.r>384||p.r%32||p.k<1||p.k>16384||p.n<p.k||p.n>248320||p.h!=2560||p.row==0)throw std::runtime_error("draft router: invalid profile");return p;}
__global__ void decode_q8(const unsigned char* q,float* x,int H){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=H)return;const unsigned char*b=q+(i/32)*36;const __half d=*(const __half*)b;x[i]=__half2float(d)*(float)((const signed char*)b)[4+i%32];}
__global__ void add_bias(float* scores,const float* bias,int N){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)scores[i]+=bias[i];}
__global__ void normalize_query(float*x,int H){
 __shared__ float part[8],rs;const int lane=threadIdx.x&31,warp=threadIdx.x>>5;float ss=0;
 for(int i=threadIdx.x;i<H;i+=256)ss+=x[i]*x[i];for(int d=16;d>0;d>>=1)ss+=__shfl_xor_sync(0xffffffffu,ss,d);
 if(!lane)part[warp]=ss;__syncthreads();if(!threadIdx.x){float sum=0;for(int w=0;w<8;++w)sum+=part[w];rs=rsqrtf(fmaxf(sum/H,1e-20f));}__syncthreads();for(int i=threadIdx.x;i<H;i+=256)x[i]*=rs;
}

__global__ void decode_normalize_query(const unsigned char* q,float* x,int H){
 for(int i=threadIdx.x;i<H;i+=256){const auto* b=q+(i/32)*36;
  x[i]=__half2float(*(const __half*)b)*float(((const signed char*)b)[4+i%32]);}
 __syncthreads();
 __shared__ float part[8],rs;const int lane=threadIdx.x&31,warp=threadIdx.x>>5;float ss=0;
 for(int i=threadIdx.x;i<H;i+=256)ss+=x[i]*x[i];
 for(int d=16;d>0;d>>=1)ss+=__shfl_xor_sync(0xffffffffu,ss,d);
 if(!lane)part[warp]=ss;__syncthreads();
 if(!threadIdx.x){float sum=0;for(int w=0;w<8;++w)sum+=part[w];rs=rsqrtf(fmaxf(sum/H,1e-20f));}
 __syncthreads();for(int i=threadIdx.x;i<H;i+=256)x[i]*=rs;
}

__global__ void word_ids(const int* indices,const int* vocab,int* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=vocab?vocab[indices[i]]:indices[i];}
} // namespace
struct DraftHeadRouter::State {
    Profile p;const void* head=nullptr;const int32_t* base_vocab=nullptr;
    int device=0,sort_begin=0;bool indexed=false,fused_query=false;std::vector<void*> allocations;
    uint16_t* projection=nullptr;uint8_t *coarse=nullptr,*cq=nullptr,*selected=nullptr;
    float *x=nullptr,*latent=nullptr,*scores=nullptr,*sorted=nullptr,*bias=nullptr;
    int32_t *identity=nullptr,*indices=nullptr,*words=nullptr;void* temp=nullptr;size_t temp_bytes=0;
    ~State(){int old=0;cudaGetDevice(&old);cudaSetDevice(device);for(void* a:allocations)cudaFree(a);cudaSetDevice(old);}
    template<class T>T* alloc(size_t n){T* a=nullptr;ck(cudaMalloc((void**)&a,n*sizeof(T)));allocations.push_back(a);return a;}
    template<class T>T* upload(const void* data,size_t n){T* a=alloc<T>(n);ck(cudaMemcpy(a,data,n*sizeof(T),cudaMemcpyHostToDevice));return a;}
};
DraftHeadRouter::~DraftHeadRouter(){delete state_;}
uint64_t DraftHeadRouter::reserve_bytes(const std::string& path,uint64_t row_bytes){
    try{auto p=profile(path);return (uint64_t)p.n*(p.r/32*34+32)+(uint64_t)p.r*p.h*2+(uint64_t)p.k*(row_bytes+4)+16*1024*1024;}
    catch(...){return 64*1024*1024;}
}

bool DraftHeadRouter::load(const std::string& path,const void* head,const int32_t* vocab,int type,int rows,int cols,uint64_t row_bytes,std::string& err){
    if(state_){err="draft router: already loaded";return false;}
    try{
        auto s=std::make_unique<State>();s->p=profile(path);auto& p=s->p;ck(cudaGetDevice(&s->device));
        if(const char* v=std::getenv("STRATA_MTP_ROUTER_SORT_BEGIN_BIT"))s->sort_begin=std::clamp(std::atoi(v),0,16);
        if(const char* v=std::getenv("STRATA_MTP_ROUTER_INDEXED"))s->indexed=std::atoi(v)==1;
        if(const char* v=std::getenv("STRATA_MTP_ROUTER_FUSED_QUERY"))s->fused_query=std::atoi(v)==1;
        if(!head||rows!=p.n||cols!=p.h||type!=p.type||row_bytes!=p.row)throw std::runtime_error("draft router: model/head geometry does not match profile");
        std::vector<uint8_t> weights((size_t)p.n*row_bytes);ck(cudaMemcpy(weights.data(),head,weights.size(),cudaMemcpyDeviceToHost));
        std::vector<int32_t> ids(p.n);if(vocab)ck(cudaMemcpy(ids.data(),vocab,ids.size()*4,cudaMemcpyDeviceToHost));else std::iota(ids.begin(),ids.end(),0);
        if(fnv(weights.data(),weights.size())!=p.head_hash||fnv(ids.data(),ids.size()*4)!=p.vocab_hash)throw std::runtime_error("draft router: head weights/vocabulary do not match calibration profile");
        const auto raw=read(path+"/projection-f32.bin",(size_t)p.r*p.h*4);std::vector<uint16_t> bf((size_t)p.r*p.h);
        for(size_t i=0;i<bf.size();++i){uint32_t bits;std::memcpy(&bits,raw.data()+i*4,4);bf[i]=(uint16_t)((bits+0x7fff+((bits>>16)&1))>>16);}
        s->projection=s->upload<uint16_t>(bf.data(),bf.size());
        auto cw=read(path+"/coarse-q8.bin",(size_t)p.n*(p.r/32*34));s->coarse=s->upload<uint8_t>(cw.data(),cw.size());
        auto bw=read(path+"/bias-f32.bin",(size_t)p.n*4);s->bias=s->upload<float>(bw.data(),p.n);
        s->head=head;s->base_vocab=vocab;s->cq=s->alloc<uint8_t>(p.r/32*36);
        if(!s->indexed)s->selected=s->alloc<uint8_t>((size_t)p.k*row_bytes);
        s->x=s->alloc<float>(p.h);s->latent=s->alloc<float>(p.r);s->scores=s->alloc<float>(p.n);s->sorted=s->alloc<float>(p.n);
        std::iota(ids.begin(),ids.end(),0);s->identity=s->upload<int32_t>(ids.data(),p.n);s->indices=s->alloc<int32_t>(p.n);s->words=s->alloc<int32_t>(p.k);
        ck(cub::DeviceRadixSort::SortPairsDescending(nullptr,s->temp_bytes,s->scores,s->sorted,s->identity,s->indices,p.n,s->sort_begin,32));s->temp=s->alloc<uint8_t>(s->temp_bytes);
        std::fprintf(stderr,"strata mtp: approximate draft-only router rank%d, %d candidates of %d; head/vocabulary profile matches; target verifier unchanged\n",p.r,p.k,p.n);
        state_=s.release();return true;
    }catch(const std::exception& e){err=e.what();return false;}
}
bool DraftHeadRouter::run(const void* xq,float* logits,void* stream,std::string& err){
    if(!state_||!xq||!logits||!stream){err="draft router: invalid launch";return false;}
    try{auto& s=*state_;const auto& p=s.p;auto cs=(cudaStream_t)stream;
        if(s.fused_query)decode_normalize_query<<<1,256,0,cs>>>((const uint8_t*)xq,s.x,p.h);
        else {decode_q8<<<(p.h+255)/256,256,0,cs>>>((const uint8_t*)xq,s.x,p.h);normalize_query<<<1,256,0,cs>>>(s.x,p.h);}
        k::bf16_gemv_fp32_mmvf(s.x,s.projection,s.latent,p.h,p.r,cs);k::native_quantize_q8_1(s.latent,s.cq,p.r,1,cs);
        k::native_mmvq(8,s.coarse,s.cq,s.scores,p.r,p.n,1,cs);add_bias<<<(p.n+255)/256,256,0,cs>>>(s.scores,s.bias,p.n);
        ck(cub::DeviceRadixSort::SortPairsDescending(s.temp,s.temp_bytes,s.scores,s.sorted,s.identity,s.indices,p.n,s.sort_begin,32,cs));
        if(s.indexed)k::native_mmvq_indexed(p.type,s.head,xq,s.indices,logits,p.h,p.k,cs);
        else {k::gather_rows((const uint8_t*)s.head,p.row,s.indices,p.k,s.selected,cs);k::native_mmvq(p.type,s.selected,xq,logits,p.h,p.k,1,cs);}
        word_ids<<<(p.k+255)/256,256,0,cs>>>(s.indices,s.base_vocab,s.words,p.k);ck(cudaGetLastError());return true;
    }catch(const std::exception& e){err=e.what();return false;}
}
int DraftHeadRouter::size()const{return state_?state_->p.k:0;}
const int32_t* DraftHeadRouter::vocab()const{return state_?state_->words:nullptr;}
} // namespace strata::core
