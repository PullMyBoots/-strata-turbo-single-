#include "strata/core/draft_residual_forecast.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::core {
namespace {
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Profile { int rank=0, streams=0, width=0; uint64_t norm_hash=0; };
Profile profile(const std::string& path) {
    Profile p; std::string tag,extra; std::ifstream f(path+"/forecast.txt");
    if(!(f>>tag>>p.rank>>p.streams>>p.width>>std::hex>>p.norm_hash) || tag!="SRF1" || (f>>extra) ||
       p.rank<32 || p.rank>256 || p.rank%32 || p.streams!=4 || p.width!=2560)
        throw std::runtime_error("draft residual forecast: invalid profile");
    return p;
}
std::vector<float> read(const std::string& path,size_t n) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()!=std::streamoff(n*4)) throw std::runtime_error("draft residual forecast: asset size: "+path);
    f.seekg(0);std::vector<float> v(n);
    if(!f.read(reinterpret_cast<char*>(v.data()),n*4) || !std::all_of(v.begin(),v.end(),[](float x){return std::isfinite(x);}))
        throw std::runtime_error("draft residual forecast: nonfinite or truncated asset: "+path);
    return v;
}
uint64_t fnv(const void* data,size_t n) {
    uint64_t h=14695981039346656037ull;const auto* p=static_cast<const uint8_t*>(data);
    for(size_t i=0;i<n;++i)h=(h^p[i])*1099511628211ull;return h;
}
__global__ void normalize_center(const float* residual,const float* mean,float* normalized,float* centered,int width) {
    __shared__ float part[8],rs;
    const int base=blockIdx.x*width,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    float s=0;
    for(int i=threadIdx.x;i<width;i+=256){float x=residual[base+i];s=fmaf(x,x,s);}
    for(int d=16;d;d>>=1)s+=__shfl_xor_sync(0xffffffffu,s,d);
    if(!lane)part[warp]=s;__syncthreads();
    if(!threadIdx.x){float sum=0;for(int w=0;w<8;++w)sum+=part[w];rs=rsqrtf(fmaxf(sum/width,1e-24f));}
    __syncthreads();
    for(int i=threadIdx.x;i<width;i+=256){float x=residual[base+i]*rs;normalized[base+i]=x;centered[base+i]=x-mean[base+i];}
}
__global__ void apply(float* residual,const float* normalized,const float* correction,const float* bias,int count,float gain) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)residual[i]=fmaf(gain,correction[i]+bias[i],normalized[i]);
}
}
struct DraftResidualForecast::State {
    Profile p; int device=0; float gain=1;
    uint16_t *projection=nullptr,*decoder=nullptr;
    float *mean=nullptr,*bias=nullptr,*normalized=nullptr,*centered=nullptr,*latent=nullptr,*correction=nullptr;
    std::vector<void*> allocations;
    ~State(){int old=0;cudaGetDevice(&old);cudaSetDevice(device);for(void* v:allocations)cudaFree(v);cudaSetDevice(old);}
    template<class T>T* alloc(size_t n){T* v=nullptr;ck(cudaMalloc(reinterpret_cast<void**>(&v),n*sizeof(T)));allocations.push_back(v);return v;}
    float* upload(const std::vector<float>& v){auto* p=alloc<float>(v.size());ck(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice));return p;}
    uint16_t* upload_bf16(const std::vector<float>& v){std::vector<uint16_t> b(v.size());for(size_t i=0;i<v.size();++i){uint32_t u;std::memcpy(&u,&v[i],4);b[i]=uint16_t((u+0x7fff+((u>>16)&1))>>16);}auto* p=alloc<uint16_t>(v.size());ck(cudaMemcpy(p,b.data(),b.size()*2,cudaMemcpyHostToDevice));return p;}
};
DraftResidualForecast::~DraftResidualForecast(){delete state_;}
uint64_t DraftResidualForecast::reserve_bytes(const std::string& path){try{auto p=profile(path);uint64_t h=p.streams*p.width;return h*p.rank*4+h*20+p.rank*4+65536;}catch(...){return 32ull<<20;}}
bool DraftResidualForecast::load(const std::string& path,int streams,int width,const float* input_norm,std::string& err){
    if(state_){err="draft residual forecast: already loaded";return false;}
    try{
        auto s=std::make_unique<State>();s->p=profile(path);auto p=s->p;ck(cudaGetDevice(&s->device));
        const size_t h=streams*width;
        if(p.streams!=streams || p.width!=width || !input_norm)throw std::runtime_error("draft residual forecast: geometry mismatch");
        std::vector<float> norm(h);ck(cudaMemcpy(norm.data(),input_norm,h*4,cudaMemcpyDeviceToHost));
        if(fnv(norm.data(),h*4)!=p.norm_hash)throw std::runtime_error("draft residual forecast: auxiliary norm does not match profile");
        if(const char* v=std::getenv("STRATA_MTP_RESIDUAL_GAIN")){char* end=nullptr;float gain=std::strtof(v,&end);if(end==v || *end || !std::isfinite(gain) || gain<0 || gain>1)throw std::runtime_error("draft residual forecast: invalid gain");s->gain=gain;}
        s->projection=s->upload_bf16(read(path+"/projection-f32.bin",h*p.rank));
        s->decoder=s->upload_bf16(read(path+"/decoder-f32.bin",h*p.rank));
        s->mean=s->upload(read(path+"/mean-f32.bin",h));s->bias=s->upload(read(path+"/bias-f32.bin",h));
        s->normalized=s->alloc<float>(h);s->centered=s->alloc<float>(h);s->latent=s->alloc<float>(p.rank);s->correction=s->alloc<float>(h);
        std::fprintf(stderr,"strata mtp: experimental draft residual forecast rank%d gain%.3f; subsequent proposals only, target verification unchanged\n",p.rank,s->gain);
        state_=s.release();return true;
    }catch(const std::exception& e){err=e.what();return false;}
}
bool DraftResidualForecast::run(float* residual,void* stream,std::string& err){
    if(!state_ || !residual){err="draft residual forecast: invalid launch";return false;}
    try{
        auto& s=*state_;auto p=s.p;auto cs=static_cast<cudaStream_t>(stream);const int h=p.streams*p.width;
        normalize_center<<<p.streams,256,0,cs>>>(residual,s.mean,s.normalized,s.centered,p.width);
        strata::kernels::bf16_gemv_fp32_mmvf(s.centered,s.projection,s.latent,h,p.rank,cs);
        strata::kernels::bf16_gemv_fp32_mmvf(s.latent,s.decoder,s.correction,p.rank,h,cs);
        apply<<<(h+255)/256,256,0,cs>>>(residual,s.normalized,s.correction,s.bias,h,s.gain);
        ck(cudaGetLastError());return true;
    }catch(const std::exception& e){err=e.what();return false;}
}
}
