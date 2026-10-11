#include "strata/kernels/bf16_gemv.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

void ck(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void cb(cublasStatus_t e) { if(e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(e)); }
template<class T> struct Dev {
    T* p; explicit Dev(size_t n) { ck(cudaMalloc(&p,n*sizeof(T))); }
    ~Dev(){ cudaFree(p); }
};
__global__ void split(const float* x, __nv_bfloat16* parts, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    float r=x[i];
    for(int j=0;j<3;++j) {
        auto v=__float2bfloat16_rn(r); parts[j*count+i]=v;
        r=__fsub_rn(r,__bfloat162float(v));
    }
}
__global__ void combine(const float* products, float* out,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) out[i]=__fadd_rn(products[i],__fadd_rn(products[count+i],products[2*count+i]));
}
struct Graph {
    cudaGraph_t graph; cudaGraphExec_t exec;
    Graph(cudaStream_t s,const std::function<void()>& fn) {
        fn(); ck(cudaStreamSynchronize(s));
        ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal)); fn();
        ck(cudaStreamEndCapture(s,&graph)); ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
    ~Graph(){cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);}
    double time(cudaStream_t s) {
        for(int i=0;i<20;++i) ck(cudaGraphLaunch(exec,s));
        cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
        ck(cudaEventRecord(a,s));
        for(int i=0;i<200;++i) ck(cudaGraphLaunch(exec,s));
        ck(cudaEventRecord(b,s));ck(cudaEventSynchronize(b));
        float ms;ck(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);
        return ms*1000/200;
    }
};
int main() try {
    cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));
    if(prop.major<8) return 77;
    cudaStream_t s;ck(cudaStreamCreate(&s));
    cublasHandle_t handle;cb(cublasCreate(&handle));cb(cublasSetStream(handle,s));
    cb(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
    std::mt19937 rng(60211);std::normal_distribution<float> norm;
    bool pass=true;
    for(auto shape:std::vector<std::pair<int,int>>{{2560,48},{2560,128},{2560,512},{2560,2560},{10240,320}}) {
        int K=shape.first,M=shape.second;
        std::vector<__nv_bfloat16> w(size_t(M)*K);
        for(auto& v:w) v=__float2bfloat16_rn(norm(rng)*0.02f);
        // Include an exact cancelling row pair in the matrix.
        for(int j=0;j<K;++j) w[K+j]=__float2bfloat16_rn(-__bfloat162float(w[j]));
        Dev<__nv_bfloat16> dw(w.size());ck(cudaMemcpy(dw.p,w.data(),w.size()*2,cudaMemcpyHostToDevice));
        for(int T:{1,2,4,6,8}) {
            std::vector<float>x(size_t(T)*K),a(size_t(T)*M),b(a.size());
            for(size_t i=0;i<x.size();++i) x[i]=norm(rng)*(i%199==0?32.f:1.f);
            if(T>1) std::fill(x.begin(),x.begin()+K,0.f);
            Dev<float> dx(x.size()), da(a.size()), db(b.size()), products(3*b.size());
            Dev<__nv_bfloat16> parts(3*x.size());
            ck(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));
            auto baseline=[&]{strata::kernels::bf16_gemv_fp32_mmvf_multi(dx.p,K,(uint16_t*)dw.p,da.p,M,K,M,T,s);};
            auto candidate=[&]{
                split<<<(x.size()+255)/256,256,0,s>>>(dx.p,parts.p,x.size());
                float alpha=1.f,beta=0.f;
                cb(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,M,T,K,&alpha,
                   dw.p,CUDA_R_16BF,K,0,parts.p,CUDA_R_16BF,K,x.size(),&beta,
                   products.p,CUDA_R_32F,M,b.size(),3,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                combine<<<(b.size()+255)/256,256,0,s>>>(products.p,db.p,b.size());
                ck(cudaGetLastError());
            };
            Graph ga(s,baseline),gb(s,candidate);
            std::vector<double>ta,tb;
            for(int repeat=0;repeat<3;++repeat){ta.push_back(ga.time(s));tb.push_back(gb.time(s));tb.push_back(gb.time(s));ta.push_back(ga.time(s));}
            std::sort(ta.begin(),ta.end());std::sort(tb.begin(),tb.end());
            double am=(ta[2]+ta[3])/2,bm=(tb[2]+tb[3])/2;
            ck(cudaMemcpy(a.data(),da.p,a.size()*4,cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(b.data(),db.p,b.size()*4,cudaMemcpyDeviceToHost));
            std::vector<__nv_bfloat16> hp(3*x.size());ck(cudaMemcpy(hp.data(),parts.p,hp.size()*2,cudaMemcpyDeviceToHost));
            double reconstruction=0;
            for(size_t i=0;i<x.size();++i){double v=0;for(int j=0;j<3;++j)v+=__bfloat162float(hp[j*x.size()+i]);reconstruction=std::max(reconstruction,std::abs(v-x[i]));}
            double ea=0,eb=0,energy=0,maxabs=0;
            for(int t=0;t<T;++t)for(int r=0;r<M;++r){
                double ref=0;for(int k=0;k<K;++k)ref+=double(x[t*K+k])*__bfloat162float(w[size_t(r)*K+k]);
                double av=a[t*M+r],bv=b[t*M+r];
                if(!std::isfinite(av)||!std::isfinite(bv))throw std::runtime_error("nonfinite result");
                ea+=(av-ref)*(av-ref);eb+=(bv-ref)*(bv-ref);energy+=ref*ref;maxabs=std::max(maxabs,std::abs(bv-ref));
            }
            double ra=sqrt(ea/std::max(energy,1e-30)),rb=sqrt(eb/std::max(energy,1e-30));
            bool ok=reconstruction==0&&rb<=2*ra+1e-7;pass&=ok;
            printf("{\"K\":%d,\"M\":%d,\"T\":%d,\"baseline_us\":%.6f,\"candidate_us\":%.6f,\"speedup\":%.6f,\"baseline_rms\":%.9g,\"candidate_rms\":%.9g,\"max_abs\":%.9g,\"reconstruction_max_abs\":%.9g,\"numeric_pass\":%s}\n",K,M,T,am,bm,am/bm,ra,rb,maxabs,reconstruction,ok?"true":"false");fflush(stdout);
        }
    }
    cb(cublasDestroy(handle));ck(cudaStreamDestroy(s));return pass?0:1;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 2;}
