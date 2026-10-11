#include "packed_gdn.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

void ck(cudaError_t e) { if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);} }
template<typename T> T* alloc(size_t n) {T* p=nullptr;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
struct Slot {
    float *sr,*st,*h,*gate,*beta,*z,*yr,*yt;
    int32_t* keep;
    size_t state_size,out_size;
    int T;
    Slot(int tokens,int hk,int hv,std::mt19937& rng):T(tokens){
        const int C=128*(2*hk+hv);
        state_size=(size_t)128*hv*128;out_size=(size_t)T*hv*128;
        sr=alloc<float>(state_size);st=alloc<float>(state_size);h=alloc<float>((size_t)T*C);
        gate=alloc<float>((size_t)T*hv);beta=alloc<float>((size_t)T*hv);z=alloc<float>(out_size);
        yr=alloc<float>(out_size);yt=alloc<float>(out_size);keep=alloc<int32_t>(1);
        std::uniform_real_distribution<float> uniform(-.05f,.05f);
        auto init=[&](float* p,size_t n,float shift){std::vector<float> v(n);for(auto&x:v)x=uniform(rng)+shift;ck(cudaMemcpy(p,v.data(),n*4,cudaMemcpyHostToDevice));};
        init(sr,state_size,0);ck(cudaMemcpy(st,sr,state_size*4,cudaMemcpyDeviceToDevice));
        init(h,(size_t)T*C,0);init(gate,(size_t)T*hv,-.06f);init(beta,(size_t)T*hv,.08f);init(z,out_size,0);
        ck(cudaMemset(yr,0,out_size*4));ck(cudaMemset(yt,0,out_size*4));
        int k=T;ck(cudaMemcpy(keep,&k,4,cudaMemcpyHostToDevice));
    }
    ~Slot(){for(auto p:{sr,st,h,gate,beta,z,yr,yt})cudaFree(p);cudaFree(keep);}
};
bool same(float* a,float* b,size_t n){
    std::vector<float>x(n),y(n);ck(cudaMemcpy(x.data(),a,n*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),b,n*4,cudaMemcpyDeviceToHost));
    if(std::memcmp(x.data(),y.data(),n*4)==0)return true;
    size_t fail=0;for(size_t i=0;i<n;++i)fail+=std::memcmp(&x[i],&y[i],4)!=0;
    std::printf("DIFFERENT %zu of %zu floats\n",fail,n);return false;
}

int main(){
    ck(cudaSetDevice(0));cudaStream_t stream;ck(cudaStreamCreate(&stream));
    std::mt19937 rng(20261011);int cases=0,failed=0;size_t bytes=0;
    for(auto [hk,hv]:std::vector<std::pair<int,int>>{{8,32},{16,48}}){
        float* gamma=alloc<float>(128);std::vector<float> g(128,1.0f);ck(cudaMemcpy(gamma,g.data(),512,cudaMemcpyHostToDevice));
        for(int n=1;n<=6;++n)for(int pattern=0;pattern<4;++pattern){
            std::vector<int> Ts(n,1);int spare=8-n;
            for(int j=0;j<spare;++j)if(Ts[(j+pattern)%n]<4)++Ts[(j+pattern)%n];
            std::vector<Slot*> slots;for(int T:Ts)slots.push_back(new Slot(T,hk,hv,rng));
            strata::turbo::GdnGroups groups{};
            for(int i=0;i<n;++i){auto&s=*slots[i];int k=(pattern+i)%(s.T+1);ck(cudaMemcpy(s.keep,&k,4,cudaMemcpyHostToDevice));groups.group[i]={s.st,s.h,s.gate,s.beta,s.z,s.yt,s.keep,s.T};}
            const int C=128*(2*hk+hv);
            // Commit does not consume y. Reference uses a distinct scratch so
            // its diagnostic writes cannot overwrite the verification output.
            float* scratch=alloc<float>(8*hv*128);
            auto ref=[&]{for(auto*s:slots){
                strata::kernels::gdn_step_norm_multi(s->sr,s->h,C,s->gate,s->beta,s->z,gamma,1e-6f,s->yr,hk,hv,s->T,nullptr,stream,0,nullptr);
                strata::kernels::gdn_step_norm_multi(s->sr,s->h,C,s->gate,s->beta,s->z,gamma,1e-6f,scratch,hk,hv,s->T,s->keep,stream,s->T>1?s->T:0,nullptr);
            }};
            auto candidate=[&]{strata::turbo::gdn_groups_run(groups,n,C,hk,hv,gamma,1e-6f,stream);strata::turbo::gdn_groups_commit(groups,n,C,hk,hv,stream);};
            cudaGraph_t gr,gt;cudaGraphExec_t er,et;
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));ref();ck(cudaStreamEndCapture(stream,&gr));ck(cudaGraphInstantiate(&er,gr,0));
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));candidate();ck(cudaStreamEndCapture(stream,&gt));ck(cudaGraphInstantiate(&et,gt,0));
            for(int repeat=0;repeat<3;++repeat){
                if(repeat==0){ref();candidate();}else{ck(cudaGraphLaunch(er,stream));ck(cudaGraphLaunch(et,stream));}
                ck(cudaStreamSynchronize(stream));bool ok=true;
                for(auto*s:slots){ok=same(s->yr,s->yt,s->out_size)&&ok;ok=same(s->sr,s->st,s->state_size)&&ok;bytes+=(s->out_size+s->state_size)*4;}
                failed+=!ok;++cases;
            }
            if(hv==32 && pattern==0 && (n==2||n==4||n==6)){
                auto time=[&](cudaGraphExec_t e){cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));ck(cudaEventRecord(a,stream));for(int j=0;j<200;++j)ck(cudaGraphLaunch(e,stream));ck(cudaEventRecord(b,stream));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000/200;};
                float r=time(er),t=time(et);std::printf("GROUP_BENCH sessions=%d reference_us=%.2f fused_us=%.2f ratio=%.3f\n",n,r,t,r/t);
            }
            ck(cudaGraphExecDestroy(er));ck(cudaGraphExecDestroy(et));ck(cudaGraphDestroy(gr));ck(cudaGraphDestroy(gt));cudaFree(scratch);for(auto*s:slots)delete s;
        }
        cudaFree(gamma);
    }
    cudaStreamDestroy(stream);std::printf("PARITY cases=%d bytes=%zu failed=%d\n",cases,bytes,failed);return failed?1:0;
}
