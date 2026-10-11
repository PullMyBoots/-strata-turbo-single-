#include "resident_dispatch.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void require(bool yes, const char* message) {
    if (!yes) throw std::runtime_error(message);
}
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { if (p) cudaFree(p); }
};
template<class T> struct Mapped {
    T* h = nullptr;
    T* d = nullptr;
    explicit Mapped(size_t n) {
        check(cudaHostAlloc(&h, n * sizeof(T), cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(&d, h, 0));
    }
    ~Mapped() { if (h) cudaFreeHost(h); }
};
}

int main() {
    try {
        check(cudaSetDevice(0));
        constexpr int STEPS=16, E=64, CAP=80, STRIDE=96, H=4096, BLOB=256;
        const int ptr_off=(4+(CAP+1)+2*CAP+1)&~1;
        const int plan_words=ptr_off+4*CAP+(CAP+1)+1;
        Device<int32_t> ids(STEPS*CAP), residency(STEPS*E), plan(STEPS*plan_words);
        Device<uint32_t> skip(1);
        Device<float> x(STEPS*8*H), weights(STEPS*CAP);
        Device<uint8_t> cache(E*BLOB*2);
        Device<unsigned long long> offsets(E);
        Mapped<int32_t> traces(STEPS*STRIDE), host_ids(CAP);
        Mapped<float> host_x(8*H), host_w(CAP);
        Mapped<uint32_t> seq(1);
        std::vector<int32_t> hi(STEPS*CAP), hr(STEPS*E), hp(STEPS*plan_words);
        std::vector<float> hx(STEPS*8*H), hw(STEPS*CAP);
        std::vector<unsigned long long> ho(E);
        for (int s=0; s<E; ++s) ho[s]=(unsigned long long)(E-1-s)*BLOB*2;
        for (size_t i=0;i<hx.size();++i) hx[i]=float(i%8192)*0.125f;
        for (size_t i=0;i<hw.size();++i) hw[i]=float((i%127)+1)*0.00390625f;
        check(cudaMemcpy(x.p,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(weights.p,hw.data(),hw.size()*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(offsets.p,ho.data(),ho.size()*8,cudaMemcpyHostToDevice));
        cudaStream_t stream;
        check(cudaStreamCreate(&stream));
        int cases=0, resident_groups=0, fallback_groups=0;
        uint64_t checked_bytes=0;
        for (bool parallel : {false,true}) {
        strata::kernels::resident_plan_set_parallel(parallel);
        for (int T=1;T<=8;++T) for (int pattern=0;pattern<3;++pattern)
        for (bool variable_offsets : {false,true}) for (bool graph_mode : {false,true}) {
            const int n=T*10;
            for (int s=0;s<STEPS;++s) for (int i=0;i<CAP;++i) {
                const int e=pattern==0 ? (i+s)%E : pattern==1 ? (i%7+s)%E : ((i*17)%13+s)%E;
                hi[s*CAP+i]=e;
            }
            check(cudaMemcpy(ids.p,hi.data(),hi.size()*4,cudaMemcpyHostToDevice));
            auto launch=[&] {
                for (int s=0;s<STEPS;++s) {
                    strata::kernels::resident_plan(ids.p+s*CAP,n,10,residency.p+s*E,E,
                        cache.p,variable_offsets ? offsets.p : nullptr,BLOB,plan.p+s*plan_words,
                        CAP,skip.p,s+1,stream);
                    strata::kernels::resident_dispatch_publish(x.p+s*8*H,ids.p+s*CAP,
                        weights.p+s*CAP,(int64_t)T*H,n,host_x.d,host_ids.d,host_w.d,
                        seq.d,skip.p,s+1,traces.d+s*STRIDE,stream);
                }
            };
            cudaGraph_t graph=nullptr;
            cudaGraphExec_t exec=nullptr;
            if (graph_mode) {
                check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                launch();
                check(cudaStreamEndCapture(stream,&graph));
                check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            }
            // Replay the same captured graph with changed actual residency.
            for (int replay=0;replay<5;++replay) {
                std::vector<bool> resident(STEPS,true);
                int last_miss=-1;
                for (int s=0;s<STEPS;++s) {
                    for (int e=0;e<E;++e) hr[s*E+e]=(e*7+s)%E;
                    if (replay==4 || (replay<3 && (s+replay)%3==1)) {
                        hr[s*E+hi[s*CAP+n/2]]=-1;
                        resident[s]=false;last_miss=s;
                    }
                }
                std::fill(hp.begin(),hp.end(),-1515870811);
                std::fill(traces.h,traces.h+STEPS*STRIDE,-1);
                for(int s=0;s<STEPS;++s) traces.h[s*STRIDE]=0;
                std::fill(host_x.h,host_x.h+8*H,-123.0f);
                std::fill(host_ids.h,host_ids.h+CAP,-123);
                std::fill(host_w.h,host_w.h+CAP,-123.0f);
                *seq.h=0;
                check(cudaMemcpy(plan.p,hp.data(),hp.size()*4,cudaMemcpyHostToDevice));
                check(cudaMemcpy(residency.p,hr.data(),hr.size()*4,cudaMemcpyHostToDevice));
                if (graph_mode) check(cudaGraphLaunch(exec,stream)); else launch();
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(hp.data(),plan.p,hp.size()*4,cudaMemcpyDeviceToHost));
                require(*seq.h==STEPS,"global diagnostic ring mismatch");
                for (int s=0;s<STEPS;++s) {
                    const int32_t* trace=traces.h+s*STRIDE;
                    require(trace[0]==s+1,"unstable per-step completion ring");
                    require(trace[1]==int(resident[s]),"residency decision mismatch");
                    if (!resident[s]) {
                        ++fallback_groups;
                        for(int i=0;i<plan_words;++i)
                            require(hp[s*plan_words+i]==-1515870811,"partial plan overwritten");
                        continue;
                    }
                    ++resident_groups;
                    require(!std::memcmp(trace+2,hi.data()+s*CAP,n*4),"resident route trace overwritten");
                    checked_bytes+=n*4;
                    const int32_t* p=hp.data()+s*plan_words;
                    const int32_t *start=p+4,*dst=start+CAP+1,*tok=dst+CAP;
                    const auto* ptr=(const unsigned long long*)(p+ptr_off);
                    std::vector<int> order;
                    for(int i=0;i<n;++i)
                        if(std::find(order.begin(),order.end(),hi[s*CAP+i])==order.end())
                            order.push_back(hi[s*CAP+i]);
                    require(p[0]==int(order.size())&&p[1]==n&&p[2]==0,"resident plan count mismatch");
                    int at=0;
                    for(size_t group=0;group<order.size();++group) {
                        require(start[group]==at,"resident plan order mismatch");
                        const int slot=hr[s*E+order[group]];
                        const auto expected=(unsigned long long)(cache.p+(variable_offsets?ho[slot]:slot*BLOB));
                        require(ptr[group]==expected,"resident weight pointer mismatch");
                        for(int i=0;i<n;++i) if(hi[s*CAP+i]==order[group]) {
                            require(dst[at]==i&&tok[at]==i/10,"resident entry order mismatch");++at;
                        }
                    }
                    require(start[order.size()]==n&&p[ptr_off+4*CAP]==n,"resident plan end mismatch");
                }
                if (last_miss>=0) {
                    require(!std::memcmp(host_x.h,hx.data()+last_miss*8*H,T*H*4),"fallback activation bytes differ");
                    require(!std::memcmp(host_ids.h,hi.data()+last_miss*CAP,n*4),"fallback route bytes differ");
                    require(!std::memcmp(host_w.h,hw.data()+last_miss*CAP,n*4),"fallback weight bytes differ");
                    checked_bytes+=(uint64_t)(T*H+2*n)*4;
                } else {
                    for(int i=0;i<8*H;++i) require(host_x.h[i]==-123.0f,"all-hit activation staging touched");
                    for(int i=0;i<CAP;++i) require(host_ids.h[i]==-123&&host_w.h[i]==-123.0f,"all-hit route staging touched");
                }
                ++cases;
            }
            if(exec) check(cudaGraphExecDestroy(exec));
            if(graph) check(cudaGraphDestroy(graph));
        }
        }
        check(cudaStreamDestroy(stream));
        std::cout<<"{\"passed\":true,\"cases\":"<<cases
                 <<",\"resident_groups\":"<<resident_groups
                 <<",\"fallback_groups\":"<<fallback_groups
                 <<",\"checked_payload_bytes\":"<<checked_bytes<<"}\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"resident parity failed: "<<e.what()<<"\n";return 1;
    }
}
