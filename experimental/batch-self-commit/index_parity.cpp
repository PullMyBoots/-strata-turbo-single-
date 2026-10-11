// Compare singleton forward state with the old tail-restore/append commit.
#include "strata/kernels/native_qsa_indexer.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace strata::kernels;
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T>T* alloc(size_t n){T*p=nullptr;ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));return p;}
bool same(const void*a,const void*b,size_t bytes){std::vector<char>x(bytes),y(bytes);ck(cudaMemcpy(x.data(),a,bytes,cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),b,bytes,cudaMemcpyDeviceToHost));return std::memcmp(x.data(),y.data(),bytes)==0;}
struct State {
    QsaIndexerBuffers b;
    explicit State(int cells):b{alloc<float>(3*128),alloc<float>(128),alloc<float>((cells/4+1)*128),alloc<int32_t>(1)}{}
    ~State(){cudaFree(b.tail);cudaFree(b.dead);cudaFree(b.pooled);cudaFree(b.block_pos);}
};
int main(){
    cudaStream_t stream;ck(cudaStreamCreate(&stream));
    constexpr int cells=132,dim=128;
    auto shape=qsa_real_shapes();
    float*raw=alloc<float>(dim),*gamma=alloc<float>(dim),*snapshot=alloc<float>(3*dim);
    int32_t*pos=alloc<int32_t>(1);
    std::mt19937 rng(20261012);std::normal_distribution<float> normal;
    std::vector<float> values(dim),g(dim);
    for(auto&v:g)v=1.f+normal(rng)*0.1f;
    ck(cudaMemcpy(gamma,g.data(),dim*4,cudaMemcpyHostToDevice));
    size_t bytes=0;int cases=0,failed=0;
    for(int base:{0,4096})for(bool graph:{false,true}){
        State ref(cells),candidate(cells);
        auto old=[&]{
            ck(cudaMemcpyAsync(snapshot,ref.b.tail,3*dim*4,cudaMemcpyDeviceToDevice,stream));
            native_qsa_indexer_append(raw,pos,base,gamma,1e-6f,ref.b,shape,cells,rope_scaling(),stream);
            ck(cudaMemcpyAsync(ref.b.tail,snapshot,3*dim*4,cudaMemcpyDeviceToDevice,stream));
            native_qsa_indexer_append(raw,pos,base,gamma,1e-6f,ref.b,shape,cells,rope_scaling(),stream);
        };
        auto fast=[&]{native_qsa_indexer_append(raw,pos,base,gamma,1e-6f,candidate.b,shape,cells,rope_scaling(),stream);};
        cudaGraph_t gr=nullptr,gt=nullptr;cudaGraphExec_t er=nullptr,et=nullptr;
        if(graph){
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));old();ck(cudaStreamEndCapture(stream,&gr));ck(cudaGraphInstantiate(&er,gr,0));
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));fast();ck(cudaStreamEndCapture(stream,&gt));ck(cudaGraphInstantiate(&et,gt,0));
        }
        for(int p=0;p<cells;++p){
            for(auto&v:values)v=(p%17==0?0.f:normal(rng));
            ck(cudaMemcpyAsync(raw,values.data(),dim*4,cudaMemcpyHostToDevice,stream));
            ck(cudaMemcpyAsync(pos,&p,4,cudaMemcpyHostToDevice,stream));
            if(graph){ck(cudaGraphLaunch(er,stream));ck(cudaGraphLaunch(et,stream));}else{old();fast();}
            ck(cudaStreamSynchronize(stream));
            bool ok=same(ref.b.tail,candidate.b.tail,3*dim*4)&&same(ref.b.dead,candidate.b.dead,dim*4)
                &&same(ref.b.pooled,candidate.b.pooled,(cells/4+1)*dim*4)&&same(ref.b.block_pos,candidate.b.block_pos,4);
            ++cases;failed+=!ok;bytes+=(4+cells/4+1)*dim*4+4;
            if(!ok)printf("MISMATCH base=%d graph=%d position=%d\n",base,int(graph),p);
        }
        if(graph){cudaGraphExecDestroy(er);cudaGraphExecDestroy(et);cudaGraphDestroy(gr);cudaGraphDestroy(gt);}
    }
    cudaFree(raw);cudaFree(gamma);cudaFree(snapshot);cudaFree(pos);cudaStreamDestroy(stream);
    printf("INDEX_PARITY cases=%d bytes=%zu failed=%d\n",cases,bytes,failed);return failed?1:0;
}
