#include "strata/core/conversation_snapshot.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace strata::core {
bool conversation_state_transfer(const SessionState&, SessionState&, const ModelGeometry&,
                                 const std::vector<int32_t>&, std::string&);
}
using namespace strata::core;
void check(bool v, const char* s) { if (!v) { std::fprintf(stderr,"FAIL %s\n",s); std::exit(1); } }
void cu(cudaError_t e) { check(e == cudaSuccess, cudaGetErrorString(e)); }
struct Fixture {
    SessionState s;
    std::vector<QsaState> qs;
    std::vector<void*> allocations;
    ConversationStateSizes z;
    template<class T> void alloc(T*& p,size_t n) {
        void* v=nullptr; if (!n) return; cu(cudaMalloc(&v,n)); p=static_cast<T*>(v);
        allocations.push_back(v); cu(cudaMemset(v,0xa5,n));
    }
    Fixture(const ModelGeometry& g,bool ple,bool partial) : qs(g.n_qsa_layers()) {
        s.max_cells=262144; s.layer_lo=partial?4:0; s.layer_hi=g.n_layers;
        s.gdn_ord0=partial?3:0; s.gdn_alloc=g.n_gdn_layers()-s.gdn_ord0;
        s.qsa_ord0=partial?1:0; s.qsa_alloc=g.n_qsa_layers()-s.qsa_ord0;
        s.qsa_states=qs.data(); std::string e;
        check(conversation_session_sizes(g,s,z,e),e.c_str());
        alloc(s.gdn_state,z.gdn); if(ple) alloc(s.ple_hist,z.ple);
        for(int j=s.qsa_ord0;j<s.qsa_ord0+s.qsa_alloc;++j) {
            auto& q=qs[j]; q.max_cells=s.max_cells; q.idx_pooled_rows=s.max_cells/4+1;
            alloc(q.idx_tail,z.tail); alloc(q.idx_dead,z.dead); alloc(q.idx_block_pos,z.block_pos);
            alloc(q.idx_pooled,q.idx_pooled_rows*g.idx_key_dim*sizeof(float));
        }
    }
    ~Fixture(){ for(auto p:allocations) cudaFree(p); }
};
size_t bytes=0;
void equal(const ConversationCheckpoint& a,const ConversationCheckpoint& b) {
    check(a.gdn==b.gdn && a.ple==b.ple && a.tails==b.tails && a.dead==b.dead && a.block_pos==b.block_pos,"checkpoint equality");
    bytes+=a.gdn.size()+a.ple.size()+a.tails.size()+a.dead.size()+a.block_pos.size();
}
int main() {
    setenv("STRATA_DIRECT_SLOT_STATE","1",1);
    int cases=0;
    for (bool full : {false,true}) for(bool ple:{false,true}) for(bool partial:{false,true}) {
        ModelGeometry g;
        if(!full) { g.n_layers=8;g.ssm_state_size=2;g.ssm_v_heads=2;g.ssm_conv_channels=8; }
        Fixture src(g,ple,partial), ref(g,ple,partial), dst(g,ple,partial);
        for(auto p:src.allocations) { (void)p; }
        cu(cudaMemset(src.s.gdn_state,0x37,src.z.gdn));
        if(ple) cu(cudaMemset(src.s.ple_hist,0x73,src.z.ple));
        for(auto& q:src.qs) if(q.idx_tail) {
            cu(cudaMemset(q.idx_tail,0x21,src.z.tail)); cu(cudaMemset(q.idx_dead,0x42,src.z.dead));
            cu(cudaMemset(q.idx_block_pos,0x13,src.z.block_pos));
        }
        cu(cudaDeviceSynchronize());
        for(int n:{0,1,2,3,4,5,31,32,33,32767,32768,262143,262144}) {
            std::vector<int32_t> ids(n,19); if(n>1) ids[n-2]=23;
            ConversationCheckpoint snap; snap.ids=ids; std::string e;
            check(conversation_checkpoint_save(snap,src.s,g,e),e.c_str());
            check(conversation_checkpoint_restore(snap,ref.s,g,e),e.c_str());
            check(conversation_state_transfer(src.s,dst.s,g,ids,e),e.c_str());
            ConversationCheckpoint a,b; a.ids=b.ids=ids;
            check(conversation_checkpoint_save(a,ref.s,g,e),e.c_str());
            check(conversation_checkpoint_save(b,dst.s,g,e),e.c_str()); equal(a,b);
            check(ref.s.ple_prev[0]==dst.s.ple_prev[0]&&ref.s.ple_prev[1]==dst.s.ple_prev[1],"host history");
            for(int j=src.s.qsa_ord0;j<src.s.qsa_ord0+src.s.qsa_alloc;++j) {
                std::vector<unsigned char> x(src.z.dead),y(src.z.dead);
                cu(cudaMemcpy(x.data(),ref.qs[j].idx_pooled+(n/4)*g.idx_key_dim,x.size(),cudaMemcpyDefault));
                cu(cudaMemcpy(y.data(),dst.qs[j].idx_pooled+(n/4)*g.idx_key_dim,y.size(),cudaMemcpyDefault));
                check(x==y,"spare pooled row"); bytes+=x.size();
            }
            ++cases;
        }
        std::string e; std::vector<int32_t> ids(55,7);
        // Validation must reject before any destination mutation.
        ConversationCheckpoint before,after; before.ids=after.ids=ids;
        check(conversation_checkpoint_save(before,dst.s,g,e),e.c_str());
        ++dst.s.layer_lo;
        check(!conversation_state_transfer(src.s,dst.s,g,ids,e),"reject layer mismatch");
        --dst.s.layer_lo;
        check(conversation_checkpoint_save(after,dst.s,g,e),e.c_str()); equal(before,after);
        if(full && ple && !partial) {
            for(int mode:{0,1,0,1}) {
                auto t=std::chrono::steady_clock::now();
                for(int r=0;r<20;++r) {
                    if(mode) check(conversation_state_transfer(src.s,dst.s,g,ids,e),e.c_str());
                    else { ConversationCheckpoint c;c.ids=ids;check(conversation_checkpoint_save(c,src.s,g,e),e.c_str());check(conversation_checkpoint_restore(c,dst.s,g,e),e.c_str()); }
                }
                double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/20;
                std::printf("mode=%s mean_ms=%.4f\n",mode?"direct":"host",ms);
            }
        }
    }
    std::printf("PASS cases=%d bytes=%zu mismatches=0\n",cases,bytes);
}
