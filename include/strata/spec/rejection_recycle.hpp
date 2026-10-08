// Reuse unused verifier predictions as proposals, never as accepted outputs.
// Learning and timing belong only to the current request.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
namespace strata::spec {
class RejectionRecycle {
public:
    explicit RejectionRecycle(int mode) : mode_(std::clamp(mode,0,3)) {}
    void observe(int t,int accepted,double verification_ms,double preparation_ms,bool recycled) {
        if(t<1||t>8||accepted<0||accepted>=t||!std::isfinite(verification_ms)||verification_ms<=0)return;
        const double full=verification_ms+std::max(0.0,preparation_ms);
        if(recycled){
            for(int j=0;j<std::min(t-1,accepted+1);++j){ok_[j]*=.985;bad_[j]*=.985;(j<accepted?ok_[j]:bad_[j])+=1;}
            ++recycle_n_;
        }else if(normal_n_++>0){
            if(normal_ms_==0){normal_tokens_=accepted+1;normal_ms_=full;}
            else {normal_tokens_=.95*normal_tokens_+.05*(accepted+1);normal_ms_=.95*normal_ms_+.05*full;}
        }
    }
    template<class Cost> int choose(int available,Cost verify_cost) {
        if(!mode_||available<2)return 0;
        available=std::clamp(available,0,7);
        if(mode_==1)return available;
        if(mode_==2)return available>=3?available:0;
        if(normal_n_<4||normal_ms_<=0)return 0;
        const double baseline=normal_tokens_/normal_ms_;
        double expected=1,survive=1,best=baseline*1.02;int k_best=0;
        for(int j=0;j<available;++j){
            const double q=(ok_[j]+2*.8)/(ok_[j]+bad_[j]+2);
            survive*=q;expected+=survive;
            // Auxiliary KV catch-up is measured separately and is tiny; use
            // a conservative initial reserve until engine measurements exist.
            if(j>=1){const double rate=expected/(verify_cost(j+2)+.15);if(rate>best){best=rate;k_best=j+1;}}
        }
        // A small bounded number of probes prevents a wrong prior from
        // concealing useful recycling. All probes still use exact verification.
        if(!k_best&&recycle_n_<3&&++opportunities_%16==0)return std::min(available,3);
        return k_best;
    }
private:
    int mode_=0,normal_n_=0,recycle_n_=0,opportunities_=0;
    double normal_tokens_=0,normal_ms_=0;
    std::array<double,7>ok_{},bad_{};
};
}
