// Cost-aware MTP stopping. This only schedules proposals; target verification
// and acceptance remain unchanged. Costs and calibration are request-local.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>

namespace strata::spec {

class AdaptiveLookahead {
public:
    static constexpr int kMaxT = 8, kBins = 10;
    explicit AdaptiveLookahead(int max_t,int policy=-1) : max_t_(std::clamp(max_t, 2, kMaxT)),
        rate_on_(policy<0?env_on("STRATA_MTP_RATE_POLICY"):(policy&1)!=0),
        shared_on_(policy<0?env_on("STRATA_MTP_CALIB_SHARED"):(policy&2)!=0),
        log_on_(policy<0?env_on("STRATA_MTP_CALIB_LOG"):(policy&4)!=0) {}

    double acceptance(int depth, float probability) const {
        depth = std::clamp(depth, 0, kMaxT-2);
        const int b = bin(probability);
        // Confidence is not acceptance. This weak prior yields to observed
        // target decisions, conditioned on both confidence and draft depth.
        const double p = log_on_?(std::isfinite(probability)?std::clamp(double(probability),0.0,1.0):0.0):(b + 0.5) / kBins;
        const double prior = 0.2 + 0.8 * std::sqrt(p);
        if (shared_calibration()) {
            const double common=(common_ok_[b]+2*prior)/(common_ok_[b]+common_bad_[b]+2);
            return (ok_[depth][b]+4*common)/(ok_[depth][b]+bad_[depth][b]+4);
        }
        return (ok_[depth][b] + 2.0 * prior) /
               (ok_[depth][b] + bad_[depth][b] + 2.0);
    }

    int choose(const float* probabilities, int available) const {
        available = std::clamp(available, 0, max_t_-1);
        double expected = 1.0, survive = 1.0;
        const bool rate=rate_policy() && rate_samples_>=4;
        const double rho=rate_ms_>0?rate_tokens_/rate_ms_:0;
        double best = rate?expected-rho*verify_cost(1):expected/round_cost(1);
        int best_t = 1;
        for (int j=0;j<available;++j) {
            survive *= acceptance(j, probabilities[j]);
            expected += survive;
            const double score = rate?expected-rho*verify_cost(j+2):expected/round_cost(j+2);
            if (rate?score>best+0.005:score>best*1.005) { best=score; best_t=j+2; }
        }
        return best_t;
    }

    bool continue_draft(const float* probabilities, int available, int limit) {
        if (available >= std::min(limit, max_t_-1)) return false;
        const int current = choose(probabilities, available);
        if (current != available+1) return false;
        double expected=1.0, survive=1.0;
        for (int j=0;j<available;++j) {
            survive *= acceptance(j, probabilities[j]);
            expected += survive;
        }
        // Until another confidence is known, estimate its conditional
        // acceptance from the last confidence at the next depth. The next
        // decision uses the confidence that was actually produced.
        const double next_q=acceptance(available,probabilities[available-1]);
        if(rate_policy() && rate_samples_>=4 && rate_ms_>0) {
            const double rho=rate_tokens_/rate_ms_;
            const int k=std::max(1,current-1);
            const double draft_delta=draft_cost_[k+1]>0 && draft_cost_[k]>0?
                std::clamp(draft_cost_[k+1]-draft_cost_[k],0.05,2.0):0.45;
            const double extra=std::max(0.0,verify_cost(current+1)-verify_cost(current))+draft_delta;
            const bool probe=(++decisions_%48==0) && survive>0.5 && seen_[current+1]<3;
            return probe || survive*next_q>rho*extra*1.01;
        }
        const double old_rate=expected/round_cost(current);
        const double new_rate=(expected+survive*next_q)/round_cost(current+1);
        // An occasional measured probe prevents an unseen length's cost
        // estimate from locking the controller into a short window forever.
        const bool probe=(++decisions_%48==0) && survive>(affine_cost()?0.25:0.5) && seen_[current+1]<3;
        return probe || new_rate > old_rate*1.01;
    }

    void observe(int t,int accepted,const float* probabilities,double verify_commit_ms,bool mtp) {
        t=std::clamp(t,1,kMaxT);
        pending_tokens_=std::clamp(accepted,0,t-1)+1;
        pending_ms_=verify_commit_ms;
        if (std::isfinite(verify_commit_ms) && verify_commit_ms>0) {
            ++seen_[t];
            // First call at each size may capture/upload a graph.
            if (seen_[t]>1) {
                if (robust_cost()) {
                    const int n=std::min(seen_[t]-1,5);
                    recent_[t][(seen_[t]-2)%5]=verify_commit_ms;
                    auto sorted=recent_[t];
                    std::sort(sorted.begin(),sorted.begin()+n);
                    cost_[t]=n==5?(sorted[1]+sorted[2]+sorted[3])/3.0:sorted[n/2];
                } else cost_[t]=cost_[t]>0?0.9*cost_[t]+0.1*verify_commit_ms:verify_commit_ms;
            }
        }
        if (!mtp || !probabilities) return;
        accepted=std::clamp(accepted,0,t-1);
        const int known=std::min(t-1,accepted+1);
        for (int j=0;j<known;++j) {
            const int b=bin(probabilities[j]);
            ok_[j][b]*=0.985;bad_[j][b]*=0.985;
            (j<accepted?ok_[j][b]:bad_[j][b])+=1.0;
            common_ok_[b]*=0.985;common_bad_[b]*=0.985;
            (j<accepted?common_ok_[b]:common_bad_[b])+=1.0;
        }
        // Rows after the first rejected draft were never tested against their
        // valid history. They must not be counted as rejected observations.
    }

    void observe_draft(int count,double ms) {
        if (count<1 || !std::isfinite(ms) || ms<=0) return;
        if(pending_tokens_>0 && std::isfinite(pending_ms_) && pending_ms_>0) {
            const double full=pending_ms_+ms;
            if(++rate_samples_>1) {
                if(rate_ms_==0) {rate_tokens_=pending_tokens_;rate_ms_=full;}
                else {rate_tokens_=0.95*rate_tokens_+0.05*pending_tokens_;rate_ms_=0.95*rate_ms_+0.05*full;}
            }
            pending_tokens_=0;
        }
        if (++draft_seen_[count]>1)
            draft_cost_[count]=draft_cost_[count]>0?0.9*draft_cost_[count]+0.1*ms:ms;
    }

    double verify_cost(int t) const {
        t=std::clamp(t,1,max_t_);
        if (cost_[t]>0 && (!robust_cost() || seen_[t]>=4)) return cost_[t];
        // All-resident GPU prior, replaced by actual per-size costs. Unknown
        // sizes are scaled from nearby observations, with distance weighting.
        constexpr double old_shape[9]={0,1,1.28,1.55,1.83,2.15,2.48,2.8,3.13};
        // Resident weights and fixed graph work are shared by the window.
        // Missing sizes should not assume that all work scales per token.
        // This is a weak latency prior; measured costs still replace it.
        constexpr double resident_shape[9]={0,1,1.34,1.50,1.66,1.82,1.98,2.14,2.30};
        const double* shape=affine_cost()?resident_shape:old_shape;
        double sum=0,weight=0;
        for (int u=1;u<=max_t_;++u) if(cost_[u]>0 && (!robust_cost() || seen_[u]>=4)) {
            const double w=std::min(seen_[u],20)/(1.0+std::abs(u-t));
            sum+=w*cost_[u]*shape[t]/shape[u];weight+=w;
        }
        const double prior=weight>0?sum/weight:7.0*shape[t];
        // A single cold streaming/capture stall must not permanently suppress
        // the very window that would provide its next cost observation. Once
        // three costs exist, their robust measured estimate controls directly.
        if (cost_[t]>0) return weight>0?std::min(cost_[t],prior*1.15):cost_[t];
        return prior;
    }

private:
    bool rate_policy() const {return rate_on_;}
    bool shared_calibration() const {return shared_on_;}
    static bool env_on(const char* name) {const char* v=std::getenv(name);return v&&v[0]=='1';}
    static bool robust_cost() {
        static const bool enabled=[] {const char* v=std::getenv("STRATA_MTP_COST_ROBUST");return v&&std::atoi(v)==1;}();
        return enabled;
    }
    static bool affine_cost() {
        static const bool enabled=[] {const char* v=std::getenv("STRATA_MTP_COST_AFFINE");return v&&std::atoi(v)==1;}();
        return enabled;
    }
    int bin(float p) const {
        if(!std::isfinite(p))return 0;
        p=std::clamp(p,0.0f,1.0f);
        if(!log_on_)return std::clamp(int(p*kBins),0,kBins-1);
        // Equal coarse probability widths blur near-certainty with merely
        // likely drafts. Thresholds are fixed; observations stay request-local.
        constexpr double edges[kBins-1]={0.3934693402873666,0.6321205588285577,
            0.7768698398515702,0.8646647167633873,0.950212931632136,
            0.9816843611112658,0.9975212478233336,0.9996645373720975,0.9999938557876467};
        int b=0;for(double edge:edges)b+=p>=edge;return b;
    }
    double round_cost(int t) const {
        t=std::clamp(t,1,max_t_);
        const int k=std::max(1,t-1);
        if(draft_cost_[k]>0) return verify_cost(t)+draft_cost_[k];
        double base=1.3;
        if(draft_cost_[1]>0)base=draft_cost_[1];
        else for(int j=2;j<max_t_;++j)if(draft_cost_[j]>0) {
            base=std::max(0.1,draft_cost_[j]-0.45*(j-1));break;
        }
        return verify_cost(t)+base+0.45*(k-1);
    }
    int max_t_,decisions_=0;
    bool rate_on_=false,shared_on_=false,log_on_=false;
    int rate_samples_=0,pending_tokens_=0;
    double rate_tokens_=0,rate_ms_=0,pending_ms_=0;
    std::array<double,kBins> common_ok_{},common_bad_{};
    std::array<std::array<double,kBins>,kMaxT-1> ok_{},bad_{};
    std::array<double,kMaxT+1> cost_{},draft_cost_{};
    std::array<std::array<double,5>,kMaxT+1> recent_{};
    std::array<int,kMaxT+1> seen_{},draft_seen_{};
};

} // namespace strata::spec
