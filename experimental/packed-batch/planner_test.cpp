#include "planner.hpp"
#include <cassert>
#include <numeric>

int main() {
    using strata::turbo::ProposalBudget;
    using strata::turbo::packed_rows;
    // Two sessions can verify three proposals each without losing a base row.
    assert((packed_rows({{0,{.9f,.8f,.7f}},{1,{.9f,.8f,.7f}}},8) ==
            std::vector<int>{4,4}));
    // Six sessions all advance, with only two extra verified proposals.
    auto six = packed_rows({{0,{.1f}},{1,{.2f}},{2,{.9f}},
                            {3,{.8f}},{4,{.4f}},{5,{.3f}}},8);
    assert((six == std::vector<int>{1,1,2,2,1,1}));
    // Descendants compete by prefix confidence, not individual confidence.
    assert((packed_rows({{0,{.8f,.8f,.8f}},{1,{.7f}}},4) ==
            std::vector<int>{2,2}));
    assert((packed_rows({{0,{}},{1,{.7f}}},8) == std::vector<int>{1,2}));
    assert(packed_rows({{0,{}},{1,{}}},1).empty());
    // Exhaustively check capacity and contiguous-prefix bounds for 1..6 slots.
    for (int n=1;n<=6;++n) for (int mask=0;mask<(1<<n);++mask) {
        std::vector<ProposalBudget> slots;
        for (int i=0;i<n;++i) slots.push_back({i,std::vector<float>(mask&(1<<i)?3:0,.8f)});
        auto rows=packed_rows(slots,8);
        assert(std::accumulate(rows.begin(),rows.end(),0)<=8);
        for(int i=0;i<n;++i) assert(rows[i]>=1 && rows[i]<=1+(int)slots[i].probabilities.size());
    }
}
