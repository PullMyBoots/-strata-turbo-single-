#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace strata::turbo {

struct ProposalBudget {
    int slot;
    std::vector<float> probabilities;
};

// Every live session receives its base row. Remaining rows extend contiguous
// proposal prefixes; a low confidence descendant cannot displace its ancestor.
// These scores choose work only. Acceptance always uses the target sampler.
inline std::vector<int> packed_rows(const std::vector<ProposalBudget>& slots,
                                    int row_budget) {
    std::vector<int> rows(slots.size(), 1);
    if ((int) slots.size() > row_budget) return {};
    for (int free = row_budget - (int) slots.size(); free > 0; --free) {
        int best = -1;
        float best_score = -1.0f;
        for (int i = 0; i < (int) slots.size(); ++i) {
            const auto& probabilities = slots[i].probabilities;
            if (rows[i] > (int) probabilities.size()) continue;
            float score = 1.0f;
            for (int j = 0; j < rows[i]; ++j)
                score *= std::clamp(probabilities[j], 0.0f, 1.0f);
            if (std::isfinite(score) && score > best_score) {
                best = i;
                best_score = score;
            }
        }
        if (best < 0) break;
        ++rows[best];
    }
    return rows;
}

} // namespace strata::turbo
