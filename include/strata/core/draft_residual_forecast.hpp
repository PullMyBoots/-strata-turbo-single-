#pragma once
#include <cstdint>
#include <string>

namespace strata::core {
// Experimental auxiliary-only map. No target residual or target logits use it.
class DraftResidualForecast {
public:
    ~DraftResidualForecast();
    static uint64_t reserve_bytes(const std::string& path);
    bool load(const std::string& path, int streams, int width,
              const float* input_norm, std::string& err);
    bool run(float* residual, void* stream, std::string& err);
private:
    struct State;
    State* state_ = nullptr;
};
}
