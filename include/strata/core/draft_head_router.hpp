#pragma once
#include <cstdint>
#include <string>

namespace strata::core {
// Approximate proposal selection only. The target model still verifies every token.
class DraftHeadRouter {
public:
    DraftHeadRouter()=default;
    ~DraftHeadRouter();
    DraftHeadRouter(const DraftHeadRouter&)=delete;
    DraftHeadRouter& operator=(const DraftHeadRouter&)=delete;
    static uint64_t reserve_bytes(const std::string& path, uint64_t row_bytes);
    bool load(const std::string& path, const void* head, const int32_t* vocab,
              int type, int rows, int cols, uint64_t row_bytes, std::string& err);
    bool run(const void* xq, float* logits, void* stream, std::string& err);
    int size() const;
    const int32_t* vocab() const;
private:
    struct State;
    State* state_=nullptr;
};
}
