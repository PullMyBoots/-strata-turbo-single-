#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <functional>
#include <string>

namespace strata::core {
// Opt-in local CUDA IPC pool. Only immutable model weights may use these
// functions. Session state, sampling, histories and workspaces stay private.
bool shared_weights_enabled();
bool shared_weights_reader();
cudaError_t shared_weight_alloc(void** out, size_t bytes, const std::string& key);
cudaError_t shared_weight_upload(void** out, const void* src, size_t bytes, const std::string& key);
cudaError_t shared_weight_publish(void* ptr, size_t bytes, const std::string& key);
cudaError_t shared_weight_free(void* ptr);
bool shared_weight_imported(const void* ptr);
// Mutable prefill workspaces may be imported only with this process-wide
// interprocess lease held throughout prompt processing, including MTP catch-up.
class SharedPrefillLease {
public:
    explicit SharedPrefillLease(std::function<bool()> waiting={});
    ~SharedPrefillLease();
    void release();
    bool acquired() const { return fd_>=0; }
    SharedPrefillLease(const SharedPrefillLease&)=delete;
    SharedPrefillLease& operator=(const SharedPrefillLease&)=delete;
private:
    int fd_=-1;
};
// Private stable virtual address; physical prompt scratch exists only during
// prefill. No session data or weights may be placed here. A SharedPrefillLease
// bounds concurrent physical allocation across the pool.
class EphemeralPrefillWorkspace {
public:
    EphemeralPrefillWorkspace();
    ~EphemeralPrefillWorkspace();
    EphemeralPrefillWorkspace(const EphemeralPrefillWorkspace&)=delete;
    EphemeralPrefillWorkspace& operator=(const EphemeralPrefillWorkspace&)=delete;
    bool reserve(size_t bytes, std::string& err);
    bool commit(std::string& err, size_t requested=0);
    bool release(std::string& err);
    void* data() const;
    size_t bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
