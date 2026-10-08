#include "strata/core/shared_weights.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_set>
#include <thread>
#include <chrono>
#include <cerrno>
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
#include <cuda.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#include <stdexcept>
#endif

namespace strata::core {
namespace {
std::mutex guard;
std::unordered_set<const void*> imported;
const char* directory() { const char* d=std::getenv("STRATA_SHARED_WEIGHTS_DIR");return d && *d ? d : nullptr; }
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
struct Record {
    char magic[8]={'S','T','R','I','P','C','1',0};
    uint64_t bytes=0;
    int64_t pid=0;
    cudaUUID_t uuid{};
    cudaIpcMemHandle_t handle{};
    char key[320]{};
};
std::string path_for(const std::string& key) {
    int dev=0;cudaGetDevice(&dev);cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,dev);const cudaUUID_t uuid=prop.uuid;
    uint64_t h=1469598103934665603ull;
    for(unsigned char c:key)h=(h^c)*1099511628211ull;
    for(unsigned char c:uuid.bytes)h=(h^c)*1099511628211ull;
    char b[32];std::snprintf(b,sizeof b,"/%016llx.ipc",(unsigned long long)h);
    return std::string(directory())+b;
}
cudaError_t import(void** out,size_t bytes,const std::string& key) {
    if(key.size()>=sizeof(Record{}.key))return cudaErrorInvalidValue;
    const std::string path=path_for(key);struct stat st{};
    if(stat(path.c_str(),&st)!=0 || st.st_uid!=getuid() || !S_ISREG(st.st_mode) || (st.st_mode&0077)) {
        std::fprintf(stderr,"strata IPC: missing or unsafe weight record %s\n",path.c_str());return cudaErrorInvalidValue;
    }
    Record r;std::ifstream f(path,std::ios::binary);f.read((char*)&r,sizeof r);
    int dev=0;cudaGetDevice(&dev);cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,dev);const cudaUUID_t uuid=prop.uuid;
    if(!f || std::memcmp(r.magic,"STRIPC1",8)!=0 || r.bytes!=bytes ||
       std::strncmp(r.key,key.c_str(),sizeof r.key)!=0 || std::memcmp(&uuid,&r.uuid,sizeof uuid)!=0 ||
       r.pid<=0 || kill((pid_t)r.pid,0)!=0) {
        std::fprintf(stderr,"strata IPC: stale or mismatched weight record %s\n",path.c_str());return cudaErrorInvalidValue;
    }
    const cudaError_t e=cudaIpcOpenMemHandle(out,r.handle,cudaIpcMemLazyEnablePeerAccess);
    if(e==cudaSuccess){std::lock_guard<std::mutex> lock(guard);imported.insert(*out);}
    return e;
}
#endif
}
bool shared_weights_enabled() {
#if defined(_WIN32) || defined(STRATA_USE_HIP)
    return false;
#else
    return directory()!=nullptr;
#endif
}
bool shared_weights_reader() {
    const char* m=std::getenv("STRATA_SHARED_WEIGHTS_MODE");
    return shared_weights_enabled() && m && std::strcmp(m,"reader")==0;
}
bool shared_weight_imported(const void* ptr) {std::lock_guard<std::mutex> lock(guard);return imported.count(ptr)>0;}
cudaError_t shared_weight_alloc(void** out,size_t bytes,const std::string& key) {
    *out=nullptr;
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    if(shared_weights_reader())return import(out,bytes,key);
#endif
    return cudaMalloc(out,bytes);
}
cudaError_t shared_weight_publish(void* ptr,size_t bytes,const std::string& key) {
    if(!shared_weights_enabled() || shared_weights_reader())return cudaSuccess;
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    if(key.size()>=sizeof(Record{}.key))return cudaErrorInvalidValue;
    Record r;r.bytes=bytes;r.pid=getpid();std::strcpy(r.key,key.c_str());
    int dev=0;cudaGetDevice(&dev);cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,dev);r.uuid=prop.uuid;
    cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)return e;
    e=cudaIpcGetMemHandle(&r.handle,ptr);if(e!=cudaSuccess)return e;
    const auto path=path_for(key),tmp=path+".tmp."+std::to_string(getpid());
    std::ofstream f(tmp,std::ios::binary|std::ios::trunc);if(!f)return cudaErrorInvalidValue;
    f.write((const char*)&r,sizeof r);f.close();
    if(!f || chmod(tmp.c_str(),0600)!=0 || rename(tmp.c_str(),path.c_str())!=0){unlink(tmp.c_str());return cudaErrorInvalidValue;}
#endif
    return cudaSuccess;
}
cudaError_t shared_weight_upload(void** out,const void* src,size_t bytes,const std::string& key) {
    cudaError_t e=shared_weight_alloc(out,bytes,key);
    if(e!=cudaSuccess || shared_weights_reader())return e;
    e=cudaMemcpy(*out,src,bytes,cudaMemcpyHostToDevice);
    if(e==cudaSuccess)e=shared_weight_publish(*out,bytes,key);
    if(e!=cudaSuccess){cudaFree(*out);*out=nullptr;}
    return e;
}
cudaError_t shared_weight_free(void* ptr) {
    if(!ptr)return cudaSuccess;
    bool was_imported=false;
    {std::lock_guard<std::mutex> lock(guard);was_imported=imported.erase(ptr)>0;}
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    if(was_imported)return cudaIpcCloseMemHandle(ptr);
#endif
    return cudaFree(ptr);
}
SharedPrefillLease::SharedPrefillLease(std::function<bool()> waiting) {
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    if(!shared_weights_enabled())return;
    const std::string path=std::string(directory())+"/prefill.lock";
    fd_=open(path.c_str(),O_CREAT|O_RDWR|O_CLOEXEC,0600);
    if(fd_<0)throw std::runtime_error("cannot open shared prefill lease");
    while(flock(fd_,LOCK_EX|LOCK_NB)!=0){
        if(errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR){
            close(fd_);fd_=-1;throw std::runtime_error("cannot lock shared prefill workspace");
        }
        if(waiting && !waiting()){close(fd_);fd_=-1;return;}
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#else
    (void)waiting;
#endif
}
void SharedPrefillLease::release() {
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    if(fd_>=0){flock(fd_,LOCK_UN);close(fd_);fd_=-1;}
#endif
}
SharedPrefillLease::~SharedPrefillLease(){release();}

struct EphemeralPrefillWorkspace::Impl {
    int device=-1;
    size_t size=0;
    size_t granularity=0, mapped_size=0;
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    CUdeviceptr ptr=0;
    CUmemGenericAllocationHandle handle=0;
    CUmemAllocationProp prop{};
    bool mapped=false;
#endif
};
EphemeralPrefillWorkspace::EphemeralPrefillWorkspace():impl_(std::make_unique<Impl>()){}
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
namespace {
bool vmm_ok(CUresult e,const char* operation,std::string& err) {
    if(e==CUDA_SUCCESS)return true;
    const char* s=nullptr;cuGetErrorString(e,&s);
    err=std::string("ephemeral prefill: ")+operation+": "+(s?s:"driver error");return false;
}
}
#endif
bool EphemeralPrefillWorkspace::reserve(size_t bytes,std::string& err) {
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    auto& m=*impl_;
    if(m.ptr || !bytes){err="ephemeral prefill: invalid or repeated reservation";return false;}
    CUdevice dev=0;
    if(!vmm_ok(cuCtxGetDevice(&dev),"device",err))return false;
    int supported=0;
    if(!vmm_ok(cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,dev),"support",err))return false;
    if(!supported){err="ephemeral prefill: CUDA VMM is unavailable";return false;}
    m.device=(int)dev;m.prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
    m.prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;m.prop.location.id=dev;
    size_t gran=0;
    if(!vmm_ok(cuMemGetAllocationGranularity(&gran,&m.prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM),"granularity",err))return false;
    m.granularity=gran;
    m.size=((bytes+gran-1)/gran)*gran;
    return vmm_ok(cuMemAddressReserve(&m.ptr,m.size,gran,0,0),"reserve",err);
#else
    (void)bytes;err="ephemeral prefill requires Linux CUDA";return false;
#endif
}
bool EphemeralPrefillWorkspace::commit(std::string& err,size_t requested) {
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    auto& m=*impl_;
    if(!m.ptr){err="ephemeral prefill: workspace is not reserved";return false;}
    const size_t want=requested?((requested+m.granularity-1)/m.granularity)*m.granularity:m.size;
    if(want>m.size){err="ephemeral prefill: mapping exceeds its reservation";return false;}
    if(m.mapped && m.mapped_size==want)return true;
    if(m.mapped && !release(err))return false;
    if(!vmm_ok(cuMemCreate(&m.handle,want,&m.prop,0),"allocate",err))return false;
    if(!vmm_ok(cuMemMap(m.ptr,want,0,m.handle,0),"map",err)){
        cuMemRelease(m.handle);m.handle=0;return false;
    }
    m.mapped=true;m.mapped_size=want;
    CUmemAccessDesc access{};access.location=m.prop.location;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if(!vmm_ok(cuMemSetAccess(m.ptr,want,&access,1),"access",err)){
        cuMemUnmap(m.ptr,want);cuMemRelease(m.handle);m.mapped=false;m.mapped_size=0;m.handle=0;return false;
    }
    return true;
#else
    (void)requested;err="ephemeral prefill requires Linux CUDA";return false;
#endif
}
bool EphemeralPrefillWorkspace::release(std::string& err) {
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    auto& m=*impl_;
    if(!m.mapped){
        if(m.handle && !vmm_ok(cuMemRelease(m.handle),"free",err))return false;
        m.handle=0;return true;
    }
    // Wait for this context's compute/copy/MTP streams. Other MPS clients keep
    // generating; their addresses and allocations are independent.
    const cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){err=std::string("ephemeral prefill: synchronize: ")+cudaGetErrorString(e);return false;}
    if(!vmm_ok(cuMemUnmap(m.ptr,m.mapped_size),"unmap",err))return false;
    m.mapped=false;m.mapped_size=0;
    if(!vmm_ok(cuMemRelease(m.handle),"free",err))return false;
    m.handle=0;return true;
#else
    (void)err;return true;
#endif
}
EphemeralPrefillWorkspace::~EphemeralPrefillWorkspace(){
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    auto& m=*impl_;
    if(m.device<0)return;
    int previous=0;cudaGetDevice(&previous);cudaSetDevice(m.device);
    std::string err;
    if(release(err) && m.ptr)cuMemAddressFree(m.ptr,m.size);
    cudaSetDevice(previous);
#endif
}
void* EphemeralPrefillWorkspace::data() const {
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    return reinterpret_cast<void*>(impl_->ptr);
#else
    return nullptr;
#endif
}
size_t EphemeralPrefillWorkspace::bytes() const{return impl_->size;}
}
