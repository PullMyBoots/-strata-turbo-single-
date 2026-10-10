// CPU regression test: SVE2 visual prefixes survive thinking-budget text continuations.
#include "strata/core/native_media.hpp"
#include <cassert>
#include <cstring>
#include <iostream>

std::vector<unsigned char> fixture() {
    std::vector<unsigned char> bytes;
    auto append = [&](const void* p, size_t n) {
        auto* b = static_cast<const unsigned char*>(p); bytes.insert(bytes.end(), b, b+n);
    };
    int32_t h[5]={0x32455653,4,2,1,4}, ids[4]={1,248056,248056,2};
    int32_t pos[12]={0,0,0,1,1,1,1,1,2,3,3,3}, segment[2]={1,2};
    float rows[4]={0.1f,0.2f,0.3f,0.4f};
    append(h,sizeof(h)); append(ids,sizeof(ids)); append(pos,sizeof(pos));
    append(segment,sizeof(segment)); append(rows,sizeof(rows)); return bytes;
}
bool parse(const std::vector<unsigned char>& data, const std::vector<int64_t>& ids={1,248056,248056,2}) {
    auto* file=std::tmpfile();assert(file);
    assert(std::fwrite(data.data(),1,data.size(),file)==data.size());std::rewind(file);
    strata::core::NativeMediaPrompt result;std::string error;
    bool ok=strata::core::read_native_media(file,2,ids,32,result,error);std::fclose(file);
    if(ok){
        assert(result.segments.size()==1 && result.rows.size()==4);
        assert(result.positions.size()==ids.size() && result.next_position==(int32_t)ids.size());
        assert((result.positions[2]==std::array<int32_t,3>{1,1,2}));
        for(size_t i=4;i<ids.size();++i)
            assert((result.positions[i]==std::array<int32_t,3>{(int32_t)i,(int32_t)i,(int32_t)i}));
    }
    return ok;
}
int main(){
    auto good=fixture();assert(parse(good));
    assert(parse(good,{1,248056,248056,2,17,19,20}));
    assert(!parse(good,{1,248056,248056,3,17}));
    assert(!parse(good,{1,248056,248056}));
    assert(!parse(good,{1,248056,248056,2,248056}));
    auto overlong=std::vector<int64_t>{1,248056,248056,2};overlong.resize(33,17);
    assert(!parse(good,overlong));
    for(size_t length:{size_t(0),size_t(8),size_t(22),size_t(40),good.size()-1}){
        auto broken=good;broken.resize(length);assert(!parse(broken));
    }
    assert(!parse(good,{1,248056,248056,3}));
    auto extra=good;extra.push_back(0);assert(!parse(extra));
    auto badpos=good;int32_t negative=-1;std::memcpy(badpos.data()+36,&negative,4);assert(!parse(badpos));
    auto overlap=good;int32_t before=-1;std::memcpy(overlap.data()+84,&before,4);assert(!parse(overlap));
    auto huge=good;int32_t large=1000000;std::memcpy(huge.data()+4,&large,4);assert(!parse(huge));
    std::cout<<"PASS SVE2 prefix continuation, native positions, row mapping and malformed-input bounds\n";
}
