// Validated SVE2 import. The positions come from llama.cpp mtmd; the engine does not reconstruct them.
#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::core {
struct NativeMediaSegment { int32_t start, count; size_t offset; };
struct NativeMediaPrompt {
    int32_t next_position = 0;
    std::vector<std::array<int32_t, 3>> positions;
    std::vector<NativeMediaSegment> segments;
    std::vector<float> rows;
};
inline bool read_native_media(std::FILE* f, int32_t width, const std::vector<int64_t>& ids,
                              int64_t max_context, NativeMediaPrompt& out, std::string& error) {
    auto fail = [&](const char* why) { error = why; return false; };
    int32_t h[5];
    if (std::fread(h, sizeof(int32_t), 5, f) != 5 || h[0] != 0x32455653 ||
        h[1] < 1 || h[1] > max_context || (size_t)h[1] > ids.size() || ids.size() > (size_t)max_context || h[2] != width ||
        h[3] < 1 || h[3] > h[1] || h[4] < 1 || h[4] > h[1]) return fail("invalid native media header");
    std::vector<int32_t> exported_ids(h[1]);
    if (std::fread(exported_ids.data(), sizeof(int32_t), exported_ids.size(), f) != exported_ids.size())
        return fail("short native media token list");
    for (size_t i = 0; i < exported_ids.size(); ++i)
        if (exported_ids[i] != ids[i]) return fail("native media tokens differ from request");
    out.next_position = h[4];
    out.positions.resize(h[1]);
    if (std::fread(out.positions.data(), sizeof(out.positions[0]), out.positions.size(), f) != out.positions.size())
        return fail("short native media positions");
    for (const auto& p : out.positions) for (auto component : p)
        if (component < 0 || component >= max_context) return fail("native rotary position outside context");
    // A thinking-budget continuation reuses this visual prefix and appends generated text + </think>.
    // Keep every exported media position intact; text continues from mtmd's next position on all axes.
    const size_t prefix_size = out.positions.size();
    if ((int64_t)out.next_position + (int64_t)(ids.size() - prefix_size) > max_context)
        return fail("native continuation position outside context");
    out.positions.reserve(ids.size());
    for (size_t i = prefix_size; i < ids.size(); ++i) {
        if (ids[i] == 248056) return fail("native continuation cannot append image tokens");
        const int32_t position = out.next_position++;
        out.positions.push_back({position, position, position});
    }
    int32_t previous_end = 0;
    for (int32_t k = 0; k < h[3]; ++k) {
        int32_t segment[2];
        if (std::fread(segment, sizeof(int32_t), 2, f) != 2 || segment[0] < previous_end ||
            segment[1] < 1 || (int64_t)segment[0] + segment[1] > h[1]) return fail("invalid native media segment");
        for (int64_t i = segment[0]; i < (int64_t)segment[0] + segment[1]; ++i)
            if (ids[i] != 248056) return fail("native image rows require image token ids");
        const size_t offset = out.rows.size(), count = (size_t)segment[1] * width;
        out.rows.resize(offset + count);
        if (std::fread(out.rows.data() + offset, sizeof(float), count, f) != count)
            return fail("short native media embeddings");
        out.segments.push_back({segment[0], segment[1], offset});
        previous_end = segment[0] + segment[1];
    }
    if (std::fgetc(f) != EOF || std::ferror(f)) return fail("unexpected native media trailing bytes");
    return true;
}
} // namespace strata::core
