// Export llama.cpp's complete mtmd prompt, including merged video frames and decoder positions.
// SVE2: int32 {magic, prompt_tokens, width, image_chunks, next_position}, token ids,
//       int32[prompt_tokens][3] (t,h,w), then {start,count} and float[count][width] per image chunk.
#pragma once
#include <nlohmann/json.hpp>
#include <array>
#include <fstream>
#include <stdexcept>

inline void native_media_prepare(mtmd_context* ctx, int width, const std::string& input,
                                 const std::string& output) {
    using json = nlohmann::json;
    std::ifstream request_file(input);
    const json req = json::parse(request_file);
    const std::string prompt = req.at("prompt");
    const int limit = req.at("max_context");
    if (limit < 1 || limit > 1048576) throw std::runtime_error("invalid context limit");
    std::vector<mtmd_helper_bitmap_wrapper> media;
    std::vector<const mtmd_bitmap*> bitmaps;
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    struct Cleanup {
        std::vector<mtmd_helper_bitmap_wrapper>& media;
        mtmd_input_chunks* chunks;
        ~Cleanup() {
            mtmd_input_chunks_free(chunks);
            for (auto& item : media) {
                if (item.bitmap) mtmd_bitmap_free(item.bitmap);
                if (item.video_ctx) mtmd_helper_video_free(item.video_ctx);
            }
        }
    } cleanup{media, chunks};
    auto opt = mtmd_helper_init_opt_default();
    opt.video_params.fps_target = req.value("fps", 4.0f);
    opt.video_params.timestamp_interval_ms = req.value("timestamp_interval_ms", int64_t(5000));
    for (const auto& path : req.at("media")) {
        auto item = mtmd_helper_bitmap_init_from_file(ctx, path.get<std::string>().c_str(), false, opt);
        media.push_back(item);
        if (!item.bitmap) throw std::runtime_error("media cannot be decoded");
        bitmaps.push_back(item.bitmap);
    }
    mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
    if (mtmd_tokenize(ctx, chunks, &txt, bitmaps.data(), bitmaps.size()) != 0)
        throw std::runtime_error("native mtmd tokenization failed");
    const size_t n = mtmd_helper_get_n_tokens(chunks);
    if (!n || n > (size_t) limit) throw std::runtime_error("native media prompt exceeds context; reduce video fps or resolution");
    std::vector<int32_t> ids;
    std::vector<std::array<int32_t, 3>> positions;
    int32_t next = 0, image_count = 0;
    for (size_t k = 0; k < mtmd_input_chunks_size(chunks); ++k) {
        const auto* ch = mtmd_input_chunks_get(chunks, k);
        const size_t count = mtmd_input_chunk_get_n_tokens(ch);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t length;
            const auto* tokens = mtmd_input_chunk_get_tokens_text(ch, &length);
            ids.insert(ids.end(), tokens, tokens + length);
            for (size_t j = 0; j < length; ++j) positions.push_back({next + (int32_t) j, next + (int32_t) j, next + (int32_t) j});
        } else if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            ++image_count;
            const auto* image = mtmd_input_chunk_get_tokens_image(ch);
            ids.insert(ids.end(), count, 248056); // this Qwen model's image token id, also used by Strata's PLE
            for (size_t j = 0; j < count; ++j) {
                const auto p = mtmd_image_tokens_get_decoder_pos(image, next, j);
                positions.push_back({(int32_t)p.t, (int32_t)p.y, (int32_t)p.x}); // mtmd's (t,h,w) ordering
            }
        } else throw std::runtime_error("audio is not supported");
        next += mtmd_input_chunk_get_n_pos(ch);
    }
    if (ids.size() != n || positions.size() != n) throw std::runtime_error("native prompt size mismatch");
    std::ofstream out(output, std::ios::binary);
    auto write = [&](const void* data, size_t bytes) {
        out.write(static_cast<const char*>(data), bytes);
        if (!out) throw std::runtime_error("cannot write native media export");
    };
    const int32_t header[5] = {0x32455653, (int32_t)n, width, image_count, next};
    write(header, sizeof(header));
    write(ids.data(), ids.size() * sizeof(int32_t));
    write(positions.data(), positions.size() * sizeof(positions[0]));
    int32_t start = 0;
    json segments = json::array();
    for (size_t k = 0; k < mtmd_input_chunks_size(chunks); ++k) {
        const auto* ch = mtmd_input_chunks_get(chunks, k);
        const int32_t count = (int32_t)mtmd_input_chunk_get_n_tokens(ch);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            if (mtmd_encode_chunk(ctx, ch) != 0) throw std::runtime_error("native video/image encoding failed");
            const int32_t segment[2] = {start, count};
            write(segment, sizeof(segment));
            write(mtmd_get_output_embd(ctx), (size_t)count * width * sizeof(float));
            segments.push_back({{"start", start}, {"tokens", count}});
        }
        start += count;
    }
    out.close();
    std::ofstream metadata(output + ".json");
    metadata << json{{"ids", ids}, {"next_position", next}, {"segments", segments}}.dump();
    if (!metadata) throw std::runtime_error("cannot write native media metadata");
    std::printf("OKPRE %zu %d %d\n", n, image_count, next);
}
