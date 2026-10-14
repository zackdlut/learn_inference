#pragma once

#include "inference/tensor.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace infr_test
{

inline void append_le(std::vector<std::byte> &buf, const void *src, std::size_t n)
{
    const auto *p = static_cast<const std::byte *>(src);
    buf.insert(buf.end(), p, p + n);
}

template <typename T> inline void append_le(std::vector<std::byte> &buf, T v)
{
    append_le(buf, &v, sizeof(T));
}

struct TensorSpec
{
    std::string name;
    inference::Shape shape;
    std::vector<float> data;
};

inline std::vector<std::byte> pack_infr(std::uint32_t version,
                                        const std::vector<TensorSpec> &tensors)
{
    std::vector<std::byte> header;
    const char magic[] = "INFR";
    append_le(header, magic, 4);
    append_le(header, version);
    append_le(header, static_cast<std::uint32_t>(tensors.size()));

    std::vector<std::byte> body;
    std::uint64_t cursor = 0;
    for (const auto &t : tensors)
    {
        const auto name_bytes = t.name;
        append_le(header, static_cast<std::uint32_t>(name_bytes.size()));
        append_le(header, name_bytes.data(), name_bytes.size());
        append_le(header, static_cast<std::uint32_t>(0)); // FP32
        append_le(header, static_cast<std::uint32_t>(t.shape.size()));
        for (const auto dim : t.shape)
        {
            append_le(header, dim);
        }
        const auto nbytes = static_cast<std::uint64_t>(
            static_cast<std::size_t>(inference::numel_of(t.shape)) * sizeof(float));
        append_le(header, cursor);
        append_le(header, nbytes);
        append_le(body, t.data.data(), t.data.size() * sizeof(float));
        cursor += nbytes;
    }

    const std::uint64_t data_start =
        static_cast<std::uint64_t>(header.size() + sizeof(std::uint64_t));
    append_le(header, data_start);
    header.insert(header.end(), body.begin(), body.end());
    return header;
}

inline std::string write_temp_file(const std::vector<std::byte> &bytes, const char *tag)
{
    const auto path =
        std::filesystem::temp_directory_path() / std::filesystem::path("learn_inference_") / tag;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out.good())
    {
        throw std::runtime_error("failed to write temp INFR file");
    }
    return path.string();
}

inline std::string model_dir()
{
#ifdef INFERENCE_SOURCE_DIR
    return std::string(INFERENCE_SOURCE_DIR) + "/data/models";
#else
    return "data/models";
#endif
}

inline bool file_exists(const std::string &path)
{
    return std::filesystem::exists(path);
}

} // namespace infr_test
