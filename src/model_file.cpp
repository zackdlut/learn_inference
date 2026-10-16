#include "inference/model_file.hpp"
#include "inference/dtype.hpp"
#include "inference/tensor.hpp"
#include <cstring>
#include <fstream>
#include <stdexcept>
namespace inference
{

namespace
{

// 抛出异常
[[noreturn]] void fail(const std::string &msg)
{
    throw std::runtime_error("[inference::ModelFile] " + msg);
}

// 读取小端字节序的值
template <typename T> T read_le(const std::vector<std::byte> &buf, std::size_t &pos)
{
    if (pos + sizeof(T) > buf.size())
    {
        fail("Read out of bounds");
    }
    T v{};
    std::memcpy(&v, buf.data() + pos, sizeof(T));
    pos += sizeof(T);
    if constexpr (std::endian::native == std::endian::big)
    {
        auto *p = reinterpret_cast<std::byte *>(&v);
        std::reverse(p, p + sizeof(T));
    }
    return v;
}
// 读取字符串
std::string read_string(const std::vector<std::byte> &buf, std::size_t &pos, std::size_t length)
{
    if (pos + length > buf.size())
    {
        fail("Read String out of bounds");
    }
    std::string s(reinterpret_cast<const char *>(buf.data() + pos), length);
    pos += length;
    return s;
}
} // namespace

ModelFile ModelFile::load(const std::string &path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        fail("Failed to open file: " + path);
    }
    const auto size = static_cast<std::size_t>(file.tellg());
    file.seekg(0);
    std::vector<std::byte> buf(size);
    file.read(reinterpret_cast<char *>(buf.data()), size);
    if (!file.good())
    {
        fail("Failed to read file: " + path);
    }

    ModelFile model_file;
    // Read and check magic number INFR
    std::size_t pos = 0;
    if (size < 12 || std::memcmp(buf.data(), "INFR", 4) != 0)
    {
        fail("Invalid INFR file: " + path);
    }
    pos += 4;
    // Read and check version
    model_file.version_ = read_le<std::uint32_t>(buf, pos);
    if (model_file.version_ != 1)
    {
        fail("Unsupported INFR version: " + path);
    }
    // read and check number of tensors
    const auto num_tensors = read_le<std::uint32_t>(buf, pos);
    if (num_tensors == 0)
    {
        fail("No tensors in INFR file: " + path);
    }
    model_file.tensor_infos.reserve(num_tensors);
    // read and check tensors
    for (std::size_t i = 0; i < num_tensors; ++i)
    {
        TensorInfo info;
        const auto name_len = read_le<std::uint32_t>(buf, pos);
        info.name = read_string(buf, pos, name_len);
        const auto dtype_code = read_le<std::uint32_t>(buf, pos);
        if (dtype_code != 0)
        {
            fail("Unsupported dtype code: " + path);
        }
        info.dtype = DType::FP32;
        const auto ndim = read_le<std::uint32_t>(buf, pos);
        info.shape.reserve(ndim);
        for (std::size_t j = 0; j < ndim; ++j)
        {
            info.shape.push_back(read_le<std::int64_t>(buf, pos));
        }
        info.offset = read_le<std::uint64_t>(buf, pos);
        info.length = read_le<std::uint64_t>(buf, pos);

        // check offset and length
        const auto expect = numel_of(info.shape) * dtype_size(info.dtype);
        if (expect != info.length)
        {
            fail("Invalid tensor length: " + path);
        }
        model_file.tensor_infos.push_back(info);
        model_file.name_map.emplace(info.name, i);
    }
    // read and check data start
    const auto data_start = read_le<std::uint64_t>(buf, pos);
    if (data_start > size)
    {
        fail("Invalid data start: " + path);
    }
    model_file.data.assign(buf.begin() + data_start, buf.end());
    return model_file;
}

Tensor ModelFile::get_tensor(const std::string &name) const
{
    const auto it = name_map.find(name);
    if (it == name_map.end())
    {
        fail("Tensor not found: " + name);
    }
    const auto &info = tensor_infos[it->second];
    Tensor ret = Tensor::zeros(info.shape, info.dtype);
    std::memcpy(ret.data().data(), data.data() + info.offset, info.length);
    return ret;
}

std::string ModelFile::to_string() const noexcept
{
    std::string file_info;
    file_info += "INFR version: " + std::to_string(version_) + "\n";
    file_info += "Number of tensors: " + std::to_string(tensor_infos.size()) + "\n";
    file_info += "Data start: " + std::to_string(data.size()) + "\n";
    return file_info;
}

} // namespace inference