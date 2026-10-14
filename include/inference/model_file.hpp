#pragma once

#include "inference/tensor.hpp"
#include <unordered_map>

namespace inference
{
struct TensorInfo
{
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    uint64_t offset;
    uint64_t length;
};

class ModelFile
{
  public:
    static ModelFile load(const std::string &path);
    std::uint32_t version() const noexcept
    {
        return version_;
    }
    std::string info() const noexcept;
    bool contains(const std::string &name) const noexcept
    {
        return name_map.contains(name);
    }
    Tensor get_tensor(const std::string &name) const;
    const std::vector<TensorInfo> &get_tensor_infos() const noexcept
    {
        return tensor_infos;
    }

  private:
    std::uint32_t version_ = 0;
    std::vector<TensorInfo> tensor_infos;
    std::unordered_map<std::string, std::size_t> name_map;
    std::vector<std::byte> data;
};

} // namespace inference