#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace inference
{

// 数据类型枚举 对应 dtype_size 和 dtype_name
// 0: FP32, 1: FP16, 2: INT32, 3: INT8
enum class DType : std::uint8_t
{
    FP32 = 0,
    FP16 = 1,
    INT32 = 2,
    INT8 = 3
};

// 获取数据类型大小
[[nodiscard]] constexpr std::size_t dtype_size(DType dtype) noexcept
{
    switch (dtype)
    {
    case DType::FP32:
        return 4;
    case DType::FP16:
        return 2;
    case DType::INT32:
        return 4;
    case DType::INT8:
        return 1;
    default:
        return 0;
    }
}
// 获取数据类型名称
[[nodiscard]] constexpr std::string_view dtype_name(DType dtype) noexcept
{
    switch (dtype)
    {
    case DType::FP32:
        return "fp32";
    case DType::FP16:
        return "fp16";
    case DType::INT32:
        return "int32";
    case DType::INT8:
        return "int8";
    default:
        return "unknown";
    }
}

// C++ 元素类型 → DType。fp16 没有标准类型，按 IEEE half 的位模式存在 uint16_t 里
template <class T> [[nodiscard]] constexpr DType dtype_of() noexcept
{
    if constexpr (std::is_same_v<T, float>)
        return DType::FP32;
    else if constexpr (std::is_same_v<T, std::uint16_t>)
        return DType::FP16;
    else if constexpr (std::is_same_v<T, std::int32_t>)
        return DType::INT32;
    else
    {
        static_assert(std::is_same_v<T, std::int8_t>, "不支持的元素类型");
        return DType::INT8;
    }
}
} // namespace inference