#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace inference {

// 数据类型枚举 对应 dtype_size 和 dtype_name
// 0: FP32, 1: FP16, 2: INT32, 3: INT8
enum class DType : std::uint8_t { FP32 = 0, FP16 = 1, INT32 = 2, INT8 = 3 };

// 获取数据类型大小
[[nodiscard]] constexpr std::size_t dtype_size(DType dtype) noexcept {
    switch (dtype) {
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
[[nodiscard]] constexpr std::string_view dtype_name(DType dtype) noexcept {
    switch (dtype) {
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
} // namespace inference