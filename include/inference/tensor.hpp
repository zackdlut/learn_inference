#pragma once

#include "dtype.hpp"
#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace inference {
// 定义int64类型
using int64 = std::int64_t;
// 定义形状：shape = {2, 3} 表示 2 行 3 列。标量的 shape 是空的 {} 阶段1临时使用，后续定义Shape类
using Shape = std::vector<int64>;
std::string shape_string(const Shape &shape);
Shape contiguous_strides(const Shape &shape);
int64 numel_of(const Shape &shape);

// ---------------------------------------------------------------------------
// Tensor：多维数组
//
// 一个张量 = 一块连续内存(data_) + 一副"怎么读这块内存"的说明书(shape_/strides_/offset_)。
// 多个张量可以共享同一块内存，各自拿着不同的说明书 —— 这就是 view(视图)，
// 转置、切片、reshape 都靠它做到零拷贝。
// ---------------------------------------------------------------------------
class Tensor {
  public:
    //=============================================================================
    // 张量构造函数
    //=============================================================================
    // 默认构造函数
    Tensor() noexcept = default;
    // 创建全0张量
    static Tensor zeros(Shape shape, DType dtype = DType::FP32);
    // 创建全value张量
    static Tensor full(Shape shape, float value);
    // 创建全1张量
    static Tensor ones(Shape shape, DType dtype = DType::FP32);
    // 创建等差数列张量 [0, 1, 2, ..., n-1]
    static Tensor arange(int64 n);
    // 从数组创建张量
    static Tensor from(Shape shape, std::span<const float> data);
    // 从初始化列表创建张量
    static Tensor from(Shape shape, std::initializer_list<float> data);
    //=============================================================================
    // 获取张量元信息
    //=============================================================================
    // 获取张量形状
    [[nodiscard]] const Shape &shape() const noexcept {
        return shape_;
    }
    // 获取张量步长
    [[nodiscard]] const Shape &strides() const noexcept {
        return strides_;
    }
    // 获取张量维度
    [[nodiscard]] int64 dim(int64 i) const noexcept {
        return shape_.at(static_cast<std::size_t>(i));
    }
    // 获取张量维度数量
    [[nodiscard]] int64 ndim() const noexcept {
        return static_cast<int64>(shape_.size());
    }
    // 获取张量元素数量
    [[nodiscard]] int64 numel() const noexcept {
        return numel_;
    }
    // 获取张量数据类型
    [[nodiscard]] DType dtype() const noexcept {
        return dtype_;
    }
    // 获取张量是否为空
    [[nodiscard]] bool empty() const noexcept {
        return numel_ == 0;
    }
    [[nodiscard]] bool is_contiguous() const noexcept;
    //=============================================================================
    // 获取张量数据
    //=============================================================================
    [[nodiscard]] std::span<float> data();
    [[nodiscard]] std::span<const float> data() const;

    // 获取张量元素
    template <std::integral... Index> [[nodiscard]] float &operator()(Index... index) {
        const int64 raw[] = {static_cast<int64>(index)...};
        return raw_ptr()[offset_of(std::span<const int64>{raw, sizeof...(Index)})];
    }
    // 获取张量元素
    template <std::integral... Index> [[nodiscard]] const float &operator()(Index... index) const {
        const int64 raw[] = {static_cast<int64>(index)...};
        return raw_ptr()[offset_of(std::span<const int64>{raw, sizeof...(Index)})];
    }
    // 重塑张量形状。new_shape 中可以有一个 -1，表示由元素总数自动推断该维。
    [[nodiscard]] Tensor reshape(Shape new_shape) const;
    // 转置张量
    [[nodiscard]] Tensor transpose(int64 a, int64 b) const;
    // 按 new_order 重排轴。new_order 必须是 0..ndim-1 的一个排列（不能缺、不能重复）。
    [[nodiscard]] Tensor permute(Shape new_order) const;

    // 切片张量
    [[nodiscard]] Tensor slice(int64 axis, int64 start, int64 end) const;

    // 唯一会真正拷贝内存的操作：把 view 落实成一块紧凑内存。
    [[nodiscard]] Tensor contiguous() const;
    //=============================================================================
    // 打印张量信息
    //=============================================================================
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string meta_string() const;

  private:
    std::shared_ptr<std::byte[]> data_;
    Shape shape_;
    Shape strides_;
    int64 offset_ = 0;
    int64 numel_ = 0;
    DType dtype_ = DType::FP32;

    [[nodiscard]] float *raw_ptr() noexcept {
        return reinterpret_cast<float *>(data_.get());
    }
    [[nodiscard]] const float *raw_ptr() const noexcept {
        return reinterpret_cast<const float *>(data_.get());
    }
    [[nodiscard]] int64 offset_of(std::span<const int64> index) const;
};

} // namespace inference