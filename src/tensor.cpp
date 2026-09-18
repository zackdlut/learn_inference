#include "inference/tensor.hpp"
#include <algorithm>
#include <format>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

namespace inference {
//=============================================================================
// Shape 相关的函数
//=============================================================================

std::string shape_string(const Shape &shape) {
    std::string ret = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        ret += std::format("{}{}", shape[i], i + 1 < shape.size() ? ", " : "");
    }
    ret += ")";
    return ret;
}

Shape contiguous_strides(const Shape &shape) {
    Shape strides(shape.size());
    int64 acc = 1;
    for (int64 i = static_cast<int64>(strides.size()) - 1; i >= 0; --i) {
        strides[static_cast<std::size_t>(i)] = acc;
        acc *= shape[static_cast<std::size_t>(i)];
    }
    return strides;
}

int64 numel_of(const Shape &shape) {
    return std::accumulate(shape.begin(), shape.end(), int64{1}, std::multiplies<int64>{});
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

//=============================================================================
// 张量 excpetion
//=============================================================================
[[noreturn]] void fail(const std::string &msg) {
    throw std::invalid_argument("inference::Tensor: " + msg);
}
//=============================================================================
// 张量构造函数
//=============================================================================
Tensor Tensor::zeros(Shape shape, DType dtype) {
    if (std::ranges::any_of(shape, [](int64 dim) { return dim < 0; })) {
        fail("shape must be non-negative");
    }
    Tensor t;
    t.numel_ = numel_of(shape);
    t.shape_ = shape;
    t.dtype_ = dtype;
    t.strides_ = contiguous_strides(shape);
    t.shape_ = std::move(shape);
    t.data_ = std::make_shared<std::byte[]>(static_cast<std::size_t>(t.numel_) * dtype_size(dtype));
    return t;
}

Tensor Tensor::full(Shape shape, float value) {
    Tensor t = zeros(std::move(shape), DType::FP32);
    std::ranges::fill(t.data(), value);
    return t;
}

Tensor Tensor::arange(int64 n) {
    Tensor t = zeros({n}, DType::FP32);
    std::span<float> d = t.data();
    for (int64 i = 0; i < n; ++i) {
        d[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    return t;
}

Tensor Tensor::from(Shape shape, std::span<const float> src) {
    Tensor t = zeros(std::move(shape), DType::FP32);
    if (static_cast<int64>(src.size()) != t.numel_) {
        fail(std::format("data size must be {} but got {}", t.numel_,
                         static_cast<int64>(src.size())));
    }
    std::ranges::copy(src, t.data().begin());
    return t;
}

Tensor Tensor::from(Shape shape, std::initializer_list<float> src) {
    return from(std::move(shape), std::span<const float>(src.begin(), src.size()));
}

//=============================================================================
// 获取张量元信息
//=============================================================================
bool Tensor::is_contiguous() const noexcept {
    return strides_ == contiguous_strides(shape_);
}

//=============================================================================
// 获取张量数据
//=============================================================================
std::span<float> Tensor::data() {
    if (!is_contiguous()) {
        fail("tensor is not contiguous");
    }
    return {raw_ptr() + offset_, static_cast<std::size_t>(numel_)};
}

std::span<const float> Tensor::data() const {
    if (!is_contiguous()) {
        fail("tensor is not contiguous");
    }
    return {raw_ptr() + offset_, static_cast<std::size_t>(numel_)};
}

int64 Tensor::offset_of(std::span<const int64> index) const {
    if (static_cast<int64>(index.size()) != ndim()) {
        fail("index size must be equal to shape size");
    }
    // 核心公式：线性偏移 = Σ 下标[i] * 步长[i]
    int64 offset = 0;
    for (int64 i = 0; i < static_cast<int64>(index.size()); ++i) {
        if (index[i] < 0 || index[i] >= shape_[i]) {
            fail("index out of range");
        }
        offset += index[i] * strides_[i];
    }
    return offset;
}

Tensor Tensor::reshape(Shape new_shape) const {
    if (!is_contiguous()) {
        fail("非连续张量不能 reshape，请先调用 contiguous()");
    }
    if (numel_of(new_shape) != numel_) {
        fail(std::format("reshape {} -> {} 元素个数不一致（{} vs {}）", shape_string(shape_),
                         shape_string(new_shape), numel_, numel_of(new_shape)));
    }

    Tensor t = *this;
    t.strides_ = contiguous_strides(new_shape);
    t.shape_ = std::move(new_shape);
    return t;
}

Tensor Tensor::transpose(int64 a, int64 b) const {
    if (a < 0 || a >= ndim() || b < 0 || b >= ndim()) {
        fail("index out of range");
    }

    Tensor t = *this;
    std::swap(t.strides_[a], t.strides_[b]);
    std::swap(t.shape_[a], t.shape_[b]);
    return t;
}
// 公式：线性偏移 = Σ 下标[i] * 步长[i]，其中 i 是 axis 的索引 offset_ 为起始偏移
Tensor Tensor::slice(int64 axis, int64 start, int64 end) const {
    if (axis < 0 || axis >= ndim()) {
        fail("index out of range");
    }
    if (start < 0 || start >= shape_[axis] || end < 0 || end >= shape_[axis]) {
        fail("index out of range");
    }
    if (start >= end) {
        fail("start must be less than end");
    }
    Tensor t = *this;
    t.offset_ += start * strides_[axis];
    t.shape_[axis] = end - start;
    t.numel_ = numel_of(t.shape_);

    return t;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this; // 已经紧凑，不用白拷一遍
    }
    Tensor out = zeros(shape_, dtype_);
    std::span<float> dst = out.data();

    Shape idx(ndim(), 0);
    for (int64 i = 0; i < numel_; i++) {
        int64 temp = i;
        for (int64 j = ndim() - 1; j >= 0; --j) {
            idx[j] = temp % shape_[j];
            temp /= shape_[j];
        }
        dst[static_cast<std::size_t>(i)] = raw_ptr()[offset_of(idx)];
    }

    return out;
}

std::string Tensor::meta_string() const {
    return std::format("Tensor(shape={}, strides={}, offset={}, numel={}, dtype={}, contiguous={})",
                       shape_string(shape_), shape_string(strides_), offset_, numel_,
                       dtype_name(dtype_), is_contiguous());
}

std::string Tensor::to_string() const {
    std::string out = meta_string() + "\n";
    if (numel_ == 0)
        return out + "[]";

    // 逐元素遍历（用 offset_of 所以非连续张量也能正确打印），
    // 每换一行插一个换行符，方便肉眼核对矩阵。
    const int64 last = ndim() > 0 ? shape_.back() : 1;
    Shape idx(static_cast<std::size_t>(ndim()), 0);
    for (int64 linear = 0; linear < numel_; ++linear) {
        if (linear % last == 0)
            out += "  [ ";
        out += std::format("{:8.4f} ", raw_ptr()[offset_of(idx)]);
        if ((linear + 1) % last == 0)
            out += "]\n";

        for (int64 d = ndim() - 1; d >= 0; --d) {
            const auto ud = static_cast<std::size_t>(d);
            if (++idx[ud] < shape_[ud])
                break;
            idx[ud] = 0;
        }
    }
    return out;
}

} // namespace inference