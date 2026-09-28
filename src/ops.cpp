#include "inference/ops.hpp"
#include "inference/kernel.hpp"
#include "inference/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

namespace inference::ops {

[[noreturn]] void fail(const std::string &msg) {
    throw std::invalid_argument("[inference::ops] " + msg);
}

std::string shape_str(const Shape &s) {
    std::string out = "(";
    for (std::size_t i = 0; i < s.size(); ++i) {
        out += std::format("{}{}", s[i], i + 1 < s.size() ? ", " : "");
    }
    return out + ")";
}

void check_same_shape(const Tensor &a, const Tensor &b, const char *op) {
    if (a.shape() != b.shape()) {
        fail(std::format("{}: 两个输入形状必须一致，但拿到 {} 和 {}", op, shape_str(a.shape()),
                         shape_str(b.shape())));
    }
}

// kernel 层假设输入连续，所以这里统一做一次「保证连续」。
// 已经连续时 contiguous() 直接返回自身，不会有额外拷贝。
Tensor as_contiguous(const Tensor &t) {
    return t.contiguous();
}

struct RowView {
    int64 rows;
    int64 cols;
};

RowView row_view(const Tensor &x, const char *op) {
    if (x.ndim() == 0) {
        fail(std::format("{}: 不能作用在标量上", op));
    }
    const int64 cols = x.shape().back();
    return {cols == 0 ? 0 : x.numel() / cols, cols};
}
// ---------------------------------------------------------------------------
// 逐元素
// ---------------------------------------------------------------------------
Tensor add(const Tensor &a, const Tensor &b) {
    check_same_shape(a, b, "add");
    const Tensor ca = as_contiguous(a);
    const Tensor cb = as_contiguous(b);
    Tensor out = Tensor::zeros(a.shape());
    kernel::add_f32(ca.data(), cb.data(), out.data());
    return out;
}

Tensor mul(const Tensor &a, const Tensor &b) {
    check_same_shape(a, b, "mul");
    const Tensor ca = as_contiguous(a);
    const Tensor cb = as_contiguous(b);
    Tensor out = Tensor::zeros(a.shape());
    kernel::mul_f32(ca.data(), cb.data(), out.data());
    return out;
}

Tensor add_scalar(const Tensor &a, float s) {
    const Tensor ca = as_contiguous(a);
    Tensor out = Tensor::zeros(a.shape());
    kernel::add_scalar_f32(ca.data(), s, out.data());
    return out;
}

Tensor relu(const Tensor &x) {
    const Tensor cx = as_contiguous(x);
    Tensor out = Tensor::zeros(x.shape());
    kernel::relu_f32(cx.data(), out.data());
    return out;
}

Tensor gelu(const Tensor &x) {
    const Tensor cx = as_contiguous(x);
    Tensor out = Tensor::zeros(x.shape());
    kernel::gelu_f32(cx.data(), out.data());
    return out;
}

Tensor softmax(const Tensor &x) {
    const auto [rows, cols] = row_view(x, "softmax");
    const Tensor cx = as_contiguous(x);
    Tensor out = Tensor::zeros(x.shape());
    std::span<const float> x_span = cx.data();
    std::span<float> dst = out.data();
    for (int64 i = 0; i < rows; ++i) {
        const auto off = static_cast<std::size_t>(i * cols);
        const auto n = static_cast<std::size_t>(cols);
        kernel::softmax_f32(x_span.subspan(off, n), dst.subspan(off, n));
    }
    return out;
}

Tensor layernorm(const Tensor &x, const Tensor &w, const Tensor &b, float eps) {
    const auto [rows, cols] = row_view(x, "layernorm");
    if (w.numel() != cols || b.numel() != cols) {
        fail(std::format("layernorm: weight/bias 长度应为 {}，实际是 {} 和 {}", cols, w.numel(),
                         b.numel()));
    }
    check_same_shape(x, w, "layernorm");
    check_same_shape(x, b, "layernorm");
    const Tensor cx = as_contiguous(x);
    const Tensor cw = as_contiguous(w);
    const Tensor cb = as_contiguous(b);
    Tensor out = Tensor::zeros(x.shape());
    std::span<const float> src = cx.data();
    std::span<float> dst = out.data();
    for (int64 r = 0; r < rows; ++r) {
        const auto off = static_cast<std::size_t>(r * cols);
        const auto n = static_cast<std::size_t>(cols);
        kernel::layernom_f32(src.subspan(off, n), cw.data(), cb.data(), dst.subspan(off, n), eps);
    }
    return out;
}

Tensor rmsnorm(const Tensor &x, const Tensor &w, float eps) {
    const auto [rows, cols] = row_view(x, "rmsnorm");
    if (w.numel() != cols) {
        fail(std::format("rmsnorm: weight 长度应为 {}，实际是 {}", cols, w.numel()));
    }
    const Tensor cx = as_contiguous(x);
    const Tensor cw = as_contiguous(w);
    Tensor out = Tensor::zeros(x.shape());

    std::span<const float> src = cx.data();
    std::span<float> dst = out.data();
    for (int64 r = 0; r < rows; ++r) {
        const auto off = static_cast<std::size_t>(r * cols);
        const auto n = static_cast<std::size_t>(cols);
        kernel::rmsnorm_f32(src.subspan(off, n), cw.data(), dst.subspan(off, n), eps);
    }
    return out;
}

// ---------------------------------------------------------------------------
// 矩阵乘法
// ---------------------------------------------------------------------------
Tensor matmul(const Tensor &a, const Tensor &b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        fail(std::format("matmul: 目前只支持二维，拿到 {} 维和 {} 维", a.ndim(), b.ndim()));
    }
    const int64 M = a.dim(0), K = a.dim(1), N = b.dim(1);
    if (b.dim(0) != K) {
        // 这是整个项目里最高频的错误，报错信息一定要把两边形状都打出来
        fail(std::format("matmul: 形状对不上，{} × {} —— "
                         "第一个矩阵的列数({})必须等于第二个矩阵的行数({})",
                         shape_str(a.shape()), shape_str(b.shape()), K, b.dim(0)));
    }
    const Tensor ca = as_contiguous(a);
    const Tensor cb = as_contiguous(b);
    Tensor out = Tensor::zeros({M, N});
    kernel::matmul_f32(ca.data(), cb.data(), out.data(), M, N, K);
    return out;
}
Tensor linear(const Tensor &x, const Tensor &w) {
    if (x.ndim() != 2 || w.ndim() != 2) {
        fail(std::format("linear: x 和 w 都必须是二维，拿到 {} 维和 {} 维", x.ndim(), w.ndim()));
    }
    const int64 M = x.dim(0), K = x.dim(1), N = w.dim(0);
    if (w.dim(1) != K) {
        fail(std::format("linear: x 的特征数是 {}，但权重 {} 期望输入特征数是 {}", K,
                         shape_str(w.shape()), w.dim(1)));
    }
    const Tensor cx = as_contiguous(x);
    const Tensor cw = as_contiguous(w);
    Tensor out = Tensor::zeros({M, N});
    kernel::matmul_nt_f32(cx.data(), cw.data(), out.data(), M, N, K);
    return out;
}
Tensor linear(const Tensor &x, const Tensor &w, const Tensor &bias) {
    Tensor out = linear(x, w);
    const int64 N = out.dim(1);
    if (bias.numel() != N) {
        fail(std::format("linear: bias 长度应为 {}，实际是 {}", N, bias.numel()));
    }
    // bias 是 (N,)，要加到输出的每一行上 —— 这其实就是一次广播。
    // 等你做完阶段 1 的练习 5（broadcast），这段可以直接换成 add()。
    const Tensor cb = as_contiguous(bias);
    std::span<const float> bv = cb.data();
    std::span<float> ov = out.data();
    for (int64 r = 0; r < out.dim(0); ++r) {
        for (int64 j = 0; j < N; ++j) {
            ov[static_cast<std::size_t>(r * N + j)] += bv[static_cast<std::size_t>(j)];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 杂项
// ---------------------------------------------------------------------------
int64 argmax(const Tensor &x) {
    if (x.numel() == 0)
        fail("argmax: 张量是空的");
    const Tensor cx = as_contiguous(x);
    std::span<const float> v = cx.data();
    return static_cast<int64>(std::distance(v.begin(), std::ranges::max_element(v)));
}
float max_abs_diff(const Tensor &a, const Tensor &b) {
    check_same_shape(a, b, "max_abs_diff");
    const Tensor ca = as_contiguous(a);
    const Tensor cb = as_contiguous(b);
    std::span<const float> va = ca.data();
    std::span<const float> vb = cb.data();
    float worst = 0.0F;
    for (std::size_t i = 0; i < va.size(); ++i) {
        worst = std::max(worst, std::fabs(va[i] - vb[i]));
    }
    return worst;
}
bool allclose(const Tensor &a, const Tensor &b, float atol) {
    return max_abs_diff(a, b) <= atol;
}
} // namespace inference::ops