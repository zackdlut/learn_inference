#pragma once

#include "inference/tensor.hpp"

namespace inference::ops {
// ---------------------------------------------------------------------------
// Ops 层：认识 Tensor，负责
//   1. 校验形状，形状不对就抛带详细信息的异常
//   2. 把非连续输入自动 contiguous 一下
//   3. 分配输出张量
//   4. 调用 kernel 层干实际的计算
//
// 换句话说，ops 层管「对不对」，kernel 层管「快不快」。
// ---------------------------------------------------------------------------
// --- 逐元素：要求两个输入形状完全一致（广播留作练习）-------------------------
[[nodiscard]] Tensor add(const Tensor &a, const Tensor &b);
[[nodiscard]] Tensor mul(const Tensor &a, const Tensor &b);
[[nodiscard]] Tensor add_scalar(const Tensor &a, float s);
[[nodiscard]] Tensor relu(const Tensor &x);
[[nodiscard]] Tensor gelu(const Tensor &x);

// --- 归约：沿最后一维处理，前面的维度全当成「行」------------------------------
// 例如 x 的形状是 (2, 3, 4)，就当作 6 行、每行 4 个元素来做。
[[nodiscard]] Tensor softmax(const Tensor &x);
[[nodiscard]] Tensor layernorm(const Tensor &x, const Tensor &w, const Tensor &b,
                               float eps = 1e-5F);
[[nodiscard]] Tensor rmsnorm(const Tensor &x, const Tensor &w, float eps = 1e-5F);

// --- 矩阵乘法 -------------------------------------------------------------
// a: (M, K), b: (K, N)  ->  (M, N)
[[nodiscard]] Tensor matmul(const Tensor &a, const Tensor &b);

// 全连接层：y = x @ Wᵀ + bias
//   x:    (M, in)
//   w:    (out, in)   ← 注意是 PyTorch 的 nn.Linear 布局，权重按输出通道存一行
//   bias: (out,)
// 阶段 3 加载 PyTorch 权重时直接就能对上，不用转置。
[[nodiscard]] Tensor linear(const Tensor &x, const Tensor &w);
[[nodiscard]] Tensor linear(const Tensor &x, const Tensor &w, const Tensor &bias);

// --- 杂项 -----------------------------------------------------------------
// 返回最大值所在的下标。分类模型最后一步就靠它把概率变成类别。
[[nodiscard]] int64 argmax(const Tensor &x);

// 逐元素比较，调试和对拍时用
[[nodiscard]] bool allclose(const Tensor &a, const Tensor &b, float atol = 1e-5F);

// 返回两个张量间的最大绝对误差，对拍失败时用来看差多少
[[nodiscard]] float max_abs_diff(const Tensor &a, const Tensor &b);

} // namespace inference::ops