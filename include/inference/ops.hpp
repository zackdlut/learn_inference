#pragma once

#include "inference/tensor.hpp"

namespace inference::ops
{
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

// 仿射变换：y = x @ W + bias，注意 W **没有**转置
//   x:    (M, in)
//   w:    (in, out)   ← HuggingFace 的 GPT-2 用 Conv1D，权重就是这个布局
//   bias: (out,)
//
// 和上面的 linear 只差一个转置，但加载 GPT-2 权重时必须用对，
// 否则形状能对上（768×768 是方阵！）却算出一堆垃圾 —— 这是最难查的一类 bug。
[[nodiscard]] Tensor affine(const Tensor &x, const Tensor &w, const Tensor &bias);

// --- 阶段 6：Transformer ----------------------------------------------------

// 词嵌入查表：table 是 (n_vocab, C)，返回 (T, C)。
// 它其实就是 one-hot 向量乘矩阵，但没人真那么算 —— 直接按行拷贝快几千倍。
[[nodiscard]] Tensor embedding(const Tensor &table, std::span<const int> ids);

// 多头因果注意力。q/k/v 形状 (T*, n_head*head_size)，返回和 q 同形。
// q_offset 是 q 第一行的绝对位置，用 KV-Cache 增量解码时不为 0。
[[nodiscard]] Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, int n_head,
                               int q_offset = 0);

// --- 杂项 -----------------------------------------------------------------
// 返回最大值所在的下标。分类模型最后一步就靠它把概率变成类别。
[[nodiscard]] int64 argmax(const Tensor &x);

// 逐元素比较，调试和对拍时用
[[nodiscard]] bool allclose(const Tensor &a, const Tensor &b, float atol = 1e-5F);

// 返回两个张量间的最大绝对误差，对拍失败时用来看差多少
[[nodiscard]] float max_abs_diff(const Tensor &a, const Tensor &b);

} // namespace inference::ops