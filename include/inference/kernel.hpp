#pragma once

#include <cstdint>
#include <span>

namespace inference::kernel {

using int64 = std::int64_t;

// --- 逐元素（elementwise）-------------------------------------------------
void add_f32(std::span<const float> a, std::span<const float> b, std::span<float> out);
void mul_f32(std::span<const float> a, std::span<const float> b, std::span<float> out);
void add_scalar_f32(std::span<const float> a, float s, std::span<float> out);
void relu_f32(std::span<const float> x, std::span<float> out);
void gelu_f32(std::span<const float> x, std::span<float> out); // tanh 近似版，GPT-2 用的就是这个
void gelu_exact_f32(std::span<const float> x,
                    std::span<float> out); // erf 精确版，用来验证近似版误差

// --- 归约类（reduction）：都只处理「一行」，多行由上层循环 -------------------
// softmax：把一行数字变成和为 1 的概率分布
void softmax_f32(std::span<const float> x, std::span<float> out);
// layernorm：对一行做标准化后再缩放平移。w/b 长度必须等于 x
void layernom_f32(std::span<const float> x, std::span<const float> w, std::span<const float> b,
                  std::span<float> out, float eps);
// rmsnorm：LLaMA 系模型用的简化版归一化，阶段 6 会用到
void rmsnorm_f32(std::span<const float> x, std::span<const float> w, std::span<float> out,
                 float eps);

// --- 矩阵乘法 -------------------------------------------------------------
// C[M,N] = A[M,K] × B[K,N]，全部行优先连续存储。
// 提供两个循环顺序不同、结果完全相同的版本。
// 它们的性能差距是阶段 5 的第一课 —— 跑 examples/02_ops 亲眼看看差多少。
void matmul_f32_ijk(std::span<const float> a, std::span<const float> b, std::span<float> c, int64 M,
                    int64 N, int64 K);
void matmul_f32_ikj(std::span<const float> a, std::span<const float> b, std::span<float> c, int64 M,
                    int64 N, int64 K);
// 默认实现，目前指向较快的那个。阶段 5 会把它换成优化版。
void matmul_f32(std::span<const float> a, std::span<const float> b, std::span<float> c, int64 M,
                int64 N, int64 K);
// C[M,N] = A[M,K] × Wᵀ，其中 W 的形状是 [N,K]。
// 为什么要单独写一个转置版？因为 PyTorch 的 Linear 层权重就存成 [out, in]，
// 而这个布局对 cache 反而更友好：A 和 W 都是按行连续读取的。
void matmul_nt_f32(std::span<const float> a, std::span<const float> w, std::span<float> c, int64 M,
                   int64 N, int64 K);

} // namespace inference::kernel