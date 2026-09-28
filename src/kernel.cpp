#include "inference/kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace inference::kernel {
// ---------------------------------------------------------------------------
// 逐元素运算
// ---------------------------------------------------------------------------
void add_f32(std::span<const float> a, std::span<const float> b, std::span<float> out) {
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = a[i] + b[i];
}
void mul_f32(std::span<const float> a, std::span<const float> b, std::span<float> out) {
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = a[i] * b[i];
}
void add_scalar_f32(std::span<const float> a, float s, std::span<float> out) {
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = a[i] + s;
}
void relu_f32(std::span<const float> x, std::span<float> out) {
    // 写成 max(0, x) 而不是 if/else，是因为编译器能把它变成一条无分支的
    // SIMD 指令（vmaxps）。分支预测失败的代价比计算本身大得多。
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = std::max(0.0F, x[i]);
}
void gelu_f32(std::span<const float> x, std::span<float> out) {
    // GELU 的 tanh 近似式（GPT-2 / BERT 的官方实现用的就是这个）：
    //   gelu(x) ≈ 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
    // 直觉：ReLU 在 0 处硬拐弯，GELU 是它的平滑版本，负值不会被一刀切掉。
    constexpr float kSqrt2OverPi = 0.7978845608028654F; // √(2/π)
    constexpr float kCoeff = 0.044715F;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const float v = x[i];
        const float inner = kSqrt2OverPi * (v + kCoeff * v * v * v);
        out[i] = 0.5F * v * (1.0F + std::tanh(inner));
    }
}
void gelu_exact_f32(std::span<const float> x, std::span<float> out) {
    // 精确定义：gelu(x) = x * Φ(x) = 0.5x * (1 + erf(x/√2))
    // 比 tanh 版慢，但可以用来衡量近似版的误差（约 1e-3 量级）。
    const float inv_sqrt2 = 1.0F / std::numbers::sqrt2_v<float>;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = 0.5F * x[i] * (1.0F + std::erf(x[i] * inv_sqrt2));
    }
}

// ---------------------------------------------------------------------------
// 归约运算
// ---------------------------------------------------------------------------
void softmax_f32(std::span<const float> x, std::span<float> out) {
    if (x.empty())
        return;
    // 【数值稳定性：本阶段最重要的一课】
    //
    // softmax 的数学定义是 exp(xᵢ) / Σexp(xⱼ)。直接照抄会炸：
    // float 最大约 3.4e38，而 exp(89) 就已经溢出成 inf 了。
    // 大模型 logits 上百是家常便饭，一炸就得到 inf/inf = nan，
    // 而 nan 会像病毒一样污染后面所有计算，且不会报错。
    //
    // 解决办法：分子分母同时除以 exp(max)，等价于每个 x 先减去最大值。
    //   exp(xᵢ - max) / Σexp(xⱼ - max)
    // 数学上完全等价，但现在指数最大是 exp(0)=1，永远不会溢出。
    const float max_v = *std::max_element(x.begin(), x.end());
    float sum = 0.0F;
    for (std::size_t i = 0; i < x.size(); i++) {
        out[i] = std::exp(x[i] - max_v);
        sum += out[i];
    }
    const float inv_sum = 1.0F / sum;
    for (float &v : out)
        v *= inv_sum;
}

void layernom_f32(std::span<const float> x, std::span<const float> w, std::span<const float> b,
                  std::span<float> out, float eps) {
    const auto n = static_cast<float>(x.size());
    float mean = 0.0F;
    for (float v : x)
        mean += v;
    mean /= n;
    // 这里用「先减均值再平方」而不是 E[x²]-E[x]² 的快捷公式。
    // 后者只需遍历一次，但当均值远大于方差时会发生灾难性抵消，精度全丢。
    // 推理引擎里正确性优先于这点速度。
    float var = 0.0F;
    for (float v : x) {
        const float d = v - mean;
        var += d * d;
    }
    var /= n;
    const float inv_std = 1.0F / std::sqrt(var + eps);
    for (std::size_t i = 0; i < x.size(); i++) {
        out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
    }
}

void rmsnorm_f32(std::span<const float> x, std::span<const float> w, std::span<float> out,
                 float eps) {
    // LayerNorm 的简化版：不减均值，只按均方根缩放。
    // 少算一遍均值，效果几乎一样，所以 LLaMA 之后的模型基本都换成了它
    const auto n = static_cast<float>(x.size());
    float sum_sq = 0.0F;
    for (float v : x)
        sum_sq += v * v;
    const float scale = 1.0F / std::sqrt(sum_sq / n + eps);
    for (std::size_t i = 0; i < x.size(); i++) {
        out[i] = x[i] * scale * w[i];
    }
}

// ---------------------------------------------------------------------------
// 矩阵乘法
// ---------------------------------------------------------------------------
void matmul_f32_ijk(std::span<const float> a, std::span<const float> b, std::span<float> c, int64 M,
                    int64 N, int64 K) {
    // 教科书写法，和数学定义一一对应：C[i][j] = Σₖ A[i][k]·B[k][j]
    //
    // 它慢在哪？看最内层的 b[k*N + j]：k 每加 1，内存地址就跳 N 个 float。
    // N=1024 时每次跳 4KB，等于每读一个数就 miss 一次 cache，
    // CPU 绝大部分时间都在等内存，算力完全闲置。
    for (int64 i = 0; i < M; i++) {
        for (int64 j = 0; j < N; j++) {
            float sum = 0.0F;
            for (int64 k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

void matmul_f32_ikj(std::span<const float> a, std::span<const float> b, std::span<float> c, int64 M,
                    int64 N, int64 K) {
    // 只是把 j 和 k 两层循环换了个顺序，数学结果完全一样，但快好几倍。
    //
    // 现在最内层是 j：b[k*N + j] 和 c[i*N + j] 都是「相邻地址连续读写」。
    // 一次 cache line（64 字节）拉进来能用满 16 个 float，
    // 而且这种模式编译器能自动向量化成 AVX 指令。
    //
    // 代价是不能再用局部变量累加 sum，必须先把 C 清零再累加。
    std::ranges::fill(c, 0.0F);
    for (int64 i = 0; i < M; ++i) {
        for (int64 k = 0; k < K; ++k) {
            const float aik = a[static_cast<std::size_t>(i * K + k)]; // 提到循环外，只读一次
            for (int64 j = 0; j < N; ++j) {
                c[static_cast<std::size_t>(i * N + j)] +=
                    aik * b[static_cast<std::size_t>(k * N + j)];
            }
        }
    }
}
void matmul_f32(std::span<const float> a, std::span<const float> b, std::span<float> c, int64 M,
                int64 N, int64 K) {
    matmul_f32_ikj(a, b, c, M, N, K);
}

void matmul_nt_f32(std::span<const float> a, std::span<const float> w, std::span<float> c, int64 M,
                   int64 N, int64 K) {
    // C[i][j] = Σₖ A[i][k] · W[j][k]   注意 W 的下标是 [j][k] 不是 [k][j]
    //
    // 这个布局下 a 和 w 都沿着 k 连续读，是两个向量的点积。
    // 既 cache 友好，又能直接用 SIMD 做（阶段 5 会改成 FMA 版本）。
    for (int64 i = 0; i < M; ++i) {
        for (int64 j = 0; j < N; ++j) {
            float sum = 0.0F;
            for (int64 k = 0; k < K; ++k) {
                sum +=
                    a[static_cast<std::size_t>(i * K + k)] * w[static_cast<std::size_t>(j * K + k)];
            }
            c[static_cast<std::size_t>(i * N + j)] = sum;
        }
    }
}
} // namespace inference::kernel