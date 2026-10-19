#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "inference/kernel.hpp"
#include "minitest.hpp"

namespace kernel = inference::kernel;
using inference::kernel::int64;

namespace
{

std::vector<float> random_vec(std::size_t n, unsigned seed, float lo = -1.0F, float hi = 1.0F)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (float &x : v)
        x = dist(rng);
    return v;
}

float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b)
{
    float m = 0.0F;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

float sum_of(const std::vector<float> &v)
{
    float s = 0.0F;
    for (float x : v)
        s += x;
    return s;
}

bool all_finite(const std::vector<float> &v)
{
    return std::ranges::all_of(v, [](float x) { return std::isfinite(x); });
}

} // namespace

// ---------------------------------------------------------------------------
// 逐元素
// ---------------------------------------------------------------------------

TEST(kernel_add_f32)
{
    const std::vector<float> a{1, 2, 3, -4};
    const std::vector<float> b{10, 20, 30, 4};
    std::vector<float> out(4);
    kernel::add_f32(a, b, out);
    CHECK((out == std::vector<float>{11, 22, 33, 0}));
}

TEST(kernel_mul_f32)
{
    const std::vector<float> a{1, 2, 3, -4};
    const std::vector<float> b{10, 0.5F, 0, 4};
    std::vector<float> out(4);
    kernel::mul_f32(a, b, out);
    CHECK((out == std::vector<float>{10, 1, 0, -16}));
}

TEST(kernel_add_scalar_f32)
{
    const std::vector<float> a{1, -2, 0};
    std::vector<float> out(3);
    kernel::add_scalar_f32(a, 2.5F, out);
    CHECK((out == std::vector<float>{3.5F, 0.5F, 2.5F}));
}

TEST(kernel_elementwise_in_place)
{
    // out 和输入是同一块内存时也要正确：逐元素运算只读写同一下标
    std::vector<float> a{1, 2, 3};
    const std::vector<float> b{4, 5, 6};
    kernel::add_f32(a, b, a);
    CHECK((a == std::vector<float>{5, 7, 9}));
    kernel::relu_f32(std::vector<float>{-1, 0, 2}, a);
    CHECK((a == std::vector<float>{0, 0, 2}));
}

TEST(kernel_relu_f32)
{
    const std::vector<float> x{-3, -0.0F, 0, 1e-7F, 5};
    std::vector<float> out(x.size());
    kernel::relu_f32(x, out);
    CHECK_NEAR(out[0], 0.0, 0.0);
    CHECK_NEAR(out[1], 0.0, 0.0);
    CHECK_NEAR(out[2], 0.0, 0.0);
    CHECK_NEAR(out[3], 1e-7, 1e-12);
    CHECK_NEAR(out[4], 5.0, 0.0);
}

TEST(kernel_gelu_known_values)
{
    const std::vector<float> x{-3, -1, 0, 1, 3};
    std::vector<float> approx(x.size());
    std::vector<float> exact(x.size());
    kernel::gelu_f32(x, approx);
    kernel::gelu_exact_f32(x, exact);

    // 精确值: x * Φ(x)
    CHECK_NEAR(exact[0], -0.0040497, 1e-6);
    CHECK_NEAR(exact[1], -0.1586553, 1e-6);
    CHECK_NEAR(exact[2], 0.0, 0.0);
    CHECK_NEAR(exact[3], 0.8413447, 1e-6);
    CHECK_NEAR(exact[4], 2.9959503, 1e-6);

    CHECK(max_abs_diff(approx, exact) < 1e-3F);
}

TEST(kernel_gelu_asymptotics)
{
    // 大正数 → 恒等，大负数 → 0；两个版本都不应出现 nan
    const std::vector<float> x{-20, 20};
    std::vector<float> approx(2);
    std::vector<float> exact(2);
    kernel::gelu_f32(x, approx);
    kernel::gelu_exact_f32(x, exact);
    CHECK_NEAR(approx[0], 0.0, 1e-6);
    CHECK_NEAR(approx[1], 20.0, 1e-5);
    CHECK_NEAR(exact[0], 0.0, 1e-6);
    CHECK_NEAR(exact[1], 20.0, 1e-5);
}

TEST(kernel_gelu_simd_matches_std_tanh)
{
    // 元素数不是 8 的倍数：前面走 AVX2（自己实现的 exp/tanh），尾巴走标量 std::tanh。
    // 和直接用 std::tanh 的公式逐个比，覆盖 [-12, 12] 以及原地计算（x 和 out 同一块内存）
    constexpr std::size_t n = 24001;
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = -12.0F + 24.0F * static_cast<float>(i) / static_cast<float>(n - 1);
    std::vector<float> expect(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const float v = x[i];
        expect[i] = 0.5F * v * (1.0F + std::tanh(0.7978845608028654F * (v + 0.044715F * v * v * v)));
    }
    std::vector<float> got = x;
    kernel::gelu_f32(got, got);
    const float diff = max_abs_diff(got, expect);
    PRINT("gelu SIMD vs std::tanh 最大绝对误差 {:.2e}", diff);
    CHECK(diff < 2e-6F);
    CHECK(std::none_of(got.begin(), got.end(), [](float v) { return std::isnan(v); }));
}

TEST(kernel_attention_strided_matches_contiguous)
{
    // q/k/v 交错存放在一块 [T, 3C] 里（c_attn 的输出布局），按行距 3C 读，
    // 结果必须和拆成三个连续 [T, C] 后调普通版完全一致
    constexpr int64 T = 7, n_head = 3, hs = 8, C = n_head * hs;
    const auto qkv = random_vec(static_cast<std::size_t>(T * 3 * C), 77);
    std::vector<float> q(T * C), k(T * C), v(T * C);
    for (int64 t = 0; t < T; ++t)
        for (int64 j = 0; j < C; ++j)
        {
            q[t * C + j] = qkv[t * 3 * C + j];
            k[t * C + j] = qkv[t * 3 * C + C + j];
            v[t * C + j] = qkv[t * 3 * C + 2 * C + j];
        }
    std::vector<float> ref(T * C), got(T * C);
    kernel::attention_f32(q, k, v, ref, T, T, n_head, hs, 0);
    kernel::attention_f32_strided(qkv.data(), 3 * C, qkv.data() + C, 3 * C, qkv.data() + 2 * C,
                                  3 * C, got.data(), C, T, T, n_head, hs, 0);
    CHECK(got == ref);
}

// ---------------------------------------------------------------------------
// softmax
// ---------------------------------------------------------------------------

TEST(kernel_softmax_basic)
{
    const std::vector<float> x{1, 2, 3};
    std::vector<float> out(3);
    kernel::softmax_f32(x, out);
    CHECK_NEAR(out[0], 0.0900306, 1e-6);
    CHECK_NEAR(out[1], 0.2447285, 1e-6);
    CHECK_NEAR(out[2], 0.6652410, 1e-6);
    CHECK_NEAR(sum_of(out), 1.0, 1e-6);
}

TEST(kernel_softmax_shift_invariant)
{
    // softmax(x + c) == softmax(x)，c 再大也不能溢出
    const auto x = random_vec(64, 1, -5.0F, 5.0F);
    std::vector<float> shifted(x.size());
    kernel::add_scalar_f32(x, 1000.0F, shifted);

    std::vector<float> y1(x.size());
    std::vector<float> y2(x.size());
    kernel::softmax_f32(x, y1);
    kernel::softmax_f32(shifted, y2);

    CHECK(all_finite(y2));
    CHECK(max_abs_diff(y1, y2) < 1e-5F);
}

TEST(kernel_softmax_uniform_and_single)
{
    std::vector<float> out(4);
    kernel::softmax_f32(std::vector<float>(4, 3.0F), out);
    CHECK((out == std::vector<float>(4, 0.25F)));

    std::vector<float> one(1);
    kernel::softmax_f32(std::vector<float>{-123.0F}, one);
    CHECK_NEAR(one[0], 1.0, 0.0);
}

TEST(kernel_softmax_with_neg_inf_mask)
{
    // attention 的 causal mask 会把位置填成 -inf，对应概率必须是精确的 0
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    const std::vector<float> x{0, kNegInf, 0, kNegInf};
    std::vector<float> out(4);
    kernel::softmax_f32(x, out);
    CHECK_NEAR(out[0], 0.5, 1e-7);
    CHECK_NEAR(out[1], 0.0, 0.0);
    CHECK_NEAR(out[2], 0.5, 1e-7);
    CHECK_NEAR(out[3], 0.0, 0.0);
}

TEST(kernel_softmax_empty_is_noop)
{
    std::vector<float> out;
    kernel::softmax_f32(std::vector<float>{}, out);
    CHECK(out.empty());
}

TEST(kernel_softmax_in_place)
{
    std::vector<float> x{1, 2, 3};
    kernel::softmax_f32(x, x);
    CHECK_NEAR(x[2], 0.6652410, 1e-6);
    CHECK_NEAR(sum_of(x), 1.0, 1e-6);
}

// ---------------------------------------------------------------------------
// 归一化
// ---------------------------------------------------------------------------

TEST(kernel_layernorm_zero_mean_unit_var)
{
    const auto x = random_vec(128, 2, -3.0F, 7.0F);
    const std::vector<float> w(x.size(), 1.0F);
    const std::vector<float> b(x.size(), 0.0F);
    std::vector<float> out(x.size());
    kernel::layernorm_f32(x, w, b, out, 1e-5F);

    const auto n = static_cast<float>(out.size());
    const float mean = sum_of(out) / n;
    float var = 0.0F;
    for (float v : out)
        var += (v - mean) * (v - mean);
    var /= n;
    CHECK_NEAR(mean, 0.0, 1e-5);
    CHECK_NEAR(var, 1.0, 1e-3);
}

TEST(kernel_layernorm_weight_bias_per_channel)
{
    // x={1,2,3}: mean=2, std=sqrt(2/3) → 归一化后 {-1.22474, 0, 1.22474}
    const std::vector<float> x{1, 2, 3};
    const std::vector<float> w{1, 2, 3};
    const std::vector<float> b{0, 10, -1};
    std::vector<float> out(3);
    kernel::layernorm_f32(x, w, b, out, 0.0F);
    CHECK_NEAR(out[0], -1.2247449, 1e-5);
    CHECK_NEAR(out[1], 10.0, 1e-5);
    CHECK_NEAR(out[2], 3 * 1.2247449 - 1, 1e-5);
}

TEST(kernel_layernorm_large_mean_is_stable)
{
    // 均值远大于方差：用 E[x²]-E[x]² 的写法在 float 下会彻底失真
    const std::vector<float> x{10000.0F, 10001.0F, 10002.0F};
    const std::vector<float> w(3, 1.0F);
    const std::vector<float> b(3, 0.0F);
    std::vector<float> out(3);
    kernel::layernorm_f32(x, w, b, out, 0.0F);
    CHECK_NEAR(out[0], -1.2247449, 1e-4);
    CHECK_NEAR(out[1], 0.0, 1e-4);
    CHECK_NEAR(out[2], 1.2247449, 1e-4);
}

TEST(kernel_layernorm_constant_row_uses_eps)
{
    const std::vector<float> x(8, 5.0F);
    const std::vector<float> w(8, 1.0F);
    const std::vector<float> b(8, 0.5F);
    std::vector<float> out(8);
    kernel::layernorm_f32(x, w, b, out, 1e-5F);
    CHECK(all_finite(out));
    CHECK_NEAR(out[0], 0.5, 1e-6);
}

TEST(kernel_rmsnorm_basic)
{
    // rms = sqrt((1+4+9)/3) = sqrt(14/3)
    const std::vector<float> x{1, 2, 3};
    const std::vector<float> w{1, 1, 2};
    std::vector<float> out(3);
    kernel::rmsnorm_f32(x, w, out, 0.0F);
    const double rms = std::sqrt(14.0 / 3.0);
    CHECK_NEAR(out[0], 1.0 / rms, 1e-6);
    CHECK_NEAR(out[1], 2.0 / rms, 1e-6);
    CHECK_NEAR(out[2], 2.0 * 3.0 / rms, 1e-6);
}

TEST(kernel_rmsnorm_unit_rms_and_scale_invariance)
{
    // 输出的均方根应为 1；输入整体放大 k 倍结果不变（eps 足够小时）
    const auto x = random_vec(96, 3);
    const std::vector<float> w(x.size(), 1.0F);
    std::vector<float> y1(x.size());
    kernel::rmsnorm_f32(x, w, y1, 1e-8F);

    float sum_sq = 0.0F;
    for (float v : y1)
        sum_sq += v * v;
    CHECK_NEAR(std::sqrt(sum_sq / static_cast<float>(y1.size())), 1.0, 1e-5);

    std::vector<float> x_scaled(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        x_scaled[i] = x[i] * 100.0F;
    std::vector<float> y2(x.size());
    kernel::rmsnorm_f32(x_scaled, w, y2, 1e-8F);
    CHECK(max_abs_diff(y1, y2) < 1e-5F);
}

TEST(kernel_rmsnorm_zero_row_uses_eps)
{
    const std::vector<float> x(4, 0.0F);
    const std::vector<float> w(4, 1.0F);
    std::vector<float> out(4, -1.0F);
    kernel::rmsnorm_f32(x, w, out, 1e-6F);
    CHECK((out == std::vector<float>(4, 0.0F)));
}

// ---------------------------------------------------------------------------
// 朴素矩阵乘法（kernel.cpp 里的参考实现）
// ---------------------------------------------------------------------------

TEST(kernel_matmul_ijk_ikj_hand_computed)
{
    // [1 2 3]   [ 7  8]   [ 58  64]
    // [4 5 6] × [ 9 10] = [139 154]
    //           [11 12]
    const std::vector<float> a{1, 2, 3, 4, 5, 6};
    const std::vector<float> b{7, 8, 9, 10, 11, 12};
    const std::vector<float> expect{58, 64, 139, 154};

    std::vector<float> c1(4, -1.0F);
    std::vector<float> c2(4, -1.0F); // 预填垃圾值：ikj 必须自己先清零
    kernel::matmul_f32_ijk(a, b, c1, 2, 2, 3);
    kernel::matmul_f32_ikj(a, b, c2, 2, 2, 3);
    CHECK(c1 == expect);
    CHECK(c2 == expect);
}

TEST(kernel_matmul_nt_naive_hand_computed)
{
    // W 的形状是 [N,K]，C = A × Wᵀ。W 取上一个用例 B 的转置，结果应相同
    const std::vector<float> a{1, 2, 3, 4, 5, 6};
    const std::vector<float> w{7, 9, 11, 8, 10, 12};
    std::vector<float> c(4, -1.0F);
    kernel::matmul_nt_f32_naive(a, w, c, 2, 2, 3);
    CHECK((c == std::vector<float>{58, 64, 139, 154}));
}

TEST(kernel_matmul_degenerate_shapes)
{
    // 向量点积 (1,K)×(K,1)、外积 (M,1)×(1,N)
    const std::vector<float> u{1, 2, 3};
    const std::vector<float> v{4, 5, 6};

    std::vector<float> dot(1);
    kernel::matmul_f32_ijk(u, v, dot, 1, 1, 3);
    CHECK_NEAR(dot[0], 32.0, 0.0);

    std::vector<float> outer(9);
    kernel::matmul_f32_ikj(u, v, outer, 3, 3, 1);
    CHECK((outer == std::vector<float>{4, 5, 6, 8, 10, 12, 12, 15, 18}));

    std::vector<float> outer_nt(9);
    kernel::matmul_nt_f32_naive(u, v, outer_nt, 3, 3, 1);
    CHECK(outer_nt == outer);
}

TEST(kernel_matmul_naive_versions_agree)
{
    constexpr int64 M = 23, N = 31, K = 19;
    const auto a = random_vec(M * K, 10);
    const auto b = random_vec(K * N, 11);

    std::vector<float> bt(N * K);
    for (int64 k = 0; k < K; ++k)
        for (int64 j = 0; j < N; ++j)
            bt[j * K + k] = b[k * N + j];

    std::vector<float> c_ijk(M * N);
    std::vector<float> c_ikj(M * N);
    std::vector<float> c_nt(M * N);
    kernel::matmul_f32_ijk(a, b, c_ijk, M, N, K);
    kernel::matmul_f32_ikj(a, b, c_ikj, M, N, K);
    kernel::matmul_nt_f32_naive(a, bt, c_nt, M, N, K);

    CHECK(max_abs_diff(c_ijk, c_ikj) < 1e-5F);
    CHECK(max_abs_diff(c_ijk, c_nt) < 1e-5F);
}
