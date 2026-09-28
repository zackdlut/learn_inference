#include <cmath>
#include <random>

#include "inference/kernel.hpp"
#include "inference/ops.hpp"
#include "inference/tensor.hpp"
#include "minitest.hpp"

using inference::Tensor;
namespace ops = inference::ops;

// ---------------------------------------------------------------------------
// 逐元素
// ---------------------------------------------------------------------------

TEST(elementwise_add_mul) {
    Tensor a = Tensor::from({2, 2}, {1, 2, 3, 4});
    Tensor b = Tensor::from({2, 2}, {10, 20, 30, 40});

    Tensor s = ops::add(a, b);
    CHECK_NEAR(s(0, 0), 11.0, 1e-6);
    CHECK_NEAR(s(1, 1), 44.0, 1e-6);

    Tensor p = ops::mul(a, b);
    CHECK_NEAR(p(0, 1), 40.0, 1e-6);
}

TEST(shape_mismatch_throws) {
    Tensor a = Tensor::zeros({2, 3});
    Tensor b = Tensor::zeros({3, 2});
    CHECK_THROWS(ops::add(a, b));
}

TEST(relu_clips_negatives) {
    Tensor x = Tensor::from({5}, {-2, -0.5F, 0, 0.5F, 2});
    Tensor y = ops::relu(x);
    CHECK_NEAR(y(0), 0.0, 1e-6);
    CHECK_NEAR(y(1), 0.0, 1e-6);
    CHECK_NEAR(y(3), 0.5, 1e-6);
    CHECK_NEAR(y(4), 2.0, 1e-6);
}

TEST(gelu_known_values) {
    Tensor x = Tensor::from({3}, {-1, 0, 1});
    Tensor y = ops::gelu(x);
    CHECK_NEAR(y(0), -0.15880, 1e-4);
    CHECK_NEAR(y(1), 0.0, 1e-6);
    CHECK_NEAR(y(2), 0.84119, 1e-4);
}

TEST(gelu_tanh_approx_matches_exact) {
    // tanh 近似版和 erf 精确版的误差应在 1e-3 量级 —— 这就是 GPT-2 接受的精度
    Tensor x = Tensor::zeros({201});
    for (inference::int64 i = 0; i < 201; ++i)
        x(i) = static_cast<float>(i - 100) * 0.05F;

    Tensor approx = Tensor::zeros({201});
    Tensor exact = Tensor::zeros({201});
    inference::kernel::gelu_f32(x.data(), approx.data());
    inference::kernel::gelu_exact_f32(x.data(), exact.data());

    CHECK(ops::max_abs_diff(approx, exact) < 1e-3F);
}

// ---------------------------------------------------------------------------
// softmax：数值稳定性是本阶段最重要的一课
// ---------------------------------------------------------------------------

TEST(softmax_basic) {
    Tensor x = Tensor::from({3}, {1, 2, 3});
    Tensor y = ops::softmax(x);
    CHECK_NEAR(y(0), 0.09003, 1e-5);
    CHECK_NEAR(y(1), 0.24473, 1e-5);
    CHECK_NEAR(y(2), 0.66524, 1e-5);
    CHECK_NEAR(y(0) + y(1) + y(2), 1.0, 1e-6); // 概率之和必须是 1
}

TEST(softmax_survives_huge_inputs) {
    // 【关键测试】直接算 exp(1002) 会溢出成 inf，inf/inf = nan。
    // 减去最大值之后，结果应该和 softmax({1,2,3}) 完全一样 —— 因为
    // softmax 只关心元素之间的差值，整体平移不改变结果。
    Tensor x = Tensor::from({3}, {1000, 1001, 1002});
    Tensor y = ops::softmax(x);

    CHECK(!std::isnan(y(0)));
    CHECK(!std::isinf(y(0)));
    CHECK_NEAR(y(0), 0.09003, 1e-5);
    CHECK_NEAR(y(2), 0.66524, 1e-5);
}

TEST(softmax_handles_negatives) {
    Tensor x = Tensor::from({3}, {-1000, -1001, -1002});
    Tensor y = ops::softmax(x);
    CHECK(!std::isnan(y(0)));
    CHECK_NEAR(y(0), 0.66524, 1e-5); // 这次最大的是第 0 个
}

TEST(softmax_is_row_wise) {
    // (2,3) 应当被当成 2 行分别做 softmax，而不是把 6 个数一起归一化
    Tensor x = Tensor::from({2, 3}, {1, 2, 3, 1, 2, 3});
    Tensor y = ops::softmax(x);
    CHECK_NEAR(y(0, 0) + y(0, 1) + y(0, 2), 1.0, 1e-6);
    CHECK_NEAR(y(1, 0) + y(1, 1) + y(1, 2), 1.0, 1e-6);
}

// ---------------------------------------------------------------------------
// 归一化
// ---------------------------------------------------------------------------

TEST(layernorm_produces_zero_mean_unit_var) {
    Tensor x = Tensor::from({3}, {1, 2, 3});
    Tensor w = Tensor::full({3}, 1.0F);
    Tensor b = Tensor::zeros({3});
    Tensor y = ops::layernorm(x, w, b);

    // mean=2, var=2/3, std≈0.8165  →  (-1.2247, 0, 1.2247)
    CHECK_NEAR(y(0), -1.22474, 1e-4);
    CHECK_NEAR(y(1), 0.0, 1e-5);
    CHECK_NEAR(y(2), 1.22474, 1e-4);
    CHECK_NEAR(y(0) + y(1) + y(2), 0.0, 1e-5); // 均值归零
}

TEST(layernorm_applies_weight_and_bias) {
    Tensor x = Tensor::from({3}, {1, 2, 3});
    Tensor w = Tensor::full({3}, 2.0F);
    Tensor b = Tensor::full({3}, 10.0F);
    Tensor y = ops::layernorm(x, w, b);
    CHECK_NEAR(y(1), 10.0, 1e-5); // 0 * 2 + 10
    CHECK_NEAR(y(2), 2 * 1.22474 + 10, 1e-4);
}

TEST(layernorm_constant_row_does_not_divide_by_zero) {
    // 一整行数字全一样时方差是 0，没有 eps 就会除零得到 nan
    Tensor x = Tensor::full({4}, 7.0F);
    Tensor w = Tensor::full({4}, 1.0F);
    Tensor b = Tensor::zeros({4});
    Tensor y = ops::layernorm(x, w, b);
    CHECK(!std::isnan(y(0)));
    CHECK_NEAR(y(0), 0.0, 1e-5);
}

TEST(rmsnorm_basic) {
    Tensor x = Tensor::from({3}, {1, 2, 3});
    Tensor w = Tensor::full({3}, 1.0F);
    Tensor y = ops::rmsnorm(x, w);
    // rms = sqrt((1+4+9)/3) = sqrt(4.6667) = 2.1602
    CHECK_NEAR(y(0), 1.0 / 2.16025, 1e-4);
    CHECK_NEAR(y(2), 3.0 / 2.16025, 1e-4);
}

// ---------------------------------------------------------------------------
// 矩阵乘法
// ---------------------------------------------------------------------------

TEST(matmul_hand_computed) {
    // [1 2] × [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    Tensor a = Tensor::from({2, 2}, {1, 2, 3, 4});
    Tensor b = Tensor::from({2, 2}, {5, 6, 7, 8});
    Tensor c = ops::matmul(a, b);
    CHECK_NEAR(c(0, 0), 19.0, 1e-5);
    CHECK_NEAR(c(0, 1), 22.0, 1e-5);
    CHECK_NEAR(c(1, 0), 43.0, 1e-5);
    CHECK_NEAR(c(1, 1), 50.0, 1e-5);
}

TEST(matmul_non_square) {
    // (2,3) × (3,2) -> (2,2)
    Tensor a = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b = Tensor::from({3, 2}, {1, 2, 3, 4, 5, 6});
    Tensor c = ops::matmul(a, b);
    CHECK((c.shape() == inference::Shape{2, 2}));
    CHECK_NEAR(c(0, 0), 22.0, 1e-5); // 1*1+2*3+3*5
    CHECK_NEAR(c(0, 1), 28.0, 1e-5); // 1*2+2*4+3*6
    CHECK_NEAR(c(1, 0), 49.0, 1e-5); // 4*1+5*3+6*5
    CHECK_NEAR(c(1, 1), 64.0, 1e-5); // 4*2+5*4+6*6
}

TEST(matmul_shape_mismatch_throws) {
    CHECK_THROWS(ops::matmul(Tensor::zeros({2, 3}), Tensor::zeros({4, 5})));
    CHECK_THROWS(ops::matmul(Tensor::zeros({2, 3, 4}), Tensor::zeros({4, 5})));
}

TEST(matmul_identity) {
    // 乘单位矩阵应该原样返回，这是最省事的自检手段
    Tensor a = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor eye = Tensor::zeros({3, 3});
    for (int i = 0; i < 3; ++i)
        eye(i, i) = 1.0F;
    CHECK(ops::allclose(ops::matmul(a, eye), a));
}

TEST(matmul_loop_orders_agree) {
    // ijk 和 ikj 只是循环顺序不同，结果必须逐位一致（浮点累加顺序不同，
    // 所以给一点容差而不是要求完全相等 —— 这本身也是个知识点：
    // 浮点加法不满足结合律，(a+b)+c ≠ a+(b+c)）。
    constexpr inference::int64 M = 37, N = 41, K = 53;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    Tensor a = Tensor::zeros({M, K});
    Tensor b = Tensor::zeros({K, N});
    for (float &v : a.data())
        v = dist(rng);
    for (float &v : b.data())
        v = dist(rng);

    Tensor c1 = Tensor::zeros({M, N});
    Tensor c2 = Tensor::zeros({M, N});
    inference::kernel::matmul_f32_ijk(a.data(), b.data(), c1.data(), M, N, K);
    inference::kernel::matmul_f32_ikj(a.data(), b.data(), c2.data(), M, N, K);

    CHECK(ops::max_abs_diff(c1, c2) < 1e-4F);
}

TEST(matmul_matches_transposed_version) {
    // matmul(a, b) 应该等于 matmul_nt(a, bᵀ)
    constexpr inference::int64 M = 5, N = 7, K = 3;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-2.0F, 2.0F);

    Tensor a = Tensor::zeros({M, K});
    Tensor b = Tensor::zeros({K, N});
    for (float &v : a.data())
        v = dist(rng);
    for (float &v : b.data())
        v = dist(rng);

    Tensor c1 = ops::matmul(a, b);
    Tensor bt = b.transpose(0, 1).contiguous(); // (N, K)
    Tensor c2 = ops::linear(a, bt);

    CHECK(ops::max_abs_diff(c1, c2) < 1e-5F);
}

// ---------------------------------------------------------------------------
// linear
// ---------------------------------------------------------------------------

TEST(linear_with_bias) {
    Tensor x = Tensor::from({1, 2}, {1, 2});
    Tensor w = Tensor::from({3, 2}, {1, 0,   // 输出通道 0
                                     0, 1,   // 输出通道 1
                                     1, 1}); // 输出通道 2
    Tensor bias = Tensor::from({3}, {1, 2, 3});
    Tensor y = ops::linear(x, w, bias);

    CHECK((y.shape() == inference::Shape{1, 3}));
    CHECK_NEAR(y(0, 0), 2.0, 1e-5); // 1*1+2*0 + 1
    CHECK_NEAR(y(0, 1), 4.0, 1e-5); // 1*0+2*1 + 2
    CHECK_NEAR(y(0, 2), 6.0, 1e-5); // 1*1+2*1 + 3
}

TEST(linear_batch) {
    Tensor x = Tensor::from({2, 2}, {1, 2, 3, 4});
    Tensor w = Tensor::from({2, 2}, {1, 1, 0, 1});
    Tensor bias = Tensor::from({2}, {0, 0});
    Tensor y = ops::linear(x, w, bias);
    CHECK_NEAR(y(0, 0), 3.0, 1e-5); // 1+2
    CHECK_NEAR(y(0, 1), 2.0, 1e-5); // 2
    CHECK_NEAR(y(1, 0), 7.0, 1e-5); // 3+4
    CHECK_NEAR(y(1, 1), 4.0, 1e-5); // 4
}

TEST(linear_bad_bias_throws) {
    CHECK_THROWS(ops::linear(Tensor::zeros({1, 2}), Tensor::zeros({3, 2}), Tensor::zeros({5})));
}

// ---------------------------------------------------------------------------
// 杂项
// ---------------------------------------------------------------------------

TEST(argmax_finds_index) {
    Tensor x = Tensor::from({5}, {0.1F, 0.7F, 0.05F, 0.9F, 0.2F});
    CHECK(ops::argmax(x) == 3);
}

TEST(ops_accept_non_contiguous_input) {
    // ops 层应当自动处理非连续输入，调用方不需要操心
    Tensor a = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor at = a.transpose(0, 1); // (3,2)，非连续
    Tensor b = Tensor::from({2, 4}, {1, 0, 0, 1, 0, 1, 1, 0});
    Tensor c = ops::matmul(at, b); // (3,2) × (2,4) -> (3,4)
    CHECK((c.shape() == inference::Shape{3, 4}));
    CHECK_NEAR(c(0, 0), 1.0, 1e-5); // at[0]=[1,4] · b[:,0]=[1,0] = 1
    CHECK_NEAR(c(0, 1), 4.0, 1e-5); // [1,4] · [0,1] = 4
}
