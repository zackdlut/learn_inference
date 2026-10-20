#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "inference/kernel.hpp"
#include "minitest.hpp"

namespace kernel = inference::kernel;
using inference::kernel::int64;

namespace
{

using MatmulFn = std::function<void(std::span<const float>, std::span<const float>,
                                    std::span<float>, int64, int64, int64)>;

std::vector<float> random_vec(int64 n, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> v(static_cast<std::size_t>(n));
    for (float &x : v)
        x = dist(rng);
    return v;
}

// 用 double 累加的参考实现，误差远小于被测的 float 版本
std::vector<float> reference_nn(const std::vector<float> &a, const std::vector<float> &b, int64 M,
                                int64 N, int64 K)
{
    std::vector<float> c(static_cast<std::size_t>(M * N));
    for (int64 i = 0; i < M; ++i)
        for (int64 j = 0; j < N; ++j)
        {
            double s = 0.0;
            for (int64 k = 0; k < K; ++k)
                s += static_cast<double>(a[i * K + k]) * b[k * N + j];
            c[i * N + j] = static_cast<float>(s);
        }
    return c;
}

std::vector<float> transpose(const std::vector<float> &b, int64 rows, int64 cols)
{
    std::vector<float> t(b.size());
    for (int64 r = 0; r < rows; ++r)
        for (int64 c = 0; c < cols; ++c)
            t[c * rows + r] = b[r * cols + c];
    return t;
}

float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b)
{
    float m = 0.0F;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const float d = std::abs(a[i] - b[i]);
        if (std::isnan(d))
            return std::numeric_limits<float>::infinity();
        m = std::max(m, d);
    }
    return m;
}

// 能覆盖各条代码路径的形状：
//   M 非 4 的倍数 → 走下方边角；N 非 16/2 的倍数 → 走右侧边角；
//   K 非 8 的倍数 → NT 版标量补尾；任一维 > 64 → 跨越多个 tile
struct Shape3
{
    int64 M, N, K;
};
const std::vector<Shape3> kShapes = {
    {1, 1, 1},   {1, 17, 3},   {3, 5, 7},    {4, 16, 8},    {4, 16, 1},   {5, 17, 9},
    {8, 32, 16}, {7, 33, 13},  {16, 2, 64},  {64, 64, 64},  {65, 63, 66}, {130, 70, 129},
    {2, 200, 5}, {200, 3, 17}, {33, 48, 100},
    // K > 256、N > 512：覆盖 avx2 版沿 K 分段累加、沿 N 分多块打包的路径
    {5, 530, 300}, {13, 517, 513}};

// 输出预填 NaN：既能发现「没写到的元素」，也能发现「忘了清零就累加」
// 容差按 K 放宽：float 累加误差大致随 K 线性增长
void check_nn(const char *name, const MatmulFn &fn, unsigned seed)
{
    float worst = 0.0F;
    for (const auto &[M, N, K] : kShapes)
    {
        const auto a = random_vec(M * K, seed);
        const auto b = random_vec(K * N, seed + 1);
        const auto expect = reference_nn(a, b, M, N, K);
        std::vector<float> c(static_cast<std::size_t>(M * N),
                             std::numeric_limits<float>::quiet_NaN());
        fn(a, b, c, M, N, K);
        const float diff = max_abs_diff(c, expect);
        const auto tol = 1e-6F * static_cast<float>(K + 8);
        if (diff > tol)
            throw minitest::failure(std::string(name) + " 在 M=" + std::to_string(M) +
                                    " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                    " 误差 " + std::to_string(diff) + " > " +
                                    std::to_string(tol));
        worst = std::max(worst, diff / tol);
    }
    PRINT("{}: {} 个形状全部通过，最大误差占容差 {:.1f}%", name, kShapes.size(), worst * 100);
}

void check_nt(const char *name, const MatmulFn &fn, unsigned seed)
{
    float worst = 0.0F;
    for (const auto &[M, N, K] : kShapes)
    {
        const auto a = random_vec(M * K, seed);
        const auto b = random_vec(K * N, seed + 1);
        const auto w = transpose(b, K, N); // [N,K]
        const auto expect = reference_nn(a, b, M, N, K);
        std::vector<float> c(static_cast<std::size_t>(M * N),
                             std::numeric_limits<float>::quiet_NaN());
        fn(a, w, c, M, N, K);
        const float diff = max_abs_diff(c, expect);
        const auto tol = 1e-6F * static_cast<float>(K + 8);
        if (diff > tol)
            throw minitest::failure(std::string(name) + " 在 M=" + std::to_string(M) +
                                    " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                    " 误差 " + std::to_string(diff) + " > " +
                                    std::to_string(tol));
        worst = std::max(worst, diff / tol);
    }
    PRINT("{}: {} 个形状全部通过，最大误差占容差 {:.1f}%", name, kShapes.size(), worst * 100);
}

// 测试里改线程数，退出作用域时恢复，避免影响同进程里的其它用例
struct ThreadGuard
{
    int saved = kernel::get_num_threads();
    explicit ThreadGuard(int n)
    {
        kernel::set_num_threads(n);
    }
    ~ThreadGuard()
    {
        kernel::set_num_threads(saved);
    }
    ThreadGuard(const ThreadGuard &) = delete;
    ThreadGuard &operator=(const ThreadGuard &) = delete;
};

} // namespace

// ---------------------------------------------------------------------------
// 配置接口
// ---------------------------------------------------------------------------

TEST(matmul_build_info)
{
    PRINT("has_avx2_support = {}, hardware threads = {}", kernel::has_avx2_support(),
          kernel::get_num_threads());
    CHECK(kernel::get_num_threads() >= 1);
}

TEST(matmul_set_num_threads_clamps)
{
    ThreadGuard guard(4);
    CHECK(kernel::get_num_threads() == 4);
    kernel::set_num_threads(0);
    CHECK(kernel::get_num_threads() == 1);
    kernel::set_num_threads(-5);
    CHECK(kernel::get_num_threads() == 1);
}

// ---------------------------------------------------------------------------
// C = A × B
// ---------------------------------------------------------------------------

TEST(matmul_tiled_hand_computed)
{
    const std::vector<float> a{1, 2, 3, 4};
    const std::vector<float> b{5, 6, 7, 8};
    std::vector<float> c(4, -1.0F);
    kernel::matmul_f32_tiled(a, b, c, 2, 2, 2);
    CHECK((c == std::vector<float>{19, 22, 43, 50}));
}

TEST(matmul_tiled_matches_reference)
{
    check_nn("matmul_f32_tiled", kernel::matmul_f32_tiled, 100);
}

TEST(matmul_avx2_matches_reference)
{
    check_nn("matmul_f32_avx2", kernel::matmul_f32_avx2, 200);
}

TEST(matmul_avx2_integer_exact)
{
    // 小整数的乘加在 float 里是精确的，FMA 与标量路径必须给出逐位相同的结果
    constexpr int64 M = 9, N = 35, K = 11;
    std::vector<float> a(M * K);
    std::vector<float> b(K * N);
    for (int64 i = 0; i < M * K; ++i)
        a[i] = static_cast<float>(i % 7 - 3);
    for (int64 i = 0; i < K * N; ++i)
        b[i] = static_cast<float>(i % 5 - 2);
    std::vector<float> c(M * N);
    kernel::matmul_f32_avx2(a, b, c, M, N, K);
    CHECK(c == reference_nn(a, b, M, N, K));
}

TEST(matmul_default_single_thread)
{
    ThreadGuard guard(1);
    check_nn("matmul_f32 (1 thread)", kernel::matmul_f32, 300);
}

TEST(matmul_default_multi_thread)
{
    // M*N*K ≥ 2^20 才会真的开线程，所以这里用较大的形状
    ThreadGuard guard(4);
    constexpr int64 M = 130, N = 100, K = 97;
    static_assert(M * N * K >= (1 << 20));
    const auto a = random_vec(M * K, 400);
    const auto b = random_vec(K * N, 401);

    std::vector<float> c(M * N, std::numeric_limits<float>::quiet_NaN());
    kernel::matmul_f32(a, b, c, M, N, K);
    CHECK(max_abs_diff(c, reference_nn(a, b, M, N, K)) < 1e-4F);

    // 切分后每段的「4 行主块 / 边角行」划分会变，同一行可能走 FMA 或标量路径，
    // 所以和单线程结果只能近似相等，不能要求逐位一致
    std::vector<float> c1(M * N);
    kernel::set_num_threads(1);
    kernel::matmul_f32(a, b, c1, M, N, K);
    CHECK(max_abs_diff(c, c1) < 1e-4F);
}

TEST(matmul_more_threads_than_rows)
{
    // 线程数 > M 时，每个线程至多分到一行，不能越界也不能漏行
    ThreadGuard guard(16);
    constexpr int64 M = 5, N = 512, K = 512;
    static_assert(M * N * K >= (1 << 20));
    const auto a = random_vec(M * K, 500);
    const auto b = random_vec(K * N, 501);
    std::vector<float> c(M * N, std::numeric_limits<float>::quiet_NaN());
    kernel::matmul_f32(a, b, c, M, N, K);
    CHECK(max_abs_diff(c, reference_nn(a, b, M, N, K)) < 1e-3F);
}

TEST(matmul_parallel_matches_reference)
{
    // 小形状低于开线程阈值，走串行分支；不同线程数下都必须正确
    for (const int nt : {1, 3, 8})
    {
        ThreadGuard guard(nt);
        check_nn("matmul_f32_parallel", kernel::matmul_f32_parallel, 1100 + nt);
    }
}

TEST(matmul_parallel_single_thread_equals_avx2)
{
    // 单线程时 parallel 直接把整块交给 avx2，结果应逐位一致
    ThreadGuard guard(1);
    constexpr int64 M = 130, N = 100, K = 97;
    const auto a = random_vec(M * K, 1200);
    const auto b = random_vec(K * N, 1201);
    std::vector<float> c_par(M * N);
    std::vector<float> c_avx2(M * N);
    kernel::matmul_f32_parallel(a, b, c_par, M, N, K);
    kernel::matmul_f32_avx2(a, b, c_avx2, M, N, K);
    CHECK(c_par == c_avx2);
}

TEST(matmul_parallel_thread_counts)
{
    // 行数刻意选成不能被线程数整除，检验每段 [lo, hi) 的切分没有越界或漏行
    constexpr int64 M = 131, N = 97, K = 101;
    static_assert(M * N * K >= (1 << 20));
    const auto a = random_vec(M * K, 1300);
    const auto b = random_vec(K * N, 1301);
    const auto expect = reference_nn(a, b, M, N, K);
    for (const int nt : {2, 3, 7, 16, 200})
    {
        ThreadGuard guard(nt);
        std::vector<float> c(M * N, std::numeric_limits<float>::quiet_NaN());
        kernel::matmul_f32_parallel(a, b, c, M, N, K);
        const float diff = max_abs_diff(c, expect);
        PRINT("threads={} max_abs_diff={:.3g}", nt, diff);
        CHECK(diff < 1e-4F);
    }
}

TEST(matmul_all_versions_agree)
{
    constexpr int64 M = 67, N = 45, K = 71;
    const auto a = random_vec(M * K, 600);
    const auto b = random_vec(K * N, 601);
    std::vector<float> c_ikj(M * N);
    std::vector<float> c_tiled(M * N);
    std::vector<float> c_avx2(M * N);
    std::vector<float> c_default(M * N);
    kernel::matmul_f32_ikj(a, b, c_ikj, M, N, K);
    kernel::matmul_f32_tiled(a, b, c_tiled, M, N, K);
    kernel::matmul_f32_avx2(a, b, c_avx2, M, N, K);
    kernel::matmul_f32(a, b, c_default, M, N, K);
    CHECK(max_abs_diff(c_ikj, c_tiled) < 1e-4F);
    CHECK(max_abs_diff(c_ikj, c_avx2) < 1e-4F);
    CHECK(max_abs_diff(c_ikj, c_default) < 1e-4F);
}

// ---------------------------------------------------------------------------
// C = A × Wᵀ
// ---------------------------------------------------------------------------

TEST(matmul_nt_avx2_hand_computed)
{
    // A (2,3) × Wᵀ，W 为 (2,3)
    const std::vector<float> a{1, 2, 3, 4, 5, 6};
    const std::vector<float> w{7, 9, 11, 8, 10, 12};
    std::vector<float> c(4, -1.0F);
    kernel::matmul_nt_f32_avx2(a, w, c, 2, 2, 3);
    CHECK((c == std::vector<float>{58, 64, 139, 154}));
}

TEST(matmul_nt_avx2_matches_reference)
{
    check_nt("matmul_nt_f32_avx2", kernel::matmul_nt_f32_avx2, 700);
}

TEST(gemv_nt_matches_reference)
{
    // 覆盖：N 非 4 的倍数（单行点积尾巴）、K 非 16 / 8 的倍数（标量补尾）、
    // K<8（一次 SIMD 都不走），以及 GPT-2 的真实形状
    const std::vector<std::pair<int64, int64>> shapes = {
        {1, 1},     {3, 5},     {4, 16},    {5, 17},     {7, 24},    {8, 33},
        {13, 7},    {64, 64},   {65, 100},  {130, 129},  {2304, 768}, {768, 3072},
    };
    float worst = 0.0F;
    unsigned seed = 1800;
    for (const auto &[N, K] : shapes)
    {
        const auto x = random_vec(K, seed++);
        const auto b = random_vec(K * N, seed++);
        const auto w = transpose(b, K, N);
        const auto expect = reference_nn(x, b, 1, N, K);
        std::vector<float> y(static_cast<std::size_t>(N) + 8, 12345.0F);
        kernel::gemv_nt_f32(x, w, y, N, K);
        const bool tail_ok =
            std::all_of(y.begin() + N, y.end(), [](float v) { return v == 12345.0F; });
        y.resize(static_cast<std::size_t>(N));
        const float diff = max_abs_diff(y, expect);
        const auto tol = 1e-6F * static_cast<float>(K + 8);
        if (!tail_ok || diff > tol)
            throw minitest::failure("gemv_nt_f32 在 N=" + std::to_string(N) + " K=" +
                                    std::to_string(K) + (tail_ok ? " 误差过大" : " 越界写"));
        worst = std::max(worst, diff / tol);
    }
    PRINT("gemv_nt_f32: {} 个形状全部通过，最大误差占容差 {:.1f}%", shapes.size(), worst * 100);
}

TEST(gemv_nt_equals_matmul_nt_row)
{
    // M=1 的 matmul_nt 现在直接走 gemv，结果应该逐位相同
    constexpr int64 N = 101, K = 77;
    const auto x = random_vec(K, 1900);
    const auto w = random_vec(N * K, 1901);
    std::vector<float> y(N), c(N);
    kernel::gemv_nt_f32(x, w, y, N, K);
    kernel::matmul_nt_f32_avx2(x, w, c, 1, N, K);
    CHECK(y == c);
}

TEST(matmul_nt_parallel_small_m_splits_columns)
{
    // M < 线程数时走「按输出列切」的路径：覆盖 M=1（decode）、M=3/5（短 prompt），
    // N 不是块大小 / 4 的倍数，以及 c 后面多出来的部分不能被写
    constexpr std::size_t kTail = 32;
    constexpr float kSentinel = 12345.0F;
    const std::vector<Shape3> shapes = {
        {1, 768, 768}, {1, 2307, 100}, {3, 1001, 77}, {5, 3072, 768}, {1, 50257, 64}};
    for (const int nt : {2, 4, 7, 14})
    {
        ThreadGuard guard(nt);
        unsigned seed = 2000;
        for (const auto &[M, N, K] : shapes)
        {
            const auto a = random_vec(M * K, seed++);
            const auto w = random_vec(N * K, seed++);
            std::vector<float> expect(static_cast<std::size_t>(M * N));
            kernel::matmul_nt_f32_naive(a, w, expect, M, N, K);
            std::vector<float> c(static_cast<std::size_t>(M * N) + kTail, kSentinel);
            kernel::matmul_nt_f32_parallel(a, w, c, M, N, K);
            const bool tail_ok =
                std::all_of(c.end() - kTail, c.end(), [](float v) { return v == kSentinel; });
            c.resize(static_cast<std::size_t>(M * N));
            const float diff = max_abs_diff(c, expect);
            if (!tail_ok || diff > 1e-6F * static_cast<float>(K + 8))
                throw minitest::failure("threads=" + std::to_string(nt) + " M=" +
                                        std::to_string(M) + " N=" + std::to_string(N) +
                                        (tail_ok ? " 结果错误" : " 越界写"));
        }
    }
    PRINT("按列切的 GEMV 路径：4 种线程数 × {} 个形状全部通过", shapes.size());
}

// ---------------------------------------------------------------------------
// Q8_0 量化
// ---------------------------------------------------------------------------

TEST(q8_0_quantize_roundtrip)
{
    // 每个数的还原误差不超过本组 scale 的一半（四舍五入）；全 0 的组不能出 nan
    constexpr int64 n = 32 * 9;
    auto x = random_vec(n, 4000);
    for (int64 i = 32; i < 64; ++i)
        x[static_cast<std::size_t>(i)] = 0.0F; // 第 2 组全 0
    x[100] = 50.0F;                             // 第 4 组有个离群值
    std::vector<std::int8_t> q(n);
    std::vector<float> s(n / 32), back(n);
    kernel::quantize_row_q8_0(x.data(), q.data(), s.data(), n);
    kernel::dequantize_row_q8_0(q.data(), s.data(), back.data(), n);
    for (int64 i = 0; i < n; ++i)
    {
        const float d = s[static_cast<std::size_t>(i / 32)];
        CHECK(std::abs(back[i] - x[i]) <= d * 0.5F + 1e-7F);
    }
    CHECK(s[1] == 0.0F);
    CHECK(std::all_of(back.begin() + 32, back.begin() + 64, [](float v) { return v == 0.0F; }));
    // 离群值只拖累自己那一组：第 4 组的 scale 是别的组的 ~50 倍
    PRINT("普通组 scale {:.4f}，离群组 scale {:.4f}", s[0], s[3]);
    CHECK(s[3] > 20.0F * s[0]);
}

TEST(q8_0_matmul_matches_dequantized_reference)
{
    // 参考值：把 W 和每行 A 都「量化再反量化」，再用 double 做普通矩阵乘。
    // 被测 kernel 用整数 SIMD 算同样的东西，只应该有浮点舍入级别的差异。
    // 覆盖 M=1（decode）、M>1、N 非 4 的倍数（单行尾巴）、多线程按列切块
    struct S
    {
        int64 M, N, K;
    };
    const std::vector<S> shapes = {{1, 1, 32},    {1, 7, 64},     {3, 33, 96},
                                   {1, 768, 768}, {5, 2304, 768}, {2, 101, 3072}};
    for (const int nt : {1, 4})
    {
        ThreadGuard guard(nt);
        unsigned seed = 4100;
        for (const auto &[M, N, K] : shapes)
        {
            const auto a = random_vec(M * K, seed++);
            const auto w = random_vec(N * K, seed++);
            const int64 nb = K / 32;
            std::vector<std::int8_t> wq(N * K), aq(M * K);
            std::vector<float> ws(N * nb), as(M * nb), wd(N * K), ad(M * K);
            for (int64 r = 0; r < N; ++r)
                kernel::quantize_row_q8_0(w.data() + r * K, wq.data() + r * K, ws.data() + r * nb, K);
            for (int64 r = 0; r < M; ++r)
                kernel::quantize_row_q8_0(a.data() + r * K, aq.data() + r * K, as.data() + r * nb, K);
            kernel::dequantize_row_q8_0(wq.data(), ws.data(), wd.data(), N * K);
            kernel::dequantize_row_q8_0(aq.data(), as.data(), ad.data(), M * K);
            const auto expect = reference_nn(ad, transpose(wd, N, K), M, N, K);

            std::vector<float> c(M * N, std::numeric_limits<float>::quiet_NaN());
            kernel::matmul_nt_q8_0(a, wq.data(), ws.data(), c, M, N, K);
            const float diff = max_abs_diff(c, expect);
            if (!(diff < 1e-5F * static_cast<float>(K)))
                throw minitest::failure("threads=" + std::to_string(nt) + " M=" + std::to_string(M) +
                                        " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                        " 误差 " + std::to_string(diff));
        }
    }
    PRINT("Q8_0 matmul：{} 个形状 × 2 种线程数都和反量化参考一致", shapes.size());
}

TEST(q8_0_matmul_error_vs_fp32)
{
    // 和原始 fp32 结果比：量化误差应在 1% 量级（相对于输出的典型幅度）
    constexpr int64 M = 1, N = 3072, K = 768;
    const auto a = random_vec(M * K, 4200);
    const auto w = random_vec(N * K, 4201);
    const int64 nb = K / 32;
    std::vector<std::int8_t> wq(N * K);
    std::vector<float> ws(N * nb);
    for (int64 r = 0; r < N; ++r)
        kernel::quantize_row_q8_0(w.data() + r * K, wq.data() + r * K, ws.data() + r * nb, K);
    std::vector<float> c(N), ref(N);
    kernel::matmul_nt_q8_0(a, wq.data(), ws.data(), c, M, N, K);
    kernel::matmul_nt_f32_naive(a, w, ref, M, N, K);
    double err2 = 0, ref2 = 0;
    for (int64 i = 0; i < N; ++i)
    {
        err2 += (c[i] - ref[i]) * (c[i] - ref[i]);
        ref2 += ref[i] * ref[i];
    }
    const double rel = std::sqrt(err2 / ref2);
    PRINT("Q8_0 相对 fp32 的相对 RMS 误差 {:.3f}%", rel * 100);
    CHECK(rel < 0.02);
}

TEST(q8_0_rejects_k_not_multiple_of_32)
{
    std::vector<float> a(40), c(1);
    std::vector<std::int8_t> wq(40);
    std::vector<float> ws(2);
    CHECK_THROWS(kernel::matmul_nt_q8_0(a, wq.data(), ws.data(), c, 1, 1, 40));
}

TEST(f16_convert_roundtrip)
{
    // 能精确表示的数原样往返；一般的数相对误差 ≤ 2^-11（10 位尾数，最近舍入）
    const std::vector<float> exact = {0.0F, 1.0F, -2.0F, 0.5F, 1024.0F, -0.375F, 65504.0F};
    std::vector<std::uint16_t> h(exact.size());
    std::vector<float> back(exact.size());
    kernel::convert_f32_to_f16(exact.data(), h.data(), static_cast<int64>(exact.size()));
    kernel::convert_f16_to_f32(h.data(), back.data(), static_cast<int64>(exact.size()));
    CHECK(back == exact);
    CHECK(h[1] == 0x3C00); // 1.0 的 half 位模式

    const auto x = random_vec(1000, 4300);
    h.resize(x.size());
    back.resize(x.size());
    kernel::convert_f32_to_f16(x.data(), h.data(), 1000);
    kernel::convert_f16_to_f32(h.data(), back.data(), 1000);
    for (std::size_t i = 0; i < x.size(); ++i)
        CHECK(std::abs(back[i] - x[i]) <= std::abs(x[i]) * (1.0F / 2048.0F) + 1e-7F);
}

TEST(f16_matmul_matches_converted_reference)
{
    struct S
    {
        int64 M, N, K;
    };
    const std::vector<S> shapes = {{1, 1, 5}, {1, 7, 40}, {3, 33, 96}, {1, 3001, 768}, {4, 501, 768}};
    for (const int nt : {1, 4})
    {
        ThreadGuard guard(nt);
        unsigned seed = 4400;
        for (const auto &[M, N, K] : shapes)
        {
            const auto a = random_vec(M * K, seed++);
            const auto w = random_vec(N * K, seed++);
            std::vector<std::uint16_t> wh(N * K);
            std::vector<float> wd(N * K);
            kernel::convert_f32_to_f16(w.data(), wh.data(), N * K);
            kernel::convert_f16_to_f32(wh.data(), wd.data(), N * K);
            const auto expect = reference_nn(a, transpose(wd, N, K), M, N, K);

            std::vector<float> c(M * N, std::numeric_limits<float>::quiet_NaN());
            kernel::matmul_nt_f16(a, wh.data(), c, M, N, K);
            const float diff = max_abs_diff(c, expect);
            if (!(diff < 1e-5F * static_cast<float>(K)))
                throw minitest::failure("threads=" + std::to_string(nt) + " M=" + std::to_string(M) +
                                        " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                        " 误差 " + std::to_string(diff));
        }
    }
}

TEST(matmul_nt_default_single_thread)
{
    ThreadGuard guard(1);
    check_nt("matmul_nt_f32 (1 thread)", kernel::matmul_nt_f32, 800);
}

TEST(matmul_nt_default_multi_thread)
{
    ThreadGuard guard(4);
    constexpr int64 M = 131, N = 99, K = 101;
    static_assert(M * N * K >= (1 << 20));
    const auto a = random_vec(M * K, 900);
    const auto b = random_vec(K * N, 901);
    const auto w = transpose(b, K, N);

    std::vector<float> c(M * N, std::numeric_limits<float>::quiet_NaN());
    kernel::matmul_nt_f32(a, w, c, M, N, K);
    CHECK(max_abs_diff(c, reference_nn(a, b, M, N, K)) < 1e-4F);

    std::vector<float> c1(M * N);
    kernel::set_num_threads(1);
    kernel::matmul_nt_f32(a, w, c1, M, N, K);
    CHECK(max_abs_diff(c, c1) < 1e-4F);
}

TEST(matmul_nt_parallel_matches_reference)
{
    for (const int nt : {1, 3, 8})
    {
        ThreadGuard guard(nt);
        check_nt("matmul_nt_f32_parallel", kernel::matmul_nt_f32_parallel, 1400 + nt);
    }
}

TEST(matmul_nt_parallel_single_thread_equals_avx2)
{
    ThreadGuard guard(1);
    constexpr int64 M = 131, N = 99, K = 101;
    const auto a = random_vec(M * K, 1500);
    const auto w = random_vec(N * K, 1501);
    std::vector<float> c_par(M * N);
    std::vector<float> c_avx2(M * N);
    kernel::matmul_nt_f32_parallel(a, w, c_par, M, N, K);
    kernel::matmul_nt_f32_avx2(a, w, c_avx2, M, N, K);
    CHECK(c_par == c_avx2);
}

TEST(matmul_nt_parallel_thread_counts)
{
    constexpr int64 M = 129, N = 101, K = 103;
    static_assert(M * N * K >= (1 << 20));
    const auto a = random_vec(M * K, 1600);
    const auto b = random_vec(K * N, 1601);
    const auto w = transpose(b, K, N);
    const auto expect = reference_nn(a, b, M, N, K);
    for (const int nt : {2, 3, 7, 16, 200})
    {
        ThreadGuard guard(nt);
        std::vector<float> c(M * N, std::numeric_limits<float>::quiet_NaN());
        kernel::matmul_nt_f32_parallel(a, w, c, M, N, K);
        const float diff = max_abs_diff(c, expect);
        PRINT("threads={} max_abs_diff={:.3g}", nt, diff);
        CHECK(diff < 1e-4F);
    }
}

TEST(matmul_k_zero_gives_zeros)
{
    // K=0 时 C 应全为 0：avx2 版的 K 分段循环一次都不执行，必须单独处理
    const std::vector<float> empty;
    for (const MatmulFn &fn : {MatmulFn(kernel::matmul_f32_tiled), MatmulFn(kernel::matmul_f32_avx2),
                               MatmulFn(kernel::matmul_f32)})
    {
        std::vector<float> c(6, std::numeric_limits<float>::quiet_NaN());
        fn(empty, empty, c, 2, 3, 0);
        CHECK((c == std::vector<float>(6, 0.0F)));
    }
}

TEST(matmul_does_not_write_past_mn)
{
    // c 比 M*N 长时，多出来的部分必须原样保留。
    // 并行版把 c.subspan(lo*N) 交给每个线程，若下层越界写（例如对整个 span 清零），
    // 就会覆盖其他线程负责的行 —— 这个用例在串行路径上也能抓到这类问题
    constexpr int64 M = 13, N = 19, K = 11;
    constexpr std::size_t kTail = 64;
    constexpr float kSentinel = 12345.0F;
    const auto a = random_vec(M * K, 1700);
    const auto b = random_vec(K * N, 1701);
    const auto w = transpose(b, K, N);
    const auto expect = reference_nn(a, b, M, N, K);

    const std::vector<std::pair<const char *, MatmulFn>> nn = {
        {"ijk", kernel::matmul_f32_ijk},         {"ikj", kernel::matmul_f32_ikj},
        {"tiled", kernel::matmul_f32_tiled},     {"avx2", kernel::matmul_f32_avx2},
        {"parallel", kernel::matmul_f32_parallel}, {"default", kernel::matmul_f32}};
    const std::vector<std::pair<const char *, MatmulFn>> nt = {
        {"nt_naive", kernel::matmul_nt_f32_naive},
        {"nt_avx2", kernel::matmul_nt_f32_avx2},
        {"nt_parallel", kernel::matmul_nt_f32_parallel},
        {"nt_default", kernel::matmul_nt_f32}};

    auto run = [&](const char *name, const MatmulFn &fn, const std::vector<float> &rhs)
    {
        std::vector<float> c(static_cast<std::size_t>(M * N) + kTail, kSentinel);
        fn(a, rhs, c, M, N, K);
        const bool tail_ok =
            std::all_of(c.begin() + M * N, c.end(), [](float v) { return v == kSentinel; });
        c.resize(static_cast<std::size_t>(M * N));
        if (!tail_ok || max_abs_diff(c, expect) > 1e-4F)
            throw minitest::failure(std::string(name) + (tail_ok ? " 结果错误" : " 写出了 M*N 范围"));
    };
    for (const auto &[name, fn] : nn)
        run(name, fn, b);
    for (const auto &[name, fn] : nt)
        run(name, fn, w);
    PRINT("{} 个实现都没有越界写", nn.size() + nt.size());
}

TEST(matmul_nt_agrees_with_nn)
{
    constexpr int64 M = 21, N = 37, K = 45;
    const auto a = random_vec(M * K, 1000);
    const auto b = random_vec(K * N, 1001);
    const auto w = transpose(b, K, N);
    std::vector<float> c_nn(M * N);
    std::vector<float> c_nt(M * N);
    std::vector<float> c_naive(M * N);
    kernel::matmul_f32(a, b, c_nn, M, N, K);
    kernel::matmul_nt_f32(a, w, c_nt, M, N, K);
    kernel::matmul_nt_f32_naive(a, w, c_naive, M, N, K);
    CHECK(max_abs_diff(c_nn, c_nt) < 1e-4F);
    CHECK(max_abs_diff(c_naive, c_nt) < 1e-4F);
}
