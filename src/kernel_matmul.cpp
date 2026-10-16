#include "inference/kernel.hpp"
#include "inference/tensor.hpp"

#include <algorithm>
#include <thread>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define INFER_HAS_AVX2 1
#else
#define INFER_HAS_AVX2 0
#endif

namespace inference::kernel
{

// --- 分块尺寸 ---------------------------------------------------------------
// 本机 L1d = 48 KB/核 = 12288 个 float。
// B 的一块 kTileK×kTileN = 32×256 个 float = 32 KB，打包后留在 L1，剩余空间给 A/C。
// kTileN 取 256 而不是 64：最内层 j 循环太短（64 个 float 只有 8 个 AVX 向量）
// 时，循环开销占比明显变大。
// 这些数字要按自己的 CPU 调，没有万能值 —— 所以工业库（OpenBLAS）会针对
// 每种微架构手工标定一套参数。
constexpr int64 kTileN = 256;
constexpr int64 kTileK = 32;

// --- AVX2 微内核尺寸 --------------------------------------------------------
// 一次算 C 的 4 行 × 16 列。为什么是这个形状？
//   16 列 = 2 个 __m256（每个装 8 个 float）
//   4 行 × 2 = 8 个累加器寄存器，再加 2 个 B 向量、1 个 A 广播 = 11 个
//   YMM 寄存器总共 16 个，留了余量给编译器周转。
//   再大就会「寄存器溢出」（spill）到栈上，反而变慢。
constexpr int64 kMR = 4;
constexpr int64 kNR = 16;

int g_num_threads = 0; // 0 表示还没初始化，取硬件并发数
// 小问题不值得开线程：创建一个线程要几十微秒，
// 如果计算本身才几十微秒，那纯粹是赔本。真实引擎都有这个阈值判断。
constexpr int64 kParallelMinWork = 1 << 20; // M*N*K 低于 100 万就串行

#if INFER_HAS_AVX2
// C 的 4×16 小块全程待在寄存器里，K 循环结束才写回内存一次。
// 对比标量版每次内积都要读写一遍 C，这是最大的性能来源。

// C 的 4×16 小块全程待在寄存器里，K 循环结束才写回内存一次。
// 对比标量版每次内积都要读写一遍 C，这是最大的性能来源。
void micro_kernel_4x16(const float *a, const float *b, float *c, int64 K, int64 lda, int64 ldb,
                       int64 ldc)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();

    for (int64 k = 0; k < K; ++k)
    {
        // B 的一行 16 个数，两次 256 位加载
        const __m256 b0 = _mm256_loadu_ps(b + k * ldb);
        const __m256 b1 = _mm256_loadu_ps(b + k * ldb + 8);

        // broadcast：把 A 的一个标量复制到 8 个通道
        // fmadd(x, y, z) = x*y + z，乘加一条指令完成，还少一次舍入
        __m256 av = _mm256_set1_ps(a[0 * lda + k]);
        c00 = _mm256_fmadd_ps(av, b0, c00);
        c01 = _mm256_fmadd_ps(av, b1, c01);

        av = _mm256_set1_ps(a[1 * lda + k]);
        c10 = _mm256_fmadd_ps(av, b0, c10);
        c11 = _mm256_fmadd_ps(av, b1, c11);

        av = _mm256_set1_ps(a[2 * lda + k]);
        c20 = _mm256_fmadd_ps(av, b0, c20);
        c21 = _mm256_fmadd_ps(av, b1, c21);

        av = _mm256_set1_ps(a[3 * lda + k]);
        c30 = _mm256_fmadd_ps(av, b0, c30);
        c31 = _mm256_fmadd_ps(av, b1, c31);
    }

    _mm256_storeu_ps(c + 0 * ldc, c00);
    _mm256_storeu_ps(c + 0 * ldc + 8, c01);
    _mm256_storeu_ps(c + 1 * ldc, c10);
    _mm256_storeu_ps(c + 1 * ldc + 8, c11);
    _mm256_storeu_ps(c + 2 * ldc, c20);
    _mm256_storeu_ps(c + 2 * ldc + 8, c21);
    _mm256_storeu_ps(c + 3 * ldc, c30);
    _mm256_storeu_ps(c + 3 * ldc + 8, c31);
}

// 把 __m256 里的 8 个 float 加成一个标量。
// 做法是「折半相加」：8→4→2→1，共 3 轮，比逐个取出来加快得多。
inline float hsum256(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi); // 8 -> 4
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf); // 4 -> 2
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf); // 2 -> 1
    return _mm_cvtss_f32(sums);
}

#endif // INFER_HAS_AVX2

// 标量兜底的小块计算，处理边角料（M/N 不是微内核尺寸整数倍的部分）
void scalar_block(const float *a, const float *b, float *c, int64 m, int64 n, int64 K, int64 lda,
                  int64 ldb, int64 ldc)
{
    for (int64 i = 0; i < m; ++i)
    {
        for (int64 j = 0; j < n; ++j)
        {
            float sum = 0.0F;
            for (int64 k = 0; k < K; ++k)
                sum += a[i * lda + k] * b[k * ldb + j];
            c[i * ldc + j] = sum;
        }
    }
}

// 把一段行区间 [row_begin, row_end) 分给若干线程并行处理
template <class F> void parallel_rows(int64 total_rows, int64 work, F &&fn)
{
    const int nt = get_num_threads();
    if (nt <= 1 || work < kParallelMinWork || total_rows < 2)
    {
        fn(int64{0}, total_rows);
        return;
    }
    const auto chunks = std::min<int64>(nt, total_rows);
    const int64 per = (total_rows + chunks - 1) / chunks;

    // std::jthread 是 C++20 的新线程类型，比 std::thread 好在两点：
    //   1. 析构时自动 join，不会因为忘记 join 而 std::terminate
    //   2. 自带 stop_token，可以协作式取消（这里用不到）
    std::vector<std::jthread> workers;
    workers.reserve(static_cast<std::size_t>(chunks - 1));
    for (int64 t = 1; t < chunks; ++t)
    {
        const int64 lo = t * per;
        const int64 hi = std::min(lo + per, total_rows);
        if (lo >= hi)
            break;
        workers.emplace_back([&fn, lo, hi] { fn(lo, hi); });
    }
    fn(int64{0}, std::min(per, total_rows)); // 主线程也干一份活，别闲着
                                             // workers 析构时自动 join
}

bool has_avx2_support() noexcept
{
    return INFER_HAS_AVX2 != 0;
}
void set_num_threads(int n) noexcept
{
    g_num_threads = n < 1 ? 1 : n;
}

int get_num_threads() noexcept
{
    if (g_num_threads == 0)
    {
        const auto hw = static_cast<int>(std::thread::hardware_concurrency());
        g_num_threads = hw > 0 ? hw : 1;
    }
    return g_num_threads;
}

// ---------------------------------------------------------------------------
// v3：循环分块
// ---------------------------------------------------------------------------
void matmul_f32_tiled(std::span<const float> a, std::span<const float> b, std::span<float> c,
                      int64 M, int64 N, int64 K)
{
    std::ranges::fill(c.first(static_cast<std::size_t>(M * N)), 0.0F);

    // 外层按 (j0, k0) 切出 B 的一块，先拷贝（pack）到连续的小缓冲区，
    // 再让 A 的所有行轮流和它相乘：一块 B 只打包一次，被复用 M 次。
    //
    // 为什么一定要打包？直接在原矩阵上分块时，块内相邻两行相隔 N 个 float。
    // N=1024 时恰好是 4 KB，而 L1 的组号只看地址第 6~11 位 ——
    // 相隔 4 KB 的地址全落在同一组里，几十行抢 12 路，块根本留不在 L1。
    // 打包后块内各行首尾相接，就不受原矩阵 N 是不是 2 的幂影响了。
    std::vector<float> packed(static_cast<std::size_t>(kTileK * kTileN));
    for (int64 j0 = 0; j0 < N; j0 += kTileN)
    {
        const int64 jn = std::min(kTileN, N - j0);
        for (int64 k0 = 0; k0 < K; k0 += kTileK)
        {
            const int64 kn = std::min(kTileK, K - k0);
            for (int64 k = 0; k < kn; ++k)
            {
                std::copy_n(b.data() + (k0 + k) * N + j0, jn, packed.data() + k * jn);
            }

            // 块内仍是 i-k-j：每个 C[i][j] 的累加顺序和 ikj 版完全一致
            for (int64 i = 0; i < M; ++i)
            {
                float *ci = c.data() + i * N + j0;
                const float *ai = a.data() + i * K + k0;
                for (int64 k = 0; k < kn; ++k)
                {
                    const float aik = ai[k];
                    const float *bk = packed.data() + k * jn;
                    for (int64 j = 0; j < jn; ++j)
                    {
                        ci[j] += aik * bk[j];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v4：AVX2 + FMA 微内核
// ---------------------------------------------------------------------------
void matmul_f32_avx2(std::span<const float> a, std::span<const float> b, std::span<float> c,
                     int64 M, int64 N, int64 K)
{
#if INFER_HAS_AVX2
    const float *pa = a.data();
    const float *pb = b.data();
    float *pc = c.data();

    const int64 m_main = M / kMR * kMR; // 能被微内核整除的部分
    const int64 n_main = N / kNR * kNR;

    for (int64 i = 0; i < m_main; i += kMR)
    {
        for (int64 j = 0; j < n_main; j += kNR)
        {
            micro_kernel_4x16(pa + i * K, pb + j, pc + i * N + j, K, K, N, N);
        }
        // 右边剩下不足 16 列的边角
        if (n_main < N)
        {
            scalar_block(pa + i * K, pb + n_main, pc + i * N + n_main, kMR, N - n_main, K, K, N, N);
        }
    }
    // 下面剩下不足 4 行的边角
    if (m_main < M)
    {
        scalar_block(pa + m_main * K, pb, pc + m_main * N, M - m_main, N, K, K, N, N);
    }
#else
    matmul_f32_tiled(a, b, c, M, N, K); // 没有 AVX2 就退回分块版
#endif
}

// ---------------------------------------------------------------------------
// v5：多线程
// ---------------------------------------------------------------------------
void matmul_f32_parallel(std::span<const float> a, std::span<const float> b, std::span<float> c,
                         int64 M, int64 N, int64 K)
{
    // 按行切分。为什么切行不切列？因为 C 是行优先存储，
    // 不同线程写的是**不同的行**，落在不同的 cache line 上，
    // 不会发生 false sharing（两个核抢同一条 cache line 的所有权，性能暴跌）。
    parallel_rows(M, M * N * K,
                  [&](int64 lo, int64 hi)
                  {
                      const int64 rows = hi - lo;
                      if (rows <= 0)
                          return;
                      // subspan 必须带长度：下层的 tiled 回退路径会 fill(c, 0)，
                      // 不限长就会清掉后面其他线程负责的行
                      matmul_f32_avx2(a.subspan(static_cast<std::size_t>(lo * K),
                                                static_cast<std::size_t>(rows * K)),
                                      b,
                                      c.subspan(static_cast<std::size_t>(lo * N),
                                                static_cast<std::size_t>(rows * N)),
                                      rows, N, K);
                  });
}

void matmul_f32(std::span<const float> a, std::span<const float> b, std::span<float> c, int64 M,
                int64 N, int64 K)
{
    matmul_f32_parallel(a, b, c, M, N, K);
}

// ---------------------------------------------------------------------------
// NT 版本（C = A × Wᵀ，linear 层用的就是它）
// ---------------------------------------------------------------------------
void matmul_nt_f32_avx2(std::span<const float> a, std::span<const float> w, std::span<float> c,
                        int64 M, int64 N, int64 K)
{
#if INFER_HAS_AVX2
    const float *pa = a.data();
    const float *pw = w.data();
    float *pc = c.data();

    // NT 布局下 A 和 W 都沿 k 连续，所以直接沿 k 向量化，最后水平求和。
    // 这里做 4 行 A × 2 行 W 的寄存器分块：
    //   A 的每一行读进来能服务 2 个输出列，W 的每一行能服务 4 个输出行，
    //   内存流量降到朴素版的约 1/3。
    //   寄存器用量: 8 个累加器 + 4 个 a + 2 个 w = 14 个，YMM 共 16 个，刚好。
    constexpr int64 MR = 4, NR = 2;
    const int64 m_main = M / MR * MR;
    const int64 n_main = N / NR * NR;
    const int64 k_main = K / 8 * 8;

    for (int64 i = 0; i < m_main; i += MR)
    {
        for (int64 j = 0; j < n_main; j += NR)
        {
            __m256 acc[MR][NR];
            for (auto &row : acc)
                for (auto &v : row)
                    v = _mm256_setzero_ps();

            for (int64 k = 0; k < k_main; k += 8)
            {
                const __m256 a0 = _mm256_loadu_ps(pa + (i + 0) * K + k);
                const __m256 a1 = _mm256_loadu_ps(pa + (i + 1) * K + k);
                const __m256 a2 = _mm256_loadu_ps(pa + (i + 2) * K + k);
                const __m256 a3 = _mm256_loadu_ps(pa + (i + 3) * K + k);
                const __m256 w0 = _mm256_loadu_ps(pw + (j + 0) * K + k);
                const __m256 w1 = _mm256_loadu_ps(pw + (j + 1) * K + k);

                acc[0][0] = _mm256_fmadd_ps(a0, w0, acc[0][0]);
                acc[0][1] = _mm256_fmadd_ps(a0, w1, acc[0][1]);
                acc[1][0] = _mm256_fmadd_ps(a1, w0, acc[1][0]);
                acc[1][1] = _mm256_fmadd_ps(a1, w1, acc[1][1]);
                acc[2][0] = _mm256_fmadd_ps(a2, w0, acc[2][0]);
                acc[2][1] = _mm256_fmadd_ps(a2, w1, acc[2][1]);
                acc[3][0] = _mm256_fmadd_ps(a3, w0, acc[3][0]);
                acc[3][1] = _mm256_fmadd_ps(a3, w1, acc[3][1]);
            }

            for (int64 r = 0; r < MR; ++r)
            {
                for (int64 t = 0; t < NR; ++t)
                {
                    float sum = hsum256(acc[r][t]);
                    // K 不是 8 的倍数时，剩下的几个标量补上
                    for (int64 k = k_main; k < K; ++k)
                    {
                        sum += pa[(i + r) * K + k] * pw[(j + t) * K + k];
                    }
                    pc[(i + r) * N + (j + t)] = sum;
                }
            }
        }
        // 右边不足 2 列的边角
        for (int64 j = n_main; j < N; ++j)
        {
            for (int64 r = 0; r < MR; ++r)
            {
                float sum = 0.0F;
                for (int64 k = 0; k < K; ++k)
                    sum += pa[(i + r) * K + k] * pw[j * K + k];
                pc[(i + r) * N + j] = sum;
            }
        }
    }
    // 下面不足 4 行的边角
    for (int64 i = m_main; i < M; ++i)
    {
        for (int64 j = 0; j < N; ++j)
        {
            float sum = 0.0F;
            for (int64 k = 0; k < K; ++k)
                sum += pa[i * K + k] * pw[j * K + k];
            pc[i * N + j] = sum;
        }
    }
#else
    matmul_nt_f32_naive(a, w, c, M, N, K);
#endif
}

void matmul_nt_f32_parallel(std::span<const float> a, std::span<const float> w, std::span<float> c,
                            int64 M, int64 N, int64 K)
{
    parallel_rows(M, M * N * K,
                  [&](int64 lo, int64 hi)
                  {
                      const int64 rows = hi - lo;
                      if (rows <= 0)
                          return;
                      matmul_nt_f32_avx2(a.subspan(static_cast<std::size_t>(lo * K),
                                                   static_cast<std::size_t>(rows * K)),
                                         w,
                                         c.subspan(static_cast<std::size_t>(lo * N),
                                                   static_cast<std::size_t>(rows * N)),
                                         rows, N, K);
                  });
}

void matmul_nt_f32(std::span<const float> a, std::span<const float> w, std::span<float> c, int64 M,
                   int64 N, int64 K)
{
    matmul_nt_f32_parallel(a, w, c, M, N, K);
}

} // namespace inference::kernel