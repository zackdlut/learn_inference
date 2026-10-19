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

// --- GotoBLAS 分块尺寸（v4 用）-----------------------------------------------
// 三级存储各放一样东西，大小按 cache 容量来定：
//   L1  : B 的一条竖条  kKC×kNR = 256×16  个 float = 16 KB
//   L2  : A 的一块      kMC×kKC = 64×256  个 float = 64 KB
//   L2/L3: B 的一块     kKC×kNC = 256×512 个 float = 512 KB
// 一条 B 竖条在 L1 里被 kMC/4 条 A 横条复用，一块 A 在 L2 里被 kNC/16 条竖条复用。
//
// kNC 没有按单线程最优取更大：多线程时每个线程各打包一份 B 块，
// 14 线程 × 2 MB（kNC=2048）= 28 MB，远超 12 MB 的 L3，实测反而慢 15~30%。
// kMC / kNC 必须分别是 kMR / kNR 的倍数，否则补零的最后一组会写出缓冲区。
constexpr int64 kKC = 256;
constexpr int64 kMC = 64;
constexpr int64 kNC = 512;
static_assert(kMC % kMR == 0 && kNC % kNR == 0);

#if INFER_HAS_AVX2
// 把 B[0:kc, 0:nc]（行距 ldb）打包成竖条：第 s 列起的竖条第 k 行位于
//   packed[s*kc + k*16 + 0..15]
// 打包后微内核每次读 B 都是相邻的 64 字节，不再按行距 N 跳着读 ——
// N=1024 时那种「行距恰好 4 KB、全落在同一个 L1 组」的冲突也就没了。
// 不足 16 列的最后一条补 0，这样微内核不用分情况。
void pack_b(const float *b, int64 ldb, int64 kc, int64 nc, float *packed)
{
    for (int64 s = 0; s < nc; s += kNR)
    {
        const int64 nr = std::min(kNR, nc - s);
        for (int64 k = 0; k < kc; ++k)
        {
            float *dst = packed + s * kc + k * kNR;
            std::copy_n(b + k * ldb + s, nr, dst);
            std::fill(dst + nr, dst + kNR, 0.0F);
        }
    }
}

// 把 A[0:mc, 0:kc]（行距 lda）打包成 4 行一组的横条，组内按 k 交错：
//   第 r0 行起的横条，第 k 列的 4 个数位于 packed[r0*kc + k*4 + 0..3]
// 微内核每个 k 要的正是这 4 行在同一列上的数，交错后它们挨在一起，
// 一条 cache line 能喂 4 次迭代；不打包时这 4 个数分散在 4 行、相隔 lda。
// 不足 4 行的最后一组补 0。
void pack_a(const float *a, int64 lda, int64 mc, int64 kc, float *packed)
{
    for (int64 r0 = 0; r0 < mc; r0 += kMR)
    {
        const int64 mr = std::min(kMR, mc - r0);
        float *dst = packed + r0 * kc;
        for (int64 k = 0; k < kc; ++k)
        {
            for (int64 r = 0; r < kMR; ++r)
                dst[k * kMR + r] = r < mr ? a[(r0 + r) * lda + k] : 0.0F;
        }
    }
}

// C 的 4×16 小块全程待在寄存器里，kc 循环结束才写回内存一次。
// 对比标量版每次内积都要读写一遍 C，这是最大的性能来源。
// pa / pb 都是打包后的数据，整个 kc 循环里两者都是顺序读。
// accumulate=false 时直接覆盖 C；为 true 时加到 C 原有的值上（K 被切成多段时用）。
void micro_kernel_4x16(const float *pa, const float *pb, int64 kc, float *c, int64 ldc,
                       bool accumulate)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();

    for (int64 k = 0; k < kc; ++k)
    {
        // 打包后竖条的一行 16 个数是连续的，两次 256 位加载
        const __m256 b0 = _mm256_loadu_ps(pb + k * kNR);
        const __m256 b1 = _mm256_loadu_ps(pb + k * kNR + 8);
        const float *ak = pa + k * kMR;

        // broadcast：把 A 的一个标量复制到 8 个通道
        // fmadd(x, y, z) = x*y + z，乘加一条指令完成，还少一次舍入
        __m256 av = _mm256_broadcast_ss(ak + 0);
        c00 = _mm256_fmadd_ps(av, b0, c00);
        c01 = _mm256_fmadd_ps(av, b1, c01);

        av = _mm256_broadcast_ss(ak + 1);
        c10 = _mm256_fmadd_ps(av, b0, c10);
        c11 = _mm256_fmadd_ps(av, b1, c11);

        av = _mm256_broadcast_ss(ak + 2);
        c20 = _mm256_fmadd_ps(av, b0, c20);
        c21 = _mm256_fmadd_ps(av, b1, c21);

        av = _mm256_broadcast_ss(ak + 3);
        c30 = _mm256_fmadd_ps(av, b0, c30);
        c31 = _mm256_fmadd_ps(av, b1, c31);
    }

    if (accumulate)
    {
        c00 = _mm256_add_ps(c00, _mm256_loadu_ps(c + 0 * ldc));
        c01 = _mm256_add_ps(c01, _mm256_loadu_ps(c + 0 * ldc + 8));
        c10 = _mm256_add_ps(c10, _mm256_loadu_ps(c + 1 * ldc));
        c11 = _mm256_add_ps(c11, _mm256_loadu_ps(c + 1 * ldc + 8));
        c20 = _mm256_add_ps(c20, _mm256_loadu_ps(c + 2 * ldc));
        c21 = _mm256_add_ps(c21, _mm256_loadu_ps(c + 2 * ldc + 8));
        c30 = _mm256_add_ps(c30, _mm256_loadu_ps(c + 3 * ldc));
        c31 = _mm256_add_ps(c31, _mm256_loadu_ps(c + 3 * ldc + 8));
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

// 边角块（不足 4 行或不足 16 列）：打包时已经补过 0，微内核照常算，
// 只是结果先写进 4×16 的临时区，再把有效的 mr×nr 部分写回 C，避免写出 C 的边界。
void edge_tile(const float *pa, const float *pb, int64 kc, float *c, int64 ldc, int64 mr, int64 nr,
               bool accumulate)
{
    float tmp[kMR * kNR];
    micro_kernel_4x16(pa, pb, kc, tmp, kNR, false);
    for (int64 r = 0; r < mr; ++r)
    {
        for (int64 j = 0; j < nr; ++j)
        {
            c[r * ldc + j] = accumulate ? c[r * ldc + j] + tmp[r * kNR + j] : tmp[r * kNR + j];
        }
    }
}

// 用已打包好的 B 块（kc×nc，起始列 jc）计算 C[ic:ic+mc, jc+j0 : jc+j1]。
// 先把 A[ic:ic+mc, p0:p0+kc] 打包进 packed_a，再走 jr / ir 两层循环。
// j0 必须是 kNR 的倍数（竖条边界），j1 可以是任意值（最后一条按 nr 截断）。
void gemm_block(const float *a, int64 K, const float *packed_b, float *packed_a, float *c, int64 N,
                int64 ic, int64 mc, int64 jc, int64 j0, int64 j1, int64 p0, int64 kc,
                bool accumulate)
{
    pack_a(a + ic * K + p0, K, mc, kc, packed_a);
    for (int64 jr = j0; jr < j1; jr += kNR)
    {
        const int64 nr = std::min(kNR, j1 - jr);
        const float *bs = packed_b + jr * kc;
        for (int64 ir = 0; ir < mc; ir += kMR)
        {
            const int64 mr = std::min(kMR, mc - ir);
            const float *as = packed_a + ir * kc;
            float *cij = c + (ic + ir) * N + jc + jr;
            if (mr == kMR && nr == kNR)
                micro_kernel_4x16(as, bs, kc, cij, N, accumulate);
            else
                edge_tile(as, bs, kc, cij, N, mr, nr, accumulate);
        }
    }
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

    if (K == 0)
    {
        std::ranges::fill(c.first(static_cast<std::size_t>(M * N)), 0.0F);
        return;
    }

    // GotoBLAS 的五层循环，由外到内：
    //   jc：按 kNC 列切 B          ┐ 打包 B 的一块（→ L3），
    //   p0：按 kKC 切 K            ┘ 被 A 的所有行复用
    //   ic：按 kMC 行切 A            打包 A 的一块（→ L2），被这块 B 的所有竖条复用
    //   jr：B 块里的每条 16 列竖条    （→ L1），被 A 块里所有 4 行横条复用
    //   ir：A 块里的每条 4 行横条     交给微内核算 4×16
    std::vector<float> packed_b(static_cast<std::size_t>(kKC * kNC));
    std::vector<float> packed_a(static_cast<std::size_t>(kMC * kKC));
    for (int64 jc = 0; jc < N; jc += kNC)
    {
        const int64 nc = std::min(kNC, N - jc);
        for (int64 p0 = 0; p0 < K; p0 += kKC)
        {
            const int64 kc = std::min(kKC, K - p0);
            const bool accumulate = p0 > 0; // 第一段覆盖 C，之后的段累加，省掉提前清零
            pack_b(pb + p0 * N + jc, N, kc, nc, packed_b.data());

            for (int64 ic = 0; ic < M; ic += kMC)
            {
                gemm_block(pa, K, packed_b.data(), packed_a.data(), pc, N, ic,
                           std::min(kMC, M - ic), jc, 0, nc, p0, kc, accumulate);
            }
        }
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
    //
    // 每个线程调用 v4，各自打包一整份 B —— 看起来重复干了 nt 倍的活，
    // 但 kNC=512 时 14 份 B 块一共才 7 MB，L3 装得下；多出来的打包量
    // 只占每个线程计算量的 1/（分到的行数）。
    // 试过「所有线程共用一份 B、每轮用 std::barrier 对齐」的写法：
    // 每轮要等最慢的核（低功耗 E 核）做完，K=3072 时一次矩阵乘要同步 24 次，
    // 只在 2048³ 这类大方阵上占优，1000³、128×768×3072 等多数形状反而慢 15~30%。
    // 同步的代价比省下的打包更贵。
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
// GEMV：y = W · x（M=1 的 NT 矩阵乘，decode 阶段的主角）
//
// M=1 时每个权重只用一次，没有任何复用可言，GEMM 那套打包、寄存器分块都成了纯开销。
// 这里唯一的目标是让 W 以最快速度顺序流过 CPU：
//   - 不打包，W 一行一行顺序读
//   - x 只有 K 个 float（≤ 12 KB），全程留在 L1
//   - 一次算 4 个输出：读一次 x 的 8 个数，喂给 4 行 W，x 的加载开销摊薄到 1/4
//   - 每行 2 个累加器、k 一次走 16：共 8 条独立的 FMA 依赖链。
//     FMA 延迟 4 周期、每周期能发 2 条，至少要 8 条链并行才不会互相等。
// ---------------------------------------------------------------------------
#if INFER_HAS_AVX2
namespace
{
// 一行 W 和 x 的点积，用于 N 不是 4 的倍数时剩下的行
float dot_row_avx2(const float *x, const float *w, int64 K)
{
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int64 k = 0;
    for (; k + 16 <= K; k += 16)
    {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k), _mm256_loadu_ps(w + k), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 8), _mm256_loadu_ps(w + k + 8), acc1);
    }
    for (; k + 8 <= K; k += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k), _mm256_loadu_ps(w + k), acc0);
    float sum = hsum256(_mm256_add_ps(acc0, acc1));
    for (; k < K; ++k)
        sum += x[k] * w[k];
    return sum;
}
} // namespace
#endif

void gemv_nt_f32(std::span<const float> x, std::span<const float> w, std::span<float> y, int64 N,
                 int64 K)
{
#if INFER_HAS_AVX2
    const float *px = x.data();
    const float *pw = w.data();
    float *py = y.data();
    const int64 n_main = N / 4 * 4;
    const int64 k16 = K / 16 * 16;

    for (int64 n = 0; n < n_main; n += 4)
    {
        const float *w0 = pw + (n + 0) * K;
        const float *w1 = pw + (n + 1) * K;
        const float *w2 = pw + (n + 2) * K;
        const float *w3 = pw + (n + 3) * K;
        __m256 a0 = _mm256_setzero_ps(), b0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();

        for (int64 k = 0; k < k16; k += 16)
        {
            const __m256 xa = _mm256_loadu_ps(px + k);
            const __m256 xb = _mm256_loadu_ps(px + k + 8);
            a0 = _mm256_fmadd_ps(xa, _mm256_loadu_ps(w0 + k), a0);
            b0 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(w0 + k + 8), b0);
            a1 = _mm256_fmadd_ps(xa, _mm256_loadu_ps(w1 + k), a1);
            b1 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(w1 + k + 8), b1);
            a2 = _mm256_fmadd_ps(xa, _mm256_loadu_ps(w2 + k), a2);
            b2 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(w2 + k + 8), b2);
            a3 = _mm256_fmadd_ps(xa, _mm256_loadu_ps(w3 + k), a3);
            b3 = _mm256_fmadd_ps(xb, _mm256_loadu_ps(w3 + k + 8), b3);
        }

        float s[4] = {hsum256(_mm256_add_ps(a0, b0)), hsum256(_mm256_add_ps(a1, b1)),
                      hsum256(_mm256_add_ps(a2, b2)), hsum256(_mm256_add_ps(a3, b3))};
        const float *rows[4] = {w0, w1, w2, w3};
        // K 不是 16 的倍数时剩下的 0~15 个数
        for (int64 k = k16; k < K; ++k)
            for (int r = 0; r < 4; ++r)
                s[r] += px[k] * rows[r][k];
        for (int r = 0; r < 4; ++r)
            py[n + r] = s[r];
    }
    for (int64 n = n_main; n < N; ++n)
        py[n] = dot_row_avx2(px, pw + n * K, K);
#else
    matmul_nt_f32_naive(x, w, y, 1, N, K);
#endif
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
    // 下面不足 4 行的边角：每行就是一次 GEMV。
    // decode（M=1）整个矩阵乘都走这里，所以它不能是标量循环。
    for (int64 i = m_main; i < M; ++i)
    {
        gemv_nt_f32(a.subspan(static_cast<std::size_t>(i * K), static_cast<std::size_t>(K)), w,
                    c.subspan(static_cast<std::size_t>(i * N), static_cast<std::size_t>(N)), N,
                    K);
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