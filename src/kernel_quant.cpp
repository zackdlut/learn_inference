#include "inference/kernel.hpp"
#include "simd_math.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(__F16C__)
#include <immintrin.h>
#endif

namespace inference::kernel
{

// ---------------------------------------------------------------------------
// Q8_0 量化
//
// 每 32 个数一组：d = max|x| / 127，q = round(x / d) ∈ [-127, 127]，还原时 x ≈ q × d。
// 为什么分组而不是整行一个 scale：一个离群的大值会把同一组里其它数的精度全压没，
// 组越小，离群值的影响范围越小；32 是精度和 scale 存储开销之间的折中（llama.cpp 同款）。
// ---------------------------------------------------------------------------
void quantize_row_q8_0(const float *x, std::int8_t *q, float *scale, int64 n)
{
    for (int64 b = 0; b < n / kQ8Block; ++b)
    {
        const float *xb = x + b * kQ8Block;
        float amax = 0.0F;
        for (int64 i = 0; i < kQ8Block; ++i)
            amax = std::max(amax, std::fabs(xb[i]));
        const float d = amax / 127.0F;
        const float id = d > 0.0F ? 1.0F / d : 0.0F;
        scale[b] = d;
        for (int64 i = 0; i < kQ8Block; ++i)
            q[b * kQ8Block + i] = static_cast<std::int8_t>(std::lrint(xb[i] * id));
    }
}

void dequantize_row_q8_0(const std::int8_t *q, const float *scale, float *x, int64 n)
{
    for (int64 i = 0; i < n; ++i)
        x[i] = static_cast<float>(q[i]) * scale[i / kQ8Block];
}

namespace
{

#if INFER_SIMD_AVX2
// 32 个 int8 × 32 个 int8 的点积，结果是 8 个 int32 部分和（转成 float）。
//
// AVX2 没有「有符号 × 有符号」的 8 位乘加，只有 maddubs：无符号 u8 × 有符号 s8，
// 相邻两对相加成 s16。技巧是把 x 的符号挪到 w 上：
//   x · w = |x| · (w × sign(x))
// |x| ≤ 127 当无符号数用，w × sign(x) 仍在 [-127, 127]。
// 两对相加最大 2 × 127 × 127 = 32258 < 32767，s16 不会饱和。
// 再用 madd(·, 1) 把相邻两个 s16 加成 s32，8 个 int32 转 float 后乘上两边的 scale。
inline __m256 dot_block_q8(__m256i ax, __m256i xv, const std::int8_t *w)
{
    const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w));
    const __m256i p16 = _mm256_maddubs_epi16(ax, _mm256_sign_epi8(wv, xv));
    const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
    return _mm256_cvtepi32_ps(p32);
}
#endif

// y[0:n_rows] = W[0:n_rows, :] · x，W 和 x 都是 Q8_0
void gemv_q8_0(const std::int8_t *xq, const float *xs, const std::int8_t *wq, const float *ws,
               float *y, int64 n_rows, int64 K)
{
    const int64 nb = K / kQ8Block;
    int64 n = 0;
#if INFER_SIMD_AVX2
    // 一次算 4 行：x 的加载和 |x| 只算一次，给 4 行 W 共用
    for (; n + 4 <= n_rows; n += 4)
    {
        const std::int8_t *w0 = wq + (n + 0) * K, *w1 = wq + (n + 1) * K;
        const std::int8_t *w2 = wq + (n + 2) * K, *w3 = wq + (n + 3) * K;
        const float *s0 = ws + (n + 0) * nb, *s1 = ws + (n + 1) * nb;
        const float *s2 = ws + (n + 2) * nb, *s3 = ws + (n + 3) * nb;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        for (int64 b = 0; b < nb; ++b)
        {
            const __m256i xv =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq + b * kQ8Block));
            const __m256i ax = _mm256_sign_epi8(xv, xv);
            const float d = xs[b];
            const int64 off = b * kQ8Block;
            a0 = _mm256_fmadd_ps(dot_block_q8(ax, xv, w0 + off), _mm256_set1_ps(d * s0[b]), a0);
            a1 = _mm256_fmadd_ps(dot_block_q8(ax, xv, w1 + off), _mm256_set1_ps(d * s1[b]), a1);
            a2 = _mm256_fmadd_ps(dot_block_q8(ax, xv, w2 + off), _mm256_set1_ps(d * s2[b]), a2);
            a3 = _mm256_fmadd_ps(dot_block_q8(ax, xv, w3 + off), _mm256_set1_ps(d * s3[b]), a3);
        }
        y[n + 0] = simd::hsum256_ps(a0);
        y[n + 1] = simd::hsum256_ps(a1);
        y[n + 2] = simd::hsum256_ps(a2);
        y[n + 3] = simd::hsum256_ps(a3);
    }
    for (; n < n_rows; ++n)
    {
        const std::int8_t *w = wq + n * K;
        const float *s = ws + n * nb;
        __m256 acc = _mm256_setzero_ps();
        for (int64 b = 0; b < nb; ++b)
        {
            const __m256i xv =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq + b * kQ8Block));
            const __m256i ax = _mm256_sign_epi8(xv, xv);
            acc = _mm256_fmadd_ps(dot_block_q8(ax, xv, w + b * kQ8Block),
                                  _mm256_set1_ps(xs[b] * s[b]), acc);
        }
        y[n] = simd::hsum256_ps(acc);
    }
#else
    for (; n < n_rows; ++n)
    {
        float sum = 0.0F;
        for (int64 b = 0; b < nb; ++b)
        {
            std::int32_t isum = 0;
            for (int64 i = 0; i < kQ8Block; ++i)
                isum += static_cast<std::int32_t>(xq[b * kQ8Block + i]) *
                        static_cast<std::int32_t>(wq[n * K + b * kQ8Block + i]);
            sum += static_cast<float>(isum) * xs[b] * ws[n * nb + b];
        }
        y[n] = sum;
    }
#endif
}

} // namespace

void matmul_nt_q8_0(std::span<const float> a, const std::int8_t *wq, const float *ws,
                    std::span<float> c, int64 M, int64 N, int64 K)
{
    if (K % kQ8Block != 0)
        throw std::invalid_argument("matmul_nt_q8_0: K 必须是 32 的倍数");
    const int64 nb = K / kQ8Block;

    // 先把 A 的每一行量化成 Q8_0：只做一次，被 W 的全部 N 行复用。
    // 缓冲区跨调用复用，热路径上不分配内存（只允许一个线程调 kernel，和线程池的约束一致）
    static std::vector<std::int8_t> aq;
    static std::vector<float> as;
    aq.resize(static_cast<std::size_t>(M * K));
    as.resize(static_cast<std::size_t>(M * nb));
    for (int64 m = 0; m < M; ++m)
        quantize_row_q8_0(a.data() + m * K, aq.data() + m * K, as.data() + m * nb, K);

    // 和 fp32 的 GEMV 一样按输出列切块交给线程池；块内对 M 行依次做 GEMV。
    // 一块 W（≥16 行 × K 个 int8）只有十几 KB，留在 L1/L2 里被 M 行复用
    const int nt = get_num_threads();
    const int64 blk = std::max<int64>(16, (N / (int64{std::max(nt, 1)} * 4) + 3) / 4 * 4);
    const int64 n_blocks = (N + blk - 1) / blk;
    auto task = [&](int64 bi)
    {
        const int64 n0 = bi * blk;
        const int64 rows = std::min(blk, N - n0);
        for (int64 m = 0; m < M; ++m)
            gemv_q8_0(aq.data() + m * K, as.data() + m * nb, wq + n0 * K, ws + n0 * nb,
                      c.data() + m * N + n0, rows, K);
    };
    if (N * K < (int64{1} << 15))
    {
        for (int64 bi = 0; bi < n_blocks; ++bi)
            task(bi);
        return;
    }
    parallel_for(n_blocks, task);
}

// ---------------------------------------------------------------------------
// fp16 权重
// ---------------------------------------------------------------------------
namespace
{

#if defined(__F16C__)
inline std::uint16_t f32_to_f16(float f)
{
    return static_cast<std::uint16_t>(_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT));
}
inline float f16_to_f32(std::uint16_t h)
{
    return _cvtsh_ss(h);
}
#else
// 软件版（只处理正常数和 0，权重里不会出现 inf/nan）：
// half = 1 位符号 + 5 位指数（偏置 15）+ 10 位尾数；float 是 8 位指数（偏置 127）+ 23 位尾数
inline std::uint16_t f32_to_f16(float f)
{
    const std::uint32_t u = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (u >> 16) & 0x8000U;
    const int exp = static_cast<int>((u >> 23) & 0xFF) - 127 + 15;
    std::uint32_t mant = u & 0x7FFFFFU;
    if (exp <= 0) // 太小，按 0 处理（会丢掉 half 的次正规数，误差 < 6e-5）
        return static_cast<std::uint16_t>(sign);
    if (exp >= 31)
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    // 截掉 13 位尾数，最近偶数舍入；进位溢出到指数位正好也是对的
    std::uint32_t h = (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13);
    const std::uint32_t rest = mant & 0x1FFFU;
    if (rest > 0x1000U || (rest == 0x1000U && (h & 1U)))
        ++h;
    return static_cast<std::uint16_t>(sign | h);
}
inline float f16_to_f32(std::uint16_t h)
{
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1FU;
    const std::uint32_t mant = h & 0x3FFU;
    if (exp == 0)
        return std::bit_cast<float>(sign); // 次正规数当 0
    return std::bit_cast<float>(sign | ((exp - 15 + 127) << 23) | (mant << 13));
}
#endif

} // namespace

void convert_f32_to_f16(const float *x, std::uint16_t *y, int64 n)
{
    for (int64 i = 0; i < n; ++i)
        y[i] = f32_to_f16(x[i]);
}

void convert_f16_to_f32(const std::uint16_t *x, float *y, int64 n)
{
    for (int64 i = 0; i < n; ++i)
        y[i] = f16_to_f32(x[i]);
}

// fp16 → fp32 的向量转换（vcvtph2ps）属于 F16C 扩展，不在 AVX2 里，要单独判断
#if INFER_SIMD_AVX2 && defined(__F16C__)
#define INFER_SIMD_F16C 1
#else
#define INFER_SIMD_F16C 0
#endif

namespace
{

#if INFER_SIMD_F16C
inline __m256 load_f16x8(const std::uint16_t *p)
{
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
}
#endif

float dot_f16_scalar(const float *x, const std::uint16_t *w, int64 n)
{
    float s = 0.0F;
    for (int64 i = 0; i < n; ++i)
        s += x[i] * f16_to_f32(w[i]);
    return s;
}

// y[0:n_rows] = W[0:n_rows, :] · x，结构和 fp32 的 gemv_nt 一样：
// 4 行一组共用 x 的加载，每行 2 条 FMA 链（共 8 条）盖住 FMA 的延迟
void gemv_f16(const float *x, const std::uint16_t *w, float *y, int64 n_rows, int64 K)
{
    int64 n = 0;
#if INFER_SIMD_F16C
    const int64 K16 = K / 16 * 16;
    for (; n + 4 <= n_rows; n += 4)
    {
        const std::uint16_t *w0 = w + (n + 0) * K, *w1 = w + (n + 1) * K;
        const std::uint16_t *w2 = w + (n + 2) * K, *w3 = w + (n + 3) * K;
        __m256 a0 = _mm256_setzero_ps(), b0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
        for (int64 k = 0; k < K16; k += 16)
        {
            const __m256 x0 = _mm256_loadu_ps(x + k), x1 = _mm256_loadu_ps(x + k + 8);
            a0 = _mm256_fmadd_ps(load_f16x8(w0 + k), x0, a0);
            b0 = _mm256_fmadd_ps(load_f16x8(w0 + k + 8), x1, b0);
            a1 = _mm256_fmadd_ps(load_f16x8(w1 + k), x0, a1);
            b1 = _mm256_fmadd_ps(load_f16x8(w1 + k + 8), x1, b1);
            a2 = _mm256_fmadd_ps(load_f16x8(w2 + k), x0, a2);
            b2 = _mm256_fmadd_ps(load_f16x8(w2 + k + 8), x1, b2);
            a3 = _mm256_fmadd_ps(load_f16x8(w3 + k), x0, a3);
            b3 = _mm256_fmadd_ps(load_f16x8(w3 + k + 8), x1, b3);
        }
        y[n + 0] = simd::hsum256_ps(_mm256_add_ps(a0, b0)) + dot_f16_scalar(x + K16, w0 + K16, K - K16);
        y[n + 1] = simd::hsum256_ps(_mm256_add_ps(a1, b1)) + dot_f16_scalar(x + K16, w1 + K16, K - K16);
        y[n + 2] = simd::hsum256_ps(_mm256_add_ps(a2, b2)) + dot_f16_scalar(x + K16, w2 + K16, K - K16);
        y[n + 3] = simd::hsum256_ps(_mm256_add_ps(a3, b3)) + dot_f16_scalar(x + K16, w3 + K16, K - K16);
    }
#endif
    for (; n < n_rows; ++n)
        y[n] = dot_f16_scalar(x, w + n * K, K);
}

} // namespace

void matmul_nt_f16(std::span<const float> a, const std::uint16_t *w, std::span<float> c, int64 M,
                   int64 N, int64 K)
{
    const int nt = get_num_threads();
    const int64 blk = std::max<int64>(16, (N / (int64{std::max(nt, 1)} * 4) + 3) / 4 * 4);
    const int64 n_blocks = (N + blk - 1) / blk;
    auto task = [&](int64 bi)
    {
        const int64 n0 = bi * blk;
        const int64 rows = std::min(blk, N - n0);
        for (int64 m = 0; m < M; ++m)
            gemv_f16(a.data() + m * K, w + n0 * K, c.data() + m * N + n0, rows, K);
    };
    if (N * K < (int64{1} << 15))
    {
        for (int64 bi = 0; bi < n_blocks; ++bi)
            task(bi);
        return;
    }
    parallel_for(n_blocks, task);
}

} // namespace inference::kernel
