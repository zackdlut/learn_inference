#pragma once
// kernel 内部共用的 AVX2 数学函数，不对外暴露。

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define INFER_SIMD_AVX2 1
#else
#define INFER_SIMD_AVX2 0
#endif

namespace inference::kernel::simd
{

#if INFER_SIMD_AVX2
// 一次算 8 个 exp(x)。标准库没有 SIMD 版的 exp/tanh，逐个调用每个要 ~20 ns。
//
// 做法（Cephes 库的 expf，相对误差 ~2e-7，和 std::exp 同一量级）：
//   1. 拆成 exp(x) = 2^n × exp(r)，n = round(x / ln2)，|r| ≤ ln2/2
//   2. exp(r) 在这么小的区间上用 5 次多项式就足够精确
//   3. 2^n 直接拼 IEEE 754 的指数位：(n + 127) << 23
// 输入会被截到 [-88.38, 88.38]：exp(-inf) 得到 ~1e-38 而不是精确的 0。
inline __m256 exp256_ps(__m256 x)
{
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949F));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949F));

    __m256 n = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341F), _mm256_set1_ps(0.5F));
    n = _mm256_floor_ps(n);
    // r = x - n·ln2，ln2 拆成高低两部分减，避免大 n 时丢精度
    x = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375F), x);
    x = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4F), x);

    __m256 y = _mm256_set1_ps(1.9875691500E-4F);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3F));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3F));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2F));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1F));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1F));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0F)));

    const __m256i e =
        _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

// 8 个 float 加成一个：8→4→2→1 折半相加
inline float hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// 8 个 float 取最大
inline float hmax256_ps(__m256 v)
{
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(m);
}
#endif

} // namespace inference::kernel::simd
