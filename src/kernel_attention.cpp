#include "inference/kernel.hpp"
#include "simd_math.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace inference::kernel
{
// 一行的 softmax，原地做。减最大值是为了数值稳定 —— 这一课阶段 2 讲过：
// exp(800) 会溢出成 inf，减掉最大值后指数最大是 exp(0)=1，永远安全。
void softmax_inplace(float *s, int64 n)
{
    float m = s[0];
    for (int64 i = 1; i < n; ++i)
        m = std::max(m, s[i]);
    float sum = 0.0F;
    for (int64 i = 0; i < n; ++i)
    {
        s[i] = std::exp(s[i] - m);
        sum += s[i];
    }
    const float inv = 1.0F / sum;
    for (int64 i = 0; i < n; ++i)
        s[i] *= inv;
}

// ---------------------------------------------------------------------------
// 朴素版：三重标量循环、单线程。保留下来当参考实现（测试对拍、benchmark 对比）。
// ---------------------------------------------------------------------------
void attention_f32_naive(std::span<const float> q, std::span<const float> k,
                         std::span<const float> v, std::span<float> out, int64 Tq, int64 Tk,
                         int64 n_head, int64 head_size, int64 q_offset)
{
    const int64 C = n_head * head_size;
    // 1/√d 这个缩放很关键：点积的方差随维度 d 线性增长，不缩放的话
    // softmax 的输入会越来越大，梯度/概率分布退化成 one-hot。
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));

    std::vector<float> scores(static_cast<std::size_t>(Tk));

    for (int64 h = 0; h < n_head; ++h)
    {
        const int64 hoff = h * head_size;

        for (int64 i = 0; i < Tq; ++i)
        {
            const float *qi = q.data() + i * C + hoff;
            // 因果掩码：第 i 个查询的绝对位置是 q_offset+i，只能看到到这里为止
            const int64 visible = std::min(q_offset + i + 1, Tk);

            for (int64 j = 0; j < visible; ++j)
            {
                const float *kj = k.data() + j * C + hoff;
                float dot = 0.0F;
                for (int64 d = 0; d < head_size; ++d)
                    dot += qi[d] * kj[d];
                scores[static_cast<std::size_t>(j)] = dot * scale;
            }

            softmax_inplace(scores.data(), visible);

            // 按注意力权重把 V 加权求和
            float *oi = out.data() + i * C + hoff;
            std::fill(oi, oi + head_size, 0.0F);
            for (int64 j = 0; j < visible; ++j)
            {
                const float w = scores[static_cast<std::size_t>(j)];
                const float *vj = v.data() + j * C + hoff;
                for (int64 d = 0; d < head_size; ++d)
                    oi[d] += w * vj[d];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 快速版：SIMD + 按 (头, 查询) 并行
//
// 每个 (头 h, 查询 i) 的计算完全独立：读 q 的一段、K/V 的同一列段，写 out 的一段，
// 互不重叠。所以把 n_head × Tq 个这样的任务交给线程池，不需要任何同步。
//   decode：Tq = 1，12 个头 = 12 个任务
//   prefill：Tq = T，12 × T 个任务，越长越能喂饱所有线程
//
// 每个任务内部三步都用 AVX2：
//   1. 分数 s_j = q·k_j：head_size = 64 = 8 个 __m256，点积沿 d 向量化
//   2. softmax：max / exp / 求和 / 缩放都 8 个一组，exp 用 simd::exp256_ps
//   3. out = Σ s_j · v_j：沿 d 向量化的 axpy，out 的 64 个数常驻 L1
// ---------------------------------------------------------------------------
namespace
{

// 任务太小时并行不划算：分发一批任务有几微秒的固定开销，12 个头分给 7 个线程还要跑两轮。
// 按「总共要读的 K/V 元素数」n_head × Tq × Tk × head_size 估算。
// 本机实测（热缓存、冷缓存都测过）decode 的交叉点在 Tk ≈ 384~512：
// Tk ≤ 256 并行反而慢，Tk = 384 持平，Tk ≥ 512 开始赚。300k ≈ decode 的 Tk = 390。
constexpr int64 kAttnParallelMinWork = 300'000;
int64 g_attn_parallel_min_work = kAttnParallelMinWork;

float dot(const float *a, const float *b, int64 n)
{
    int64 d = 0;
    float s = 0.0F;
#if INFER_SIMD_AVX2
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (; d + 16 <= n; d += 16)
    {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d + 8), _mm256_loadu_ps(b + d + 8), acc1);
    }
    for (; d + 8 <= n; d += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), acc0);
    s = simd::hsum256_ps(_mm256_add_ps(acc0, acc1));
#endif
    for (; d < n; ++d)
        s += a[d] * b[d];
    return s;
}

// y += w * x
void axpy(float w, const float *x, float *y, int64 n)
{
    int64 d = 0;
#if INFER_SIMD_AVX2
    const __m256 wv = _mm256_set1_ps(w);
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(y + d, _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + d), _mm256_loadu_ps(y + d)));
#endif
    for (; d < n; ++d)
        y[d] += w * x[d];
}

void softmax_fast(float *s, int64 n)
{
    int64 i = 0;
    float m = s[0];
#if INFER_SIMD_AVX2
    if (n >= 8)
    {
        __m256 mv = _mm256_loadu_ps(s);
        for (i = 8; i + 8 <= n; i += 8)
            mv = _mm256_max_ps(mv, _mm256_loadu_ps(s + i));
        m = simd::hmax256_ps(mv);
    }
#endif
    for (; i < n; ++i)
        m = std::max(m, s[i]);

    i = 0;
    float sum = 0.0F;
#if INFER_SIMD_AVX2
    const __m256 mv = _mm256_set1_ps(m);
    __m256 sv = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
    {
        const __m256 e = simd::exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(s + i), mv));
        _mm256_storeu_ps(s + i, e);
        sv = _mm256_add_ps(sv, e);
    }
    sum = simd::hsum256_ps(sv);
#endif
    for (; i < n; ++i)
    {
        s[i] = std::exp(s[i] - m);
        sum += s[i];
    }

    const float inv = 1.0F / sum;
    i = 0;
#if INFER_SIMD_AVX2
    const __m256 iv = _mm256_set1_ps(inv);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(s + i, _mm256_mul_ps(_mm256_loadu_ps(s + i), iv));
#endif
    for (; i < n; ++i)
        s[i] *= inv;
}

} // namespace

void attention_f32_strided(HeadView q, HeadView k, HeadView v, float *out, int64 ldo, int64 Tq,
                           int64 Tk, int64 n_head, int64 head_size, int64 q_offset)
{
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));

    auto one_task = [&](int64 task)
    {
        const int64 h = task / Tq;
        const int64 i = task % Tq;
        const float *qi = q.data + h * q.head_stride + i * q.row_stride;
        const float *kh = k.data + h * k.head_stride;
        const float *vh = v.data + h * v.head_stride;
        const int64 visible = std::min(q_offset + i + 1, Tk);

        // 每个线程一份分数缓冲区，跨调用复用，不在热路径上分配内存
        thread_local std::vector<float> scores;
        if (scores.size() < static_cast<std::size_t>(visible))
            scores.resize(static_cast<std::size_t>(visible));
        float *s = scores.data();

        for (int64 j = 0; j < visible; ++j)
            s[j] = dot(qi, kh + j * k.row_stride, head_size) * scale;

        softmax_fast(s, visible);

        float *oi = out + i * ldo + h * head_size;
        std::fill(oi, oi + head_size, 0.0F);
        for (int64 j = 0; j < visible; ++j)
            axpy(s[j], vh + j * v.row_stride, oi, head_size);
    };

    const int64 n_tasks = n_head * Tq;
    if (n_head * Tq * Tk * head_size < g_attn_parallel_min_work)
    {
        for (int64 t = 0; t < n_tasks; ++t)
            one_task(t);
        return;
    }
    parallel_for(n_tasks, one_task);
}

int64 set_attention_parallel_min_work(int64 work) noexcept
{
    const int64 old = g_attn_parallel_min_work;
    g_attn_parallel_min_work = work < 0 ? kAttnParallelMinWork : work;
    return old;
}

void attention_f32(std::span<const float> q, std::span<const float> k, std::span<const float> v,
                   std::span<float> out, int64 Tq, int64 Tk, int64 n_head, int64 head_size,
                   int64 q_offset)
{
    const int64 C = n_head * head_size;
    attention_f32_strided({q.data(), C, head_size}, {k.data(), C, head_size},
                          {v.data(), C, head_size}, out.data(), C, Tq, Tk, n_head, head_size,
                          q_offset);
}

} // namespace inference::kernel
