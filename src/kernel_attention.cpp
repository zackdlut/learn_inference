#include "inference/kernel.hpp"
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

void attention_f32_strided(const float *q, int64 ldq, const float *k, int64 ldk, const float *v,
                           int64 ldv, float *out, int64 ldo, int64 Tq, int64 Tk, int64 n_head,
                           int64 head_size, int64 q_offset)
{
    // 1/√d 这个缩放很关键：点积的方差随维度 d 线性增长，不缩放的话
    // softmax 的输入会越来越大，梯度/概率分布退化成 one-hot。
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));

    std::vector<float> scores(static_cast<std::size_t>(Tk));

    for (int64 h = 0; h < n_head; ++h)
    {
        const int64 hoff = h * head_size;

        for (int64 i = 0; i < Tq; ++i)
        {
            const float *qi = q + i * ldq + hoff;
            // 因果掩码：第 i 个查询的绝对位置是 q_offset+i，只能看到到这里为止
            const int64 visible = std::min(q_offset + i + 1, Tk);

            for (int64 j = 0; j < visible; ++j)
            {
                const float *kj = k + j * ldk + hoff;
                float dot = 0.0F;
                for (int64 d = 0; d < head_size; ++d)
                    dot += qi[d] * kj[d];
                scores[static_cast<std::size_t>(j)] = dot * scale;
            }

            softmax_inplace(scores.data(), visible);

            // 按注意力权重把 V 加权求和
            float *oi = out + i * ldo + hoff;
            std::fill(oi, oi + head_size, 0.0F);
            for (int64 j = 0; j < visible; ++j)
            {
                const float w = scores[static_cast<std::size_t>(j)];
                const float *vj = v + j * ldv + hoff;
                for (int64 d = 0; d < head_size; ++d)
                    oi[d] += w * vj[d];
            }
        }
    }
}

void attention_f32(std::span<const float> q, std::span<const float> k, std::span<const float> v,
                   std::span<float> out, int64 Tq, int64 Tk, int64 n_head, int64 head_size,
                   int64 q_offset)
{
    const int64 C = n_head * head_size;
    attention_f32_strided(q.data(), C, k.data(), C, v.data(), C, out.data(), C, Tq, Tk, n_head,
                          head_size, q_offset);
}

} // namespace inference::kernel