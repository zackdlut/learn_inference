#include "inference/gpt.hpp"
#include "inference/kernel.hpp"
#include "inference/op.hpp"
#include "inference/ops.hpp"
#include "inference/tensor.hpp"

#include <cmath>
#include <format>
#include <stdexcept>
#include <vector>

namespace inference
{
namespace
{

Tensor as_contiguous(const Tensor &x)
{
    return x.is_contiguous() ? x : x.contiguous();
}

// 图里的 Value 只能是 float Tensor，token id 和位置都以 float 形式进来。
// 2^24 以内的整数 float 都能精确表示，GPT-2 的词表（50257）和上下文（1024）远小于它。
std::vector<int> to_ids(const Tensor &t)
{
    const Tensor c = as_contiguous(t);
    std::vector<int> ids;
    ids.reserve(static_cast<std::size_t>(c.numel()));
    for (const float v : c.data())
        ids.push_back(static_cast<int>(std::lround(v)));
    return ids;
}

} // namespace

// 词嵌入查表：ids [T] + table [V, C] → [T, C]。位置嵌入也用它，ids 换成位置即可
class EmbeddingOp final : public Op
{
  public:
    std::string type() const override
    {
        return "embedding";
    }
    int num_inputs() const override
    {
        return 2;
    }
    Shape inference_shape(std::span<const Shape> input_shapes,
                          const Attributes & /*attributes*/) const override
    {
        const Shape &ids = input_shapes[0];
        const Shape &table = input_shapes[1];
        if (ids.size() != 1 || table.size() != 2)
        {
            throw std::runtime_error(std::format("embedding: ids 要一维、table 要二维，拿到 {} 和 {}",
                                                 to_string(ids), to_string(table)));
        }
        return Shape{ids[0], table[1]};
    }
    Tensor compute(const OpContext *context) const override
    {
        const std::vector<int> ids = to_ids(*context->inputs[0]);
        return ops::embedding(*context->inputs[1], ids);
    }
};

// 多头因果注意力，直接吃 c_attn 的输出 [T, 3C]（每行是 [q | k | v]），不先拆成三个张量。
//   输入:  qkv [T, 3C]，positions [T]（这批 token 的绝对位置，只用第一个当 q_offset）
//   属性:  n_head，layer（用第几层的 KV-Cache）
//   输出:  [T, C]
// OpContext::kv_cache 为空时 k/v 就是 qkv 里这 T 行；不为空时先把新 k/v 写进缓存，
// 再对缓存里的全部历史做注意力 —— 和 GPT::forward_impl 是同一套逻辑。
class AttentionQKVOp final : public Op
{
  public:
    std::string type() const override
    {
        return "attention_qkv";
    }
    int num_inputs() const override
    {
        return 2;
    }
    Shape inference_shape(std::span<const Shape> input_shapes,
                          const Attributes &attributes) const override
    {
        const Shape &qkv = input_shapes[0];
        const Shape &pos = input_shapes[1];
        if (qkv.size() != 2 || qkv[1] % 3 != 0)
        {
            throw std::runtime_error(
                std::format("attention_qkv: qkv 应为 [T, 3C]，拿到 {}", to_string(qkv)));
        }
        if (pos != Shape{qkv[0]})
        {
            throw std::runtime_error(std::format("attention_qkv: positions 应为 [{}]，拿到 {}",
                                                 qkv[0], to_string(pos)));
        }
        const int64 C = qkv[1] / 3;
        const int64 n_head = attributes.get<std::int64_t>("n_head");
        if (n_head <= 0 || C % n_head != 0)
        {
            throw std::runtime_error(
                std::format("attention_qkv: 通道数 {} 无法被 {} 个头整除", C, n_head));
        }
        return Shape{qkv[0], C};
    }
    Tensor compute(const OpContext *context) const override
    {
        const Tensor qkv_t = as_contiguous(*context->inputs[0]);
        const Tensor pos_t = as_contiguous(*context->inputs[1]);
        const int64 T = qkv_t.dim(0);
        const int64 C3 = qkv_t.dim(1);
        const int64 C = C3 / 3;
        const int64 H = context->attributes->get<std::int64_t>("n_head");
        const int64 hs = C / H;
        const int start_pos = static_cast<int>(std::lround(pos_t.data()[0]));
        const float *qkv = qkv_t.data().data();

        kernel::HeadView kv_k{qkv + C, C3, hs};
        kernel::HeadView kv_v{qkv + 2 * C, C3, hs};
        int64 Tk = T;
        if (KVCache *cache = context->kv_cache)
        {
            const int layer = static_cast<int>(context->attributes->get<std::int64_t>("layer"));
            cache->append_strided(layer, start_pos, qkv + C, qkv + 2 * C, static_cast<int>(T), C3);
            kv_k = {cache->keys(layer).data(), hs, cache->head_stride()};
            kv_v = {cache->values(layer).data(), hs, cache->head_stride()};
            Tk = cache->length();
        }

        Tensor out = Tensor::zeros({T, C});
        kernel::attention_f32_strided({qkv, C3, hs}, kv_k, kv_v, out.data().data(), C, T, Tk, H, hs,
                                      start_pos);
        return out;
    }
};

// 取最后 n 行：[R, C] → [n, C]。生成时只要最后一个位置的 logits，
// 在 ln_f 之前切掉，后面的 layernorm 和 lm_head 都只算 n 行
class LastRowsOp final : public Op
{
  public:
    std::string type() const override
    {
        return "last_rows";
    }
    int num_inputs() const override
    {
        return 1;
    }
    Shape inference_shape(std::span<const Shape> input_shapes,
                          const Attributes &attributes) const override
    {
        const Shape &x = input_shapes[0];
        const int64 n = attributes.get<std::int64_t>("n");
        if (x.size() != 2 || n <= 0 || n > x[0])
        {
            throw std::runtime_error(
                std::format("last_rows: 不能从 {} 里取最后 {} 行", to_string(x), n));
        }
        return Shape{n, x[1]};
    }
    Tensor compute(const OpContext *context) const override
    {
        const Tensor &x = *context->inputs[0];
        const int64 n = context->attributes->get<std::int64_t>("n");
        return x.slice(0, x.dim(0) - n, x.dim(0)).contiguous();
    }
};

REGISTER_OP(EmbeddingOp)
REGISTER_OP(AttentionQKVOp)
REGISTER_OP(LastRowsOp)

} // namespace inference
