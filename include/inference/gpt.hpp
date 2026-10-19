#pragma once
#include "inference/model_file.hpp"
#include "inference/tensor.hpp"
#include <functional>

namespace inference
{
// ---------------------------------------------------------------------------
// GPT-2 的结构
//
//   token ids [T]
//        │
//   ┌────▼─────────────────────────────┐
//   │ wte[token] + wpe[position]       │  词嵌入 + 位置嵌入
//   └────┬─────────────────────────────┘
//        │  x: [T, 768]
//   ╔════▼═════════════════════════════╗
//   ║  Transformer Block  × 12          ║
//   ║                                   ║
//   ║    x ──┬──→ LayerNorm             ║
//   ║        │        │                 ║
//   ║        │   Multi-Head Attention   ║   ← 唯一让 token 之间交换信息的地方
//   ║        │        │                 ║
//   ║        └────→  (+)  残差           ║
//   ║                 │                 ║
//   ║        ┌────────┤                 ║
//   ║        │   LayerNorm              ║
//   ║        │        │                 ║
//   ║        │       MLP (768→3072→768) ║   ← 每个 token 独立处理
//   ║        │        │                 ║
//   ║        └────→  (+)  残差           ║
//   ╚════┬══════════════════════════════╝
//        │
//   ┌────▼─────────────────────────────┐
//   │ LayerNorm                        │
//   │ × wteᵀ  (和词嵌入共享权重)         │  输出 logits [T, 50257]
//   └──────────────────────────────────┘
//
// 124M 参数里，注意力占 28M，MLP 占 57M，词嵌入占 38M。
// 所以推理时间大头在 MLP，不是很多人以为的注意力。
// ---------------------------------------------------------------------------

struct GPTConfig
{
    int n_layer = 12;
    int n_head = 12;
    int n_embd = 768;
    int n_vocab = 50257;
    int n_ctx = 1024;
    int head_size() const
    {
        return n_embd / n_head;
    }
};

// ---------------------------------------------------------------------------
// KV-Cache：LLM 推理最重要的优化
//
//   不用 cache：每生成一个词，整段重算一遍
//     第 1 步: [The]                       算 1 个
//     第 2 步: [The capital]               算 2 个  ← "The" 白算了
//     第 3 步: [The capital of]            算 3 个  ← 前两个又白算
//     总计算量 O(n²)
//
//   用 cache：把每层算过的 K、V 存起来
//     ┌──────── KV Cache ────────┐
//     │ K: [The][capital]  ...   │
//     │ V: [The][capital]  ...   │
//     └──────────────────────────┘
//     第 3 步: 只算 [of] 的 Q/K/V，和 cache 拼起来做注意力
//     总计算量 O(n)
//
// 代价是内存：每层要存 2 × n_ctx × n_embd 个 float。
// GPT-2 满上下文时 = 12 层 × 2 × 1024 × 768 × 4 字节 ≈ 72 MB。
// 大模型上 KV-Cache 往往比权重还占地方，所以有了 MQA/GQA/PagedAttention 这些优化。
// ---------------------------------------------------------------------------

class KVCache
{
  public:
    KVCache(const GPTConfig &cfg, int max_len);
    int length() const noexcept
    {
        return len_;
    }
    int capacity() const noexcept
    {
        return max_len_;
    }
    void reset() noexcept
    {
        len_ = 0;
    }
    // 把若干行 K/V 写进第 layer 层缓存的 [pos, pos + k.size()/n_embd) 位置。
    // prefill 时一次写整段，decode 时一次写一行，走的是同一个函数。
    void append(int layer, int pos, std::span<const float> k, std::span<const float> v);
    void set_length(int n) noexcept
    {
        len_ = n;
    }
    std::span<const float> keys(int layer) const;
    std::span<const float> values(int layer) const;
    std::size_t bytes() const noexcept;

  private:
    const GPTConfig &cfg_;
    int len_ = 0;
    int max_len_ = 0;
    std::vector<Tensor> k_;
    std::vector<Tensor> v_;
};

// 采样策略：模型给出 50257 个 logits，怎么从里面挑一个 token
struct SamplerConfig
{

    float temperature = 1.0F; // 越低越保守，0 表示贪心（永远选最大的）
    int top_k = 0;
    float top_p = 1.0f;
    std::uint64_t seed = 42;
};

class Sampler
{
  public:
    explicit Sampler(const SamplerConfig &cfg) : cfg_(cfg), state_(cfg.seed ? cfg.seed : 1) {}
    int sample(std::span<const float> logits);

  private:
    SamplerConfig cfg_;
    std::uint64_t state_;
    float next_uniform();
};

// 按算子类别累加的耗时（毫秒）。优化的第一步永远是测量：先知道时间花在哪。
struct OpProfile
{
    double embed = 0;
    double layernorm = 0;
    double qkv = 0;       // c_attn 矩阵乘
    double copy = 0;      // 拆 q/k/v、写 KV-Cache、把 cache 拷出来
    double attention = 0; // 注意力核心（QKᵀ、softmax、×V）
    double attn_proj = 0;
    double residual = 0;
    double mlp_fc = 0;
    double gelu = 0;
    double mlp_proj = 0;
    double lm_head = 0;
    double sample = 0;
    int forwards = 0; // forward_impl 被调用的次数

    double total() const noexcept
    {
        return embed + layernorm + qkv + copy + attention + attn_proj + residual + mlp_fc + gelu +
               mlp_proj + lm_head + sample;
    }
};

class GPT
{
  public:
    static GPT load(const std::string &model_path);
    const GPTConfig &config() const noexcept
    {
        return cfg_;
    }

    // 一次处理整个序列（prefill 阶段），返回全部位置的 logits [T, n_vocab]。
    // 不用 KV-Cache，用于对拍和理解基本流程。
    Tensor forward(std::span<const int> tokens);

    // 带 KV-Cache 的单步推理：只处理一个新 token，返回它的 logits [n_vocab]。
    // 这是实际生成时走的路径
    Tensor forward_one(int token, int pos, KVCache &cache);

    // 生成：先把 prompt 喂进去（prefill），再逐个 token 往外吐（decode）
    struct GenStats
    {
        double prefill_ms = 0;
        double decode_ms = 0;
        int generated = 0;
        OpProfile prefill_ops;
        OpProfile decode_ops;
    };

    std::vector<int> generate(std::span<const int> prompt, int max_new_tokens, Sampler &sampler,
                              GenStats *stats = nullptr,
                              const std::function<void(int)> &on_token = {});

    // 逐层中间结果，用来和 Python 参考实现对拍
    struct Trace
    {
        Tensor embed;
        Tensor l0_attn_out;
        Tensor l0_out;
        Tensor final_norm;
        Tensor logits;
    };
    Trace forward_trace(std::span<const int> tokens);

  private:
    GPTConfig cfg_;

    struct Layer
    {
        Tensor ln1_w, ln1_b, ln2_w, ln2_b;
        // 矩阵权重在加载时已从 Conv1D 的 [in, out] 转成 [out, in]，配合 ops::linear 使用
        Tensor attn_w, attn_b; // c_attn: [3*n_embd, n_embd]
        Tensor attn_proj_w, attn_proj_b;
        Tensor fc_w, fc_b;     // c_fc:   [4*n_embd, n_embd]
        Tensor proj_w, proj_b; // c_proj: [n_embd, 4*n_embd]
    };
    std::vector<Layer> layers_;
    Tensor wte_, wpe_, lnf_w_, lnf_b_;
    OpProfile *profile_ = nullptr; // 非空时 forward_impl 把各算子耗时累加进去

    // 前向计算的中间结果缓冲区，跨调用复用。decode 每个 token 都是 T=1，
    // 第一次分配好之后就再也不用向系统要内存了（llama.cpp 的 compute buffer 是同一个思路）。
    struct Workspace
    {
        std::vector<float> x;   // [T, C]   残差流，所有子层的输出都加到这里
        std::vector<float> h;   // [T, C]   layernorm / 投影的输出
        std::vector<float> qkv; // [T, 3C]  c_attn 输出，q/k/v 按行距 3C 直接读
        std::vector<float> att; // [T, C]   注意力输出
        std::vector<float> fc;  // [T, 4C]  MLP 中间层
    };
    Workspace ws_;
    // forward / forward_one / forward_trace 全都走这一个实现，区别只在参数：
    //   cache == nullptr → 不用缓存，从头算整段
    //   cache != nullptr → 读写缓存，只算新来的这几个 token
    Tensor forward_impl(std::span<const int> tokens, int start_pos, KVCache *cache, Trace *trace,
                        bool last_only);
};

} // namespace inference