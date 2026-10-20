#pragma once
#include "inference/graph.hpp"
#include "inference/model_file.hpp"
#include "inference/tensor.hpp"
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

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
    // 同上，但输入第 r 行从 k[r * row_stride] 开始（长 n_embd），共 n_rows 行。
    // 这样可以直接传 c_attn 输出 [T, 3C] 里的 k/v 段（行距 3C），不用先拆出来。
    void append_strided(int layer, int pos, const float *k, const float *v, int n_rows,
                        std::int64_t row_stride);
    void set_length(int n) noexcept
    {
        len_ = n;
    }
    // 第 layer 层的整块缓存，按头存放：[n_head][capacity][head_size]。
    // 第 h 个头、第 t 个位置的向量从 h * head_stride() + t * head_size 开始。
    //
    // 为什么不按位置存 [capacity][n_embd]：注意力是一个头一个头算的。按位置存时，
    // 一个头的 64 个数夹在每行 768 个数中间，读一个头要每隔 3 KB 跳着取 256 字节，
    // 硬件预取器跟不上；按头存时一个头的全部历史是一整段连续内存。
    std::span<const float> keys(int layer) const;
    std::span<const float> values(int layer) const;
    std::int64_t head_stride() const noexcept
    {
        return static_cast<std::int64_t>(max_len_) * cfg_.head_size();
    }
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
    // 只有计算图版本有：Executor 自身的耗时（组输入、查表、校验形状、释放中间结果），手写版恒为 0。
    // 另外图版本的 copy 恒为 0：写 KV-Cache 在 attention_qkv 节点里，计入 attention
    double overhead = 0;
    int forwards = 0; // forward 被调用的次数

    double total() const noexcept
    {
        return embed + layernorm + qkv + copy + attention + attn_proj + residual + mlp_fc + gelu +
               mlp_proj + lm_head + sample + overhead;
    }
};

// Q8_0 量化后的矩阵 [rows, cols]：每行每 32 个数一组，q 存 int8，scale 存每组的缩放系数。
// 用 Tensor 存而不是 std::vector，计算图可以直接把它们登记成权重（共享内存，不拷贝）。
struct QMatrixQ8
{
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    Tensor q;     // int8 [rows, cols]
    Tensor scale; // fp32 [rows, cols / 32]

    static QMatrixQ8 quantize(const Tensor &w); // w 是连续的 [rows, cols]，cols 是 32 的倍数
    bool empty() const noexcept
    {
        return q.empty();
    }
    std::size_t bytes() const noexcept
    {
        return static_cast<std::size_t>(q.numel()) +
               static_cast<std::size_t>(scale.numel()) * sizeof(float);
    }
};

// 量化哪些矩阵、用什么格式。默认值就是 quantize_q8_0() 采用的方案；
// 其它组合用来做实验（bench/bench_quant.cpp 逐个矩阵排查精度损失）。
struct QuantOptions
{
    // 每层的 4 类矩阵是否量化成 Q8_0
    bool qkv = true;
    bool attn_proj = true;
    bool mlp_fc = true;
    bool mlp_proj = true;
    enum class LmHead
    {
        fp32,
        fp16,
        q8_0
    };
    LmHead lm_head = LmHead::fp16;
    // 只量化权重：量化后立即反量化回 fp32，仍走 fp32 kernel，激活不量化。
    // 速度没有任何提升，只用来把「权重误差」和「激活误差」拆开看
    bool weight_only = false;
};

class GPT
{
  public:
    static GPT load(const std::string &model_path);

    // 每层的 4 个矩阵量化成 Q8_0 并释放 fp32 版本；lm_head 存一份 fp16。
    // lm_head 不用 Q8_0：它的输入（ln_f 输出）有比普通通道大 ~300 倍的离群通道，
    // 激活和权重按组量化都会被它毁掉精度（困惑度 +8%），fp16 + fp32 激活则几乎无损。
    // 词嵌入 wte 仍保留 fp32 用来查表。decode 每 token 读的权重从 ~494 MB 降到 ~173 MB。
    void quantize_q8_0()
    {
        quantize(QuantOptions{});
    }
    // 按 opt 量化，只能调用一次
    void quantize(const QuantOptions &opt);
    bool is_quantized() const noexcept
    {
        return quantized_;
    }
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

    // ---- 计算图版本 ----
    // 和上面的手写版算的是同一个模型，但前向不再是一段写死的 C++，而是先建一张 Graph
    // （embedding → 12 × [ln → c_attn → attention → proj → + → ln → fc → gelu → proj → +] → ln_f → lm_head），
    // 再交给 Executor 按拓扑序执行。手写版保留下来做对拍和性能基线。
    //
    // Graph 的输入形状是建图时定死的，所以每种 (T, last_only) 各建一张并缓存起来：
    // decode 恒为 T=1，只建一次；prefill 每种 prompt 长度建一次。
    // 量化之后矩阵乘节点换成 linear_q8 / linear_f16（quantize() 会清掉已缓存的 fp32 图）。

    // tokens 和 positions 两个输入都是 [T]；last_only 时只输出最后一个位置的 logits [1, n_vocab]
    Graph build_graph(int T, bool last_only) const;
    Tensor forward_graph(std::span<const int> tokens);
    Tensor forward_one_graph(int token, int pos, KVCache &cache);
    // stats 非空时 Executor 给每个节点计时，按节点名归到 GenStats::*_ops 的对应项，
    // 可以和 generate 的同名项逐项对照
    std::vector<int> generate_graph(std::span<const int> prompt, int max_new_tokens,
                                    Sampler &sampler, GenStats *stats = nullptr,
                                    const std::function<void(int)> &on_token = {});

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
        // quantize_q8_0() 之后使用这几份，对应的 fp32 权重被释放
        QMatrixQ8 attn_wq, attn_proj_wq, fc_wq, proj_wq;
    };
    // 量化之后 lm_head 用下面三者之一（都是 [n_vocab, C]），都为空时直接用 wte_
    Tensor lm_head_f16_; // fp16 [n_vocab, C]
    QMatrixQ8 lm_head_q_;
    Tensor lm_head_f32_; // weight_only 实验：量化再反量化的 wte，不影响词嵌入查表
    bool quantized_ = false;
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

    // 键是 (T, last_only)
    std::map<std::pair<int, bool>, std::unique_ptr<Graph>> graphs_;
    Tensor run_graph(std::span<const int> tokens, int start_pos, KVCache *cache, bool last_only);

    // generate / generate_graph 共用的 prefill + decode 循环。
    // step(tokens, start_pos, cache) 处理这批 token 并返回最后一个位置的 logits
    using StepFn = std::function<Tensor(std::span<const int>, int, KVCache &)>;
    std::vector<int> generate_with(const StepFn &step, std::span<const int> prompt,
                                   int max_new_tokens, Sampler &sampler, GenStats *stats,
                                   const std::function<void(int)> &on_token);
};

} // namespace inference