#include "inference/gpt.hpp"
#include "inference/graph.hpp"
#include "inference/op.hpp"
#include "inference/tensor.hpp"

#include <chrono>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace inference
{

namespace
{
[[noreturn]] void fail(const std::string &msg)
{
    throw std::runtime_error("[infer::GPT] " + msg);
}

// 节点名 → OpProfile 里的哪一项，和 forward_impl 里各个 ScopedTimer 的划分对齐。
// 按节点名而不是节点类型分：5 种矩阵乘的类型都是 linear（或 linear_q8），只看类型分不开。
double OpProfile::*profile_field(std::string_view name)
{
    if (const auto dot = name.find('.'); dot != std::string_view::npos)
        name = name.substr(dot + 1); // 去掉 "h3." 这样的层前缀
    static constexpr std::pair<std::string_view, double OpProfile::*> kTable[] = {
        {"tok_emb", &OpProfile::embed},      {"pos_emb", &OpProfile::embed},
        {"embed", &OpProfile::embed},        {"ln_1", &OpProfile::layernorm},
        {"ln_2", &OpProfile::layernorm},     {"ln_f", &OpProfile::layernorm},
        {"c_attn", &OpProfile::qkv},         {"attn", &OpProfile::attention},
        {"attn_proj", &OpProfile::attn_proj}, {"res_attn", &OpProfile::residual},
        {"res_mlp", &OpProfile::residual},   {"c_fc", &OpProfile::mlp_fc},
        {"gelu", &OpProfile::gelu},          {"mlp_proj", &OpProfile::mlp_proj},
        {"last", &OpProfile::lm_head},       {"lm_head", &OpProfile::lm_head},
    };
    for (const auto &[key, field] : kTable)
        if (key == name)
            return field;
    return &OpProfile::overhead;
}
} // namespace

// ---------------------------------------------------------------------------
// 建图：和 forward_impl 一一对应，只是每一步不再立即计算，而是登记成一个节点。
//
//   tokens ─ embedding(wte) ─┐
//                            (+) ─ x
//   positions ─ embedding(wpe)┘
//
//   每层:  h   = layer_norm(x, ln_1)
//          qkv = linear(h, c_attn)                 [T, 3C]
//          a   = attention_qkv(qkv, positions)     读写 Executor 交进来的 KV-Cache
//          x   = add(x, linear(a, attn.c_proj))
//          h   = layer_norm(x, ln_2)
//          x   = add(x, linear(gelu(linear(h, c_fc)), mlp.c_proj))
//
//   last_rows(x, 1)（只在 last_only 时）→ layer_norm(ln_f) → linear(wte) → logits
//
// 量化之后，图的结构不变，只是矩阵乘节点换了类型：
//   Q8_0 的矩阵  →  linear_q8(x, q, scale, bias)
//   fp16 lm_head →  linear_f16(x, w16)
// 选哪种节点完全由权重当前的存储决定，和 forward_impl 里 matmul_weight 的分支一一对应。
//
// 权重 Tensor 内部是 shared_ptr，add_weight 只是多一个引用，建多少张图都不拷贝权重数据。
// ---------------------------------------------------------------------------
Graph GPT::build_graph(int T, bool last_only) const
{
    if (T <= 0 || T > cfg_.n_ctx)
        fail(std::format("build_graph: T = {} 不在 [1, {}] 内", T, cfg_.n_ctx));

    const auto ln_attr = [] { return Attributes{}.set("eps", 1e-5F); };

    Graph g;
    const ValueId tokens = g.add_input("tokens", {T});
    const ValueId positions = g.add_input("positions", {T});
    const ValueId wte = g.add_weight("wte", wte_);
    const ValueId wpe = g.add_weight("wpe", wpe_);

    const ValueId tok_emb = g.add_node("embedding", "tok_emb", {tokens, wte});
    const ValueId pos_emb = g.add_node("embedding", "pos_emb", {positions, wpe});
    ValueId x = g.add_node("add", "embed", {tok_emb, pos_emb});

    for (int li = 0; li < cfg_.n_layer; ++li)
    {
        const Layer &L = layers_[static_cast<std::size_t>(li)];
        const std::string p = std::format("h{}.", li);
        const auto w = [&](const char *name, const Tensor &t) { return g.add_weight(p + name, t); };
        // 带 bias 的矩阵乘：有 Q8_0 版本就用 linear_q8，否则用 fp32 的 linear
        const auto linear = [&](const std::string &name, ValueId in, const Tensor &wf,
                                const QMatrixQ8 &wq, const Tensor &bias)
        {
            const std::string n = p + name;
            if (!wq.empty())
                return g.add_node("linear_q8", n,
                                  {in, g.add_weight(n + ".q", wq.q), g.add_weight(n + ".s", wq.scale),
                                   g.add_weight(n + ".b", bias)});
            return g.add_node("linear", n,
                              {in, g.add_weight(n + ".w", wf), g.add_weight(n + ".b", bias)});
        };

        // 注意力子层
        ValueId h = g.add_node("layer_norm", p + "ln_1", {x, w("ln_1.w", L.ln1_w), w("ln_1.b", L.ln1_b)},
                               ln_attr());
        const ValueId qkv = linear("c_attn", h, L.attn_w, L.attn_wq, L.attn_b);
        const ValueId att = g.add_node("attention_qkv", p + "attn", {qkv, positions},
                                       Attributes{}
                                           .set("n_head", std::int64_t{cfg_.n_head})
                                           .set("layer", std::int64_t{li}));
        const ValueId att_proj =
            linear("attn_proj", att, L.attn_proj_w, L.attn_proj_wq, L.attn_proj_b);
        x = g.add_node("add", p + "res_attn", {x, att_proj});

        // MLP 子层
        h = g.add_node("layer_norm", p + "ln_2", {x, w("ln_2.w", L.ln2_w), w("ln_2.b", L.ln2_b)},
                       ln_attr());
        const ValueId fc = linear("c_fc", h, L.fc_w, L.fc_wq, L.fc_b);
        const ValueId act = g.add_node("gelu", p + "gelu", {fc});
        const ValueId mlp_proj = linear("mlp_proj", act, L.proj_w, L.proj_wq, L.proj_b);
        x = g.add_node("add", p + "res_mlp", {x, mlp_proj});
    }

    // layernorm 逐行独立，先切出最后一行再 norm，结果和 norm 完再切一样，但省掉 T-1 行的计算
    if (last_only)
        x = g.add_node("last_rows", "last", {x}, Attributes{}.set("n", std::int64_t{1}));
    const ValueId h = g.add_node("layer_norm", "ln_f",
                                 {x, g.add_weight("ln_f.w", lnf_w_), g.add_weight("ln_f.b", lnf_b_)},
                                 ln_attr());
    // 没量化时 weight tying：lm_head 直接复用词嵌入那个 Value，wte 在图里只登记一次。
    // 量化后 wte 仍是 fp32（给 embedding 查表），lm_head 另用一份压缩过的
    ValueId logits = -1;
    if (!lm_head_f16_.empty())
        logits = g.add_node("linear_f16", "lm_head", {h, g.add_weight("lm_head.f16", lm_head_f16_)});
    else if (!lm_head_q_.empty())
        logits = g.add_node("linear_q8", "lm_head",
                            {h, g.add_weight("lm_head.q", lm_head_q_.q),
                             g.add_weight("lm_head.s", lm_head_q_.scale)});
    else if (lm_head_f32_.numel() > 0)
        logits = g.add_node("linear", "lm_head", {h, g.add_weight("lm_head.w", lm_head_f32_)});
    else
        logits = g.add_node("linear", "lm_head", {h, wte});
    g.mark_output(logits);
    g.finalize();
    return g;
}

Tensor GPT::run_graph(std::span<const int> tokens, int start_pos, KVCache *cache, bool last_only)
{
    const auto T = static_cast<int>(tokens.size());
    if (T == 0)
        fail("forward_graph: token 序列是空的");
    if (start_pos + T > cfg_.n_ctx)
        fail(std::format("序列长度 {} 超过 GPT-2 的上下文上限 {}", start_pos + T, cfg_.n_ctx));

    auto &slot = graphs_[{T, last_only}];
    if (!slot)
        slot = std::make_unique<Graph>(build_graph(T, last_only));

    using Clock = std::chrono::steady_clock;
    OpProfile *const prof = profile_;
    const Clock::time_point t0 = prof ? Clock::now() : Clock::time_point{};

    Tensor tok = Tensor::zeros({T});
    Tensor pos = Tensor::zeros({T});
    for (int t = 0; t < T; ++t)
    {
        tok.data()[static_cast<std::size_t>(t)] = static_cast<float>(tokens[static_cast<std::size_t>(t)]);
        pos.data()[static_cast<std::size_t>(t)] = static_cast<float>(start_pos + t);
    }

    Executor exec(*slot);
    exec.set_kv_cache(cache);
    exec.set_timing(prof != nullptr);
    Tensor logits = exec.run({{"tokens", tok}, {"positions", pos}})[0];

    if (prof)
    {
        // 整个 run_graph（组输入 + run）减去各节点 compute 之和，剩下的就是图执行本身的开销
        double nodes_ms = 0;
        const std::vector<double> &node_ms = exec.stats().node_ms;
        for (const Node &node : slot->nodes())
        {
            const double ms = node_ms[static_cast<std::size_t>(node.id)];
            prof->*profile_field(node.name) += ms;
            nodes_ms += ms;
        }
        prof->overhead += std::chrono::duration<double, std::milli>(Clock::now() - t0).count() - nodes_ms;
        ++prof->forwards;
    }
    return logits;
}

Tensor GPT::forward_graph(std::span<const int> tokens)
{
    return run_graph(tokens, 0, nullptr, /*last_only=*/false);
}

Tensor GPT::forward_one_graph(int token, int pos, KVCache &cache)
{
    const int one[1] = {token};
    return run_graph(one, pos, &cache, /*last_only=*/true).reshape({cfg_.n_vocab});
}

std::vector<int> GPT::generate_graph(std::span<const int> prompt, int max_new_tokens,
                                     Sampler &sampler, GenStats *stats,
                                     const std::function<void(int)> &on_token)
{
    return generate_with([this](std::span<const int> tokens, int start_pos, KVCache &cache)
                         { return run_graph(tokens, start_pos, &cache, /*last_only=*/true); },
                         prompt, max_new_tokens, sampler, stats, on_token);
}

} // namespace inference
