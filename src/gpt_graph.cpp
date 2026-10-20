#include "inference/gpt.hpp"
#include "inference/graph.hpp"
#include "inference/op.hpp"
#include "inference/tensor.hpp"

#include <format>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace inference
{

namespace
{
[[noreturn]] void fail(const std::string &msg)
{
    throw std::runtime_error("[infer::GPT] " + msg);
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
// 权重 Tensor 内部是 shared_ptr，add_weight 只是多一个引用，建多少张图都不拷贝权重数据。
// ---------------------------------------------------------------------------
Graph GPT::build_graph(int T, bool last_only) const
{
    if (quantized_)
        fail("build_graph: 计算图版本只支持 fp32 权重，量化后 fp32 权重已被释放");
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

        // 注意力子层
        ValueId h = g.add_node("layer_norm", p + "ln_1", {x, w("ln_1.w", L.ln1_w), w("ln_1.b", L.ln1_b)},
                               ln_attr());
        const ValueId qkv =
            g.add_node("linear", p + "c_attn", {h, w("c_attn.w", L.attn_w), w("c_attn.b", L.attn_b)});
        const ValueId att = g.add_node("attention_qkv", p + "attn", {qkv, positions},
                                       Attributes{}
                                           .set("n_head", std::int64_t{cfg_.n_head})
                                           .set("layer", std::int64_t{li}));
        const ValueId att_proj =
            g.add_node("linear", p + "attn_proj",
                       {att, w("attn_proj.w", L.attn_proj_w), w("attn_proj.b", L.attn_proj_b)});
        x = g.add_node("add", p + "res_attn", {x, att_proj});

        // MLP 子层
        h = g.add_node("layer_norm", p + "ln_2", {x, w("ln_2.w", L.ln2_w), w("ln_2.b", L.ln2_b)},
                       ln_attr());
        const ValueId fc = g.add_node("linear", p + "c_fc", {h, w("c_fc.w", L.fc_w), w("c_fc.b", L.fc_b)});
        const ValueId act = g.add_node("gelu", p + "gelu", {fc});
        const ValueId mlp_proj =
            g.add_node("linear", p + "mlp_proj", {act, w("mlp_proj.w", L.proj_w), w("mlp_proj.b", L.proj_b)});
        x = g.add_node("add", p + "res_mlp", {x, mlp_proj});
    }

    // layernorm 逐行独立，先切出最后一行再 norm，结果和 norm 完再切一样，但省掉 T-1 行的计算
    if (last_only)
        x = g.add_node("last_rows", "last", {x}, Attributes{}.set("n", std::int64_t{1}));
    const ValueId h = g.add_node("layer_norm", "ln_f",
                                 {x, g.add_weight("ln_f.w", lnf_w_), g.add_weight("ln_f.b", lnf_b_)},
                                 ln_attr());
    // weight tying：lm_head 直接复用词嵌入那个 Value，wte 在图里只登记一次
    const ValueId logits = g.add_node("linear", "lm_head", {h, wte});
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

    Tensor tok = Tensor::zeros({T});
    Tensor pos = Tensor::zeros({T});
    for (int t = 0; t < T; ++t)
    {
        tok.data()[static_cast<std::size_t>(t)] = static_cast<float>(tokens[static_cast<std::size_t>(t)]);
        pos.data()[static_cast<std::size_t>(t)] = static_cast<float>(start_pos + t);
    }

    Executor exec(*slot);
    exec.set_kv_cache(cache);
    return exec.run({{"tokens", tok}, {"positions", pos}})[0];
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
