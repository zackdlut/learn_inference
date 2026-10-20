#include "inference/gpt.hpp"
#include "inference/graph.hpp"
#include "inference/ops.hpp"
#include "minitest.hpp"

#include <cmath>
#include <string>
#include <vector>

using inference::Attributes;
using inference::Executor;
using inference::GPTConfig;
using inference::Graph;
using inference::KVCache;
using inference::Shape;
using inference::Tensor;
using inference::ValueId;
namespace ops = inference::ops;

namespace
{

// 确定性的「随机」数据，不依赖 <random>，换机器结果一致
Tensor wave(Shape shape, float phase)
{
    Tensor t = Tensor::zeros(std::move(shape));
    auto d = t.data();
    for (std::size_t i = 0; i < d.size(); ++i)
        d[i] = std::sin(static_cast<float>(i) * 0.37F + phase);
    return t;
}

Tensor positions(int start, int T)
{
    Tensor t = Tensor::zeros({T});
    for (int i = 0; i < T; ++i)
        t.data()[static_cast<std::size_t>(i)] = static_cast<float>(start + i);
    return t;
}

Graph attention_graph(int T, int C, int n_head)
{
    Graph g;
    const ValueId qkv = g.add_input("qkv", {T, 3 * C});
    const ValueId pos = g.add_input("positions", {T});
    const ValueId y = g.add_node("attention_qkv", "attn", {qkv, pos},
                                 Attributes{}.set("n_head", std::int64_t{n_head}).set("layer",
                                                                                      std::int64_t{0}));
    g.mark_output(y);
    g.finalize();
    return g;
}

Graph tiny_add_graph()
{
    Graph g;
    const ValueId a = g.add_input("a", {2, 2});
    const ValueId b = g.add_input("b", {2, 2});
    const ValueId sum = g.add_node("add", "sum", {a, b});
    g.mark_output(sum);
    g.finalize();
    return g;
}

} // namespace

TEST(graph_finalize_infers_shapes_and_exec_order)
{
    Graph g;
    const ValueId x = g.add_input("x", {1, 2});
    const ValueId w = g.add_weight("w", Tensor::from({2, 2}, {1, 0, 0, 1}));
    const ValueId h = g.add_node("matmul", "mm", {x, w});
    const ValueId y = g.add_node("relu", "act", {h});
    g.mark_output(y);
    g.finalize();

    CHECK(g.finalized());
    CHECK(g.value(h).shape == Shape({1, 2}));
    CHECK(g.value(y).shape == Shape({1, 2}));

    const auto order = g.exec_order();
    CHECK(order.size() == 2);
    CHECK(g.nodes()[order[0]].type == "matmul");
    CHECK(g.nodes()[order[1]].type == "relu");
}

TEST(graph_consumer_count_after_finalize)
{
    Graph g;
    const ValueId a = g.add_input("a", {2});
    const ValueId b = g.add_input("b", {2});
    const ValueId c = g.add_input("c", {2});
    const ValueId ab = g.add_node("add", "ab", {a, b});
    const ValueId abc = g.add_node("add", "abc", {ab, c});
    g.mark_output(abc);
    g.finalize();

    CHECK(g.consumer_count(a) == 1);
    CHECK(g.consumer_count(b) == 1);
    CHECK(g.consumer_count(c) == 1);
    CHECK(g.consumer_count(ab) == 1);
    CHECK(g.consumer_count(abc) == 1);
}

TEST(graph_info_lists_io_and_compute)
{
    const Graph g = tiny_add_graph();
    const std::string s = g.to_string();

    CHECK(s.find("Graph(") != std::string::npos);
    CHECK(s.find("输入:") != std::string::npos);
    CHECK(s.find("权重:") != std::string::npos);
    CHECK(s.find("计算:") != std::string::npos);
    CHECK(s.find("输出:") != std::string::npos);
    CHECK(s.find("add") != std::string::npos);
    CHECK(s.find("a") != std::string::npos);
    CHECK(s.find("b") != std::string::npos);
}

TEST(graph_executor_add_matches_ops)
{
    const Graph g = tiny_add_graph();
    Executor exec(g);

    const Tensor ta = Tensor::from({2, 2}, {1, 2, 3, 4});
    const Tensor tb = Tensor::from({2, 2}, {10, 20, 30, 40});
    const std::vector<Tensor> out = exec.run({{"a", ta}, {"b", tb}});

    CHECK(out.size() == 1);
    CHECK(out[0].shape() == Shape({2, 2}));
    CHECK(ops::max_abs_diff(out[0], ops::add(ta, tb)) < 1e-6F);
    CHECK(exec.stats().nodes_run == 1);
}

TEST(graph_executor_linear_with_bias)
{
    Graph g;
    const ValueId x = g.add_input("x", {1, 2});
    const ValueId w = g.add_weight("w", Tensor::from({2, 2}, {1, 0, 0, 1}));
    const ValueId b = g.add_weight("b", Tensor::from({2}, {0.5F, -0.5F}));
    const ValueId h = g.add_node("linear", "fc", {x, w, b});
    g.mark_output(h);
    g.finalize();

    const Tensor xin = Tensor::from({1, 2}, {1, 0});
    const std::vector<Tensor> out = Executor(g).run({{"x", xin}});
    const Tensor want = ops::linear(xin, g.weights().at(w), g.weights().at(b));
    CHECK(ops::max_abs_diff(out[0], want) < 1e-6F);
}

TEST(graph_finalize_requires_output)
{
    Graph g;
    g.add_input("x", {1});
    CHECK_THROWS(g.finalize());
}

TEST(graph_mutate_after_finalize_throws)
{
    Graph g;
    g.add_input("x", {1});
    const ValueId y = g.add_node("relu", "r", {0});
    g.mark_output(y);
    g.finalize();

    CHECK_THROWS(g.add_input("y", {1}));
    CHECK_THROWS(g.add_weight("w", Tensor::zeros({1})));
    CHECK_THROWS(g.add_node("relu", "r2", {0}));
}

TEST(graph_invalid_value_id)
{
    Graph g;
    g.add_input("x", {1});
    CHECK_THROWS(g.mark_output(99));
    CHECK_THROWS(g.value(-1));
    CHECK_THROWS(g.value(99));
}

TEST(graph_add_node_invalid_input_id)
{
    Graph g;
    g.add_input("x", {1});
    CHECK_THROWS(g.add_node("relu", "r", {5}));
}

TEST(graph_add_node_unknown_op_throws)
{
    Graph g;
    const ValueId x = g.add_input("x", {1});
    CHECK_THROWS(g.add_node("not_registered_op", "n", {x}));
}

TEST(graph_finalize_relu_wrong_arity)
{
    Graph g;
    const ValueId a = g.add_input("a", {2});
    const ValueId b = g.add_input("b", {2});
    const ValueId y = g.add_node("relu", "r", {a, b});
    g.mark_output(y);
    CHECK_THROWS(g.finalize());
}

TEST(graph_finalize_shape_mismatch_on_add)
{
    Graph g;
    const ValueId a = g.add_input("a", {2});
    const ValueId b = g.add_input("b", {3});
    const ValueId y = g.add_node("add", "sum", {a, b});
    g.mark_output(y);
    CHECK_THROWS(g.finalize());
}

TEST(graph_executor_not_finalized)
{
    Graph g;
    g.add_input("x", {1});
    CHECK_THROWS(Executor(g).run({}));
}

TEST(graph_executor_missing_feed)
{
    Graph g;
    g.add_input("x", {1});
    const ValueId y = g.add_node("relu", "r", {0});
    g.mark_output(y);
    g.finalize();

    CHECK_THROWS(Executor(g).run({}));
}

TEST(graph_executor_feed_shape_mismatch)
{
    Graph g;
    g.add_input("x", {2});
    const ValueId y = g.add_node("relu", "r", {0});
    g.mark_output(y);
    g.finalize();

    CHECK_THROWS(Executor(g).run({{"x", Tensor::zeros({3})}}));
}

TEST(graph_finalize_idempotent)
{
    Graph g;
    const ValueId x = g.add_input("x", {1});
    const ValueId y = g.add_node("relu", "r", {x});
    g.mark_output(y);
    g.finalize();
    const auto order1 = std::vector<int>(g.exec_order().begin(), g.exec_order().end());
    g.finalize();
    const auto order2 = std::vector<int>(g.exec_order().begin(), g.exec_order().end());
    CHECK(order1 == order2);
}

TEST(graph_embedding_matches_ops)
{
    const Tensor table = wave({10, 4}, 0.5F);
    Graph g;
    const ValueId ids = g.add_input("ids", {3});
    const ValueId w = g.add_weight("table", table);
    const ValueId y = g.add_node("embedding", "emb", {ids, w});
    g.mark_output(y);
    g.finalize();
    CHECK(g.value(y).shape == Shape({3, 4}));

    const std::vector<Tensor> out = Executor(g).run({{"ids", Tensor::from({3}, {7, 0, 7})}});
    const int want_ids[] = {7, 0, 7};
    CHECK(ops::max_abs_diff(out[0], ops::embedding(table, want_ids)) == 0.0F);
}

TEST(graph_last_rows)
{
    Graph g;
    const ValueId x = g.add_input("x", {4, 3});
    const ValueId y = g.add_node("last_rows", "last", {x}, Attributes{}.set("n", std::int64_t{1}));
    g.mark_output(y);
    g.finalize();
    CHECK(g.value(y).shape == Shape({1, 3}));

    const Tensor xin = wave({4, 3}, 0.0F);
    const std::vector<Tensor> out = Executor(g).run({{"x", xin}});
    CHECK(ops::max_abs_diff(out[0], xin.slice(0, 3, 4).contiguous()) == 0.0F);
}

TEST(graph_attention_qkv_matches_ops)
{
    constexpr int T = 5, C = 8, H = 2;
    const Graph g = attention_graph(T, C, H);
    const Tensor qkv = wave({T, 3 * C}, 1.0F);
    const std::vector<Tensor> out =
        Executor(g).run({{"qkv", qkv}, {"positions", positions(0, T)}});

    const Tensor want = ops::attention(qkv.slice(1, 0, C), qkv.slice(1, C, 2 * C),
                                       qkv.slice(1, 2 * C, 3 * C), H);
    CHECK(ops::max_abs_diff(out[0], want) < 1e-5F);
}

// 先 prefill 前 3 个 token，再一个一个 decode，每步的输出都应等于一次性整段算出的对应行
TEST(graph_attention_qkv_kv_cache_incremental)
{
    constexpr int T = 6, C = 8, H = 2, prefill = 3;
    const Tensor qkv = wave({T, 3 * C}, 2.0F);
    const Tensor full =
        Executor(attention_graph(T, C, H)).run({{"qkv", qkv}, {"positions", positions(0, T)}})[0];

    GPTConfig cfg;
    cfg.n_layer = 1;
    cfg.n_head = H;
    cfg.n_embd = C;
    cfg.n_ctx = 16;
    KVCache cache(cfg, T);

    const Graph g_prefill = attention_graph(prefill, C, H);
    Executor e_prefill(g_prefill);
    e_prefill.set_kv_cache(&cache);
    const Tensor head = e_prefill.run({{"qkv", qkv.slice(0, 0, prefill).contiguous()},
                                       {"positions", positions(0, prefill)}})[0];
    CHECK(ops::max_abs_diff(head, full.slice(0, 0, prefill).contiguous()) < 1e-5F);
    CHECK(cache.length() == prefill);

    const Graph g_decode = attention_graph(1, C, H);
    Executor e_decode(g_decode);
    e_decode.set_kv_cache(&cache);
    for (int p = prefill; p < T; ++p)
    {
        const Tensor row = e_decode.run({{"qkv", qkv.slice(0, p, p + 1).contiguous()},
                                         {"positions", positions(p, 1)}})[0];
        CHECK(ops::max_abs_diff(row, full.slice(0, p, p + 1).contiguous()) < 1e-5F);
    }
    CHECK(cache.length() == T);
}
