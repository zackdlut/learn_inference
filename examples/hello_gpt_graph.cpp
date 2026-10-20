// 用计算图跑 GPT-2，并和手写版逐位、逐算子对拍。
//
//   ./build/examples/hello_gpt_graph
//   ./build/examples/hello_gpt_graph "Once upon a time" 40

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <format>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "inference/gpt.hpp"
#include "inference/graph.hpp"
#include "inference/kernel.hpp"
#include "inference/ops.hpp"
#include "inference/tokenizer.hpp"

using inference::Executor;
using inference::GPT;
using inference::Graph;
using inference::KVCache;
using inference::OpProfile;
using inference::Sampler;
using inference::SamplerConfig;
using inference::Tensor;
using inference::Tokenizer;
namespace ops = inference::ops;

namespace
{

template <class... Args> void println(std::format_string<Args...> fmt, Args &&...args)
{
    std::cout << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

void section(const char *title)
{
    std::cout << std::format("\n\033[36m===== {} =====\033[0m\n", title);
}

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// 手写版和图版本的 OpProfile 并排打印，每项都是每次 forward 的平均毫秒数
void compare_profiles(const char *title, const OpProfile &hand, const OpProfile &graph)
{
    struct Row
    {
        const char *name;
        double OpProfile::*field;
    };
    static constexpr Row rows[] = {
        {"embed", &OpProfile::embed},         {"layernorm", &OpProfile::layernorm},
        {"qkv (c_attn)", &OpProfile::qkv},    {"copy (kv)", &OpProfile::copy},
        {"attention", &OpProfile::attention}, {"attn_proj", &OpProfile::attn_proj},
        {"residual", &OpProfile::residual},   {"mlp_fc", &OpProfile::mlp_fc},
        {"gelu", &OpProfile::gelu},           {"mlp_proj", &OpProfile::mlp_proj},
        {"lm_head", &OpProfile::lm_head},     {"sample", &OpProfile::sample},
        {"图执行开销", &OpProfile::overhead},
    };
    const double nh = std::max(hand.forwards, 1), ng = std::max(graph.forwards, 1);
    println("\n{}（每次 forward 平均 ms，手写 {} 次 / 图 {} 次）:", title, hand.forwards,
            graph.forwards);
    println("  {:<16}{:>10}{:>10}{:>10}", "算子", "手写版", "图版本", "差值");
    for (const Row &r : rows)
    {
        const double h = hand.*r.field / nh, g = graph.*r.field / ng;
        println("  {:<16}{:>10.3f}{:>10.3f}{:>+10.3f}", r.name, h, g, g - h);
    }
    println("  {:<16}{:>10.3f}{:>10.3f}{:>+10.3f}", "合计", hand.total() / nh, graph.total() / ng,
            graph.total() / ng - hand.total() / nh);
}

double decode_ms_per_token(const GPT::GenStats &s)
{
    // 第一个新 token 来自 prefill，decode 阶段实际只跑了 generated - 1 次 forward
    return s.generated > 1 ? s.decode_ms / (s.generated - 1) : 0.0;
}

// 手写版和图版本交替各跑 rounds 轮，各自保留 decode 最快的一轮（减少 CPU 频率、调度带来的抖动），
// 然后检查 token 是否一致、打印速度和逐算子对照
void compare_generate(GPT &model, const Tokenizer &tok, const std::vector<int> &ids, int max_new,
                      const std::string &prompt, int rounds = 3)
{
    // 热身：图版本第一次要建 prefill 和 decode 两张图
    {
        Sampler s{SamplerConfig{.temperature = 0.0F}};
        model.generate_graph(ids, 2, s);
        model.generate(ids, 2, s);
    }

    GPT::GenStats best_hand, best_graph;
    std::vector<int> out_hand, out_graph;
    for (int r = 0; r < rounds; ++r)
    {
        Sampler sa{SamplerConfig{.temperature = 0.0F}};
        Sampler sb{SamplerConfig{.temperature = 0.0F}};
        GPT::GenStats st_hand, st_graph;
        out_hand = model.generate(ids, max_new, sa, &st_hand);
        out_graph = model.generate_graph(ids, max_new, sb, &st_graph);
        if (r == 0 || st_hand.decode_ms < best_hand.decode_ms)
            best_hand = st_hand;
        if (r == 0 || st_graph.decode_ms < best_graph.decode_ms)
            best_graph = st_graph;
    }

    const std::vector<int> gen(out_graph.begin() + static_cast<std::ptrdiff_t>(ids.size()),
                               out_graph.end());
    println("图版本生成: '{}{}'", prompt, tok.decode(gen));
    println("token 序列与手写版{}", out_hand == out_graph ? "完全一致" : "不一致！");

    println("\n  （各跑 {} 轮取 decode 最快的一轮）", rounds);
    println("  {:<8}{:>14}{:>18}", "", "prefill (ms)", "decode (ms/token)");
    println("  {:<8}{:>14.2f}{:>18.2f}", "手写版", best_hand.prefill_ms,
            decode_ms_per_token(best_hand));
    println("  {:<8}{:>14.2f}{:>18.2f}", "图版本", best_graph.prefill_ms,
            decode_ms_per_token(best_graph));

    compare_profiles("prefill 逐算子对照", best_hand.prefill_ops, best_graph.prefill_ops);
    compare_profiles("decode 逐算子对照", best_hand.decode_ops, best_graph.decode_ops);
}

} // namespace

int main(int argc, char **argv)
{
    const std::string dir = INFER_MODEL_GPT_DIR;
    const std::string prompt = argc > 1 ? argv[1] : "The capital of France is";
    const int max_new = argc > 2 ? std::stoi(argv[2]) : 20;

    if (std::getenv("INFER_NUM_THREADS") == nullptr)
    {
        const auto hw = static_cast<int>(std::thread::hardware_concurrency());
        inference::kernel::set_num_threads(std::max(1, hw / 2));
    }

    try
    {
        Tokenizer tok = Tokenizer::load(dir + "/gpt2_vocab.bin", dir + "/gpt2_merges.bin");
        const std::vector<int> ids = tok.encode(prompt);
        GPT model = GPT::load(dir);
        const auto &cfg = model.config();
        println("prompt: '{}'（{} 个 token）", prompt, ids.size());

        // ---------------------------------------------------------------------
        section("1. decode 用的那张图（T = 1，只输出最后一个位置）");
        // ---------------------------------------------------------------------
        {
            const auto t0 = Clock::now();
            const Graph g = model.build_graph(1, /*last_only=*/true);
            const double build_ms = ms_since(t0);
            std::cout << g.to_string();
            println("\n建图 + finalize（拓扑排序、形状推断）耗时 {:.2f} ms，一次计算都没跑。", build_ms);
            println("权重只是多了一个引用（Tensor 内部是 shared_ptr），不拷贝数据。");
        }

        // ---------------------------------------------------------------------
        section("2. 不带 KV-Cache：forward_graph vs forward");
        // ---------------------------------------------------------------------
        {
            const Tensor want = model.forward(ids);
            const Tensor got = model.forward_graph(ids);
            const float d = ops::max_abs_diff(got, want);
            println("logits 形状 {}，最大绝对误差 {:.3e}（{}）", inference::to_string(got.shape()),
                    static_cast<double>(d), d < 1e-3F ? "一致" : "不符");
        }

        // ---------------------------------------------------------------------
        section("3. Executor 按节点类型计时（直接跑 decode 图 32 步）");
        // ---------------------------------------------------------------------
        {
            const Graph g = model.build_graph(1, /*last_only=*/true);
            std::map<std::string, int> type_count;
            for (const auto &node : g.nodes())
                ++type_count[node.type];

            constexpr int kSteps = 32;
            KVCache cache(cfg, kSteps);
            Executor exec(g);
            exec.set_kv_cache(&cache);
            exec.set_timing(true);
            std::map<std::string, double> type_ms;
            double run_ms = 0;
            for (int pos = 0; pos < kSteps; ++pos)
            {
                const int id = ids[static_cast<std::size_t>(pos) % ids.size()];
                exec.run({{"tokens", Tensor::from({1}, {static_cast<float>(id)})},
                          {"positions", Tensor::from({1}, {static_cast<float>(pos)})}});
                for (const auto &[type, ms] : exec.stats().type_ms)
                    type_ms[type] += ms;
                run_ms += exec.stats().run_ms;
            }

            double nodes_ms = 0;
            for (const auto &[type, ms] : type_ms)
                nodes_ms += ms;
            println("  {:<16}{:>8}{:>12}{:>8}", "节点类型", "个数", "ms/步", "占比");
            for (const auto &[type, ms] : type_ms)
                println("  {:<16}{:>8}{:>12.3f}{:>7.1f}%", type, type_count[type], ms / kSteps,
                        ms / run_ms * 100.0);
            println("  {:<16}{:>8}{:>12.3f}{:>7.1f}%", "Executor 开销", "", (run_ms - nodes_ms) / kSteps,
                    (run_ms - nodes_ms) / run_ms * 100.0);
            println("  {:<16}{:>8}{:>12.3f}", "合计", g.nodes().size(), run_ms / kSteps);
            println("\n按类型只能看出「矩阵乘占大头」：c_attn、attn_proj、c_fc、mlp_proj、lm_head");
            println("五种矩阵乘都是 linear 类型。要和手写版逐项对照，得按节点名归类 —— 见下面两节。");
        }

        // ---------------------------------------------------------------------
        section("4. fp32：generate_graph vs generate");
        // ---------------------------------------------------------------------
        compare_generate(model, tok, ids, max_new, prompt);

        // ---------------------------------------------------------------------
        section("5. 量化（层内 Q8_0 + lm_head fp16）：generate_graph vs generate");
        // ---------------------------------------------------------------------
        model.quantize_q8_0();
        {
            const Graph g = model.build_graph(1, /*last_only=*/true);
            std::map<std::string, int> type_count;
            for (const auto &node : g.nodes())
                ++type_count[node.type];
            std::string line;
            for (const auto &[type, n] : type_count)
                line += std::format(" {}×{}", type, n);
            println("量化后的 decode 图，节点类型:{}", line);
            println("结构不变，只是 linear 换成了 linear_q8（48 个）和 linear_f16（lm_head）。\n");
        }
        compare_generate(model, tok, ids, max_new, prompt);
        println("\n这一节可以直接和 hello_gpt 第 7 节（量化后再生成）对比。");
    }
    catch (const std::exception &e)
    {
        std::cerr << "出错: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
