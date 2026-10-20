// 用计算图跑 GPT-2，并和手写版逐位对拍。
//
//   ./build/examples/hello_gpt_graph
//   ./build/examples/hello_gpt_graph "Once upon a time" 40

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "inference/gpt.hpp"
#include "inference/graph.hpp"
#include "inference/kernel.hpp"
#include "inference/ops.hpp"
#include "inference/tokenizer.hpp"

using inference::GPT;
using inference::Graph;
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
        section("3. 带 KV-Cache 的贪心生成：generate_graph vs generate");
        // ---------------------------------------------------------------------
        {
            Sampler greedy_a{SamplerConfig{.temperature = 0.0F}};
            Sampler greedy_b{SamplerConfig{.temperature = 0.0F}};
            // 先各跑一次热身：图版本第一次要建 prefill 和 decode 两张图
            model.generate_graph(ids, 2, greedy_b);
            model.generate(ids, 2, greedy_a);

            GPT::GenStats st_hand, st_graph;
            const std::vector<int> hand = model.generate(ids, max_new, greedy_a, &st_hand);
            const std::vector<int> graph = model.generate_graph(ids, max_new, greedy_b, &st_graph);

            const std::vector<int> gen(graph.begin() + static_cast<std::ptrdiff_t>(ids.size()),
                                       graph.end());
            println("图版本生成: '{}{}'", prompt, tok.decode(gen));
            println("token 序列与手写版{}", hand == graph ? "完全一致" : "不一致！");

            auto per_tok = [](const GPT::GenStats &s)
            { return s.generated > 1 ? s.decode_ms / (s.generated - 1) : 0.0; };
            println("\n  {:<8}{:>14}{:>18}", "", "prefill (ms)", "decode (ms/token)");
            println("  {:<8}{:>14.2f}{:>18.2f}", "手写版", st_hand.prefill_ms, per_tok(st_hand));
            println("  {:<8}{:>14.2f}{:>18.2f}", "图版本", st_graph.prefill_ms, per_tok(st_graph));
            println("\n两者调用的是同一批 kernel，时间大头在矩阵乘，所以差距不大。");
            println("多出来的开销来自 Executor：每个节点都新分配一个输出 Tensor，还要在哈希表里查每个 Value，");
            println("而手写版全程复用一块工作区、零分配。这正是后面内存规划和算子融合要解决的。");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "出错: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
