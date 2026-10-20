// Presence / Frequency Penalty：同一个 prompt，贪心解码，对比不同惩罚下的重复程度。
//
//   ./build/examples/hello_penalty
//   ./build/examples/hello_penalty "Once upon a time" 80
//   ./build/examples/hello_penalty "Once upon a time" 80 0.5 0.3   # 只跑这一组 presence / frequency

#include <algorithm>
#include <cstdlib>
#include <format>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "inference/gpt.hpp"
#include "inference/kernel.hpp"
#include "inference/logits_processor.hpp"
#include "inference/tokenizer.hpp"

using inference::GPT;
using inference::PenaltyConfig;
using inference::PenaltyProcessor;
using inference::Sampler;
using inference::SamplerConfig;
using inference::Tokenizer;

template <class... Args> void println(std::format_string<Args...> fmt, Args &&...args)
{
    std::cout << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

// 衡量重复：distinct = 不同 token 数 / 总数；max_rep = 出现最多的 token 出现了几次
struct RepeatStats
{
    double distinct = 0;
    int max_rep = 0;
};

RepeatStats repeat_stats(const std::vector<int> &gen)
{
    std::map<int, int> cnt;
    for (int t : gen)
        ++cnt[t];
    RepeatStats s;
    s.distinct = gen.empty() ? 0.0 : static_cast<double>(cnt.size()) / static_cast<double>(gen.size());
    for (const auto &[t, c] : cnt)
        s.max_rep = std::max(s.max_rep, c);
    return s;
}

int main(int argc, char **argv)
{
    const std::string dir = INFER_MODEL_GPT_DIR;
    const std::string prompt = argc > 1 ? argv[1] : "The meaning of life is";
    const int max_new = argc > 2 ? std::stoi(argv[2]) : 60;

    std::vector<PenaltyConfig> configs;
    if (argc > 4)
    {
        configs.push_back({.presence_penalty = std::stof(argv[3]), .frequency_penalty = std::stof(argv[4])});
    }
    else
    {
        configs = {
            {.presence_penalty = 0.0F, .frequency_penalty = 0.0F},
            {.presence_penalty = 1.0F, .frequency_penalty = 0.0F},
            {.presence_penalty = 0.0F, .frequency_penalty = 0.5F},
            {.presence_penalty = 1.0F, .frequency_penalty = 0.5F},
            {.presence_penalty = 2.0F, .frequency_penalty = 1.0F},
        };
    }

    if (std::getenv("INFER_NUM_THREADS") == nullptr)
    {
        const auto hw = static_cast<int>(std::thread::hardware_concurrency());
        inference::kernel::set_num_threads(std::max(1, hw / 2));
    }

    try
    {
        Tokenizer tok = Tokenizer::load(dir + "/gpt2_vocab.bin", dir + "/gpt2_merges.bin");
        GPT model = GPT::load(dir);
        model.quantize_q8_0();
        const std::vector<int> ids = tok.encode(prompt);

        println("prompt: \033[33m{}\033[0m，贪心解码 {} 个 token\n", prompt, max_new);
        for (const PenaltyConfig &pc : configs)
        {
            Sampler greedy{SamplerConfig{.temperature = 0.0F}};
            PenaltyProcessor proc{pc};
            const std::vector<int> out = model.generate(ids, max_new, greedy, nullptr, {}, &proc);
            const std::vector<int> gen(out.begin() + static_cast<std::ptrdiff_t>(ids.size()), out.end());
            const RepeatStats rs = repeat_stats(gen);

            println("\033[36m── presence {:.1f}  frequency {:.1f}  │  distinct {:.2f}  max_rep {}\033[0m",
                    pc.presence_penalty, pc.frequency_penalty, rs.distinct, rs.max_rep);
            println("{}\n", tok.decode(gen));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
