// Logits Processor 效果对比：同一个 prompt，不同的 presence / frequency penalty 设置。
//
//   ./build-release/examples/hello_processor
//   ./build-release/examples/hello_processor "Once upon a time" 80 8   # prompt、生成长度、采样种子数
//
// 四组实验：
//   1. 单步观察：某一步的 top-5 候选词在不同惩罚下怎么变
//   2. 贪心解码：多样性 vs 流畅度（困惑度）
//   3. 随机采样：多个种子取平均
//   4. include_prompt：prompt 里的词算不算「出现过」

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <map>
#include <set>
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
using inference::Tensor;
using inference::Tokenizer;

template <class... Args> void println(std::format_string<Args...> fmt, Args &&...args)
{
    std::cout << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

void section(const char *title)
{
    std::cout << std::format("\n\033[36m===== {} =====\033[0m\n", title);
}

std::string show(const std::string &s)
{
    std::string out;
    for (char c : s)
        out += c == '\n' ? std::string("\\n") : std::string(1, c);
    return out;
}

std::string label(const PenaltyConfig &pc)
{
    return std::format("pres {:>4.1f}  freq {:>4.1f}", pc.presence_penalty, pc.frequency_penalty);
}

// ---------------------------------------------------------------------------
// 评价指标。只看生成部分，不含 prompt。
//   distinct-1 / distinct-2：不同的 1-gram / 2-gram 占比，越高越不重复
//   rep-4：已经出现过的 4-gram 占比，越低越好；整段复读时接近 1
//   max_rep：出现最多的 token 出现了几次
//   ppl：未加惩罚的原模型给这段生成文本打的困惑度，衡量「偏离模型分布有多远」。
//        复读的文本概率极高，ppl 会很低；惩罚越大 ppl 越高，过大时文本开始不通顺
// ---------------------------------------------------------------------------
struct Metrics
{
    double distinct1 = 0;
    double distinct2 = 0;
    double rep4 = 0;
    double max_rep = 0;
    double ppl = 0;

    Metrics &operator+=(const Metrics &o)
    {
        distinct1 += o.distinct1;
        distinct2 += o.distinct2;
        rep4 += o.rep4;
        max_rep += o.max_rep;
        ppl += o.ppl;
        return *this;
    }
    Metrics operator/(double n) const
    {
        return {distinct1 / n, distinct2 / n, rep4 / n, max_rep / n, ppl / n};
    }
};

// GPT-2 的 token id < 65536，n 个 id 各占 16 位拼成一个整数当作 n-gram 的键
std::uint64_t ngram_key(const std::vector<int> &g, std::size_t i, std::size_t n)
{
    std::uint64_t k = 0;
    for (std::size_t j = 0; j < n; ++j)
        k = (k << 16) | static_cast<std::uint64_t>(g[i + j]);
    return k;
}

double distinct_n(const std::vector<int> &g, std::size_t n)
{
    if (g.size() < n)
        return 0.0;
    std::set<std::uint64_t> seen;
    for (std::size_t i = 0; i + n <= g.size(); ++i)
        seen.insert(ngram_key(g, i, n));
    return static_cast<double>(seen.size()) / static_cast<double>(g.size() - n + 1);
}

double repeat_rate(const std::vector<int> &g, std::size_t n)
{
    if (g.size() < n)
        return 0.0;
    std::set<std::uint64_t> seen;
    int repeated = 0;
    for (std::size_t i = 0; i + n <= g.size(); ++i)
        repeated += seen.insert(ngram_key(g, i, n)).second ? 0 : 1;
    return static_cast<double>(repeated) / static_cast<double>(g.size() - n + 1);
}

double perplexity_of_generated(GPT &model, const std::vector<int> &out, std::size_t prompt_len)
{
    const Tensor logits = model.forward(out);
    const std::span<const float> lv = logits.data();
    const auto V = static_cast<std::size_t>(logits.dim(1));
    double nll = 0.0;
    for (std::size_t t = prompt_len - 1; t + 1 < out.size(); ++t)
    {
        const float *row = lv.data() + t * V;
        const float mx = *std::max_element(row, row + V);
        double sum = 0.0;
        for (std::size_t i = 0; i < V; ++i)
            sum += std::exp(static_cast<double>(row[i] - mx));
        nll -= static_cast<double>(row[out[t + 1]] - mx) - std::log(sum);
    }
    return std::exp(nll / static_cast<double>(out.size() - prompt_len));
}

Metrics measure(GPT &model, const std::vector<int> &out, std::size_t prompt_len)
{
    const std::vector<int> gen(out.begin() + static_cast<std::ptrdiff_t>(prompt_len), out.end());
    std::map<int, int> cnt;
    for (int t : gen)
        ++cnt[t];
    int max_rep = 0;
    for (const auto &[t, c] : cnt)
        max_rep = std::max(max_rep, c);
    return {distinct_n(gen, 1), distinct_n(gen, 2), repeat_rate(gen, 4), static_cast<double>(max_rep),
            perplexity_of_generated(model, out, prompt_len)};
}

void print_metrics_header(const char *extra = "")
{
    println("  {:<30}{:>8}{:>8}{:>8}{:>8}{:>8}{}", "setting", "dist-1", "dist-2", "rep-4", "maxrep", "ppl",
            extra);
}

void print_metrics_row(const std::string &name, const Metrics &m, const std::string &extra = "")
{
    println("  {:<30}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.1f}{:>8.2f}{}", name, m.distinct1, m.distinct2, m.rep4,
            m.max_rep, m.ppl, extra);
}

int main(int argc, char **argv)
{
    const std::string dir = INFER_MODEL_GPT_DIR;
    const std::string prompt = argc > 1 ? argv[1] : "The meaning of life is";
    const int max_new = argc > 2 ? std::stoi(argv[2]) : 60;
    const int n_seeds = argc > 3 ? std::stoi(argv[3]) : 5;

    const std::vector<PenaltyConfig> configs = {
        {.presence_penalty = 0.0F, .frequency_penalty = 0.0F},
        {.presence_penalty = -1.0F, .frequency_penalty = 0.0F},
        {.presence_penalty = 1.0F, .frequency_penalty = 0.0F},
        {.presence_penalty = 3.0F, .frequency_penalty = 0.0F},
        {.presence_penalty = 0.0F, .frequency_penalty = 0.5F},
        {.presence_penalty = 0.0F, .frequency_penalty = 1.5F},
        {.presence_penalty = 1.0F, .frequency_penalty = 0.5F},
        {.presence_penalty = 2.0F, .frequency_penalty = 1.0F},
    };

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
        const std::size_t P = ids.size();
        println("prompt: \033[33m{}\033[0m（{} 个 token），每次生成 {} 个 token", prompt, P, max_new);

        // ---------------------------------------------------------------------
        section("1. 单步观察：惩罚怎么改写 top-5");
        // ---------------------------------------------------------------------
        {
            // 先不加惩罚贪心生成一段，找到第一个「已经生成过 2 次还要再选」的位置
            Sampler greedy{SamplerConfig{.temperature = 0.0F}};
            const std::vector<int> base = model.generate(ids, max_new, greedy);
            std::map<int, int> cnt;
            std::size_t pick = 0;
            for (std::size_t i = P; i < base.size(); ++i)
            {
                if (cnt[base[i]] >= 2)
                {
                    pick = i;
                    break;
                }
                ++cnt[base[i]];
            }
            if (pick == 0)
            {
                println("这段生成里没有出现第 3 次的 token，跳过");
            }
            else
            {
                const std::vector<int> ctx(base.begin(), base.begin() + static_cast<std::ptrdiff_t>(pick));
                std::string tail;
                for (std::size_t i = std::max(P, pick > 12 ? pick - 12 : 0); i < pick; ++i)
                    tail += tok.decode_one(ctx[i]);
                println("上下文（生成部分的最后 12 个 token）: …{}\033[31m▌\033[0m", show(tail));
                println("不加惩罚时贪心会选 '{}'，它在生成部分已经出现了 {} 次\n",
                        show(tok.decode_one(base[pick])), cnt[base[pick]]);

                const Tensor all = model.forward(ctx);
                const std::span<const float> lv = all.data();
                const auto V = static_cast<std::size_t>(all.dim(1));
                const std::vector<float> raw(lv.end() - static_cast<std::ptrdiff_t>(V), lv.end());

                const PenaltyConfig probe[] = {
                    {.presence_penalty = 0.0F, .frequency_penalty = 0.0F},
                    {.presence_penalty = 1.0F, .frequency_penalty = 0.0F},
                    {.presence_penalty = 3.0F, .frequency_penalty = 0.0F},
                    {.presence_penalty = 0.0F, .frequency_penalty = 1.5F},
                };
                for (const PenaltyConfig &pc : probe)
                {
                    PenaltyProcessor proc{pc};
                    proc.begin(ids);
                    for (std::size_t i = P; i < ctx.size(); ++i)
                        proc.accept(ctx[i]);
                    std::vector<float> z = raw;
                    proc.process(z);

                    std::vector<int> order(V);
                    for (std::size_t i = 0; i < V; ++i)
                        order[i] = static_cast<int>(i);
                    std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                                      [&](int a, int b) { return z[a] > z[b]; });
                    std::string row;
                    for (int r = 0; r < 5; ++r)
                    {
                        const int id = order[static_cast<std::size_t>(r)];
                        const int c = proc.count(id);
                        row += std::format("  {:<10} {:6.2f}{}", std::format("'{}'", show(tok.decode_one(id))),
                                           z[static_cast<std::size_t>(id)],
                                           c > 0 ? std::format("(c={})", c) : std::string("     "));
                    }
                    println("  {} │{}", label(pc), row);
                }
                println("\n(c=k) 表示这个词在生成部分已经出现了 k 次，被惩罚过；没有标记的是新词。");
            }
        }

        // ---------------------------------------------------------------------
        section("2. 贪心解码：多样性 vs 流畅度");
        // ---------------------------------------------------------------------
        {
            std::vector<std::pair<std::string, Metrics>> rows;
            std::vector<int> last_gen;
            for (const PenaltyConfig &pc : configs)
            {
                Sampler greedy{SamplerConfig{.temperature = 0.0F}};
                PenaltyProcessor proc{pc};
                const std::vector<int> out = model.generate(ids, max_new, greedy, nullptr, {}, &proc);
                last_gen.assign(out.begin() + static_cast<std::ptrdiff_t>(P), out.end());
                rows.emplace_back(label(pc), measure(model, out, P));
                println("\033[36m── {}\033[0m", label(pc));
                println("{}\n", tok.decode(last_gen));
            }
            print_metrics_header();
            for (const auto &[name, m] : rows)
                print_metrics_row(name, m);

            // processor 的开销：用最后一组设置生成的 token 填好计数，单独给 process() 计时
            PenaltyProcessor proc{configs.back()};
            proc.begin(ids);
            for (int t : last_gen)
                proc.accept(t);
            std::vector<float> z(static_cast<std::size_t>(model.config().n_vocab), 0.0F);
            constexpr int kIters = 2000;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kIters; ++i)
                proc.process(z);
            const auto t1 = std::chrono::steady_clock::now();
            Sampler greedy{SamplerConfig{.temperature = 0.0F}};
            [[maybe_unused]] volatile int sink = 0;
            for (int i = 0; i < kIters; ++i)
                sink = greedy.sample(z);
            const auto t2 = std::chrono::steady_clock::now();
            const auto us = [&](auto a, auto b)
            { return std::chrono::duration<double, std::micro>(b - a).count() / kIters; };
            println("\n开销（{} 个不同的词被惩罚）: process() {:.2f} us/次，对比贪心 sample() {:.2f} us/次",
                    std::set<int>(last_gen.begin(), last_gen.end()).size(), us(t0, t1), us(t1, t2));
            println("一次 forward 约 10 ms，processor 的开销可以忽略。");
        }

        // ---------------------------------------------------------------------
        section("3. 随机采样：temperature 0.8, top-k 40, top-p 0.95");
        // ---------------------------------------------------------------------
        {
            println("每组设置跑 {} 个随机种子，指标取平均；文本只展示种子 1 的前 30 个 token。\n", n_seeds);
            std::vector<std::pair<std::string, Metrics>> rows;
            for (const PenaltyConfig &pc : configs)
            {
                Metrics sum;
                std::string first_text;
                for (int s = 1; s <= n_seeds; ++s)
                {
                    Sampler sampler{SamplerConfig{.temperature = 0.8F,
                                                  .top_k = 40,
                                                  .top_p = 0.95F,
                                                  .seed = static_cast<std::uint64_t>(s)}};
                    PenaltyProcessor proc{pc};
                    const std::vector<int> out = model.generate(ids, max_new, sampler, nullptr, {}, &proc);
                    sum += measure(model, out, P);
                    if (s == 1)
                    {
                        const auto end = out.begin() + static_cast<std::ptrdiff_t>(std::min(out.size(), P + 30));
                        first_text = show(tok.decode(std::vector<int>(out.begin() + static_cast<std::ptrdiff_t>(P), end)));
                    }
                }
                rows.emplace_back(label(pc), sum / static_cast<double>(n_seeds));
                println("  {} │{}…", label(pc), first_text);
            }
            println("");
            print_metrics_header();
            for (const auto &[name, m] : rows)
                print_metrics_row(name, m);
            println("\n随机采样本身已经打破了大部分循环，惩罚的作用变成「连续地」压低重复，");
            println("不像贪心那样有一个突然生效的阈值。");
        }

        // ---------------------------------------------------------------------
        section("4. include_prompt：prompt 里的词算不算「出现过」");
        // ---------------------------------------------------------------------
        {
            const PenaltyConfig variants[] = {
                {.presence_penalty = 0.0F, .frequency_penalty = 1.5F, .include_prompt = false},
                {.presence_penalty = 0.0F, .frequency_penalty = 1.5F, .include_prompt = true},
            };
            std::vector<std::pair<std::string, Metrics>> rows;
            for (const PenaltyConfig &pc : variants)
            {
                Sampler greedy{SamplerConfig{.temperature = 0.0F}};
                PenaltyProcessor proc{pc};
                const std::vector<int> out = model.generate(ids, max_new, greedy, nullptr, {}, &proc);
                const std::vector<int> gen(out.begin() + static_cast<std::ptrdiff_t>(P), out.end());
                const std::string name = label(pc) + (pc.include_prompt ? "  +prompt" : "");
                rows.emplace_back(name, measure(model, out, P));
                println("\033[36m── {}\033[0m", name);
                println("{}\n", tok.decode(gen));
            }
            print_metrics_header();
            for (const auto &[name, m] : rows)
                print_metrics_row(name, m);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
