// 第 6 阶段：让模型说话。
//
//   ./build/examples/06_gpt2
//   ./build/examples/06_gpt2 "Once upon a time"
//   ./build/examples/06_gpt2 "The meaning of life is" 60 0.8

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "inference/gpt.hpp"
#include "inference/kernel.hpp"
#include "inference/model_file.hpp"
#include "inference/ops.hpp"
#include "inference/tokenizer.hpp"

using inference::GPT;
using inference::KVCache;
using inference::Sampler;
using inference::SamplerConfig;
using inference::Tensor;
using inference::Tokenizer;
namespace ops = inference::ops;

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

// 打印一个 token 的可读形式：把不可见字符显示出来，否则看日志很困惑
std::string show(const std::string &s)
{
    std::string out;
    for (char c : s)
    {
        if (c == '\n')
            out += "\\n";
        else if (c == ' ')
            out += '_';
        else
            out += c;
    }
    return out;
}

// 分算子耗时表。矩阵乘那几行额外给出「有效带宽」= 权重字节数 / 耗时：
// decode 是访存密集的，这个数离机器的内存带宽上限越近，说明内核越接近最优。
// 每个权重几字节：fp32 是 4；fp16 是 2；Q8_0 每 32 个权重存 32 个 int8 + 1 个 fp32 scale = 1.125
struct WeightBytes
{
    double layer = 4.0;   // 每层的 4 个矩阵
    double lm_head = 4.0; // 输出层
};

void print_profile(const char *title, const inference::OpProfile &p,
                   const inference::GPTConfig &cfg, WeightBytes wb = {})
{
    if (p.forwards == 0)
        return;
    const double L = cfg.n_layer, C = cfg.n_embd, V = cfg.n_vocab;
    const double B = wb.layer;
    const double n = p.forwards;
    struct Row
    {
        const char *name;
        double ms;
        double weight_bytes; // 每次 forward 读的权重字节数，0 表示不是矩阵乘
    };
    const Row rows[] = {
        {"embed", p.embed, 0},
        {"layernorm", p.layernorm, 0},
        {"qkv (c_attn)", p.qkv, L * C * 3 * C * B},
        {"copy (split/kv)", p.copy, 0},
        {"attention", p.attention, 0},
        {"attn_proj", p.attn_proj, L * C * C * B},
        {"residual add", p.residual, 0},
        {"mlp_fc", p.mlp_fc, L * C * 4 * C * B},
        {"gelu", p.gelu, 0},
        {"mlp_proj", p.mlp_proj, L * 4 * C * C * B},
        {"lm_head", p.lm_head, V * C * wb.lm_head},
        {"sample", p.sample, 0},
    };
    const double total = p.total();
    println("\n{}（{} 次 forward，每次平均）:", title, p.forwards);
    println("  {:<16}{:>10}{:>8}{:>12}", "算子", "ms", "占比", "有效带宽");
    for (const auto &r : rows)
    {
        const std::string bw =
            r.weight_bytes > 0 && r.ms > 0
                ? std::format("{:.1f} GB/s", r.weight_bytes * n / (r.ms * 1e6))
                : "";
        println("  {:<16}{:>10.3f}{:>7.1f}%{:>12}", r.name, r.ms / n, r.ms / total * 100.0, bw);
    }
    println("  {:<16}{:>10.3f}", "合计", total / n);
}

// 困惑度 = exp(平均每个 token 的负对数似然)。直观理解：模型在每个位置上
// 「相当于在多少个词里均匀瞎猜」。越低越好；量化后涨得越少，说明精度损失越小。
double perplexity(GPT &model, const std::vector<int> &tokens)
{
    const Tensor logits = model.forward(tokens);
    const std::span<const float> lv = logits.data();
    const auto V = static_cast<std::size_t>(logits.dim(1));
    double nll = 0.0;
    for (std::size_t t = 0; t + 1 < tokens.size(); ++t)
    {
        const float *row = lv.data() + t * V;
        const float mx = *std::max_element(row, row + V);
        double sum = 0.0;
        for (std::size_t i = 0; i < V; ++i)
            sum += std::exp(static_cast<double>(row[i] - mx));
        const double log_p = static_cast<double>(row[tokens[t + 1]] - mx) - std::log(sum);
        nll -= log_p;
    }
    return std::exp(nll / static_cast<double>(tokens.size() - 1));
}

std::vector<int> top_k_ids(const Tensor &logits_row, int k)
{
    const std::span<const float> v = logits_row.data();
    std::vector<int> order(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
        order[i] = static_cast<int>(i);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int a, int b) { return v[a] > v[b]; });
    order.resize(static_cast<std::size_t>(k));
    return order;
}

Tensor last_row(const Tensor &logits)
{
    return logits.slice(0, logits.dim(0) - 1, logits.dim(0)).contiguous();
}

int main(int argc, char **argv)
{
    const std::string dir = INFER_MODEL_GPT_DIR;
    const std::string prompt = argc > 1 ? argv[1] : "The capital of France is";
    const int max_new = argc > 2 ? std::stoi(argv[2]) : 40;
    const float temp = argc > 3 ? std::stof(argv[3]) : 0.0F;

    // decode 是访存密集的，内存带宽 6~8 个线程就饱和了；再多的线程只会互相等
    // （E 核掉队、自旋线程抢超线程资源）。所以推理程序默认只用一半逻辑核，
    // 想做实验可以 INFER_NUM_THREADS=14 覆盖。llama.cpp 默认也只用物理大核。
    if (std::getenv("INFER_NUM_THREADS") == nullptr)
    {
        const auto hw = static_cast<int>(std::thread::hardware_concurrency());
        inference::kernel::set_num_threads(std::max(1, hw / 2));
    }

    try
    {
        // ---------------------------------------------------------------------
        section("1. 分词：文字 → 数字");
        // ---------------------------------------------------------------------
        Tokenizer tok = Tokenizer::load(dir + "/gpt2_vocab.bin", dir + "/gpt2_merges.bin");
        println("词表大小: {}", tok.vocab_size());

        println("\n预分词（BPE 之前先按规则切块）:");
        std::string chunks;
        for (const auto &c : Tokenizer::pre_tokenize(prompt))
        {
            chunks += std::format("[{}]", show(c));
        }
        println("  {}", chunks);

        const std::vector<int> ids = tok.encode(prompt);
        println("\n编码结果 {} 个 token:", ids.size());
        for (int id : ids)
            println("  {:>6}  '{}'", id, show(tok.decode_one(id)));
        println("\n解码回去: '{}'", tok.decode(ids));
        println("往返一致: {}", tok.decode(ids) == prompt ? "是" : "否");

        // ---------------------------------------------------------------------
        section("2. 加载 124M 参数");
        // ---------------------------------------------------------------------
        auto t0 = Clock::now();
        GPT model = GPT::load(dir);
        const auto &cfg = model.config();
        println("加载耗时 {:.0f} ms", ms_since(t0));
        println("层数 {}, 注意力头 {}, 隐藏维 {}, 词表 {}, 上下文上限 {}", cfg.n_layer, cfg.n_head,
                cfg.n_embd, cfg.n_vocab, cfg.n_ctx);
        println("AVX2: {}，线程数 {}", inference::kernel::has_avx2_support() ? "开" : "关",
                inference::kernel::get_num_threads());

        // ---------------------------------------------------------------------
        section("3. 和 numpy 参考实现逐层对拍");
        // ---------------------------------------------------------------------
        try
        {
            const inference::ModelFile ref =
                inference::ModelFile::load(dir + "/gpt2_reference.inference");
            const Tensor rt = ref.get_tensor("tokens");
            std::vector<int> ref_tokens;
            for (float f : rt.data())
                ref_tokens.push_back(static_cast<int>(f));

            const GPT::Trace tr = model.forward_trace(ref_tokens);
            struct Item
            {
                const char *name;
                const Tensor *got;
            };
            const Item items[] = {
                {"embed", &tr.embed},   {"l0_attn_out", &tr.l0_attn_out},
                {"l0_out", &tr.l0_out}, {"final_norm", &tr.final_norm},
                {"logits", &tr.logits},
            };
            println("{:<14}{:>14}{:>10}", "张量", "最大绝对误差", "结果");
            bool all_ok = true;
            for (const auto &it : items)
            {
                const float d = ops::max_abs_diff(*it.got, ref.get_tensor(it.name));
                // 容差随层数放宽：误差会沿 12 层累积，而且 logits 的量级本身就在 100 上下，
                // 相对误差其实只有 1e-5 量级。
                const float tol = std::string(it.name) == "logits" ? 2e-2F : 5e-3F;
                const bool ok = d <= tol;
                all_ok = all_ok && ok;
                println("{:<14}{:>14.2e}{:>10}", it.name, d, ok ? "通过" : "不符");
            }
            println("\n逐层对齐: {}", all_ok ? "全部通过" : "有偏差");
        }
        catch (const std::exception &e)
        {
            println("跳过对拍（{}）", e.what());
        }

        // ---------------------------------------------------------------------
        section("4. 模型在想什么");
        // ---------------------------------------------------------------------
        {
            const Tensor logits = model.forward(ids);
            // 只看最后一个位置：它预测的是「下一个词」
            const Tensor last = logits.slice(0, logits.dim(0) - 1, logits.dim(0)).contiguous();
            const Tensor prob = ops::softmax(last);
            std::span<const float> pv = prob.data();

            std::vector<int> order(pv.size());
            for (std::size_t i = 0; i < pv.size(); ++i)
                order[i] = static_cast<int>(i);
            std::partial_sort(order.begin(), order.begin() + 10, order.end(),
                              [&](int a, int b) { return pv[a] > pv[b]; });

            println("'{}' 之后最可能的 10 个词:\n", prompt);
            for (int r = 0; r < 10; ++r)
            {
                const int id = order[static_cast<std::size_t>(r)];
                const float p = pv[static_cast<std::size_t>(id)];
                const int bars = static_cast<int>(p * 60.0F) + 1;
                println("  {:>2}. {:<14} {:>6.2f}%  {}", r + 1,
                        std::format("'{}'", show(tok.decode_one(id))), p * 100.0F,
                        std::string(static_cast<std::size_t>(bars), '#'));
            }
            println("\n注意 GPT-2 small 只有 124M 参数，是 2019 年最小的那一档。");
            println("它的语法很好，但事实知识很弱 —— 它更爱接万能的 ' the'，");
            println("而不是正确答案。这不是你的代码有 bug，详见 docs/06-gpt2.md。");
        }

        // ---------------------------------------------------------------------
        section("5. KV-Cache 到底省了多少");
        // ---------------------------------------------------------------------
        {
            // 不用 cache：每生成一个词都把整段重算一遍
            std::vector<int> seq = ids;
            t0 = Clock::now();
            for (int i = 0; i < 8; ++i)
            {
                const Tensor lg = model.forward(seq);
                const Tensor last = lg.slice(0, lg.dim(0) - 1, lg.dim(0)).contiguous();
                seq.push_back(static_cast<int>(ops::argmax(last)));
            }
            const double no_cache_ms = ms_since(t0);

            // 用 cache
            KVCache cache(cfg, static_cast<int>(ids.size()) + 8);
            t0 = Clock::now();
            Sampler greedy{SamplerConfig{.temperature = 0.0F}};
            GPT::GenStats st;
            model.generate(ids, 8, greedy, &st);
            const double cache_ms = ms_since(t0);

            println("生成 8 个 token:");
            println("  不用 KV-Cache: {:>8.0f} ms", no_cache_ms);
            println("  用 KV-Cache:   {:>8.0f} ms   （prefill {:.0f} + decode {:.0f}）", cache_ms,
                    st.prefill_ms, st.decode_ms);
            println("  加速 {:.2f}×", no_cache_ms / cache_ms);
            println("\n序列越长差距越大：不用 cache 是 O(n²)，用了是 O(n)。");
            println("代价是内存：这次的 cache 占 {:.1f} MB；",
                    static_cast<double>(cache.bytes()) / 1e6);
            println("如果撑满 {} 个 token 的上下文，要 {:.0f} MB。", cfg.n_ctx,
                    static_cast<double>(KVCache(cfg, cfg.n_ctx).bytes()) / 1e6);
        }

        // 生成一段文字并打印速度和分算子耗时；fp32 和量化后各跑一次
        auto run_generation = [&](WeightBytes wb) {
            SamplerConfig sc;
            sc.temperature = temp;
            sc.top_k = 40;
            sc.top_p = 0.95F;
            Sampler sampler{sc};

            std::cout << "\033[33m" << prompt << "\033[0m" << std::flush;
            GPT::GenStats st;
            model.generate(ids, max_new, sampler, &st,
                           [&](int id) { std::cout << tok.decode_one(id) << std::flush; });
            std::cout << "\n\n";

            println("prefill {} 个 token: {:.0f} ms  （{:.1f} token/s）", ids.size(), st.prefill_ms,
                    static_cast<double>(ids.size()) / st.prefill_ms * 1000.0);
            println("decode {} 个 token:  {:.0f} ms  （{:.1f} token/s）", st.generated,
                    st.decode_ms, static_cast<double>(st.generated) / st.decode_ms * 1000.0);
            print_profile("prefill 分算子耗时", st.prefill_ops, cfg, wb);
            print_profile("decode 分算子耗时", st.decode_ops, cfg, wb);
        };

        // ---------------------------------------------------------------------
        section("6. 生成");
        // ---------------------------------------------------------------------
        println("prompt: \033[33m{}\033[0m", prompt);
        println("温度 {:.1f}{}\n", temp, temp <= 0.0F ? "（贪心）" : "");
        run_generation({});
        println("\nprefill 是计算密集的（大矩阵乘），decode 是访存密集的：");
        println("每生成一个词，都要把 500 MB 权重完整读一遍。这就是为什么");
        println("大模型推理的瓶颈通常是内存带宽，而不是算力 —— 也是量化（阶段 7）的动机。");

        // ---------------------------------------------------------------------
        section("7. 量化：层内 Q8_0 + lm_head fp16");
        // ---------------------------------------------------------------------
        {
            // 一段固定的英文，用来测困惑度。量化前后用同一段，才有可比性
            const std::string text =
                "The city of Paris is the capital and most populous city of France. It is "
                "located on the Seine River, in the north of the country. Since the 17th "
                "century, Paris has been one of the major centres of finance, diplomacy, "
                "commerce, culture, fashion and gastronomy. For its leading role in the arts "
                "and sciences, it was called the capital of the world in the 19th century.";
            const std::vector<int> ppl_ids = tok.encode(text);

            const double ppl_fp32 = perplexity(model, ppl_ids);
            const Tensor ref_last = last_row(model.forward(ids));

            t0 = Clock::now();
            model.quantize_q8_0();
            const double quant_ms = ms_since(t0);

            const double L = cfg.n_layer, C = cfg.n_embd, V = cfg.n_vocab;
            const double n_layer_w = L * 12 * C * C, n_lm_w = V * C; // 每 token 读的权重个数
            println("量化耗时 {:.0f} ms（只在加载时做一次）", quant_ms);
            println("每 token 读的权重: fp32 {:.0f} MB → 量化后 {:.0f} MB",
                    (n_layer_w + n_lm_w) * 4 / 1e6, (n_layer_w * 1.125 + n_lm_w * 2) / 1e6);
            println("  层内 48 个矩阵 Q8_0（1.125 B/权重），lm_head fp16（2 B/权重）。");
            println("  lm_head 不用 Q8_0：它的输入 ln_f 输出里有个通道大到 ~240，普通通道 <1，");
            println("  按 32 个一组量化时，同组其它 31 个数会被压成 0 或 ±2，困惑度会涨 ~8%。");

            const double ppl_q8 = perplexity(model, ppl_ids);
            const Tensor q_last = last_row(model.forward(ids));

            const auto top_ref = top_k_ids(ref_last, 10);
            const auto top_q = top_k_ids(q_last, 10);
            int overlap = 0;
            for (int id : top_q)
                overlap += static_cast<int>(std::count(top_ref.begin(), top_ref.end(), id));

            println("\n精度对比（{} 个 token 的文本）:", ppl_ids.size());
            println("  困惑度   fp32 {:.3f}   量化 {:.3f}   （{:+.2f}%）", ppl_fp32, ppl_q8,
                    (ppl_q8 / ppl_fp32 - 1.0) * 100.0);
            println("  prompt 最后位置 logits 最大误差 {:.3f}（logits 量级约 ±100）",
                    ops::max_abs_diff(ref_last, q_last));
            println("  top-1 是否一致: {}，top-10 重合 {}/10", top_ref[0] == top_q[0] ? "是" : "否",
                    overlap);

            println("\n量化后再生成一次（同样的 prompt 和采样设置）:\n");
            run_generation({.layer = 1.125, .lm_head = 2.0});
            println("\n对比第 6 节：层内矩阵读的字节变成 1/3.6，lm_head 变成 1/2，");
            println("耗时也接近按比例下降 —— 访存密集的算子，省字节就是省时间。");
        }

        if (temp <= 0.0F)
        {
            println("\n\033[2m贪心解码容易陷入重复。试试: {} \"{}\" 60 0.8\033[0m", argv[0],
                    prompt);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
