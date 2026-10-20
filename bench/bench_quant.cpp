// 第 7 步：量化的精度和速度。
//
//   ./build-release/bench/bench_quant            # 三部分全跑（约 1–2 分钟）
//   ./build-release/bench/bench_quant outlier    # 只看 lm_head 输入的离群通道
//   ./build-release/bench/bench_quant ppl        # 只做逐矩阵的困惑度排查
//   ./build-release/bench/bench_quant speed      # 只比各方案的 decode 速度
//
// 1. outlier：用 forward_trace 拿到 lm_head 的输入（ln_f 的输出），统计每个通道的 max|x|
// 2. ppl：每次只量化一类矩阵，分别看「权重 + 激活都量化（W8A8）」和「只量化权重」的困惑度变化。
//    每个配置都重新加载一次模型（量化会释放 fp32 权重，不能撤销）
// 3. speed：fp32 和三种量化方案各加载一份，交替生成，比 decode 速度

#include "bench_common.hpp"
#include "inference/gpt.hpp"
#include "inference/kernel.hpp"
#include "inference/tokenizer.hpp"

#include <cmath>
#include <cstdint>
#include <optional>

using namespace bench;
using inference::GPT;
using inference::QuantOptions;
using inference::Tensor;
using inference::Tokenizer;
namespace kernel = inference::kernel;
using LmHead = QuantOptions::LmHead;

namespace
{

const std::string kDir = INFER_MODEL_GPT_DIR;

// 和 hello_gpt 第 7 节同一段文字
const std::string kText =
    "The city of Paris is the capital and most populous city of France. It is "
    "located on the Seine River, in the north of the country. Since the 17th "
    "century, Paris has been one of the major centres of finance, diplomacy, "
    "commerce, culture, fashion and gastronomy. For its leading role in the arts "
    "and sciences, it was called the capital of the world in the 19th century.";

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
        nll -= static_cast<double>(row[tokens[t + 1]] - mx) - std::log(sum);
    }
    return std::exp(nll / static_cast<double>(tokens.size() - 1));
}

GPT load_quantized(const std::optional<QuantOptions> &opt)
{
    GPT m = GPT::load(kDir);
    if (opt)
        m.quantize(*opt);
    return m;
}

QuantOptions only(bool qkv, bool attn_proj, bool fc, bool proj, LmHead lm, bool weight_only)
{
    QuantOptions o;
    o.qkv = qkv;
    o.attn_proj = attn_proj;
    o.mlp_fc = fc;
    o.mlp_proj = proj;
    o.lm_head = lm;
    o.weight_only = weight_only;
    return o;
}

void run_outlier(const Tokenizer &tok)
{
    section("1. lm_head 的输入（ln_f 输出）里有没有离群通道");
    GPT model = GPT::load(kDir);
    const auto ids = tok.encode(kText);
    const Tensor h = model.forward_trace(ids).final_norm; // [T, C]
    const int64_t T = h.dim(0), C = h.dim(1);
    const auto d = h.data();

    std::vector<double> chmax(static_cast<std::size_t>(C), 0.0);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < C; ++c)
            chmax[c] = std::max(chmax[c], static_cast<double>(std::abs(d[t * C + c])));
    std::vector<int> order(static_cast<std::size_t>(C));
    for (int i = 0; i < C; ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return chmax[a] > chmax[b]; });

    println("  {} 个位置 × {} 个通道，每个通道取所有位置上的 max|x|", T, C);
    println("  普通通道（中位数）: {:.2f}", median(chmax));
    for (int i = 0; i < 6; ++i)
        println("  通道 {:>3}: {:>7.1f}   （是中位数的 {:.0f} 倍）", order[i], chmax[order[i]],
                chmax[order[i]] / median(chmax));

    // 最后一个位置按 Q8_0 量化，看最大离群通道所在的那组发生了什么
    const float *x = d.data() + (T - 1) * C;
    std::vector<std::int8_t> q(static_cast<std::size_t>(C));
    std::vector<float> s(static_cast<std::size_t>(C / 32)), back(static_cast<std::size_t>(C));
    kernel::quantize_row_q8_0(x, q.data(), s.data(), C);
    kernel::dequantize_row_q8_0(q.data(), s.data(), back.data(), C);
    const int ch = order[0], blk = ch / 32;
    int zeros = 0;
    double err2 = 0, x2 = 0;
    for (int i = blk * 32; i < blk * 32 + 32; ++i)
        if (i != ch && q[i] == 0)
            ++zeros;
    for (int64_t c = 0; c < C; ++c)
    {
        err2 += (back[c] - x[c]) * (back[c] - x[c]);
        x2 += x[c] * x[c];
    }
    println("\n  最后一个位置按 Q8_0 量化：通道 {} 所在的第 {} 组，步长 {:.3f}", ch, blk, s[blk]);
    println("  同组其它 31 个数里有 {} 个被量化成了 0", zeros);
    println("  整行相对 RMS 误差只有 {:.2f}% —— 误差被大值「平均」掉了，看不出小值全毁了", 100 * std::sqrt(err2 / x2));
}

void run_ppl(const Tokenizer &tok)
{
    section("2. 逐个矩阵排查困惑度（每行都重新加载模型）");
    const auto ids = tok.encode(kText);
    const auto t0 = Clock::now();
    GPT base = GPT::load(kDir);
    const double ppl0 = perplexity(base, ids);
    println("  {} 个 token 的文本，fp32 困惑度 {:.3f}\n", ids.size(), ppl0);
    auto delta = [&](const QuantOptions &o)
    {
        GPT m = load_quantized(o);
        return (perplexity(m, ids) / ppl0 - 1.0) * 100.0;
    };

    struct Row
    {
        const char *name;
        bool qkv, ap, fc, mp;
        LmHead lm;
    };
    const Row rows[] = {
        {"只量化 c_attn", true, false, false, false, LmHead::fp32},
        {"只量化 attn_proj", false, true, false, false, LmHead::fp32},
        {"只量化 mlp_fc", false, false, true, false, LmHead::fp32},
        {"只量化 mlp_proj", false, false, false, true, LmHead::fp32},
        {"只量化 lm_head", false, false, false, false, LmHead::q8_0},
        {"层内全部，lm_head Q8_0", true, true, true, true, LmHead::q8_0},
    };
    println("  {:<26}{:>12}{:>16}", "配置", "W8A8", "只量化权重");
    for (const auto &r : rows)
    {
        const double a = delta(only(r.qkv, r.ap, r.fc, r.mp, r.lm, false));
        const double b = delta(only(r.qkv, r.ap, r.fc, r.mp, r.lm, true));
        println("  {:<26}{:>+11.2f}%{:>+15.2f}%", r.name, a, b);
    }
    println("");
    println("  {:<26}{:>+11.2f}%", "层内全部，lm_head fp32", delta(only(true, true, true, true, LmHead::fp32, false)));
    println("  {:<26}{:>+11.2f}%   ← 采用", "层内全部，lm_head fp16", delta(QuantOptions{}));
    println("\n  （用时 {:.0f} s）83 个 token 的困惑度有 ±0.3% 左右的噪声，小于这个的差别不用当真", ms_since(t0) / 1000);
}

void run_speed(const Tokenizer &tok)
{
    const int nt = default_threads();
    kernel::set_num_threads(nt);
    section(std::format("3. decode 速度（{} 线程，生成 60 个 token，交替 5 轮取中位数）", nt));
    struct Scheme
    {
        const char *name;
        std::optional<QuantOptions> opt;
        double mb; // 每 token 读的权重
    };
    const double layer_w = 12.0 * 12 * 768 * 768, lm_w = 50257.0 * 768;
    std::vector<Scheme> schemes = {
        {"fp32", std::nullopt, (layer_w + lm_w) * 4 / 1e6},
        {"全部 Q8_0", only(true, true, true, true, LmHead::q8_0, false), (layer_w + lm_w) * 1.125 / 1e6},
        {"层内 Q8_0 + lm_head fp32", only(true, true, true, true, LmHead::fp32, false),
         (layer_w * 1.125 + lm_w * 4) / 1e6},
        {"层内 Q8_0 + lm_head fp16", QuantOptions{}, (layer_w * 1.125 + lm_w * 2) / 1e6},
    };
    std::vector<GPT> models;
    for (const auto &s : schemes)
        models.push_back(load_quantized(s.opt));

    const auto ids = tok.encode("The capital of France is");
    std::vector<std::vector<double>> tps(schemes.size());
    for (int round = 0; round < 5; ++round)
        for (std::size_t i = 0; i < models.size(); ++i)
        {
            inference::Sampler greedy{inference::SamplerConfig{.temperature = 0.0F}};
            GPT::GenStats st;
            models[i].generate(ids, 60, greedy, &st);
            tps[i].push_back(st.generated / st.decode_ms * 1000.0);
        }
    println("  {:<28}{:>14}{:>14}", "方案", "每 token 读", "decode");
    for (std::size_t i = 0; i < schemes.size(); ++i)
        println("  {:<28}{:>11.0f} MB{:>9.1f} tok/s", schemes[i].name, schemes[i].mb, median(tps[i]));
}

} // namespace

int main(int argc, char **argv)
{
    const std::string what = arg_or(argc, argv, 1, "all");
    const Tokenizer tok = Tokenizer::load(kDir + "/gpt2_vocab.bin", kDir + "/gpt2_merges.bin");
    if (what == "all" || what == "outlier")
        run_outlier(tok);
    if (what == "all" || what == "ppl")
        run_ppl(tok);
    if (what == "all" || what == "speed")
        run_speed(tok);
}
