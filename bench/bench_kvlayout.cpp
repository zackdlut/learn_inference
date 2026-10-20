// 第 5 步：冷缓存下的 KV-Cache 布局对比，以及注意力并行阈值的交叉点。
//
//   ./build-release/bench/bench_kvlayout              # 多线程用一半逻辑核
//   ./build-release/bench/bench_kvlayout 7 200 5      # 线程数、第 3 部分生成的 token 数、轮数
//
// bench_attn 反复读同一块 K/V，数据一直热在缓存里，掩盖了真实 decode 的访存问题：
// 真实模型每个 token 要依次读 12 层各自的 K/V，读到第 l 层时它早被前面的层挤出了缓存。
// 这里照样分配 12 层各一份（两种布局各 75 MB，远超 12 MB 的 L3），一个 token = 12 层各算一次。
//
// 两种布局（一个头的 64 个数怎么放）：
//   按位置 [capacity][768]：第 h 个头夹在每行中间，读一个头要每隔 3 KB 跳着取 256 字节
//   按头   [12][capacity][64]：一个头的全部历史是一整段连续内存，硬件预取器跟得上
//
// 第 2 部分强制总是并行（set_attention_parallel_min_work(0)），在同一进程里交替测
// 1 线程和多线程，找出「并行开始划算」的 Tk。库里的默认阈值 30 万 ≈ decode 的 Tk 390。
// 第 3 部分加载真实模型，ABBA 交替生成，比较默认阈值和总是并行时的注意力耗时和 decode 总耗时。

#include "bench_common.hpp"
#include "inference/gpt.hpp"
#include "inference/kernel.hpp"
#include "inference/tokenizer.hpp"

#include <cstdint>

using namespace bench;
namespace kernel = inference::kernel;
using int64 = std::int64_t;
using kernel::HeadView;

namespace
{

constexpr int64 L = 12, H = 12, D = 64, C = H * D, kCap = 1024;

struct Layout
{
    const char *name;
    std::vector<std::vector<float>> k, v; // 每层一份
    int64 row_stride, head_stride;
};

Layout make_layout(const char *name, int64 row_stride, int64 head_stride, unsigned seed)
{
    Layout lay{name, {}, {}, row_stride, head_stride};
    for (int64 l = 0; l < L; ++l)
    {
        lay.k.push_back(random_floats(kCap * C, seed + 2 * l));
        lay.v.push_back(random_floats(kCap * C, seed + 2 * l + 1));
    }
    return lay;
}

// 一个 decode token：12 层各做一次注意力
double token_ms(const Layout &lay, const std::vector<float> &q, std::vector<float> &out, int64 Tk)
{
    return time_ms(
        [&]
        {
            for (int64 l = 0; l < L; ++l)
                kernel::attention_f32_strided(HeadView{q.data(), C, D},
                                              HeadView{lay.k[l].data(), lay.row_stride, lay.head_stride},
                                              HeadView{lay.v[l].data(), lay.row_stride, lay.head_stride},
                                              out.data(), C, 1, Tk, H, D, Tk - 1);
        });
}

} // namespace

int main(int argc, char **argv)
{
    const int nt = std::stoi(arg_or(argc, argv, 1, std::to_string(default_threads()).c_str()));
    const int gen_tokens = std::stoi(arg_or(argc, argv, 2, "200"));
    const int rounds = std::stoi(arg_or(argc, argv, 3, "5"));
    println("分配 KV-Cache：2 种布局 × 12 层 × K/V × 1024 × 768 float = 150 MB ...");
    const Layout by_pos = make_layout("按位置", C, D, 100);
    const Layout by_head = make_layout("按头", D, kCap * D, 200);
    const auto q = random_floats(C, 1);
    std::vector<float> out(C);

    section("1. 两种布局，ms / token（12 层），同一进程交替跑，20 次取中位数");
    println("{:>8}{:>8}{:>12}{:>12}{:>10}", "线程", "Tk", "按位置", "按头", "加速");
    for (int threads : {1, nt})
    {
        kernel::set_num_threads(threads);
        for (int64 Tk : {128, 256, 512, 1024})
        {
            std::vector<double> a, b;
            token_ms(by_pos, q, out, Tk); // 预热
            for (int r = 0; r < 20; ++r)
            {
                a.push_back(token_ms(by_pos, q, out, Tk));
                b.push_back(token_ms(by_head, q, out, Tk));
            }
            println("{:>8}{:>8}{:>12.3f}{:>12.3f}{:>9.2f}×", threads, Tk, median(a), median(b),
                    median(a) / median(b));
        }
    }

    section(std::format("2. 并行阈值：强制总是并行，1 线程 vs {} 线程（按头布局）", nt));
    const auto old = kernel::set_attention_parallel_min_work(0);
    println("{:>8}{:>14}{:>14}{:>12}  {}", "Tk", "1 线程 ms", "多线程 ms", "工作量", "结论");
    for (int64 Tk : {32, 64, 128, 256, 384, 512, 1024})
    {
        std::vector<double> a, b;
        for (int r = 0; r < 20; ++r)
        {
            kernel::set_num_threads(1);
            token_ms(by_head, q, out, Tk); // 换线程数会重建线程池，先空跑一次
            a.push_back(token_ms(by_head, q, out, Tk));
            kernel::set_num_threads(nt);
            token_ms(by_head, q, out, Tk);
            b.push_back(token_ms(by_head, q, out, Tk));
        }
        const double s = median(a), p = median(b);
        println("{:>8}{:>14.3f}{:>14.3f}{:>12}  {}", Tk, s, p, H * Tk * D,
                p < 0.9 * s ? "并行划算" : (p > 1.1 * s ? "并行更慢" : "差不多"));
    }
    kernel::set_attention_parallel_min_work(old);
    println("\n库里的阈值是 n_head × Tq × Tk × head_size ≥ {} 才并行（decode 时 Tk ≈ {}）", old,
            old / (H * D));
    println("注意：这里每次切换线程数后都先空跑一次，工作线程正在自旋，测到的是并行的最好情况。");

    section(std::format("3. 端到端：真实模型生成 {} 个 token，默认阈值 vs 总是并行（{} 线程）", gen_tokens, nt));
    kernel::set_num_threads(nt);
    const std::string dir = INFER_MODEL_GPT_DIR;
    inference::GPT model = inference::GPT::load(dir);
    const auto tok = inference::Tokenizer::load(dir + "/gpt2_vocab.bin", dir + "/gpt2_merges.bin");
    const std::vector<int> prompt = tok.encode("The capital of France is");
    auto attn_ms = [&](int64 threshold)
    {
        kernel::set_attention_parallel_min_work(threshold);
        inference::Sampler greedy{inference::SamplerConfig{.temperature = 0.0F}};
        inference::GPT::GenStats st;
        model.generate(prompt, gen_tokens, greedy, &st);
        return std::pair{st.decode_ops.attention / st.decode_ops.forwards,
                         st.decode_ms / st.generated};
    };
    std::vector<double> a_def, a_par, d_def, d_par;
    auto run_def = [&] { const auto [a, d] = attn_ms(-1); a_def.push_back(a), d_def.push_back(d); };
    auto run_par = [&] { const auto [a, d] = attn_ms(0); a_par.push_back(a), d_par.push_back(d); };
    for (int r = 0; r < rounds; ++r)
    {
        // ABBA 顺序：避免"总是后跑的那个赶上降频"
        if (r % 2 == 0) run_def(), run_par();
        else run_par(), run_def();
    }
    kernel::set_attention_parallel_min_work(-1);
    println("{:<16}{:>16}{:>16}", "", "注意力 ms/token", "decode ms/token");
    println("{:<16}{:>16.3f}{:>16.3f}", "默认阈值", median(a_def), median(d_def));
    println("{:<16}{:>16.3f}{:>16.3f}", "总是并行", median(a_par), median(d_par));
    println("（平均上下文长度约 {} 个 token；{} 轮交替取中位数）", gen_tokens / 2 + 5, rounds);
}
