// 第 5 步：注意力 kernel 本身，朴素标量版 vs SIMD 单线程 vs SIMD 多线程（热缓存）。
//
//   ./build-release/bench/bench_attn          # 多线程用一半逻辑核
//   ./build-release/bench/bench_attn 7
//
// 「热缓存」：同一块 K/V 反复用，数据一直在 L2/L3 里。测的是 kernel 的计算效率，
// 不是真实 decode 的访存（那个看 bench_kvlayout）。
// 每个数是一层的耗时（12 个头），单位 µs。
//
// 注意多线程那一列里，工作量低于阈值（n_head × Tq × Tk × head_size < 30 万）的形状
// 会自动走串行，所以和单线程差不多 —— 这是有意的，见 bench_kvlayout 的阈值扫描。

#include "bench_common.hpp"
#include "inference/kernel.hpp"

#include <cstdint>

using namespace bench;
namespace kernel = inference::kernel;
using int64 = std::int64_t;

namespace
{

// 按单次耗时自动定重复次数，每项测 ~0.3 秒
template <class F> double auto_median_ms(F &&f)
{
    const double once = time_ms(f);
    const int reps = static_cast<int>(std::clamp(300.0 / std::max(once, 1e-3), 3.0, 2000.0));
    return median_ms(reps, f);
}

} // namespace

int main(int argc, char **argv)
{
    const int nt = std::stoi(arg_or(argc, argv, 1, std::to_string(default_threads()).c_str()));
    constexpr int64 H = 12, D = 64, C = H * D;

    struct Case
    {
        const char *name;
        int64 Tq, Tk;
    };
    const Case cases[] = {
        {"decode Tk=64", 1, 64},     {"decode Tk=256", 1, 256},   {"decode Tk=512", 1, 512},
        {"decode Tk=1024", 1, 1024}, {"prefill T=128", 128, 128}, {"prefill T=512", 512, 512},
    };

    println("每层一次（12 个头 × head_size 64），µs；多线程 = {} 线程\n", nt);
    println("{:<16}{:>12}{:>14}{:>14}{:>10}", "形状", "朴素", "SIMD 1线程", "SIMD 多线程", "总加速");
    for (const auto &c : cases)
    {
        const auto q = random_floats(c.Tq * C, 1);
        const auto k = random_floats(c.Tk * C, 2);
        const auto v = random_floats(c.Tk * C, 3);
        std::vector<float> out(c.Tq * C);
        const int64 q_offset = c.Tk - c.Tq; // decode: Tk-1；prefill: 0

        const double t_naive = auto_median_ms(
            [&] { kernel::attention_f32_naive(q, k, v, out, c.Tq, c.Tk, H, D, q_offset); });
        kernel::set_num_threads(1);
        const double t_1 =
            auto_median_ms([&] { kernel::attention_f32(q, k, v, out, c.Tq, c.Tk, H, D, q_offset); });
        kernel::set_num_threads(nt);
        const double t_n =
            auto_median_ms([&] { kernel::attention_f32(q, k, v, out, c.Tq, c.Tk, H, D, q_offset); });
        println("{:<16}{:>12.1f}{:>14.1f}{:>14.1f}{:>9.1f}×", c.name, t_naive * 1000, t_1 * 1000,
                t_n * 1000, t_naive / t_n);
    }
    println("\n看点：SIMD 只快 ~2 倍（每个 key 只有 4 次 FMA，水平求和的依赖链成了瓶颈）；");
    println("      并行在 prefill 上收益最大（12 × T 个任务，能喂饱所有线程）。");
}
