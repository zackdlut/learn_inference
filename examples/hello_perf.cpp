#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "inference/kernel.hpp"
#include "inference/mlp.hpp"
#include "inference/model_file.hpp"
#include "inference/ops.hpp"
#include "inference/tensor.hpp"

using inference::int64;
using inference::Tensor;
namespace kernel = inference::kernel;
namespace ops = inference::ops;

template <class... Args> void println(std::format_string<Args...> fmt, Args &&...args)
{
    std::cout << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

void section(const char *title)
{
    std::cout << std::format("\n\033[36m===== {} =====\033[0m\n", title);
}

// 跑若干次取最小值。
//
// 为什么取最小值而不是平均值？因为干扰（操作系统调度、其他进程、
// cache 被别人挤掉）只会让某次变慢，不会让它变快。
// 最小值最接近「这段代码在理想条件下的真实速度」。
double bench_ms(const std::function<void()> &fn, int warmup = 1, int runs = 3)
{
    for (int i = 0; i < warmup; ++i)
        fn(); // 预热：把数据和指令都拉进 cache
    double best = 1e30;
    for (int i = 0; i < runs; ++i)
    {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

void bar(double value, double max_value, int width = 30)
{
    const int n = static_cast<int>(value / max_value * width + 0.5);
    std::cout << "\033[32m" << std::string(static_cast<std::size_t>(std::max(n, 1)), '#')
              << "\033[0m";
}

void run_matmul_suite(int64 N)
{
    const double flops = 2.0 * static_cast<double>(N * N * N);
    println("");
    println("  \033[1mN = {}\033[0m    计算量 {:.2f} GFLOP", N, flops / 1e9);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    Tensor a = Tensor::zeros({N, N});
    Tensor b = Tensor::zeros({N, N});
    for (float &v : a.data())
        v = dist(rng);
    for (float &v : b.data())
        v = dist(rng);

    Tensor ref = Tensor::zeros({N, N});
    Tensor out = Tensor::zeros({N, N});
    struct Variant
    {
        const char *name;
        const char *note;
        void (*fn)(std::span<const float>, std::span<const float>, std::span<float>, int64, int64,
                   int64);
        int warmup;
        int runs;
    };
    // note 存的是裸指针，字符串必须活得比 variants 久，不能直接用 format(...) 的临时值
    const std::string parallel_note = std::format("{} 线程并行", kernel::get_num_threads());
    const Variant variants[] = {
        // v1 一次要好几秒，跑一轮就够看出量级
        {"v1 朴素 i-j-k", "教科书写法", kernel::matmul_f32_ijk, 0, 1},
        {"v2 换循环顺序", "i-k-j，内层连续访问", kernel::matmul_f32_ikj, 1, 5},
        {"v3 循环分块", "B 块打包成连续内存，留在 L1", kernel::matmul_f32_tiled, 1, 5},
        {"v4 AVX2+FMA", "B 打包 + 4×16 微内核，累加器在寄存器", kernel::matmul_f32_avx2, 1, 7},
        {"v5 多线程", parallel_note.c_str(), kernel::matmul_f32_parallel, 1, 9},
    };
    // 先用参考实现算出标准答案
    kernel::matmul_f32_ijk(a.data(), b.data(), ref.data(), N, N, N);

    double baseline = 0.0, best_gflops = 0.0;
    std::vector<std::pair<std::string, double>> results;

    for (const auto &v : variants)
    {
        std::ranges::fill(out.data(), 0.0F);
        const double ms = bench_ms([&] { v.fn(a.data(), b.data(), out.data(), N, N, N); },
                                   v.warmup, v.runs);
        const double gflops = flops / (ms * 1e6);
        if (baseline == 0.0)
            baseline = ms;
        best_gflops = std::max(best_gflops, gflops);

        // 每个优化版都必须和参考实现对得上，否则再快也没意义
        const float diff = ops::max_abs_diff(out, ref);
        const bool ok = diff < 1e-3F;

        results.emplace_back(v.name, gflops);
        println("  {:<14} {:>9.1f} ms  {:>7.1f} GFLOPS  {:>6.1f}×   {}  {}", v.name, ms, gflops,
                baseline / ms, ok ? "\033[32m正确\033[0m" : "\033[31m错误!\033[0m", v.note);
    }

    println("");
    for (const auto &[name, g] : results)
    {
        std::cout << std::format("  {:<14} ", name);
        bar(g, best_gflops);
        std::cout << std::format(" {:.1f}\n", g);
    }
    println("");
    println("  总加速比 \033[1m{:.0f}×\033[0m，达到理论峰值的约 {:.0f}%",
            results.back().second / results.front().second, best_gflops / 672.0 * 100.0);
}

int main()
{
    println("");
    println("CPU 线程数: {}    AVX2/FMA: {}", std::thread::hardware_concurrency(),
            kernel::has_avx2_support() ? "\033[32m可用\033[0m" : "\033[31m不可用\033[0m");

    // -------------------------------------------------------------------------
    section("1. 先算清楚理论上限");
    println("  本机 CPU 每核每周期能做多少浮点运算？");
    println("    AVX2 一条指令处理 8 个 float");
    println("    FMA 一条指令同时做乘和加      → 16 FLOP/指令");
    println("    每周期能发射 2 条 FMA         → 32 FLOP/周期/核");
    println("  按 ~3 GHz、7 个物理核估算：");
    println("    单核峰值 ≈ 32 × 3e9        = \033[1m96 GFLOPS\033[0m");
    println("    全核峰值 ≈ 96 × 7          = \033[1m672 GFLOPS\033[0m");
    println("");
    println("  真实代码能到峰值的 50~70% 就算很不错了（OpenBLAS 大约 85%）。");
    println("  知道上限才知道优化到哪算够 —— 否则你永远不知道还有多少空间。");

    // -------------------------------------------------------------------------
    section("2. 矩阵乘法：五个版本");
    println("  同一组实现跑两种尺寸。1024 是 2 的幂，B 相邻两行恰好相隔 4 KB，");
    println("  未打包的分块会在 L1 里发生组冲突；1000 则没有这个问题，可以对照着看。");
    println("  每个版本先预热一次，再跑多轮取最快的一次，减少调度（P 核/E 核）带来的噪声。");
    run_matmul_suite(1024);
    run_matmul_suite(1000);
}
