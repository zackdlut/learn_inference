// 第 0 步：测本机内存读带宽的上限。
//
// decode 每个 token 要把全部权重读一遍，所以「带宽 ÷ 每 token 字节数」就是 decode 速度的天花板。
// 这个数字要自己测：CPU 手册上的理论带宽在笔记本 + WSL2 上达不到。
//
//   ./build-release/bench/bench_bw                 # 默认 496 MB，扫 1,2,4,6,... 线程
//   ./build-release/bench/bench_bw 496 1,4,8,14    # 指定大小（MB）和线程数
//
// 做法：每个线程用 AVX2 顺序读自己那一段并累加。几个要点：
//   - 数据要远大于 L3（这台机器 12 MB），否则测到的是缓存带宽
//   - 每个线程用 4 个独立累加器：只用 1 个时，测到的是加法的延迟（4 周期一次），不是带宽
//   - 结果要「被用到」，否则编译器会把整个循环删掉

#include "bench_common.hpp"

#include <immintrin.h>

using namespace bench;

namespace
{

float sum_range(const float *p, std::size_t n)
{
    std::size_t i = 0;
    float s = 0.0F;
#if defined(__AVX2__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32)
    {
        a0 = _mm256_add_ps(a0, _mm256_loadu_ps(p + i));
        a1 = _mm256_add_ps(a1, _mm256_loadu_ps(p + i + 8));
        a2 = _mm256_add_ps(a2, _mm256_loadu_ps(p + i + 16));
        a3 = _mm256_add_ps(a3, _mm256_loadu_ps(p + i + 24));
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (float t : tmp)
        s += t;
#endif
    for (; i < n; ++i)
        s += p[i];
    return s;
}

// 用 nt 个线程把 data 完整读一遍，返回耗时（毫秒）
double read_once(const std::vector<float> &data, int nt)
{
    std::vector<float> partial(static_cast<std::size_t>(nt));
    const std::size_t chunk = (data.size() + nt - 1) / nt;
    const auto t0 = Clock::now();
    {
        std::vector<std::jthread> threads;
        for (int t = 0; t < nt; ++t)
            threads.emplace_back(
                [&, t]
                {
                    const std::size_t b = std::min(data.size(), t * chunk);
                    const std::size_t e = std::min(data.size(), b + chunk);
                    partial[t] = sum_range(data.data() + b, e - b);
                });
    }
    const double ms = ms_since(t0);
    do_not_optimize(partial.data());
    return ms;
}

} // namespace

int main(int argc, char **argv)
{
    const double mb = std::stod(arg_or(argc, argv, 1, "496"));
    const auto threads = argc > 2 ? parse_int_list(argv[2]) : default_thread_list();
    const auto n = static_cast<std::size_t>(mb * 1e6 / sizeof(float));

    println("缓冲区 {:.0f} MB（GPT-2 decode 每 token 读 ~494 MB 权重）", mb);
    std::vector<float> data(n);
    for (std::size_t i = 0; i < n; ++i) // 先写一遍：让操作系统真正分配物理页
        data[i] = static_cast<float>(i & 1023) * 1e-3F;

    println("\n{:>6}{:>12}{:>14}", "线程", "最快 ms", "带宽 GB/s");
    for (int nt : threads)
    {
        double best = 1e30;
        for (int r = 0; r < 7; ++r) // 取最快一次：带宽上限关心的是「能达到多少」
            best = std::min(best, read_once(data, nt));
        println("{:>6}{:>12.2f}{:>14.1f}", nt, best, static_cast<double>(n * sizeof(float)) / (best * 1e6));
    }
    println("\n带宽在几个线程时停止增长，就是 decode 多线程的合理线程数。");
    println("fp32 decode 理论上限 ≈ 494 MB ÷ 最高带宽。");
}
