#pragma once
// bench/ 下各程序共用的小工具：计时、取中位数、解析参数、随机数据。

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <format>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace bench
{

using Clock = std::chrono::steady_clock;

inline double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

template <class F> double time_ms(F &&f)
{
    const auto t0 = Clock::now();
    f();
    return ms_since(t0);
}

inline double median(std::vector<double> v)
{
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// 先跑一次预热（让代码、数据、线程池都就位），再跑 reps 次取中位数。
// 笔记本会降频，同一配置前后能差 30%，中位数比平均值和最小值都稳。
template <class F> double median_ms(int reps, F &&f)
{
    f();
    std::vector<double> t;
    t.reserve(static_cast<std::size_t>(reps));
    for (int i = 0; i < reps; ++i)
        t.push_back(time_ms(f));
    return median(std::move(t));
}

template <class... Args> void println(std::format_string<Args...> fmt, Args &&...args)
{
    std::cout << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

inline void section(const std::string &title)
{
    std::cout << "\n===== " << title << " =====\n";
}

// 阻止编译器把「结果没人用」的计算整个删掉
inline void do_not_optimize(const void *p)
{
    asm volatile("" : : "r"(p) : "memory");
}

inline int hw_threads()
{
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
}

// 推理程序默认用一半逻辑核（和 hello_gpt 一致）；INFER_NUM_THREADS 可以覆盖
inline int default_threads()
{
    if (const char *e = std::getenv("INFER_NUM_THREADS"); e && *e)
        return std::max(1, std::atoi(e));
    return std::max(1, hw_threads() / 2);
}

// "1,4,8" → {1, 4, 8}
inline std::vector<int> parse_int_list(const std::string &s)
{
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty())
            out.push_back(std::stoi(item));
    return out;
}

// 默认要扫的线程数：1, 2, 4, 6, 8, ... 直到逻辑核数
inline std::vector<int> default_thread_list()
{
    std::vector<int> out = {1, 2, 4};
    for (int t = 6; t < hw_threads(); t += 2)
        out.push_back(t);
    if (out.back() != hw_threads())
        out.push_back(hw_threads());
    return out;
}

inline std::vector<float> random_floats(std::size_t n, unsigned seed, float lo = -1.0F,
                                        float hi = 1.0F)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto &x : v)
        x = dist(rng);
    return v;
}

inline const char *arg_or(int argc, char **argv, int i, const char *def)
{
    return argc > i ? argv[i] : def;
}

} // namespace bench
