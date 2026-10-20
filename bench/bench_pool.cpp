// 第 3 步：线程池的两种「等完成」方式对比。
//
//   ./build-release/bench/bench_pool                 # 扫 1,2,4,6,... 线程
//   ./build-release/bench/bench_pool 1,7,14          # 只测这几个线程数
//
// v1（BarrierPool，本文件里实现）：第一版的写法。调用者发完任务后，要等「每个工作线程」
//     都来报到说「这批我看过了」才返回。好处是实现简单、一定安全：没有任何线程会带着
//     上一批的状态闯进下一批。坏处是：只要有一个线程被操作系统换下去一个时间片，
//     哪怕任务早被别人做完了，调用者也得等它回来。
// v2（inference::ThreadPool，库里的实现）：调用者只等「完成的任务数」到齐。
//
// 两者的自旋 / 睡眠策略完全一样（自旋 2 ms 再睡到条件变量上），唯一的区别就是等什么。
//
// 测两个数：
//   1. 每次 parallel_for 的固定开销：任务几乎不干活，看中位数和 p99
//   2. 模拟 decode：一个 token 49 次 GEMV（12 层 × 4 + lm_head），权重共 494 MB
//
// 线程数接近或等于逻辑核数时，v1 的 p99 会明显变差（被抢占的线程拖住整批）。
// 能不能复现、差多少，取决于当时机器上有多少别的活（WSL2 下宿主机的进程也算）。

#include "bench_common.hpp"
#include "inference/kernel.hpp"
#include "inference/thread_pool.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <immintrin.h>
#include <mutex>

using namespace bench;
using int64 = std::int64_t;

namespace
{

inline void cpu_relax()
{
    _mm_pause();
}

// 第一版线程池：调用者等每个工作线程都报到
class BarrierPool
{
  public:
    explicit BarrierPool(int n_threads)
    {
        for (int i = 1; i < n_threads; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }
    ~BarrierPool()
    {
        stop_.store(true);
        {
            const std::lock_guard lk(mutex_);
        }
        cv_.notify_all();
        for (auto &t : workers_)
            t.join();
    }

    void parallel_for(int64 n, const std::function<void(int64)> &fn)
    {
        if (workers_.empty())
        {
            for (int64 i = 0; i < n; ++i)
                fn(i);
            return;
        }
        // 上一批所有工作线程都已报到，没人还在读这些字段
        job_ = &fn;
        total_ = n;
        next_.store(0, std::memory_order_relaxed);
        arrived_.store(0, std::memory_order_relaxed);
        {
            const std::lock_guard lk(mutex_);
            gen_.store(gen_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
        }
        cv_.notify_all();
        run();
        // ↓ 就是这一行：等「每个线程」，而不是等「每个任务」
        while (arrived_.load(std::memory_order_acquire) != static_cast<int>(workers_.size()))
            cpu_relax();
    }

  private:
    void run()
    {
        for (int64 i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < total_;)
            (*job_)(i);
    }
    void worker_loop()
    {
        std::uint64_t seen = 0;
        for (;;)
        {
            const auto t0 = Clock::now();
            bool woke = false;
            for (unsigned i = 0;; ++i)
            {
                if (gen_.load(std::memory_order_acquire) != seen || stop_.load())
                {
                    woke = true;
                    break;
                }
                cpu_relax();
                if ((i & 1023U) == 0 && ms_since(t0) > 2.0)
                    break;
            }
            if (!woke)
            {
                std::unique_lock lk(mutex_);
                cv_.wait(lk, [&] { return gen_.load(std::memory_order_acquire) != seen || stop_.load(); });
            }
            if (stop_.load())
                return;
            seen = gen_.load(std::memory_order_acquire);
            run();
            arrived_.fetch_add(1, std::memory_order_release);
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    std::atomic<std::uint64_t> gen_{0};
    const std::function<void(int64)> *job_ = nullptr;
    int64 total_ = 0;
    std::atomic<int64> next_{0};
    std::atomic<int> arrived_{0};
};

struct Stats
{
    double median_us, p99_us;
};

// 每次 parallel_for 的固定开销：4 × 线程数个几乎不干活的任务
template <class Pool> Stats call_overhead(Pool &pool, int nt)
{
    const int64 n_tasks = 4 * int64{nt};
    std::vector<std::int64_t> sink(static_cast<std::size_t>(n_tasks) * 16);
    const std::function<void(int64)> fn = [&](int64 i) { sink[i * 16] += i; };
    std::vector<double> t;
    const auto start = Clock::now();
    for (int i = 0; i < 3000 && ms_since(start) < 1500.0; ++i)
        t.push_back(time_ms([&] { pool.parallel_for(n_tasks, fn); }) * 1000.0);
    do_not_optimize(sink.data());
    std::sort(t.begin(), t.end());
    return {median(t), t[static_cast<std::size_t>(0.99 * static_cast<double>(t.size() - 1))]};
}

struct Mat
{
    int64 N, K;
    std::vector<float> w;
};

// 一个 token 的 49 次 GEMV，切块方式和库里的 matmul_nt_f32_parallel 一样
template <class Pool>
double decode_token_ms(Pool &pool, int nt, const std::vector<Mat> &mats, std::span<const float> x,
                       std::span<float> y)
{
    return time_ms(
        [&]
        {
            for (const Mat &m : mats)
            {
                const int64 blk = std::max<int64>(16, (m.N / (int64{nt} * 4) + 3) / 4 * 4);
                const int64 n_blocks = (m.N + blk - 1) / blk;
                pool.parallel_for(n_blocks,
                                  [&](int64 b)
                                  {
                                      const int64 n0 = b * blk, rows = std::min(blk, m.N - n0);
                                      inference::kernel::gemv_nt_f32(
                                          x.first(m.K),
                                          std::span<const float>(m.w).subspan(n0 * m.K, rows * m.K),
                                          y.subspan(n0, rows), rows, m.K);
                                  });
            }
        });
}

} // namespace

int main(int argc, char **argv)
{
    const auto threads = argc > 1 ? parse_int_list(argv[1]) : default_thread_list();

    println("准备 decode 权重（12 层 × 4 + lm_head，494 MB）...");
    std::vector<Mat> mats;
    unsigned seed = 1;
    const std::array<std::pair<int64, int64>, 4> shapes = {{{2304, 768}, {768, 768}, {3072, 768}, {768, 3072}}};
    for (int l = 0; l < 12; ++l)
        for (auto [N, K] : shapes)
            mats.push_back({N, K, random_floats(N * K, seed++, -0.1F, 0.1F)});
    mats.push_back({50257, 768, random_floats(50257 * 768, seed++, -0.1F, 0.1F)});
    const auto x = random_floats(3072, 7);
    std::vector<float> y(50257);

    println("\n每个线程数：两个池子各建一份，交替测 3 轮，取中位数。");
    println("开销 = 一次 parallel_for（4×线程数 个空任务）的耗时；decode = 49 次 GEMV 一个 token\n");
    println("{:>6} | {:>12}{:>12} | {:>12}{:>12} | {:>12}{:>12}", "线程", "v1 中位 µs", "v2 中位 µs",
            "v1 p99 µs", "v2 p99 µs", "v1 ms/tok", "v2 ms/tok");
    for (int nt : threads)
    {
        BarrierPool v1(nt);
        inference::ThreadPool v2(nt);
        std::vector<double> m1, m2, p1, p2, d1, d2;
        for (int round = 0; round < 3; ++round)
        {
            const Stats a = call_overhead(v1, nt);
            const Stats b = call_overhead(v2, nt);
            m1.push_back(a.median_us), p1.push_back(a.p99_us);
            m2.push_back(b.median_us), p2.push_back(b.p99_us);
            d1.push_back(decode_token_ms(v1, nt, mats, x, y));
            d2.push_back(decode_token_ms(v2, nt, mats, x, y));
        }
        println("{:>6} | {:>12.1f}{:>12.1f} | {:>12.1f}{:>12.1f} | {:>12.2f}{:>12.2f}", nt, median(m1),
                median(m2), median(p1), median(p2), median(d1), median(d2));
    }
    println("\n看点：");
    println("  - 线程数接近逻辑核数时，v1 的 p99 变大：被换下去的线程拖住整批");
    println("  - decode 的 ms/token 在 6–8 线程后不再下降：内存带宽饱和（对照 bench_bw）");
}
