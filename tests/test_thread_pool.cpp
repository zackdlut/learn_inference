#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "inference/thread_pool.hpp"
#include "minitest.hpp"

using inference::ThreadPool;

TEST(thread_pool_runs_every_task_once)
{
    for (const int nt : {1, 2, 4, 14})
    {
        ThreadPool pool(nt);
        CHECK(pool.size() == nt);
        for (const std::int64_t n : {0, 1, 3, 13, 100, 1000})
        {
            std::vector<std::atomic<int>> hits(static_cast<std::size_t>(n));
            pool.parallel_for(n, [&](std::int64_t i) { hits[static_cast<std::size_t>(i)]++; });
            for (const auto &h : hits)
                CHECK(h.load() == 1);
        }
    }
}

TEST(thread_pool_many_rounds)
{
    // decode 一秒钟要发上千批任务：反复发小批次，检查不丢任务、不重复、不卡死
    ThreadPool pool(6);
    std::atomic<std::int64_t> sum{0};
    constexpr int kRounds = 5000;
    for (int r = 0; r < kRounds; ++r)
        pool.parallel_for(8, [&](std::int64_t i) { sum += i + 1; });
    CHECK(sum.load() == std::int64_t{kRounds} * 36);
    PRINT("{} 批任务全部正确完成", kRounds);
}

TEST(thread_pool_results_visible_after_return)
{
    // parallel_for 返回后，工作线程写的普通（非原子）内存必须对调用者可见
    ThreadPool pool(4);
    std::vector<int> out(4096, 0);
    for (int r = 1; r <= 50; ++r)
    {
        pool.parallel_for(64,
                          [&](std::int64_t b)
                          {
                              for (std::int64_t i = b * 64; i < (b + 1) * 64; ++i)
                                  out[static_cast<std::size_t>(i)] = r;
                          });
        for (int v : out)
            CHECK(v == r);
    }
}

TEST(thread_pool_nested_call_runs_serially)
{
    ThreadPool pool(4);
    std::atomic<int> inner{0};
    pool.parallel_for(8, [&](std::int64_t)
                      { pool.parallel_for(5, [&](std::int64_t) { inner++; }); });
    CHECK(inner.load() == 40);
}

TEST(thread_pool_sleeping_workers_wake_up)
{
    // 超过自旋时间后工作线程会睡到条件变量上，新任务来了必须能把它们叫醒
    ThreadPool pool(4);
    std::atomic<int> n{0};
    pool.parallel_for(4, [&](std::int64_t) { n++; });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pool.parallel_for(100, [&](std::int64_t) { n++; });
    CHECK(n.load() == 104);
}
