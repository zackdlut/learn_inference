#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace inference
{

// ---------------------------------------------------------------------------
// 常驻线程池
//
// 为什么不每次 new 线程：decode 一个 token 要做 49 次矩阵乘，GEMV 并行后
// 一次只要 0.1~1 ms，而创建 + join 十几个线程就要几百微秒，开销比干活还大。
// 线程池在启动时建好线程，之后每次只是「发任务 → 等完成」。
//
// 工作线程等任务的方式：先自旋一小段时间，等不到再睡到条件变量上。
//   - 纯睡眠：每次唤醒要走一次系统调用，延迟 10~50 µs，decode 里每秒上千次任务扛不住
//   - 纯自旋：没活的时候也把 CPU 跑满，笔记本发热降频
//   llama.cpp / OpenMP 都是这种「先自旋、后睡眠」的折中。
//
// 任务分发是动态的：谁有空谁领下一个。本机是 P 核 + E 核混合架构，
// 静态平均切分时所有人都要等最慢的 E 核；动态领取时快的核自然多干。
//
// 调用者只等「任务全部完成」，不等「每个线程都来报到」：
// 一个没被操作系统调度到的线程不会拖住整批任务。为此「批次号、任务总数、
// 下一个任务编号」打包在同一个 64 位原子变量里，用 CAS 领任务 ——
// CAS 成功就说明领到的确实是当前批次的任务，而这批任务没做完之前，
// 调用者不会改任务描述，这时再去读它才是安全的。
//
// 限制：同一时刻只能有一个线程调用 parallel_for；任务函数不能抛异常；
// 一批最多 kMaxTasks 个任务。在任务内部再调用 parallel_for 会退化成串行执行，不会死锁。
// ---------------------------------------------------------------------------
class ThreadPool
{
  public:
    static constexpr std::int64_t kMaxTasks = (std::int64_t{1} << 20) - 1;

    // n_threads 包含调用者自己，所以实际只新建 n_threads - 1 个工作线程
    explicit ThreadPool(int n_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    int size() const noexcept
    {
        return static_cast<int>(workers_.size()) + 1;
    }

    // 对 i = 0 .. n_tasks-1 各调用一次 fn(i)，调用者也参与，全部完成后才返回
    void parallel_for(std::int64_t n_tasks, const std::function<void(std::int64_t)> &fn);

  private:
    void worker_loop();
    std::uint64_t wait_for_work(std::uint64_t seen_gen);
    void run_tasks(std::uint64_t gen);

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};

    // 位布局：[63:40] 批次号（24 位）| [39:20] 任务总数 | [19:0] 下一个任务编号
    std::atomic<std::uint64_t> state_{0};
    std::uint64_t gen_ = 0; // 只有调用者读写
    const std::function<void(std::int64_t)> *job_ = nullptr;
    std::atomic<std::int64_t> done_{0}; // 本批已完成的任务数
};

} // namespace inference
