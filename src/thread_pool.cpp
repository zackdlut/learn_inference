#include "inference/thread_pool.hpp"

#include <chrono>
#include <stdexcept>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace inference
{

namespace
{

// 当前线程是否正在执行线程池的任务，用来把嵌套的 parallel_for 降级成串行
thread_local bool t_inside_pool = false;

// 自旋时的「歇一下」：x86 的 pause 指令告诉 CPU 这是自旋等待，
// 降低功耗，也让同一物理核上的超线程兄弟拿到更多执行资源
inline void cpu_relax() noexcept
{
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

// 一批任务做完后，工作线程自旋这么久还没等到新任务才去睡。
// decode 时两次矩阵乘之间只隔几十微秒（layernorm、gelu 之类），2 ms 足够覆盖。
constexpr auto kSpinTime = std::chrono::microseconds(2000);

constexpr unsigned kIdxBits = 20;
constexpr unsigned kGenShift = 2 * kIdxBits;
constexpr std::uint64_t kIdxMask = (std::uint64_t{1} << kIdxBits) - 1;
constexpr std::uint64_t kGenMask = (std::uint64_t{1} << (64 - kGenShift)) - 1;

constexpr std::uint64_t gen_of(std::uint64_t s)
{
    return s >> kGenShift;
}
constexpr std::uint64_t total_of(std::uint64_t s)
{
    return (s >> kIdxBits) & kIdxMask;
}
constexpr std::uint64_t next_of(std::uint64_t s)
{
    return s & kIdxMask;
}
constexpr std::uint64_t make_state(std::uint64_t gen, std::uint64_t total)
{
    return (gen << kGenShift) | (total << kIdxBits);
}

} // namespace

ThreadPool::ThreadPool(int n_threads)
{
    for (int i = 1; i < n_threads; ++i)
        workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool()
{
    stop_.store(true, std::memory_order_release);
    // 先拿一下锁再通知：保证不会有工作线程「刚检查完条件、还没进入 wait」时错过这次通知
    {
        const std::lock_guard lk(mutex_);
    }
    cv_.notify_all();
    for (auto &t : workers_)
        t.join();
}

void ThreadPool::run_tasks(std::uint64_t gen)
{
    std::uint64_t s = state_.load(std::memory_order_acquire);
    for (;;)
    {
        if (gen_of(s) != gen || next_of(s) >= total_of(s))
            return; // 换批了，或者本批任务已经被领完
        // 领任务：编号 +1。失败时 s 被更新成最新值，重新判断
        if (!state_.compare_exchange_weak(s, s + 1, std::memory_order_acq_rel,
                                          std::memory_order_acquire))
            continue;
        // 领到了任务，本批就一定还没结束，job_ 仍然指向本批的函数
        (*job_)(static_cast<std::int64_t>(next_of(s)));
        // release：任务写的结果，对看到完成数的调用者可见
        done_.fetch_add(1, std::memory_order_release);
        s = state_.load(std::memory_order_acquire);
    }
}

std::uint64_t ThreadPool::wait_for_work(std::uint64_t seen_gen)
{
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned i = 0;; ++i)
    {
        const std::uint64_t g = gen_of(state_.load(std::memory_order_acquire));
        if (g != seen_gen || stop_.load(std::memory_order_acquire))
            return g;
        cpu_relax();
        // 读时钟比 pause 贵得多，每 1024 次才看一眼
        if ((i & 1023U) == 0 && std::chrono::steady_clock::now() - t0 > kSpinTime)
            break;
    }
    std::unique_lock lk(mutex_);
    cv_.wait(lk, [&]
             {
                 return gen_of(state_.load(std::memory_order_acquire)) != seen_gen ||
                        stop_.load(std::memory_order_acquire);
             });
    return gen_of(state_.load(std::memory_order_acquire));
}

void ThreadPool::worker_loop()
{
    t_inside_pool = true;
    std::uint64_t seen = 0;
    for (;;)
    {
        seen = wait_for_work(seen);
        if (stop_.load(std::memory_order_acquire))
            return;
        run_tasks(seen);
    }
}

void ThreadPool::parallel_for(std::int64_t n_tasks, const std::function<void(std::int64_t)> &fn)
{
    if (n_tasks <= 0)
        return;
    if (workers_.empty() || n_tasks == 1 || t_inside_pool)
    {
        for (std::int64_t i = 0; i < n_tasks; ++i)
            fn(i);
        return;
    }
    if (n_tasks > kMaxTasks)
        throw std::invalid_argument("ThreadPool::parallel_for: 一批最多 2^20-1 个任务");

    // 上一批的任务都已完成，没有任何线程还会读 job_，这里改它是安全的
    job_ = &fn;
    done_.store(0, std::memory_order_relaxed);
    gen_ = (gen_ + 1) & kGenMask;
    {
        // release：job_ 对 CAS 领到任务的工作线程可见。
        // 在锁里写，配合条件变量的谓词检查，保证睡着的线程不会错过这次唤醒
        const std::lock_guard lk(mutex_);
        state_.store(make_state(gen_, static_cast<std::uint64_t>(n_tasks)),
                     std::memory_order_release);
    }
    cv_.notify_all();

    t_inside_pool = true;
    run_tasks(gen_); // 调用者也干活，别闲着
    t_inside_pool = false;

    while (done_.load(std::memory_order_acquire) != n_tasks)
        cpu_relax();
}

} // namespace inference
