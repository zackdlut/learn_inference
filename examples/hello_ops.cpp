// 第 2 阶段的演示程序：算子长什么样，以及两个必须亲眼看一次的现象
//   (1) softmax 不减最大值会炸成 nan
//   (2) 只是换个循环顺序，矩阵乘法就能快好几倍
//
//   ./build/examples/02_ops

#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <random>
#include <vector>

#include "inference/kernel.hpp"
#include "inference/ops.hpp"
#include "inference/tensor.hpp"

using inference::Tensor;
namespace ops = inference::ops;

template <typename... Args> void println(std::format_string<Args...> fmt, Args &&...args) {
    std::cout << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

void section(const char *title) {
    std::cout << std::format("\n\033[36m===== {} =====\033[0m\n", title);
}

// 计时小工具：跑一次 fn，返回耗时（毫秒）
template <typename F> double time_ms(F &&fn) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// 故意写错的 softmax：直接照抄数学定义，不减最大值
std::vector<float> naive_softmax_unsafe(const std::vector<float> &x) {
    std::vector<float> out(x.size());
    float sum = 0.0F;
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = std::exp(x[i]);
        sum += out[i];
    }
    for (float &v : out)
        v /= sum;
    return out;
}

int main() {
    // -------------------------------------------------------------------------
    section("1. 逐元素算子");
    Tensor x = Tensor::from({2, 3}, {-2, -1, 0, 1, 2, 3});
    std::cout << "输入:\n" << x.to_string();
    std::cout << "relu(x):\n" << ops::relu(x).to_string();
    std::cout << "gelu(x)  —— 注意负数没有被一刀切成 0，而是平滑地趋近 0:\n"
              << ops::gelu(x).to_string();

    // -------------------------------------------------------------------------
    section("2. softmax 的数值稳定性：这是每个人都会踩一次的坑");

    const std::vector<float> small = {1.0F, 2.0F, 3.0F};
    const std::vector<float> large = {1000.0F, 1001.0F, 1002.0F};

    println("输入 {{1, 2, 3}}：");
    auto r1 = naive_softmax_unsafe(small);
    println("  照抄公式的版本  : {:.5f} {:.5f} {:.5f}   看起来没问题", r1[0], r1[1], r1[2]);

    println("");
    println("输入 {{1000, 1001, 1002}}（大模型的 logits 到这个量级很正常）：");
    auto r2 = naive_softmax_unsafe(large);
    println("  照抄公式的版本  : {:.5f} {:.5f} {:.5f}   \033[31m全是 nan！\033[0m", r2[0], r2[1],
            r2[2]);
    println("    原因: exp(1000) = {} 直接溢出，然后 inf / inf = nan", std::exp(1000.0F));

    Tensor big = Tensor::from({3}, {1000, 1001, 1002});
    Tensor ok = ops::softmax(big);
    println("  我们的稳定版本  : {:.5f} {:.5f} {:.5f}   \033[32m正确\033[0m", ok(0), ok(1), ok(2));
    println("    做法: 每个数先减去最大值。数学上完全等价（softmax 只看差值），");
    println("          但指数最大变成 exp(0)=1，永远不会溢出。");
    println("");
    println("  对照 softmax({{1,2,3}}) 的结果完全一样 —— 整体平移不改变 softmax。");

    // -------------------------------------------------------------------------
    section("3. 矩阵乘法：同样的数学，两种循环顺序");

    constexpr inference::int64 N = 512;
    println("测试规模: {0}×{0} × {0}×{0}，总共 {1:.2f} 亿次乘加", N, 2.0 * N * N * N / 1e8);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    Tensor a = Tensor::zeros({N, N});
    Tensor b = Tensor::zeros({N, N});
    for (float &v : a.data())
        v = dist(rng);
    for (float &v : b.data())
        v = dist(rng);
    Tensor c1 = Tensor::zeros({N, N});
    Tensor c2 = Tensor::zeros({N, N});

    const double t_ijk =
        time_ms([&] { inference::kernel::matmul_f32_ijk(a.data(), b.data(), c1.data(), N, N, N); });
    const double t_ikj =
        time_ms([&] { inference::kernel::matmul_f32_ikj(a.data(), b.data(), c2.data(), N, N, N); });

    const double flops = 2.0 * static_cast<double>(N) * N * N;
    println("");
    println("  i-j-k 顺序（教科书写法）: {:8.1f} ms   {:6.2f} GFLOPS", t_ijk,
            flops / (t_ijk * 1e6));
    println("  i-k-j 顺序（换了两行）  : {:8.1f} ms   {:6.2f} GFLOPS", t_ikj,
            flops / (t_ikj * 1e6));
    println("  \033[33m加速比: {:.2f}×\033[0m", t_ijk / t_ikj);
    println("  两者结果最大差异 {:.2e}（只是浮点累加顺序不同）",
            static_cast<double>(ops::max_abs_diff(c1, c2)));
    println("");
    println("  为什么差这么多？看最内层循环访问 B 的方式：");
    println("    i-j-k: b[k*N + j]  k 每加 1 就跳 {} 个 float（{} KB），每次都 cache miss", N,
            N * 4 / 1024);
    println("    i-k-j: b[k*N + j]  j 每加 1 只跳 1 个 float，顺序读，cache 全命中");
    println("  一次 cache line 是 64 字节 = 16 个 float。顺序读能用满，跳着读只用 1 个就扔。");
    println("");
    println("  这还只是第一层优化。阶段 5 会用分块 + AVX2 + 多线程再提 10 倍以上。");

    // -------------------------------------------------------------------------
    section("4. 一个迷你「模型」：把算子串起来");
    println("结构: 输入(4) -> Linear(4→3) -> ReLU -> Linear(3→2) -> Softmax");

    Tensor in = Tensor::from({1, 4}, {0.5F, -1.0F, 2.0F, 0.1F});
    Tensor w1 = Tensor::from(
        {3, 4}, {0.2F, -0.3F, 0.5F, 0.1F, -0.4F, 0.6F, -0.2F, 0.3F, 0.1F, 0.1F, 0.4F, -0.5F});
    Tensor b1 = Tensor::from({3}, {0.1F, -0.1F, 0.0F});
    Tensor w2 = Tensor::from({2, 3}, {1.0F, -1.0F, 0.5F, -0.5F, 1.0F, 0.2F});
    Tensor b2 = Tensor::from({2}, {0.0F, 0.1F});

    Tensor h = ops::linear(in, w1, b1);
    println("Linear1 输出 : [{:.4f}, {:.4f}, {:.4f}]", h(0, 0), h(0, 1), h(0, 2));
    h = ops::relu(h);
    println("ReLU 之后    : [{:.4f}, {:.4f}, {:.4f}]", h(0, 0), h(0, 1), h(0, 2));
    Tensor logits = ops::linear(h, w2, b2);
    println("Linear2 输出 : [{:.4f}, {:.4f}]  (这叫 logits，还不是概率)", logits(0, 0),
            logits(0, 1));
    Tensor prob = ops::softmax(logits);
    println("Softmax 之后 : [{:.4f}, {:.4f}]  (加起来 = {:.4f})", prob(0, 0), prob(0, 1),
            prob(0, 0) + prob(0, 1));
    println("预测类别     : {}", ops::argmax(prob));
    println("");
    println("阶段 3 要做的就是把这些权重换成真正训练出来的，输入换成一张手写数字图片。");

    std::cout << '\n';
    return 0;
}
