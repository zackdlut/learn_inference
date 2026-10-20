// 第 2 步：加载时把 Conv1D 权重从 [in, out] 转成 [out, in]，三种转置写法比速度。
//
//   ./build-release/bench/bench_transpose
//
// 12 层 × 4 个矩阵共 8500 万个 float（340 MB）。三种写法：
//   naive    按行读、按列写：读是连续的，但每写一个数都跳一整行（3–12 KB），
//            每条 64 字节的 cache line 只用上 4 字节，剩下的在被再次用到之前就被挤出缓存了
//   blocked  32×32 分块：一块读进来 4 KB、写出去 4 KB，都在 L1 里完成（src/gpt.cpp 的 to_out_in）
//   tensor   Tensor::transpose(0, 1).contiguous()：通用的多维下标循环，每个元素都要算一遍多维下标

#include "bench_common.hpp"
#include "inference/tensor.hpp"

#include <cstdint>

using namespace bench;
using inference::Tensor;
using int64 = std::int64_t;

namespace
{

void transpose_naive(const float *src, float *dst, int64 rows, int64 cols)
{
    for (int64 r = 0; r < rows; ++r)
        for (int64 c = 0; c < cols; ++c)
            dst[c * rows + r] = src[r * cols + c];
}

void transpose_blocked(const float *src, float *dst, int64 rows, int64 cols)
{
    constexpr int64 B = 32;
    for (int64 r0 = 0; r0 < rows; r0 += B)
        for (int64 c0 = 0; c0 < cols; c0 += B)
        {
            const int64 r1 = std::min(r0 + B, rows), c1 = std::min(c0 + B, cols);
            for (int64 r = r0; r < r1; ++r)
                for (int64 c = c0; c < c1; ++c)
                    dst[c * rows + r] = src[r * cols + c];
        }
}

struct Shape
{
    int64 rows, cols; // 文件里的 Conv1D 布局 [in, out]
};

} // namespace

int main()
{
    const Shape shapes[] = {{768, 2304}, {768, 768}, {768, 3072}, {3072, 768}};
    std::vector<Tensor> src;
    unsigned seed = 1;
    for (int l = 0; l < 12; ++l)
        for (const auto &s : shapes)
        {
            Tensor t = Tensor::zeros({s.rows, s.cols});
            const auto v = random_floats(static_cast<std::size_t>(s.rows * s.cols), seed++);
            std::copy(v.begin(), v.end(), t.data().begin());
            src.push_back(std::move(t));
        }
    std::vector<std::vector<float>> dst;
    for (const auto &t : src)
        dst.emplace_back(t.numel());

    auto run_all = [&](auto &&fn)
    {
        for (std::size_t i = 0; i < src.size(); ++i)
            fn(src[i], dst[i]);
    };

    println("转置 12 层 × 4 个矩阵（共 {:.0f} MB），各跑 3 次取中位数：\n", 85e6 * 4 / 1e6);
    const double t_naive = median_ms(3, [&]
                                     { run_all([](const Tensor &s, std::vector<float> &d)
                                               { transpose_naive(s.data().data(), d.data(), s.dim(0), s.dim(1)); }); });
    const auto ref = dst; // 拿朴素版的结果当参考
    const double t_blocked = median_ms(3, [&]
                                       { run_all([](const Tensor &s, std::vector<float> &d)
                                                 { transpose_blocked(s.data().data(), d.data(), s.dim(0), s.dim(1)); }); });
    const bool same = dst == ref;
    const double t_tensor = median_ms(3, [&]
                                      { run_all([](const Tensor &s, std::vector<float> &d)
                                                {
                                                    const Tensor t = s.transpose(0, 1).contiguous();
                                                    std::copy(t.data().begin(), t.data().end(), d.begin());
                                                }); });

    println("  {:<34}{:>10.0f} ms", "naive   按行读、按列写", t_naive);
    println("  {:<34}{:>10.0f} ms   （结果和 naive {}）", "blocked 32×32 分块", t_blocked,
            same ? "一致" : "不一致！");
    println("  {:<34}{:>10.0f} ms", "tensor  transpose().contiguous()", t_tensor);
    println("\n加载时间里多出来的就是这一项：分块比朴素版快 {:.1f} 倍。", t_naive / t_blocked);
}
