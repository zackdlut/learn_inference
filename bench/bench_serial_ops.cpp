// 第 4 步：矩阵乘之外的「零碎开销」各有多大。
//
//   ./build-release/bench/bench_serial_ops
//
// 1. gelu：标量 std::tanh（第 4 步之前的写法）vs kernel::gelu_f32（AVX2 exp 拼出来的 tanh）
// 2. KV-Cache 拷贝：第 4 步之前每层都把整段缓存拷进新分配的 ck / cv，
//    生成第 n 个 token 时每层拷 2 × n × 768 个 float —— 随长度线性增长
// 3. 小张量分配：每个算子都 Tensor::zeros 一个新张量，一个 token 约 160 次
//
// 都只用单线程测：这些算子在第 4 步时都跑在主线程上。

#include "bench_common.hpp"
#include "inference/kernel.hpp"
#include "inference/tensor.hpp"

#include <cmath>
#include <cstdint>

using namespace bench;
using inference::Tensor;
namespace kernel = inference::kernel;
using int64 = std::int64_t;

namespace
{

void gelu_scalar(std::span<const float> x, std::span<float> out)
{
    constexpr float k = 0.7978845608F; // sqrt(2/pi)
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const float v = x[i];
        out[i] = 0.5F * v * (1.0F + std::tanh(k * (v + 0.044715F * v * v * v)));
    }
}

} // namespace

int main()
{
    constexpr int L = 12, C = 768;
    kernel::set_num_threads(1);

    section("1. gelu：每 token 12 层 × 3072 个数");
    {
        const auto x = random_floats(L * 3072, 1, -6.0F, 6.0F);
        std::vector<float> a(x.size()), b(x.size());
        const double t_scalar = median_ms(200, [&] { gelu_scalar(x, a); });
        const double t_simd = median_ms(200, [&] { kernel::gelu_f32(x, b); });
        float err = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
            err = std::max(err, std::abs(a[i] - b[i]));
        println("  标量 std::tanh  {:>8.1f} µs/token", t_scalar * 1000);
        println("  AVX2            {:>8.1f} µs/token   （快 {:.1f} 倍，最大误差 {:.1e}）", t_simd * 1000,
                t_scalar / t_simd, err);
    }

    section("2. KV-Cache 拷贝：第 4 步之前每层把整段缓存拷出来");
    {
        constexpr int cap = 1024;
        // 12 层各一份 K/V（共 75 MB，超过 L3），和真实模型一样每层读的都是冷数据
        std::vector<std::vector<float>> kc(L), vc(L);
        for (int l = 0; l < L; ++l)
        {
            kc[l] = random_floats(cap * C, 10 + l);
            vc[l] = random_floats(cap * C, 40 + l);
        }
        println("  {:>8}{:>14}{:>16}", "已生成", "拷贝 ms/token", "每 token 拷贝量");
        for (int n : {64, 256, 512, 1024})
        {
            const double ms = median_ms(20,
                                        [&]
                                        {
                                            for (int l = 0; l < L; ++l)
                                            {
                                                Tensor ck = Tensor::zeros({n, C});
                                                Tensor cv = Tensor::zeros({n, C});
                                                std::copy_n(kc[l].begin(), n * C, ck.data().begin());
                                                std::copy_n(vc[l].begin(), n * C, cv.data().begin());
                                                do_not_optimize(ck.data().data());
                                                do_not_optimize(cv.data().data());
                                            }
                                        });
            println("  {:>8}{:>14.3f}{:>13.1f} MB", n, ms, 2.0 * L * n * C * 4 / 1e6);
        }
        println("  第 4 步之后：注意力直接按行距读缓存，这一项是 0");
    }

    section("3. 每个算子新分配一个张量：一个 token 约 160 次 Tensor::zeros");
    {
        // 每层：ln1、qkv、q/k/v 三个切片、注意力输出、投影、残差、ln2、fc、gelu、mlp 投影、残差 ≈ 13 个
        const std::vector<int64> sizes = {C, 3 * C, C, C, C, C, C, C, C, 4 * C, 4 * C, C, C};
        const double ms = median_ms(200,
                                    [&]
                                    {
                                        for (int l = 0; l < L; ++l)
                                            for (int64 s : sizes)
                                            {
                                                Tensor t = Tensor::zeros({1, s});
                                                do_not_optimize(t.data().data());
                                            }
                                    });
        println("  {} 次分配 + 清零: {:.1f} µs/token", L * sizes.size(), ms * 1000);
        println("  分配本身不贵；真正随长度变贵的是第 2 项的拷贝");
    }
}
