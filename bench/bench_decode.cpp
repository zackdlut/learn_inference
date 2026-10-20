// 第 0 / 2 / 3 / 7 步：decode 形状（M = 1）的矩阵乘，用各步的 kernel 跑一遍「一个 token」。
//
//   ./build-release/bench/bench_decode                   # 默认线程数 = 一半逻辑核，每种 20 次
//   ./build-release/bench/bench_decode 7 20 gemv1,gemvN  # 线程数、次数、只跑某几种
//
// 每种模式都按真实 decode 的顺序把 12 层 × 4 个矩阵 + lm_head 各算一次，算「一个 token」。
// 权重是 12 层各自独立的一份（fp32 共 494 MB，远超 L3），所以和真实模型一样每次都从内存读。
// 只读少数几个矩阵反复测会让它们留在缓存里，测出来的带宽偏高。
//
// 模式（对应文档里的步骤）：
//   gemm   第 0 步的路径：层内走 GotoBLAS 的 matmul_f32（M=1 也打包权重），
//          lm_head 走标量三重循环（matmul_nt_f32_naive，当时不足 4 行的部分就是这样算的）
//   gemv1  第 2 步：gemv_nt_f32，单线程
//   gemvN  第 3 步：matmul_nt_f32，多线程按输出列切
//   q8     第 7 步采用的方案：层内 Q8_0，lm_head fp16
//   q8all  第 7 步第一版：lm_head 也用 Q8_0
//   f16    全部 fp16（参考：带宽减半的方案）
//
// 权重的数值无所谓（只测速度），所以 gemm 模式直接把 [N, K] 的缓冲区当 [K, N] 用。

#include "bench_common.hpp"
#include "inference/kernel.hpp"

#include <array>
#include <cstdint>
#include <map>

using namespace bench;
namespace kernel = inference::kernel;
using int64 = std::int64_t;

namespace
{

struct MatShape
{
    const char *name;
    int64 N, K; // W 是 [N, K]（out, in）
};

constexpr std::array<MatShape, 4> kLayerShapes = {{
    {"qkv (c_attn)", 2304, 768},
    {"attn_proj", 768, 768},
    {"mlp_fc", 3072, 768},
    {"mlp_proj", 768, 3072},
}};
constexpr MatShape kLmHead = {"lm_head", 50257, 768};
constexpr int kLayers = 12;

// 一个矩阵的各种存储格式；按需填充
struct Mat
{
    MatShape shape{};
    std::vector<float> f32;
    std::vector<std::int8_t> q;
    std::vector<float> qs;
    std::vector<std::uint16_t> h;

    void make_q8()
    {
        const int64 n = shape.N * shape.K, nb = shape.K / kernel::kQ8Block;
        q.resize(n);
        qs.resize(shape.N * nb);
        for (int64 r = 0; r < shape.N; ++r)
            kernel::quantize_row_q8_0(f32.data() + r * shape.K, q.data() + r * shape.K,
                                      qs.data() + r * nb, shape.K);
    }
    void make_f16()
    {
        h.resize(f32.size());
        kernel::convert_f32_to_f16(f32.data(), h.data(), static_cast<int64>(f32.size()));
    }
};

enum class Fmt
{
    gemm,
    naive,
    gemv1,
    gemvN,
    q8,
    f16
};

double bytes_per_weight(Fmt f)
{
    switch (f)
    {
    case Fmt::q8:
        return 1.125;
    case Fmt::f16:
        return 2.0;
    default:
        return 4.0;
    }
}

void run(Fmt f, const Mat &m, std::span<const float> x, std::span<float> y)
{
    const int64 N = m.shape.N, K = m.shape.K;
    const auto xs = x.first(static_cast<std::size_t>(K));
    const auto ys = y.first(static_cast<std::size_t>(N));
    switch (f)
    {
    case Fmt::gemm: // b 当作 [K, N]
        kernel::matmul_f32(xs, m.f32, ys, 1, N, K);
        break;
    case Fmt::naive:
        kernel::matmul_nt_f32_naive(xs, m.f32, ys, 1, N, K);
        break;
    case Fmt::gemv1:
        kernel::gemv_nt_f32(xs, m.f32, ys, N, K);
        break;
    case Fmt::gemvN:
        kernel::matmul_nt_f32(xs, m.f32, ys, 1, N, K);
        break;
    case Fmt::q8:
        kernel::matmul_nt_q8_0(xs, m.q.data(), m.qs.data(), ys, 1, N, K);
        break;
    case Fmt::f16:
        kernel::matmul_nt_f16(xs, m.h.data(), ys, 1, N, K);
        break;
    }
}

struct Mode
{
    std::string key;
    std::string title;
    Fmt layer, lm;
    int threads; // 0 表示用命令行给的线程数
    bool slow;   // 太慢的只跑 3 次
};

} // namespace

int main(int argc, char **argv)
{
    const int nt = std::stoi(arg_or(argc, argv, 1, std::to_string(default_threads()).c_str()));
    const int iters = std::stoi(arg_or(argc, argv, 2, "20"));
    const std::string only = arg_or(argc, argv, 3, "gemm,gemv1,gemvN,q8,q8all,f16");

    const std::vector<Mode> modes = {
        {"gemm", "第 0 步：GEMM 路径（打包）+ lm_head 标量", Fmt::gemm, Fmt::naive, 1, true},
        {"gemv1", "第 2 步：GEMV，单线程", Fmt::gemv1, Fmt::gemv1, 1, false},
        {"gemvN", "第 3 步：GEMV，多线程按列切", Fmt::gemvN, Fmt::gemvN, 0, false},
        {"q8", "第 7 步：层内 Q8_0 + lm_head fp16（采用）", Fmt::q8, Fmt::f16, 0, false},
        {"q8all", "第 7 步：全部 Q8_0（第一版）", Fmt::q8, Fmt::q8, 0, false},
        {"f16", "参考：全部 fp16", Fmt::f16, Fmt::f16, 0, false},
    };
    auto wanted = [&](const std::string &k) { return ("," + only + ",").find("," + k + ",") != std::string::npos; };
    bool need_q8 = false, need_f16 = false, need_lm_q8 = false;
    for (const auto &m : modes)
        if (wanted(m.key))
        {
            need_q8 |= m.layer == Fmt::q8;
            need_lm_q8 |= m.lm == Fmt::q8;
            need_f16 |= m.layer == Fmt::f16 || m.lm == Fmt::f16;
        }

    println("准备权重：12 层 × 4 个矩阵 + lm_head，fp32 共 494 MB ...");
    std::vector<std::array<Mat, 4>> layers(kLayers);
    unsigned seed = 1;
    for (auto &L : layers)
        for (std::size_t i = 0; i < 4; ++i)
        {
            L[i].shape = kLayerShapes[i];
            L[i].f32 = random_floats(kLayerShapes[i].N * kLayerShapes[i].K, seed++, -0.1F, 0.1F);
            if (need_q8)
                L[i].make_q8();
            if (need_f16)
                L[i].make_f16();
        }
    Mat lm;
    lm.shape = kLmHead;
    lm.f32 = random_floats(kLmHead.N * kLmHead.K, seed++, -0.1F, 0.1F);
    if (need_lm_q8)
        lm.make_q8();
    if (need_f16)
        lm.make_f16();

    const auto x = random_floats(3072, 7);
    std::vector<float> y(kLmHead.N);

    for (const auto &mode : modes)
    {
        if (!wanted(mode.key))
            continue;
        const int threads = mode.threads ? mode.threads : nt;
        kernel::set_num_threads(threads);
        const int n_iter = mode.slow ? std::min(iters, 3) : iters;

        // 每个类别一列耗时，每次迭代记一个数，最后取中位数
        std::map<std::string, std::vector<double>> per_cat;
        for (int it = -1; it < n_iter; ++it) // it = -1 是预热
        {
            std::map<std::string, double> t;
            for (const auto &L : layers)
                for (const auto &m : L)
                    t[m.shape.name] += time_ms([&] { run(mode.layer, m, x, y); });
            t[kLmHead.name] += time_ms([&] { run(mode.lm, lm, x, y); });
            if (it >= 0)
                for (const auto &[k, v] : t)
                    per_cat[k].push_back(v);
        }

        section(std::format("{}（{} 线程，{} 次取中位数）", mode.title, threads, n_iter));
        println("  {:<16}{:>10}{:>12}", "矩阵", "ms/token", "有效带宽");
        double total = 0;
        auto print_row = [&](const MatShape &s, Fmt f, double count)
        {
            const double ms = median(per_cat[s.name]);
            total += ms;
            const double bytes = count * static_cast<double>(s.N * s.K) * bytes_per_weight(f);
            println("  {:<16}{:>10.3f}{:>9.1f} GB/s", s.name, ms, bytes / (ms * 1e6));
        };
        for (const auto &s : kLayerShapes)
            print_row(s, mode.layer, kLayers);
        print_row(kLmHead, mode.lm, 1);
        println("  {:<16}{:>10.3f}   （只算矩阵乘 ≈ {:.0f} token/s）", "合计", total, 1000.0 / total);
    }
}
