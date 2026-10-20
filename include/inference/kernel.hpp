#pragma once

#include <cstdint>
#include <functional>
#include <span>

namespace inference::kernel
{

using int64 = std::int64_t;

// --- 逐元素（elementwise）-------------------------------------------------
void add_f32(std::span<const float> a, std::span<const float> b, std::span<float> out);
void mul_f32(std::span<const float> a, std::span<const float> b, std::span<float> out);
void add_scalar_f32(std::span<const float> a, float s, std::span<float> out);
void relu_f32(std::span<const float> x, std::span<float> out);
void gelu_f32(std::span<const float> x, std::span<float> out); // tanh 近似版，GPT-2 用的就是这个
void gelu_exact_f32(std::span<const float> x,
                    std::span<float> out); // erf 精确版，用来验证近似版误差

// --- 归约类（reduction）：都只处理「一行」，多行由上层循环 -------------------
// softmax：把一行数字变成和为 1 的概率分布
void softmax_f32(std::span<const float> x, std::span<float> out);
// layernorm：对一行做标准化后再缩放平移。w/b 长度必须等于 x
void layernorm_f32(std::span<const float> x, std::span<const float> w, std::span<const float> b,
                   std::span<float> out, float eps);
// rmsnorm：LLaMA 系模型用的简化版归一化，阶段 6 会用到
void rmsnorm_f32(std::span<const float> x, std::span<const float> w, std::span<float> out,
                 float eps);

// --- 矩阵乘法 -------------------------------------------------------------
// C[M,N] = A[M,K] × B[K,N]，全部行优先连续存储。
// 提供两个循环顺序不同、结果完全相同的版本。
// 它们的性能差距是阶段 5 的第一课 —— 跑 examples/02_ops 亲眼看看差多少。
void matmul_f32_ijk(std::span<const float> a, std::span<const float> b, std::span<float> c, int64 M,
                    int64 N, int64 K);
void matmul_f32_ikj(std::span<const float> a, std::span<const float> b, std::span<float> c, int64 M,
                    int64 N, int64 K);
// 默认实现，目前指向较快的那个。阶段 5 会把它换成优化版。
void matmul_f32(std::span<const float> a, std::span<const float> b, std::span<float> c, int64 M,
                int64 N, int64 K);
// C[M,N] = A[M,K] × Wᵀ，其中 W 的形状是 [N,K]。
// 为什么要单独写一个转置版？因为 PyTorch 的 Linear 层权重就存成 [out, in]，
// 而这个布局对 cache 反而更友好：A 和 W 都是按行连续读取的。
void matmul_nt_f32(std::span<const float> a, std::span<const float> w, std::span<float> c, int64 M,
                   int64 N, int64 K);

// y[N] = W[N,K] · x[K]，即 M=1 的 matmul_nt。decode 每生成一个 token 都走这里，
// 它是访存密集的：目标是把 W 以内存带宽的速度读完，而不是算得快。
void gemv_nt_f32(std::span<const float> x, std::span<const float> w, std::span<float> y, int64 N,
                 int64 K);

void matmul_nt_f32_naive(std::span<const float> a, std::span<const float> w, std::span<float> c,
                         int64 M, int64 N, int64 K);
void matmul_nt_f32_tiled(std::span<const float> a, std::span<const float> w, std::span<float> c,
                         int64 M, int64 N, int64 K);
void matmul_nt_f32_avx2(std::span<const float> a, std::span<const float> w, std::span<float> c,
                        int64 M, int64 N, int64 K);
void matmul_nt_f32_parallel(std::span<const float> a, std::span<const float> w, std::span<float> c,
                            int64 M, int64 N, int64 K);

// v3: 循环分块。让参与运算的数据块小到能待在 L1 cache 里
void matmul_f32_tiled(std::span<const float> a, std::span<const float> b, std::span<float> c,
                      int64 M, int64 N, int64 K);

// v4: AVX2 + FMA 微内核。一条指令算 8 个 float，累加器常驻寄存器
void matmul_f32_avx2(std::span<const float> a, std::span<const float> b, std::span<float> c,
                     int64 M, int64 N, int64 K);

// v5: 多线程。按行切给不同核
void matmul_f32_parallel(std::span<const float> a, std::span<const float> b, std::span<float> c,
                         int64 M, int64 N, int64 K);

// --- 阶段 6：Transformer 需要的算子 ------------------------------------------
// 多头因果注意力。这一个函数同时覆盖两种场景，差别只在参数上：
//
//   prefill（一次处理整个 prompt）：Tq = T, Tk = T, q_offset = 0
//   decode （每次处理一个新词）  ：Tq = 1, Tk = 已缓存长度, q_offset = Tk-1
//
// 布局都是 [T, n_head * head_size]，同一个位置的所有头挨着放。
//
//   q 的第 i 行 ─┐
//                ├─→ 和 k 的第 0..(q_offset+i) 行做点积 → 分数
//   k 的各行  ───┘         │ softmax
//                          ▼
//   v 的各行  ──────→ 按分数加权求和 → out 的第 i 行
//
// causal（因果）的意思是第 i 个位置只能看到自己和前面的位置。
// 没有这个限制，模型预测下一个词时就能偷看答案了。

void attention_f32(std::span<const float> q, std::span<const float> k, std::span<const float> v,
                   std::span<float> out, int64 Tq, int64 Tk, int64 n_head, int64 head_size,
                   int64 q_offset);

// 朴素版：三重标量循环、单线程。只用作参考实现（测试对拍、benchmark 对比）
void attention_f32_naive(std::span<const float> q, std::span<const float> k,
                         std::span<const float> v, std::span<float> out, int64 Tq, int64 Tk,
                         int64 n_head, int64 head_size, int64 q_offset);

// 按「头」切分的矩阵视图：第 h 个头、第 t 行的 head_size 个数从
//   data + h * head_stride + t * row_stride
// 开始。用两个跨度就能描述注意力里出现的所有布局，都不用先拷成连续的 [T, C]：
//   c_attn 输出 [T, 3C] 里的 q 段     row_stride = 3C, head_stride = head_size
//   连续的 [T, C]                      row_stride = C,  head_stride = head_size
//   KV-Cache [头][位置][head_size]     row_stride = head_size, head_stride = 容量 × head_size
struct HeadView
{
    const float *data;
    int64 row_stride;
    int64 head_stride;
};

// 同 attention_f32，但 q/k/v 用 HeadView 描述；out 是 [Tq, C] 布局，行距 ldo
void attention_f32_strided(HeadView q, HeadView k, HeadView v, float *out, int64 ldo, int64 Tq,
                           int64 Tk, int64 n_head, int64 head_size, int64 q_offset);

// 注意力的并行阈值：n_head × Tq × Tk × head_size 小于它时串行。返回旧值。
// 只给实验用（bench/bench_kvlayout.cpp 扫交叉点）：传 0 = 总是并行，传负数 = 恢复默认 30 万
int64 set_attention_parallel_min_work(int64 work) noexcept;

// --- 量化：Q8_0 ---------------------------------------------------------------
// 每 32 个数一组，存 32 个 int8 和 1 个 float scale：x ≈ q × scale。
// 一个权重平均 1 + 4/32 = 1.125 字节，是 fp32 的 28%。decode 是访存密集的，
// 少读多少字节就快多少。
constexpr int64 kQ8Block = 32;

// 量化 / 反量化一行，n 必须是 32 的倍数。scale 长度为 n / 32
void quantize_row_q8_0(const float *x, std::int8_t *q, float *scale, int64 n);
void dequantize_row_q8_0(const std::int8_t *q, const float *scale, float *x, int64 n);

// C[M,N] = A[M,K] × Wᵀ，W 是按行量化的 Q8_0 [N,K]（wq: N×K 个 int8，ws: N×K/32 个 scale）。
// A 每行也会先量化成 Q8_0，点积全程用 8 位整数 SIMD。K 必须是 32 的倍数。
void matmul_nt_q8_0(std::span<const float> a, const std::int8_t *wq, const float *ws,
                    std::span<float> c, int64 M, int64 N, int64 K);

// --- 半精度权重：fp16 ----------------------------------------------------------
// 每个权重 2 字节（IEEE half 的位模式存在 uint16 里），激活保持 fp32 不量化。
// 用在对激活离群值敏感、不适合 Q8_0 的矩阵上（GPT-2 的 lm_head）。
void convert_f32_to_f16(const float *x, std::uint16_t *y, int64 n);
void convert_f16_to_f32(const std::uint16_t *x, float *y, int64 n);

// C[M,N] = A[M,K] × Wᵀ，W 是 fp16 [N,K]，A、C 是 fp32
void matmul_nt_f16(std::span<const float> a, const std::uint16_t *w, std::span<float> c, int64 M,
                   int64 N, int64 K);

// 运行期查询本次构建是否带 AVX2（-march=native 关掉时会回退到标量版）
bool has_avx2_support() noexcept;
// 多线程用几个线程。默认取硬件并发数，设成 1 就等于串行。
void set_num_threads(int n) noexcept;
int get_num_threads() noexcept;

// 用全局线程池对 i = 0..n_tasks-1 各执行一次 fn(i)，全部完成后返回。
// 任务之间必须互不干扰（写不同的内存），任务函数不能抛异常。
void parallel_for(int64 n_tasks, const std::function<void(int64)> &fn);

} // namespace inference::kernel