# GPT-2 推理加速笔记：从 1.3 token/s 到逼近内存带宽上限

> 对应代码：[`src/gpt.cpp`](../src/gpt.cpp)、[`src/ops.cpp`](../src/ops.cpp)、
> [`src/kernel_matmul.cpp`](../src/kernel_matmul.cpp)、[`src/kernel_attention.cpp`](../src/kernel_attention.cpp)
>
> 前置阅读：[矩阵乘法优化笔记](matmul_optimization.md)（GotoBLAS 打包、AVX2 微内核、多线程）
>
> 机器：Intel Core Ultra 5 135U（2P + 8E + 2LP-E，14 线程，15 W 笔记本），WSL2

---

## 0. 一页看懂

起点（`build/`，Debug 构建）：

```
prefill 5 个 token: 1376 ms  （3.6 token/s）
decode 60 个 token: 45060 ms （1.3 token/s）
```

优化路线（每一步都**先测量、再改、再测量**）：

| 步骤 | 做什么 | 解决什么瓶颈 | decode 每 token 预期 |
|------|--------|--------------|----------------------|
| 0 | 搭测量工具：分算子计时 | 不知道时间花在哪 | — |
| 1 | 用 Release 构建 | `-O0` + ASan 让每次访存慢几倍到几十倍 | 750 ms → **131 ms**（已实测） |
| 2 | 给 M=1 写专用 GEMV 内核 | decode 走 GEMM 路径：整块打包权重、lm_head 掉进标量循环 | → ~60 ms |
| 3 | GEMV 按输出列多线程 + 线程池 | decode 时只有 1 个线程在干活；每次矩阵乘都新建线程 | → ~15–20 ms |
| 4 | 去掉多余拷贝和内存分配 | KV-Cache 每层整段拷贝、每个算子都 `zeros` 新张量 | → ~14 ms，且不随序列长度变慢 |
| 5 | 注意力内核 SIMD + 多头并行 | 长上下文时注意力变成大头 | 长序列下收益明显 |
| 6 | 小 M 的 prefill 走「按列切」 | M=5 时 5 个线程各打包一份完整 B | prefill 219 ms → ~20 ms |
| 7 | 权重量化 int8 / int4 | 已经打满带宽，只能**少读字节** | → ~7 ms（int8）/ ~4 ms（int4） |
| 8 | 进阶：fp16、算子融合、投机解码 | — | — |

> 表里第 2 步以后的数字是按带宽模型算的**估计值**，每做完一步我们用实测替换它。

---

## 1. 现状诊断：时间到底花在哪

### 1.1 第一个元凶：你跑的是 Debug 构建

`build/CMakeCache.txt` 里是 `CMAKE_BUILD_TYPE=Debug`，按 [`CMakeLists.txt`](../CMakeLists.txt) 的配置它意味着：

```cmake
$<$<CONFIG:Debug>:-g -fsanitize=address,undefined>   # 没有 -O，默认 -O0
```

- **`-O0`**：不内联、不做寄存器分配优化。`std::span::operator[]`、`Tensor::data()`
  这些在 Release 里会消失成一条指令的小函数，每次都是真正的函数调用；
  `_mm256_fmadd_ps` 这类 intrinsic 也会被编译成「从栈上读→算→写回栈」。
- **ASan**：每一次内存读写前都插入一段检查「影子内存」的代码，还会在每个栈对象周围放红区。
  对于矩阵乘这种几乎全是访存的代码，开销是 2–5 倍。
- **UBSan**：每次整数运算、下标计算都检查溢出。

同一台机器、同一份代码，`build-release/` 实测：

```
           Debug (build/)    Release (build-release/)
prefill    1376 ms           219 ms     （6.3×）
decode     751 ms/token      131 ms/token（5.7×）
```

**规矩**：Debug 用来找 bug（终端里那次 `stack-use-after-scope` 就是 ASan 抓到的），
性能只在 Release 下测。

### 1.2 先算天花板：decode 的理论上限

decode 每生成一个 token，要把**所有权重完整读一遍**：

```
每个 Transformer 层：
  c_attn     768 × 2304
  attn_proj  768 × 768
  c_fc       768 × 3072
  mlp_proj  3072 × 768
  合计 7.08M 参数 × 4 字节 = 28.3 MB，× 12 层 = 340 MB
lm_head（复用 wte）：50257 × 768 × 4 字节 = 154 MB   ← 一个矩阵占了 31%！
-------------------------------------------------------------
每 token 读权重 ≈ 494 MB
```

本机实测内存读带宽（`/tmp/bench_bw.cpp`，AVX2 顺序读 496 MB）：

```
 1 线程   15.6 GB/s
 4 线程   32.5 GB/s
 8 线程   37.5 GB/s
14 线程   38.7 GB/s     ← 8 个线程就已经饱和
```

所以 fp32 decode 的**天花板**是：

\[
\frac{494\ \text{MB}}{38\ \text{GB/s}} \approx 13\ \text{ms/token} \approx 77\ \text{token/s}
\]

而现在是 131 ms/token —— **还有 10 倍空间**。

### 1.3 为什么 decode 是「访存密集」：算术强度

一个 `[M,K] × [K,N]` 的矩阵乘：

- 计算量：\(2MNK\) FLOP
- 至少要读的权重：\(4KN\) 字节

\[
\text{算术强度} = \frac{2MNK}{4KN} = \frac{M}{2}\ \text{FLOP/字节}
\]

本机多线程峰值约 250 GFLOPS，带宽 38 GB/s，「机器平衡点」约 \(250/38 \approx 6.5\) FLOP/字节。

- decode：M = 1 → 0.5 FLOP/字节，远低于 6.5 → **带宽说了算**，算得再快也没用。
- prefill：M = T（prompt 长度），要 \(T/2 > 6.5\)，即 **T ≳ 13** 才真正变成计算密集。

> 所以程序里那句「prefill 是计算密集的」只对长 prompt 成立。
> 5 个 token 的 prefill 依然是访存密集，理想耗时应该和**一次 decode 差不多**（~15 ms），
> 而不是现在的 219 ms。

这张图叫 **Roofline 模型**，是分析所有推理性能问题的起点：

```
 性能
(GFLOPS)
  250 ┤                    ┌──────────────── 算力屋顶
      │                  ╱
      │                ╱
      │              ╱  ← 带宽斜坡：性能 = 带宽 × 算术强度
      │            ╱
      │          ╱
   19 ┤ ● decode (M=1)
      │      ╱
      └──────┬───────────┬─────────→ 算术强度 (FLOP/字节)
            0.5         6.5
```

### 1.4 实测：decode 形状的矩阵乘有多慢

`/tmp/bench_decode.cpp` 直接调库里的 `kernel::matmul_f32` / `matmul_nt_f32`，用 decode 的真实形状：

```
---- M=1（decode）----
c_attn    768x2304   1.237 ms  (5.7 GB/s)
attn_proj 768x768    0.309 ms  (7.6 GB/s)
c_fc      768x3072   2.259 ms  (4.2 GB/s)
mlp_proj 3072x768    2.080 ms  (4.5 GB/s)
一层合计 5.88 ms，× 12 层 = 70.6 ms
lm_head 50257x768   26.391 ms  (5.9 GB/s)
```

有效带宽只有 4–7 GB/s，是上限 38 GB/s 的 **1/6 ~ 1/9**。
70.6 + 26.4 ≈ 97 ms，占了 131 ms 的 3/4。而且这还是权重在 L3 里有热缓存的乐观数字。

### 1.5 读代码找原因

**原因 A：decode 时只有 1 个线程在干活。**

`matmul_f32_parallel` 和 `matmul_nt_f32_parallel` 都是**按行（M）切**：

```cpp
template <class F> void parallel_rows(int64 total_rows, int64 work, F &&fn)
{
    const int nt = get_num_threads();
    if (nt <= 1 || work < kParallelMinWork || total_rows < 2)   // M=1 → 直接串行
```

M=1 时只有一行，没法切，14 个核里 13 个在睡觉。单线程带宽上限只有 15.6 GB/s。

**原因 B：GEMM 路径对 M=1 做了大量无用功。**

`affine()` → `matmul_f32` → `matmul_f32_avx2`，GotoBLAS 五层循环会先 `pack_b`：
把整块权重**拷贝**一遍到打包缓冲区，再让微内核去读。

打包的意义是「打包一次，被 A 的很多行复用」。M=1 时只复用 1 次，
等于每个权重被读 2 次、写 1 次，访存量翻 3 倍。再加上 4×16 微内核只有 1 行有效，
另外 3 行在算 0（`edge_tile`），算力也浪费 3/4。

**原因 C：lm_head 掉进了标量循环。**

`matmul_nt_f32_avx2` 一次处理 4 行，不足 4 行的部分走这里：

```cpp
// 下面不足 4 行的边角
for (int64 i = m_main; i < M; ++i)
    for (int64 j = 0; j < N; ++j)
    {
        float sum = 0.0F;
        for (int64 k = 0; k < K; ++k)       // 纯标量，没有 AVX
            sum += pa[i * K + k] * pw[j * K + k];
```

decode 和 `last_only` 的 prefill 都是 M=1，于是**整个 154 MB 的 lm_head 都在跑标量代码**，
而且单线程。（浮点加法有依赖链，编译器不开 `-ffast-math` 不会自动向量化这个求和。）

**原因 D：每次矩阵乘都新建 14 个线程。**

`parallel_rows` 每次调用都 `emplace_back` 一批 `std::jthread`。一个 token 要做
12 × 4 + 1 = 49 次矩阵乘；prefill 时每次都建线程，创建 + join 一个线程几十微秒。
现在 decode 因为原因 A 碰巧没建线程，一旦修好 A，这里就会变成新瓶颈。

**原因 E：每层都把整个 KV-Cache 拷一遍，每个算子都分配新内存。**

```cpp
cache->append(li, start_pos, k.data(), v.data());
Tensor ck = Tensor::zeros({cache->length(), C});   // 分配 + 清零
Tensor cv = Tensor::zeros({cache->length(), C});
std::ranges::copy(cache->keys(li), ck.data().begin());   // 整段拷贝
std::ranges::copy(cache->values(li), cv.data().begin());
```

生成第 n 个 token 时，每层拷 \(2 \times n \times 768 \times 4\) 字节 —— 这正是 KV-Cache
想避免的 O(n) 开销又被加了回来，序列越长越明显。另外 `layernorm`、`affine`、`add`、
`gelu`、`slice().contiguous()` 每一步都 `Tensor::zeros` 一个新张量（`make_shared<byte[]>`
会做值初始化 = 清零），一个 token 几百次堆分配。

**prefill 的额外问题（原因 F）：** M=5 时 `parallel_rows` 开 5 个线程，每个线程只分到 1 行，
却各自**完整打包一份 B**。5 倍冗余打包，而且每个线程又回到了原因 B 的情形。

---

## 2. 分步优化方案

每一步的流程都一样：

1. 写/改 benchmark，记录**改之前**的数字
2. 改代码
3. 跑 `ctest` 和 `hello_gpt` 第 3 节的「逐层对拍」，确认**结果没变**
4. 记录**改之后**的数字，写进本文档对应小节

### 第 0 步：测量工具

**目标**：能回答「一个 decode token 的 131 ms 里，每个算子各占多少」。

做法：给 `forward_impl` 加一个可选的计时器，按算子类别累加耗时：

```cpp
struct Profile
{
    double embed = 0, layernorm = 0, qkv = 0, attn = 0, attn_proj = 0,
           mlp_fc = 0, gelu = 0, mlp_proj = 0, residual = 0, kv_copy = 0, lm_head = 0;
};

// 用法：一个 RAII 小工具，作用域结束时把耗时加到某个字段上
struct ScopedTimer
{
    double *acc;
    Clock::time_point t0 = Clock::now();
    ~ScopedTimer() { if (acc) *acc += ms_since(t0); }
};
```

`generate` 结束后打印每一项占比，以及每个矩阵乘的「有效带宽」（权重字节数 / 耗时）。
**有效带宽是 decode 优化最重要的指标** —— 它直接告诉你离 38 GB/s 还有多远。

**已实现**：`OpProfile`（`include/inference/gpt.hpp`）+ `ScopedTimer`（`src/gpt.cpp`），
`generate()` 把 prefill / decode 分别累加进 `GenStats::prefill_ops` / `decode_ops`，
`hello_gpt` 第 6 节打印报告。不传 `stats` 时指针为空，计时器什么都不做。

**实测结果**（Release，温度 0，decode 59 次 forward 取平均）：

```
                      prefill (M=5)              decode (M=1)
算子              ms     占比   有效带宽      ms     占比   有效带宽
qkv (c_attn)     41.4   18.5%   2.1 GB/s     22.5   17.9%   3.8 GB/s
attn_proj        17.5    7.8%   1.6 GB/s      7.4    5.9%   3.8 GB/s
mlp_fc           54.5   24.3%   2.1 GB/s     27.9   22.2%   4.1 GB/s
mlp_proj         54.1   24.1%   2.1 GB/s     27.5   21.8%   4.1 GB/s
lm_head          49.6   22.1%   3.1 GB/s     38.4   30.5%   4.0 GB/s
其它（layernorm、gelu、拷贝、注意力、采样…）  ~7    3%                ~2.3   1.8%
合计            224                         126
```

结论：

- decode 时 **5 个矩阵乘占 98%**，有效带宽只有 ~4 GB/s，是上限 38 GB/s 的 1/10。
  注意力、拷贝、layernorm 现在都不值得碰 —— 第 2、3 步的方向是对的。
- prefill 的有效带宽（~2 GB/s）**比 decode 还低**，印证了原因 F：5 个线程各自打包一份完整权重。
- `lm_head` 一个矩阵就占 decode 的 30%，修掉原因 C（M<4 走标量）是最划算的一处。

测量注意事项（详见 [matmul_optimization.md 第 7 节](matmul_optimization.md)）：

- 笔记本会降频、P/E 核混合，同一配置前后能差 30%。每次对比**多跑几次取中位数**。
- 生成用固定种子 + 温度 0，保证每次走同样的 token 序列。
- decode 测 60 个 token 以上，避免偶然因素。

### 第 1 步：Release 构建（已完成）

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j --target hello_gpt
./build-release/examples/hello_gpt "The capital of France is" 60 0.8
```

实测：prefill 219 ms（22.9 token/s），decode 131 ms/token（7.6 token/s）。

### 第 2 步：decode 专用 GEMV 内核

**原理**：M=1 时矩阵乘退化成**矩阵 × 向量**（GEMV）。它没有任何数据复用可言，
每个权重只被用一次。所以优化目标只有一个：**把权重以最快速度顺序流过 CPU**。

- 不打包（打包是为复用服务的，没复用就是纯开销）
- 顺序读、每次读满 cache line
- 用多个独立累加器隐藏 FMA 延迟（FMA 延迟 4 周期，吞吐每周期 2 条，至少要 8 个独立累加器才能跑满）

**两种权重布局的 GEMV 写法**：

```
(a) W 是 [K, N]（in, out，HuggingFace Conv1D 布局，affine 现在用的）
    y[0:N] = Σ_k x[k] · W[k, 0:N]          ← 每一步是一次 axpy，沿 N 向量化
    读 W：一行一行顺序读 ✓
    缺点：y 要反复读写（N=3072 时 12 KB，在 L1 里，问题不大）

(b) W 是 [N, K]（out, in，PyTorch Linear 布局，linear/lm_head 用的）
    y[n] = dot(x[0:K], W[n, 0:K])           ← 每个输出是一次点积，沿 K 向量化
    读 W：也是顺序读 ✓
    x 只有 3–12 KB，常驻 L1
```

两种都是顺序读权重，都能打满带宽。建议**统一成 (b)**：加载时把 Conv1D 权重转置成 [out, in]，
这样整个模型只需要维护一个 GEMV 内核（`matmul_nt` 系列），也是 llama.cpp 的做法。
转置只在加载时做一次，代价可以忽略。

(b) 的内核骨架（一次算 4 个输出，摊薄读 x 的开销）：

```cpp
// y[n] = dot(x, W[n,:])，W 是 [N, K] 行优先
void gemv_nt_f32(const float *x, const float *w, float *y, int64 n_begin, int64 n_end, int64 K)
{
    for (int64 n = n_begin; n + 4 <= n_end; n += 4)
    {
        __m256 acc0 = _mm256_setzero_ps(), acc1 = ..., acc2 = ..., acc3 = ...;
        for (int64 k = 0; k < K; k += 8)
        {
            const __m256 xv = _mm256_loadu_ps(x + k);
            acc0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w + (n + 0) * K + k), acc0);
            acc1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w + (n + 1) * K + k), acc1);
            acc2 = ...;
            acc3 = ...;
        }
        y[n + 0] = hsum256(acc0);  // ... 其余同理
    }
    // 尾部 + K 不是 8 的倍数的情况
}
```

同时修掉原因 C：`matmul_nt_f32_avx2` 里 M < 4 的边角行改为调用这个 GEMV，而不是标量循环。

**预期**：单线程下有效带宽从 4–6 GB/s 提到接近单核上限 ~15 GB/s，
decode 131 ms → ~60 ms。

**验证**：`tests/` 里给 `gemv` 加用例，和 `matmul_nt_f32_naive` 对比，覆盖 K、N 不是 8/4 倍数的情况。

**已实现**：

- `kernel::gemv_nt_f32`（`src/kernel_matmul.cpp`）：一次算 4 个输出，每行 2 个累加器、k 一次走 16，
  共 8 条独立 FMA 链；N 的尾巴用单行点积，K 的尾巴用标量补
- `matmul_nt_f32_avx2` 里不足 4 行的部分改为逐行调用 GEMV（原来是标量三重循环）
- `GPT::load` 用 `to_out_in()` 把 4 个 Conv1D 权重转置成 `[out, in]`（32×32 分块转置），
  forward 里的 `ops::affine` 全部换成 `ops::linear`
- 测试：`gemv_nt_matches_reference`（12 个形状，含 GPT-2 真实形状、越界写检查）、
  `gemv_nt_equals_matmul_nt_row`

**实测结果**（Release，温度 0）：

```
                       改之前                    改之后
算子              ms/token  有效带宽       ms/token  有效带宽
qkv (c_attn)       22.5     3.8 GB/s        7.1    12.0 GB/s
attn_proj           7.4     3.8 GB/s        2.5    11.5 GB/s
mlp_fc             27.9     4.1 GB/s        9.3    12.1 GB/s
mlp_proj           27.5     4.1 GB/s        8.9    12.8 GB/s
lm_head            38.4     4.0 GB/s       14.0    11.0 GB/s
decode 合计       126                      44.9     （2.8×）
prefill (5 tok)   224 ms                   78–107 ms
```

几点观察：

- 有效带宽 ~12 GB/s，已经接近**单线程**带宽上限 15.6 GB/s。单线程能做的基本做完了，
  剩下的 3 倍要靠第 3 步的多线程。
- prefill 也顺带快了 2 倍多：M=5 时 `matmul_nt_f32_parallel` 把 5 行分给 5 个线程，
  每个线程都是 M=1，现在走 GEMV，不再各自打包一份权重。
- 代价：加载时多了一次转置，加载时间 ~1.1 s → ~1.5 s（朴素转置是 2.1 s，分块后降下来）。
  更彻底的做法是在 `tools/convert_gpt2.py` 里就存成 `[out, in]`，加载时零开销。
- 现在矩阵乘之外的部分（拷贝 1.5 ms、gelu 0.7 ms、注意力 0.4 ms）占 decode 的 ~7%，
  第 3 步之后比例还会上升，到时就轮到第 4 步了。

### 第 3 步：GEMV 多线程 + 线程池

**原理**：单核带宽只有 15.6 GB/s，8 核才能打满 38 GB/s。M=1 不能按行切，
就**按输出列 N 切**：每个线程负责 `y[n_begin:n_end]`，读的是 W 中互不重叠的连续行。

```
W [N, K]：
  ┌──────────────┐  线程 0 负责 y[0    : N/8]
  ├──────────────┤  线程 1 负责 y[N/8  : 2N/8]
  ├──────────────┤  ...
  └──────────────┘  每个线程顺序读自己那一段，互不干扰，也没有 false sharing
```

**为什么要线程池**：一个 token 49 次矩阵乘，每次新建 8–14 个线程，
光创建/销毁就要几毫秒，而修好之后一次 c_attn 只需要 ~0.2 ms。
线程池在启动时建好 N 个工作线程，每次只是「发任务 → 等完成」，开销降到微秒级。

线程池设计要点（这一步是学习重点）：

- 工作线程常驻，用 `std::condition_variable` 或原子变量 + 自旋等任务
- decode 时任务间隔很短（几十微秒），纯 `condition_variable` 唤醒延迟 ~10–50 µs 太贵，
  常见做法是**先自旋一小段，再睡眠**（llama.cpp / OpenMP 都这样）
- 主线程也领一份活，别闲着

**P 核 / E 核问题**：静态平均切分时，所有线程要等最慢的 E 核做完。两个对策：

1. 把 N 切成比线程数多得多的小块（比如 64 行一块），用一个原子计数器**动态领任务**，
   快的核自然多干
2. 带宽 8 线程就饱和，试试 `set_num_threads(8)` 是不是比 14 更快

**预期**：有效带宽接近 30–35 GB/s，decode → ~15–20 ms（50–65 token/s）。

**已实现**：

- `ThreadPool`（`include/inference/thread_pool.hpp`、`src/thread_pool.cpp`）：常驻工作线程，
  先自旋 2 ms 再睡到条件变量上；任务用 CAS 动态领取；调用者也干活
- `kernel_matmul.cpp` 里的全局 `pool()`：线程数跟着 `get_num_threads()` 走，变了就重建；
  `parallel_rows`（GEMM 路径）也改用线程池，不再每次新建 `std::jthread`
- `matmul_nt_f32_parallel`：**M < 线程数时按输出列 N 切**，块大小约为 N / (4 × 线程数)、
  至少 16 行、4 的倍数，每块内对 M 行依次做 GEMV；W 小于 128 KB 时不并行
- `get_num_threads()` 支持环境变量 `INFER_NUM_THREADS`；`hello_gpt` 默认用一半逻辑核
- 测试：`tests/test_thread_pool.cpp`（每个任务恰好执行一次、5000 批连续发、结果可见性、
  嵌套调用、睡眠后唤醒），`matmul_nt_parallel_small_m_splits_columns`（按列切路径 × 4 种线程数）；
  线程池另外用 ThreadSanitizer 跑过，无数据竞争

#### 踩坑：线程越多越慢

第一版线程池的写法是「调用者等**每个工作线程**都确认处理过本批任务才返回」。扫线程数的结果：

```
线程数        1     4     6     8    10    14
decode tok/s  31    38    45    43    38    9.4   ← 14 线程比单线程还慢 3 倍
```

分算子报告给出了线索：14 线程时 `lm_head`（一次 154 MB 的大矩阵乘）依然有 25 GB/s，
但 `attn_proj`（2.4 MB，实际干活 0.1 ms）每次要 ~2 ms —— **问题是每次同步的固定开销，不是带宽**。

原因：14 个线程占满 14 个逻辑核在自旋，只要操作系统（这里还隔着 WSL2 的虚拟化层）
把**任何一个**线程换下去一个时间片（毫秒级），调用者就得等它回来报到，哪怕任务早被别人做完了。
这是「自旋屏障 + 线程数 ≥ 空闲核数」的经典问题。

改进：调用者只等**任务完成数**到 n，不等线程报到。难点是一个醒晚了的线程可能读到下一批
改了一半的任务描述。解法是把「批次号 | 任务总数 | 下一个任务编号」打包进一个 64 位原子变量，
用 CAS 领任务：CAS 成功就证明领到的是当前批次的任务，而这批没做完之前调用者不会改任务描述。

改完之后（每个配置 3 次）：

```
线程数        1          4          6          8          10         12         14
decode tok/s  26–30     36–42      43–47      39–44      31–36      22–31      20–24
```

14 线程从 9.4 提到 ~21，但**最优点仍在 6–8 线程**。剩下的差距来自硬件本身：

- 带宽 6–8 个线程就饱和了（见 1.2 节），再加线程读不快
- 2 个 LP-E 核在 SoC 芯片块上，访存很慢；任务落到它们身上，整批都等它（掉队者）
- 自旋线程占满所有逻辑核，会和主线程抢超线程资源，主线程上的 layernorm / gelu 等串行算子也变慢

所以推理程序不该默认用满所有逻辑核。kernel 库的默认值保持不变（大矩阵 GEMM 用 14 线程更快，
`hello_perf` 实测 N=1024 仍有 225–337 GFLOPS），由 `hello_gpt` 自己选一半逻辑核 —— 和 llama.cpp 的 `-t` 参数同理。

**实测结果**（7 线程，温度 0）：

```
算子              第 2 步（单线程）       第 3 步（6~7 线程）
                  ms/token  有效带宽      ms/token  有效带宽
qkv (c_attn)        7.1    12.0 GB/s        2.9    29.5 GB/s
attn_proj           2.5    11.5 GB/s        1.1    27.0 GB/s
mlp_fc              9.3    12.1 GB/s        3.8    30.1 GB/s
mlp_proj            8.9    12.8 GB/s        3.6    31.3 GB/s
lm_head            14.0    11.0 GB/s        5.0    30.8 GB/s
copy (split/kv)     1.5                     1.4
gelu                0.7                     0.8
attention           0.4                     0.5
decode 合计        44.9                    19.4–22   （45 token/s 左右）
prefill (5 tok)   78–107 ms                35–51 ms
```

观察：

- 矩阵乘有效带宽 ~30 GB/s，到了实测上限 38 GB/s 的 80%。fp32 下矩阵乘还能挤的空间不多了
- 现在**矩阵乘之外**的部分（拷贝 1.4 + gelu 0.8 + 注意力 0.5 + layernorm 0.15 ≈ 3 ms）
  占 decode 的 15%，而且都是串行的 —— 这就是第 4 步要做的事

### 第 4 步：去掉多余的拷贝和分配

到这一步，矩阵乘已经接近带宽上限，剩下的「零碎开销」比例会变大。

1. **注意力直接读 KV-Cache**：`attention_f32` 接受 cache 的 span，删掉 `ck/cv` 的分配和拷贝。
   这是唯一一个**随序列长度增长**的冗余，长文本生成时影响最大。
2. **QKV 不拆成三个张量**：现在 `slice().contiguous()` 拷了 3 次。改成让 attention 接受
   带行距（stride = 3C）的 q/k/v 指针，或者让 c_attn 直接把 K/V 写进 cache 的对应位置。
3. **预分配工作区**：`x`、`h`、`qkv`、`attn_out`、`mlp_hidden`、`logits` 在模型加载时按
   最大尺寸分配一次，forward 里只复用。这是所有推理引擎的标准做法（llama.cpp 的 compute buffer）。
4. **原地残差**：`x = ops::add(x, y)` 改成 `x += y`，不新建张量。
5. **bias 融合进矩阵乘**：现在是矩阵乘写完 C，再扫一遍 C 加 bias。GEMV 算完每个输出时顺手加上即可。

**预期**：短序列下省 1–2 ms；序列越长收益越大，decode 速度不再随生成长度下降。

**已实现**：

- `GPT::Workspace`（`x`、`h`、`qkv`、`att`、`fc` 五块缓冲区）跨调用复用，`forward_impl`
  全部在上面原地算：layernorm、残差 `x += y`、gelu 都原地做；只有返回值 logits 每次新分配
- `kernel::attention_f32_strided`：q/k/v 各自带行距。q 直接读 `qkv`（行距 3C），
  用 cache 时 k/v 直接读 KV-Cache（行距 C），不用 cache 时直接读 `qkv`。
  删掉了 `slice().contiguous()` 的三次拷贝，以及每层把整段缓存拷出来的 `ck/cv`
- K/V 从 `qkv` 的对应列直接写进 cache
- 顺手做了 **AVX2 版 gelu**：标准库没有 SIMD 的 `tanh`，自己实现了 8 路 `exp`
  （Cephes 多项式，2ⁿ 直接拼浮点指数位），`tanh(u) = 1 - 2/(e^{2u}+1)`。
  和 `std::tanh` 版的最大绝对误差 3.2e-7
- 测试：`kernel_gelu_simd_matches_std_tanh`（24001 个点覆盖 [-12, 12]、原地计算）、
  `kernel_attention_strided_matches_contiguous`；逐层对拍全部通过，贪心生成文本不变

**实测结果**：这一轮机器发热降频严重，同一个程序前后能差 2 倍，所以用 git worktree 编出
第 3 步的版本，和第 4 步**交替运行** 5 轮（7 线程，ms/token）：

```
              copy    attention   gelu    layernorm   串行合计(中位)   decode(中位)
第 3 步       1.7     0.62        0.98    0.18        ~3.5             23.4
第 4 步       0.04    0.86        0.14    0.12        ~1.2             19.1
```

几点观察：

- **copy 降了 40 倍**，gelu 降了 7 倍（单线程下 0.7 ms → 0.064 ms，10 倍）
- **attention 反而变慢了 0.2 ms**：以前拷 `ck/cv` 时顺便把 KV 数据拉进了 cache，
  注意力读的是热数据；现在它自己承担 cache miss。合起来「copy + attention」1.9 → 0.9 ms，仍是净收益。
  这也是测量的一课：**省掉一步时，它承担的副作用会转移到下一步**
- **新发现：多线程时串行算子会变慢 2–3 倍**。同一时段对比：

  ```
                 attention   gelu     layernorm
  1 线程         0.52        0.064    0.083
  7 线程         1.23        0.19     0.16
  ```

  工作线程在两次矩阵乘之间自旋，要么和主线程抢同一个物理核的超线程资源，
  要么主线程被调度到了 E 核上。对策是让串行部分也并行起来（第 5 步把注意力按头并行），
  或者做线程绑核（affinity）实验

### 第 5 步：注意力内核

现在的 `attention_f32` 是纯标量三重循环、单线程。短序列（几十个 token）时它占比很小，
但上下文长到几百时，每个 token 的注意力要读 \(2 \times T \times 768 \times 4\) 字节 × 12 层，
T=1024 时约 75 MB，和权重同一量级。

优化方向：

1. `head_size = 64` = 8 个 `__m256`，点积和加权求和都用 AVX2，完全展开
2. **12 个头互相独立**，天然可以并行到 12 个线程（交给第 3 步的线程池）
3. `softmax_inplace` 里的 `std::exp` 是标量调用，可以换成 AVX2 的多项式近似 exp
4. （进阶）**KV-Cache 布局改成 [层][头][位置][64]**：同一个头的 K 在内存里连续，
   读取更友好。prefill 长序列时还可以学 **Flash Attention** 的分块 + 在线 softmax 思路

验证方法：生成 500 个 token，画出「每 token 耗时 vs 位置」的曲线，优化前应该明显上升，优化后应该很平。

**已实现**：

- `kernel_attention.cpp`：原来的标量版改名为 `attention_f32_naive`，留作参考实现。
  新的快速版对每个 (头, 查询) 任务做：AVX2 点积（2 个累加器）→ AVX2 softmax
  （max / exp / 求和 / 缩放都 8 个一组，exp 复用 `simd::exp256_ps`）→ AVX2 axpy 累加 V。
  分数缓冲区是 `thread_local`，热路径上不分配内存
- 并行：n_head × Tq 个任务交给线程池（新增 `kernel::parallel_for`）。
  总工作量 n_head × Tq × Tk × head_size < 30 万时串行
- `src/simd_math.hpp`：kernel 内部共用的 `exp256_ps` / `hsum256_ps` / `hmax256_ps`
- **KV-Cache 改成按头存放** `[n_head][capacity][head_size]`（原来是 `[capacity][n_embd]`）。
  注意力接口改成 `kernel::HeadView{data, row_stride, head_stride}`，一套 kernel 同时支持
  qkv 里的 q/k/v 段、连续 `[T, C]` 和按头存放的缓存。`KVCache::append_strided` 直接从 qkv 的
  k/v 段（行距 3C）写进缓存
- 测试：`kernel_attention_fast_matches_naive`（10 个形状 × 1/4 线程，含 decode / prefill /
  分段 prefill、head_size 非 8 的倍数、Tk = 1024），`kernel_attention_strided_matches_contiguous`
  增加按头布局的校验

**实测 1：注意力 kernel 本身**（`/tmp/bench_attn.cpp`，热缓存，每层一次，µs）：

```
形状              朴素      SIMD 1线程   SIMD 7线程
decode Tk=64      19.4      9.3          8.4        （低于阈值，走串行）
decode Tk=256     83.7      41.4         37.4       （低于阈值，走串行）
decode Tk=512     225.7     111.3        93.9
decode Tk=1024    741.6     404.3        257.7      2.9×
prefill T=128     4511      1993         642        7.0×
prefill T=512     83489     33408        7976       10.5×
```

- SIMD 只带来 **~2×**，不是 8×。短序列时每个 key 只有 4 次 FMA，之后的水平求和（8→1）
  有 ~10 条指令的依赖链，成了瓶颈；长序列时每层要读 6 MB 的 K/V，受带宽限制。
  想更快可以一次算 8 个 key 的点积，最后用转置做归约 —— 留作练习
- 并行在 prefill 上收益最大（12 × T 个任务，能喂饱所有线程）

**踩坑：并行阈值**。第一版阈值是 6.5 万（decode 的 Tk ≈ 85），结果 decode Tk=256 时
7 线程（45 µs）比单线程（39 µs）还慢：12 个头分给 7 个线程要跑两轮，每个任务才 3 µs，
同步开销占了大头。又在冷缓存下扫了一遍（强制总是并行）：

```
Tk            32     64     128    256    384    512    1024
1 线程 ms     0.16   0.31   0.70   1.24   2.79   4.28   8.10     （12 层合计，按头布局）
7 线程 ms     0.27   0.46   0.87   1.87   2.50   1.45   2.98
```

交叉点在 Tk ≈ 384~512，阈值定为 30 万（≈ decode 的 Tk 390）。

**实测 2：KV-Cache 布局**。端到端生成 900 个 token 时，注意力平均每 token 4.5 ms、占 16.7%，
但按 microbenchmark 推算只该 ~1.5 ms。差别在于 microbenchmark 反复读同一块数据（热在 L3 里），
真实模型每 token 要读 12 层共 ~33 MB 的 KV-Cache（冷的）。按位置存放时，一个头的 64 个数
夹在每行 768 个数中间，每隔 3 KB 跳着取 256 字节，硬件预取器跟不上。

冷缓存对比（`/tmp/bench_kvlayout.cpp`：12 层各一份 K/V，总量 150 MB，远超 12 MB 的 L3，
同一进程交替运行，ms / token）：

```
                     [位置][C]   [头][位置][64]
1 线程   Tk=512      26.9        12.7          2.1×
1 线程   Tk=1024     39.8        21.7          1.8×
7 线程   Tk=512      6.7         3.2           2.1×
7 线程   Tk=1024     11.9        6.3           1.9×
```

端到端 900 token：注意力占比 16.7% → 10.8–12.2%。这一轮机器降频严重，绝对值不可比，
用同一次运行里的 `mlp_fc` 做参照：注意力 / mlp_fc 从 0.93 降到 0.55–0.59，约快 1.6×。

**教训**：microbenchmark 的数据如果一直热在缓存里，会掩盖真实场景的访存问题。
测访存相关的优化时，要构造比 L3 大的数据集。

**实测 3：端到端 60 token**（7 线程，这一轮机器较凉）：decode 15.6–21 ms/token
（48–65 token/s），prefill 25–28 ms（~200 token/s），注意力 0.37–0.54 ms（第 4 步 0.86 ms）。

### 第 6 步：prefill 的小 M 问题

prefill 的矩阵乘 M = prompt 长度。按 1.3 节的分析：

- **M 小（< 16 左右）**：仍然是访存密集，应该走「按 N 切的多线程 + 一次读权重服务 M 行」的路径。
  可以把第 2 步的 GEMV 扩展成一次处理 M 个 x 向量（读一次 W 的一行，和 M 个 x 做点积），
  即 **GEMM with small M**，llama.cpp 里叫 `vec_dot` 批处理
- **M 大**：走现有的 GotoBLAS GEMM，但多线程切分要改：M 小于线程数时不要按行切。
  可以按 N 切（每个线程打包 B 的不同列块，不再冗余）

另外 `lm_head` 在 prefill 时已经做了 `last_only`，M=1，修完第 2/3 步自动受益。

**预期**：5 token 的 prefill 从 219 ms 降到和一次 decode 差不多（~20 ms）。

### 第 7 步：权重量化

做完第 3 步，decode 就**贴着带宽天花板**了。再想变快，只剩一个办法：**每个 token 少读一些字节**。

```
格式          每个权重      每 token 读    理论 decode 上限（38 GB/s）
fp32          4   字节      494 MB         ~13 ms  (77 token/s)
fp16/bf16     2   字节      247 MB         ~6.5 ms (150 token/s)
int8 (Q8_0)   1.06 字节     131 MB         ~3.5 ms (290 token/s)
int4 (Q4_0)   0.56 字节      70 MB         ~1.8 ms (540 token/s)
```

（实际达不到理论值：反量化要算力，小矩阵有同步开销。但 2–4 倍的提升是现实的。）

**Q8_0 格式**（llama.cpp 同款，建议从这个开始）：

```
每 32 个权重一组：
  scale : 1 个 fp16（或 fp32）   = max(|w|) / 127
  q[32] : 32 个 int8             = round(w / scale)
反量化：w ≈ q × scale
```

为什么要分组：一个 scale 管整行的话，一个离群大值会把其它权重的精度全压没；
32 个一组，离群值只影响自己那组。

实现步骤：

1. `tools/convert_gpt2.py` 或 C++ 加载时做量化（先做加载时，方便对比）
2. 写 `gemv_q8_0`：读 int8 → 扩展成 int16/int32 → 转 float → FMA，每组乘一次 scale。
   更快的做法是把 x 也量化成 int8，用 `_mm256_maddubs_epi16` 做整数点积（这正是 llama.cpp 的思路）
3. **精度验证**：对比 fp32 和 int8 的 logits 误差，以及在一段文本上的困惑度（perplexity）变化。
   GPT-2 上 Q8_0 的困惑度损失通常 < 0.1%

通常 lm_head 和 embedding 保持高精度（或单独用 Q8），层内矩阵可以更激进地用 Q4。

### 第 8 步：进阶方向（了解即可）

- **fp16 / bf16 权重**：用 F16C 指令 `_mm256_cvtph_ps` 读 fp16 转 fp32 再算，带宽减半、精度几乎无损
- **算子融合**：layernorm + GEMV、GEMV + GELU 合并成一个 kernel，少一次读写激活值
- **Sampler 优化**：现在每个 token 对 50257 个 logits 做 `partial_sort`，约几百微秒；
  用 `nth_element` 先选出 top-k 再排序更快
- **投机解码（speculative decoding）**：用一个小模型先猜几个 token，大模型一次 forward 验证多个。
  把访存密集的 decode 变成 M>1 的「小 prefill」，突破带宽墙
- **批处理（batching）**：服务端同时处理多个请求，M = batch size，提高算术强度。
  这是 vLLM 等推理服务的核心思路

---

## 3. 附录：本文用到的测量程序

两个临时 benchmark 都放在 `/tmp`，链接 `build-release/libinference.so`：

```bash
g++ -std=c++20 -O3 -march=native /tmp/bench_decode.cpp \
    -I include -L build-release -linference -Wl,-rpath,$PWD/build-release -o /tmp/bench_decode
g++ -std=c++20 -O3 -march=native /tmp/bench_bw.cpp -o /tmp/bench_bw
```

后续做第 0 步时，建议把它们整理进 `examples/hello_perf.cpp` 或新建 `examples/bench_gpt.cpp`，
作为每一步优化的固定回归基准。

## 4. 进度记录

| 步骤 | prefill (5 tok) | decode (ms/token) | decode token/s | 备注 |
|------|-----------------|-------------------|----------------|------|
| 起点 Debug | 1376 ms | 751 | 1.3 | `-O0` + ASan + UBSan |
| 1 Release | 219 ms | 131 | 7.6 | |
| 0 加计时后复测 | 224 ms | 126 | 8.1 | 矩阵乘占 decode 98%，有效带宽 ~4 GB/s |
| 2 GEMV + 权重转成 [out, in] | 78–107 ms | 44.9 | 22.6 | 有效带宽 ~12 GB/s，接近单线程上限 |
| 3 多线程 + 线程池 | 35–51 ms | 19.4–22 | ~45 | 7 线程；矩阵乘 ~30 GB/s；14 线程反而更慢 |
| 4 去拷贝 + 工作区 + SIMD gelu | 35–50 ms | 19.1（中位） | ~52 | 与第 3 步交替 A/B：23.4 → 19.1；串行部分 3.5 → 1.2 ms |
| 5 注意力 SIMD + 并行 + KV 按头存放 | 25–28 ms | 15.6–21 | 48–65 | 900 token 时注意力占比 16.7% → ~11% |
| 6 prefill 小 M | | | | |
| 7 量化 Q8_0 | | | | |
