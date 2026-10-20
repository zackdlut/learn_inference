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

| 步骤 | 做什么 | 解决什么瓶颈 | decode 每 token 预期 | 实测 |
|------|--------|--------------|----------------------|------|
| 0 | 搭测量工具：分算子计时 | 不知道时间花在哪 | — | 126 ms，矩阵乘占 98% |
| 1 | 用 Release 构建 | `-O0` + ASan 让每次访存慢几倍到几十倍 | 750 ms → 131 ms | 131 ms |
| 2 | 给 M=1 写专用 GEMV 内核 | decode 走 GEMM 路径：整块打包权重、lm_head 掉进标量循环 | → ~60 ms | **44.9 ms** |
| 3 | GEMV 按输出列多线程 + 线程池 | decode 时只有 1 个线程在干活；每次矩阵乘都新建线程 | → ~15–20 ms | **19.4–22 ms** |
| 4 | 去掉多余拷贝和内存分配 | KV-Cache 每层整段拷贝、每个算子都 `zeros` 新张量 | → ~14 ms，且不随序列长度变慢 | **19.1 ms**（中位，降频严重） |
| 5 | 注意力内核 SIMD + 多头并行 | 长上下文时注意力变成大头 | 长序列下收益明显 | **15.6–21 ms**；900 token 时注意力占比 16.7% → 11% |
| 6 | 小 M 的 prefill 走「按列切」 | M=5 时 5 个线程各打包一份完整 B | prefill 219 ms → ~20 ms | 第 2/3 步已顺带解决：prefill ~25 ms |
| 7 | 权重量化 int8 / int4 | 已经打满带宽，只能**少读字节** | → ~7 ms（int8）/ ~4 ms（int4） | **9.3–10.5 ms**（层内 Q8_0 + lm_head fp16） |
| 8 | 进阶：fp16、算子融合、投机解码 | — | — | — |

> 「预期」一列是动手前按带宽模型算的估计值，「实测」是做完之后的数字。
> 每一步末尾有两个小节：**复盘** 按「遇到的问题 → 怎么思考 → 用了什么测试程序 → 结果 → 怎么解决」
> 记录过程；**跟着做** 是可以直接复制执行的复现命令和预期输出。先读 [2.0 准备工作](#20-跟着复现准备工作)。
> 测量程序在 [`bench/`](../bench)，脚本在 [`bench/scripts/`](../bench/scripts)，汇总见 [第 3 节附录](#3-附录测试程序清单)。
>
> 一条命令重建所有步骤的进度表：`DEBUG=1 bench/scripts/reproduce_progress.sh`（第一次约 8 分钟）。

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

本机实测内存读带宽（[`bench/bench_bw.cpp`](../bench/bench_bw.cpp)，AVX2 顺序读 496 MB；
下面是当时的数字，重跑一次见第 0 步的「跟着做」）：

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

当时的临时程序直接调库里的 `kernel::matmul_f32` / `matmul_nt_f32`，用 decode 的真实形状
（每个矩阵反复测，权重热在缓存里，所以数字偏乐观；改进后的版本是
[`bench/bench_decode.cpp`](../bench/bench_decode.cpp) 的 `gemm` 模式，12 层权重各一份、冷缓存）：

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

每一步都用到的三道正确性关卡：

| 关卡 | 程序 | 检查什么 |
|------|------|----------|
| 单元测试 | `ctest --test-dir build-release` | 每个新 kernel 都和朴素参考实现对拍，覆盖尾部形状、多线程、越界写 |
| 逐层对拍 | `hello_gpt` 第 3 节 | 和 numpy 参考实现（`data/gpt/gpt2_reference.inference`）比 embed / 第 0 层 / final_norm / logits |
| 端到端 | `hello_gpt` 第 6 节，温度 0 | 贪心生成的文本必须和改之前**逐字相同**（量化除外，量化看困惑度） |

测试用例数的变化：122（起点）→ 124（第 2 步）→ 131（第 3 步）→ 133（第 4 步）→ 134（第 5 步）→ 140（第 7 步）。

> **注意**：测试用例是 CMake **配置阶段**扫描源文件注册的。新加了 `TEST(...)` 之后
> 要重新跑一次 `cmake -S . -B build-release`，否则 ctest 的总数不变，新用例根本没跑。
> 第 4、7 步都踩过这个坑：看到总数没涨就要警觉。

### 2.0 跟着复现：准备工作

每一步末尾都有一个 **「跟着做」** 小节，里面的命令可以直接复制执行（都在仓库根目录下运行）。
需要的东西都在仓库里：

```
bench/                    每一步的测量程序（C++，链接 libinference）
  bench_common.hpp        计时、中位数、防优化等小工具
  bench_bw.cpp            第 0 步  内存带宽上限
  bench_decode.cpp        第 0/2/3/7 步  decode 形状的矩阵乘，各步的 kernel 并排比
  bench_transpose.cpp     第 2 步  加载时转置权重：朴素 vs 分块
  bench_pool.cpp          第 3 步  第一版线程池（等每个线程报到）vs 现在的（只等任务完成）
  bench_serial_ops.cpp    第 4 步  gelu 标量 vs SIMD、KV-Cache 拷贝随长度增长
  bench_attn.cpp          第 5 步  注意力 kernel，热缓存
  bench_kvlayout.cpp      第 5 步  KV 布局 + 并行阈值，冷缓存，外加端到端验证
  bench_quant.cpp         第 7 步  离群值、逐矩阵精度排查、各量化方案的速度
bench/scripts/            把旧版本编出来、交替比较的脚本
  build_commit.sh         用 git worktree 编出某一步的 hello_gpt
  check_step.sh           某一步的三道正确性关卡（ctest、逐层对拍、贪心文本指纹）
  ab_compare.sh           两个版本 ABBA 交替跑，比 decode 中位数
  sweep_threads.sh        扫 INFER_NUM_THREADS
  tsan_thread_pool.sh     ThreadSanitizer 构建 + 跑线程池 / kernel 测试
  reproduce_progress.sh   把每一步都编出来轮流跑，重建第 5 节的进度表
```

**一次性准备**：

```bash
# 1. 模型文件。data/gpt/ 下已经有 gpt2.inference、gpt2_vocab.bin 等文件就跳过
python3 -m venv .venv && .venv/bin/pip install numpy
.venv/bin/python tools/convert_gpt2.py        # 下载 ~550 MB，生成 data/gpt/*.inference、*.bin

# 2. 编译当前版本（含全部 bench），跑测试
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j                # 包括 bench_all
ctest --test-dir build-release                # 140 个用例
```

**每一步对应的提交**（脚本里可以直接写步骤名）：

| 步骤 | 提交 | 脚本里的名字 | ctest 用例数 |
|------|------|--------------|--------------|
| 起点（优化前） | `f142a75` | `baseline` | 122 |
| 0 测量工具 | `3227974` | `0` | 122 |
| 2 GEMV + 权重转成 [out, in] | `5624734`（提交信息写的是「Step1」） | `2` | 124 |
| 3 线程池 | `89c4702` | `3` | 131 |
| 4 去拷贝 + 工作区 | `c98ce1b` | `4` | 133 |
| 5 注意力 | `3600536` | `5` | 134 |
| 7 量化 | 还没提交，就是当前工作区 | `7` 或 `.` | 140 |

> 第 7 步提交之后，把 `bench/scripts/common.sh` 里的 `[7]=.` 改成新的提交号，旧版本对比才能固定下来。

**两种跟法**：

- **看成品**：脚本用 `git worktree` 把某一步检出到 `.worktrees/<提交>/`（已在 `.gitignore`），
  软链 `data/`，编出 `hello_gpt`。你当前的工作区不受影响：

  ```bash
  bench/scripts/build_commit.sh 3     # 最后一行输出 .worktrees/89c4702/build-release/examples/hello_gpt
  .worktrees/89c4702/build-release/examples/hello_gpt "The capital of France is" 60
  ```

- **自己动手写**：从上一步的提交开一个自己的 worktree，自己实现，再用同样的脚本检查和比速度。
  脚本的「版本」参数可以是步骤名、提交号、`.`，也可以是一个源码目录：

  ```bash
  git worktree add -b my-step2 ../my-step2 3227974   # 从第 0 步出发
  git diff 3227974 5624734 --stat                    # 参考答案改了哪些文件（先别看内容）
  # …… 在 ../my-step2 里写代码 ……
  bench/scripts/check_step.sh ../my-step2            # 正确性：ctest、逐层对拍、文本指纹
  bench/scripts/ab_compare.sh 0 ../my-step2          # 速度：和第 0 步交替跑
  git diff 3227974 5624734 -- src/kernel_matmul.cpp  # 写完再对照参考答案
  ```

  `bench/` 下的测量程序只在当前版本里编（`build-release/bench/`），它们用的是当前库里的 kernel；
  旧版本的写法（GEMM 路径、第一版线程池、按位置存的 KV-Cache、标量 gelu）在 bench 里各留了一份，
  所以不用切换版本就能对比。

**测量规矩**（每次跑性能之前）：

1. 插上电源，关掉浏览器、IDE 的索引这类重活；Windows 电源模式选「最佳性能」
2. 比较两个版本一律用 `ab_compare.sh`（ABBA 交替、取中位数），**不要拿前后两次单独的运行比**
3. 下面「预期输出」都是这台机器（Core Ultra 5 135U，WSL2）上的实测值。你的机器绝对值会不同，
   要看的是**比值和趋势**
4. `hello_gpt` 第 3 步起默认用一半逻辑核（这里是 7），用 `INFER_NUM_THREADS` 覆盖

**正确性指纹**：`check_step.sh` 最后打印贪心生成 60 个 token 的 md5 前 12 位。从起点到第 7 步
（fp32 那次生成）全部是 `43d2be1cfd7f`：每一步只改了计算方式，模型的每个选择都没变。
换一台机器（不同的指令集、线程数）浮点舍入可能不同，指纹未必是这个值，但**各步之间应该一致**。
你自己实现的版本如果和上一步的指纹不一样，先找 bug，再谈速度。

```bash
for s in baseline 0 2 3 4 5 .; do echo "== $s"; bench/scripts/check_step.sh $s | grep -E "passed|逐层对齐|指纹"; done
```

```
== baseline
100% tests passed, 0 tests failed out of 122
逐层对齐: 全部通过
指纹 43d2be1cfd7f
== 0
100% tests passed, 0 tests failed out of 122
...
== .
100% tests passed, 0 tests failed out of 140
逐层对齐: 全部通过
指纹 43d2be1cfd7f
```

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

#### 复盘

- **遇到的问题**：Release 版 decode 131 ms/token，但不知道时间花在哪。机器上没有 `perf`，
  也没法用采样分析器。
- **怎么思考**：
  1. 先算**天花板**再看现状：494 MB 权重 ÷ 38 GB/s ≈ 13 ms/token，现状差 10 倍，说明有大问题可挖，
     而不是在 10% 的边角上抠
  2. 天花板要**实测**：不能用 CPU 手册上的理论带宽，WSL2 + 笔记本内存的实际值要自己测
  3. 对访存密集的算子，耗时（ms）不如**有效带宽**（权重字节 ÷ 耗时）好用：它能直接和上限比，
     一眼看出「离最优还有多远」
  4. 计时器本身不能拖慢程序：不传 `stats` 时指针为空、什么都不做；`generate` 用 RAII 在退出时把
     `profile_` 复位，防止抛异常后留下指向调用方栈上对象的悬挂指针
- **测试程序**：
  - `bench_bw`：AVX2 顺序读 496 MB，扫 1/4/8/14 线程，得到带宽上限 15.6 / 32.5 / 37.5 / 38.7 GB/s
  - `bench_decode`：直接调库里的 `matmul_f32` / `matmul_nt_f32`，用 decode 的真实形状（M=1）
    测每个矩阵乘，得到有效带宽只有 4–7 GB/s
  - `hello_gpt` 第 6 节的分算子表（本步新增）
  - 回归：ctest 122 个用例通过，生成文本不变
- **结果**：decode 时 5 个矩阵乘占 98%，有效带宽 ~4 GB/s；prefill 的有效带宽 ~2 GB/s 比 decode 还低；
  `lm_head` 一个矩阵占 30%
- **怎么解决 / 结论**：数据直接给出了优化顺序 —— 先做矩阵乘（第 2、3 步），注意力、拷贝、layernorm
  加起来不到 2%，现在碰它们是浪费时间。prefill 比 decode 还慢印证了「5 个线程各打包一份权重」的猜测

#### 跟着做

> 时间上第 1 步（换 Release）在前，建议先做完 [第 1 步的「跟着做」](#第-1-步release-构建已完成) 再回来。

**① 看这一步改了什么**（3 个文件，+181 行）：

```bash
git diff f142a75 3227974 --stat
git diff f142a75 3227974 -- include/inference/gpt.hpp   # OpProfile、GenStats::prefill_ops / decode_ops
git diff f142a75 3227974 -- src/gpt.cpp                 # ScopedTimer，forward_impl 里每个算子套一个
```

自己写的话：在 `forward_impl` 里给每个算子包一个 `ScopedTimer`，累加到 `OpProfile` 的对应字段；
`generate` 把 prefill 和 decode 分开累加；`hello_gpt` 打印每项的毫秒、占比，矩阵乘再打印有效带宽
（权重字节 ÷ 耗时）。

**② 测天花板：内存带宽**

```bash
./build-release/bench/bench_bw
```

```
  线程     最快 ms     带宽 GB/s
     1       28.18          17.6
     2       22.70          21.8
     4       16.15          30.7
     6       13.92          35.6
     8       13.41          37.0
    10       13.44          36.9
    12       12.97          38.2
    14       13.34          37.2
```

要看的两件事：最高带宽（这里 ~38 GB/s）和**在几个线程时饱和**（6–8）。
fp32 decode 的天花板 = 494 MB ÷ 38 GB/s ≈ 13 ms/token ≈ 77 token/s。

**③ 测现状：第 0 步时 decode 形状的矩阵乘**

```bash
./build-release/bench/bench_decode 1 3 gemm    # 1 线程、3 次；gemm = 第 0 步时的 kernel 路径
```

```
===== 第 0 步：GEMM 路径（打包）+ lm_head 标量（1 线程，3 次取中位数） =====
  矩阵              ms/token    有效带宽
  qkv (c_attn)        22.391      3.8 GB/s
  attn_proj            6.640      4.3 GB/s
  mlp_fc              28.096      4.0 GB/s
  mlp_proj            27.554      4.1 GB/s
  lm_head             34.627      4.5 GB/s
  合计               119.308   （只算矩阵乘 ≈ 8 token/s）
```

有效带宽 ~4 GB/s，离 38 GB/s 差 10 倍。

**④ 编出第 0 步的版本，看分算子表**

```bash
bench/scripts/build_commit.sh 0
.worktrees/3227974/build-release/examples/hello_gpt "The capital of France is" 60 \
    | sed -n '/^decode 分算子耗时/,/合计/p'
```

```
decode 分算子耗时（59 次 forward，每次平均）:
  算子                    ms    占比    有效带宽
  layernorm            0.119    0.1%
  qkv (c_attn)        24.734   19.1%    3.4 GB/s
  copy (split/kv)      1.164    0.9%
  attention            0.383    0.3%
  attn_proj            7.574    5.8%    3.7 GB/s
  mlp_fc              28.751   22.2%    3.9 GB/s
  gelu                 0.561    0.4%
  mlp_proj            29.006   22.4%    3.9 GB/s
  lm_head             37.038   28.6%    4.2 GB/s
  ...
  合计               129.497
```

和 ③ 对得上：5 个矩阵乘合计 ~127 ms，占 98%。把 `decode` 换成 `prefill` 看 prefill 的表，
有效带宽只有 ~2 GB/s，比 decode 还低（原因 F）。

**⑤ 正确性和开销**：计时器不能改变结果，也不能拖慢程序：

```bash
bench/scripts/check_step.sh 0          # 122 个用例，指纹 43d2be1cfd7f（和起点相同）
bench/scripts/ab_compare.sh baseline 0 # 预期 B/A ≈ 1.0（在噪声范围内）
```

### 第 1 步：Release 构建（已完成）

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j --target hello_gpt
./build-release/examples/hello_gpt "The capital of France is" 60 0.8
```

实测：prefill 219 ms（22.9 token/s），decode 131 ms/token（7.6 token/s）。

#### 复盘

- **遇到的问题**：prefill 5 个 token 要 1376 ms，decode 1.3 token/s，比预想慢两个数量级。
- **怎么思考**：终端输出里出现过 AddressSanitizer 的报错（`Sampler::sample` 里的
  `stack-use-after-scope`），能报 ASan 错误就说明程序是带 sanitizer 编译的。顺着这条线索查
  `build/CMakeCache.txt`，确认 `CMAKE_BUILD_TYPE=Debug`；再看 `CMakeLists.txt`，Debug 配置是
  `-g -fsanitize=address,undefined` 且没有 `-O`，也就是 `-O0`。**先排除环境问题，再分析代码**
- **测试程序**：同一条 `hello_gpt` 命令分别在 `build/` 和 `build-release/` 下跑
- **结果**：prefill 6.3×、decode 5.7×（1376 → 219 ms，751 → 131 ms/token）
- **怎么解决**：之后所有性能测量都只用 `build-release/`；Debug 构建留着抓内存错误
  （ASan 报的那个 bug 就是 Debug 的价值所在）

#### 跟着做

这一步不改代码，只换构建类型。用起点的提交分别编 Debug 和 Release：

**① 复现起点的慢**（Debug 版生成 60 个 token 约 1 分钟）：

```bash
bench/scripts/build_commit.sh baseline Debug
.worktrees/f142a75/build-debug/examples/hello_gpt "The capital of France is" 60 \
    | grep -E "^(prefill|decode) [0-9]+ 个 token"
```

```
prefill 5 个 token: 1687 ms  （3.0 token/s）
decode 60 个 token:  47956 ms  （1.3 token/s）
```

**② 找原因**：确认构建类型，再看 Debug 配置加了什么编译选项：

```bash
grep CMAKE_BUILD_TYPE: .worktrees/f142a75/build-debug/CMakeCache.txt   # CMAKE_BUILD_TYPE:STRING=Debug
grep -n 'CONFIG:Debug' CMakeLists.txt     # -g -fsanitize=address,undefined，没有 -O → -O0
```

**③ 换成 Release**：

```bash
bench/scripts/build_commit.sh baseline              # 默认 Release
.worktrees/f142a75/build-release/examples/hello_gpt "The capital of France is" 60 \
    | grep -E "^(prefill|decode) [0-9]+ 个 token"
```

```
prefill 5 个 token: 313 ms  （15.9 token/s）
decode 60 个 token:  7859 ms  （7.6 token/s）
```

decode 快 6×。prefill 只有一次 forward，噪声很大（几次运行在 195–313 ms 之间跳），看 decode 就够了。

**④ 正确性**：两种构建的贪心文本应该一样：

```bash
bench/scripts/check_step.sh baseline    # 122 个用例，指纹 43d2be1cfd7f
```

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

#### 复盘

- **遇到的问题**：decode 的矩阵乘有效带宽只有 4 GB/s，读代码找到三个原因：M=1 仍走 GotoBLAS 路径，
  每次把整块权重打包一遍；4×16 微内核只有 1 行有效；`lm_head`（M=1）掉进不足 4 行时的标量三重循环。
- **怎么思考**：
  1. 算术强度 M/2 = 0.5 FLOP/字节，每个权重只用一次，**没有复用就不需要打包**。打包是「花一次拷贝，
     换很多次复用」，M=1 时只剩拷贝的代价（权重被读 2 次、写 1 次）
  2. 目标变成「让权重以最快速度顺序流过 CPU」：顺序读、不打包、读满 cache line
  3. FMA 延迟 4 周期、每周期能发 2 条，所以至少要 **8 条独立的累加链** 才不会互相等。
     做法是一次算 4 个输出（x 的加载被 4 行共用）× 每行 2 个累加器
  4. 布局选 `[out, in]`：每个输出是一次连续点积；下一步多线程时可以按输出行切，每个线程读
     W 中互不重叠的连续一段；而且全模型只需维护一个 kernel（也是 llama.cpp 的做法）
- **测试程序**：
  - `gemv_nt_matches_reference`：12 个形状（含 GPT-2 真实形状、K 不是 8 的倍数、N 不是 4 的倍数），
    输出缓冲区末尾放哨兵值，检查有没有越界写
  - `gemv_nt_equals_matmul_nt_row`：GEMV 和整块矩阵乘的某一行结果一致
  - `hello_gpt` 第 3 节逐层对拍、第 6 节分算子表；贪心生成文本逐字不变；ctest 124 个用例通过
- **结果**：decode 126 → 44.9 ms（2.8×），有效带宽 ~12 GB/s，已接近**单线程**上限 15.6 GB/s；
  prefill 也从 224 降到 78–107 ms（M=5 分给 5 个线程，每个线程都是 M=1，现在走 GEMV 不再各自打包）
- **新问题：加载变慢**。加了转置之后加载时间 1.1 s → 2.1 s。原因是朴素转置按行读、按列写，
  写的时候每个元素跨 3–12 KB，每条 cache line 只用 4 字节。**解决**：32×32 分块转置，
  一块读写都在 L1 里完成，降到 ~1.5 s。彻底的办法是在转换脚本里直接存成 `[out, in]`
- **结论**：单线程能做的基本做完了，剩下的 3 倍只能靠多线程 → 第 3 步

#### 跟着做

**① 看这一步改了什么**（5 个文件，+180 行）：

```bash
git diff 3227974 5624734 --stat
git diff 3227974 5624734 -- src/kernel_matmul.cpp    # gemv_nt_f32；matmul_nt_f32_avx2 的边角行改调 GEMV
git diff 3227974 5624734 -- src/gpt.cpp              # to_out_in()：加载时 32×32 分块转置；affine → linear
git diff 3227974 5624734 -- tests/                   # 两个新测试
```

自己写的顺序：先写 `gemv_nt_f32` 和它的测试（和 `matmul_nt_f32_naive` 对拍，形状要覆盖
K % 16 ≠ 0、N % 4 ≠ 0），测试过了再改 `GPT::load` 和 `forward_impl`。

**② kernel 对比：GEMM 路径 vs GEMV**（都是单线程，权重 494 MB，冷缓存）：

```bash
./build-release/bench/bench_decode 1 10 gemm,gemv1
```

```
===== 第 0 步：GEMM 路径（打包）+ lm_head 标量（1 线程，3 次取中位数） =====
  合计               119.308   （只算矩阵乘 ≈ 8 token/s）

===== 第 2 步：GEMV，单线程（1 线程，10 次取中位数） =====
  矩阵              ms/token    有效带宽
  qkv (c_attn)         6.414     13.2 GB/s
  attn_proj            2.052     13.8 GB/s
  mlp_fc               8.446     13.4 GB/s
  mlp_proj             8.797     12.9 GB/s
  lm_head             11.783     13.1 GB/s
  合计                37.492   （只算矩阵乘 ≈ 27 token/s）
```

有效带宽 13 GB/s，和 `bench_bw` 的单线程 17.6 GB/s 比已经到了 75%。

**③ 加载时转置的代价**：

```bash
./build-release/bench/bench_transpose
```

```
  naive   按行读、按列写                   485 ms
  blocked 32×32 分块                       209 ms   （结果和 naive 一致）
  tensor  transpose().contiguous()        1296 ms
```

**④ 正确性**：

```bash
bench/scripts/check_step.sh 2                               # 124 个用例（+2），指纹不变
ctest --test-dir build-release -R 'gemv_nt' --output-on-failure   # 只跑这一步新增的两个测试
```

**⑤ 端到端**：

```bash
bench/scripts/ab_compare.sh 0 2
```

进度表里第 0 步 8.5 token/s → 第 2 步 24.9 token/s，预期 B/A ≈ 2.9×。

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

#### 复盘

- **遇到的问题**：单核带宽只有 15.6 GB/s，decode 却只用 1 个核（M=1 没法按行切）；
  GEMM 路径每次矩阵乘都新建一批 `std::jthread`。
- **怎么思考**：
  1. M=1 没法按行切，就**按输出列 N 切**：每个线程负责 `y[n0:n1]`，读 W 的连续若干行，互不重叠，
     没有 false sharing
  2. 一个 token 49 次矩阵乘，修好后一次 c_attn 只要 ~0.2 ms，而建线程是几十微秒级 → 必须常驻线程池
  3. 矩阵乘之间的间隔只有几十微秒，纯条件变量唤醒要 10–50 µs，所以**先自旋 2 ms 再睡**
  4. P 核 / E 核快慢不一，静态均分会等最慢的那个 → 切成约 4 × 线程数的小块，用原子变量**动态领任务**
  5. W 小于 128 KB 时同步开销大于收益，直接串行
- **测试程序**：
  - `tests/test_thread_pool.cpp`：`thread_pool_runs_every_task_once`（每个任务恰好执行一次）、
    `thread_pool_many_rounds`（5000 批连续发，专抓批次之间的竞态）、
    `thread_pool_results_visible_after_return`（内存可见性）、`thread_pool_nested_call_runs_serially`、
    `thread_pool_sleeping_workers_wake_up`（睡眠后能被唤醒）
  - `matmul_nt_parallel_small_m_splits_columns`：按列切路径 × 4 种线程数，含尾部哨兵
  - **ThreadSanitizer**：用 `-fsanitize=thread` 单独编译 `test_thread_pool` 跑一遍。手写原子操作和
    内存序时，普通测试「碰巧通过」不能说明没有数据竞争
  - `INFER_NUM_THREADS=1/4/6/8/10/12/14` 扫线程数，每个配置跑 3 次；对比 6 线程和 14 线程的分算子表
  - `hello_perf` 复测大矩阵 GEMM，确认改用线程池后没有退化（N=1024 225–337 GFLOPS）
  - ctest 131 个用例通过
- **遇到的问题 2：线程越多越慢**（14 线程 9.4 token/s，比单线程还慢 3 倍）。
  - 怎么定位：分算子表显示大矩阵 `lm_head` 在 14 线程下仍有 25 GB/s，而实际只需 0.1 ms 的
    `attn_proj` 每次要 ~2 ms。大矩阵正常、小矩阵变慢 → **是每次调用的固定开销，不是带宽问题**
  - 原因：第一版要求调用者等**每个工作线程**报到。14 个线程占满 14 个逻辑核自旋，
    WSL2 宿主机或系统进程把任何一个线程换下去一个时间片，所有人都得等它
  - 解决：调用者只等**完成的任务数**。难点是迟到的线程可能读到下一批改了一半的任务描述
    （比如读到新的任务总数，配上旧的批次号）。把「批次号 24 位 | 任务总数 20 位 | 下一个任务编号 20 位」
    打包进**一个** 64 位原子变量，用 CAS 领任务：CAS 成功就证明三者是同一时刻的一致快照
  - 结果：14 线程 9.4 → ~21 token/s，但最优仍在 6–8 线程（带宽饱和、LP-E 核掉队、自旋抢超线程）
- **怎么解决（线程数）**：kernel 库的默认值不改（大 GEMM 用 14 线程更快），由应用 `hello_gpt`
  默认选一半逻辑核，可用 `INFER_NUM_THREADS` 覆盖 —— 和 llama.cpp 的 `-t` 参数一个思路
- **结果**：decode 44.9 → 19.4–22 ms，矩阵乘有效带宽 ~30 GB/s（上限的 80%）

#### 跟着做

**① 看这一步改了什么**（7 个文件，+443 行，其中线程池 238 行）：

```bash
git diff 5624734 89c4702 --stat
git show 89c4702:include/inference/thread_pool.hpp     # 接口和设计注释
git show 89c4702:src/thread_pool.cpp                   # 64 位打包状态 + CAS 领任务 + 自旋 2 ms 再睡
git diff 5624734 89c4702 -- src/kernel_matmul.cpp      # M < 线程数时按输出列 N 切
```

自己写的顺序：先写线程池和 `tests/test_thread_pool.cpp`，**先过 TSan** 再接进矩阵乘。
第一版可以先写成 `bench/bench_pool.cpp` 里的 `BarrierPool` 那样（等每个线程报到），
跑一遍 ③ 亲眼看到它的问题，再改成只等任务完成。

**② kernel 对比：单线程 GEMV vs 多线程按列切**：

```bash
./build-release/bench/bench_decode 7 20 gemvN    # 和第 2 步 ② 的 gemv1 对比
```

```
===== 第 3 步：GEMV，多线程按列切（7 线程，20 次取中位数） =====
  矩阵              ms/token    有效带宽
  qkv (c_attn)         2.472     34.4 GB/s
  attn_proj            0.884     32.0 GB/s
  mlp_fc               3.227     35.1 GB/s
  mlp_proj             3.204     35.3 GB/s
  lm_head              4.338     35.6 GB/s
  合计                14.125   （只算矩阵乘 ≈ 71 token/s）
```

37.5 → 14.1 ms，有效带宽到了 `bench_bw` 上限的 ~92%。

**③ 复现「线程越多越慢」：两种线程池对比**

```bash
./build-release/bench/bench_pool
```

```
  线程 |   v1 中位 µs  v2 中位 µs |    v1 p99 µs   v2 p99 µs |    v1 ms/tok   v2 ms/tok
     1 |          0.1         0.1 |          0.1         0.1 |        34.80       34.54
     4 |          2.3         4.3 |         13.4        10.7 |        15.71       15.60
     8 |          5.0        10.9 |         34.8        26.1 |        15.86       14.46
    12 |          6.8        16.9 |        131.5        42.1 |        17.57       16.40
    14 |       2057.1        16.2 |       2284.7        62.0 |        94.44       31.85
```

v1 = 第一版（等每个线程报到），v2 = 现在的库（只等完成数）。14 线程（= 逻辑核数）时 v1 每次调用
要 ~2 ms —— 这是操作系统调度时间片的量级：有一个线程被换下去，调用者要一直等到它回来。
v2 不受影响，但 decode 也从 14.5 涨到 31.9 ms：线程占满所有逻辑核，掉队和抢超线程照样存在。
这个现象和当时机器上的负载有关，如果 14 线程时 v1 没变慢，开个浏览器再跑一次。

**④ 扫线程数**：

```bash
bench/scripts/sweep_threads.sh 3 "1 4 6 7 8 10 14"
```

```
threads   decode tok/s   ms/token   各轮 tok/s
      1           28.4       35.2   29.8 28.4 28.3
      4           46.9       21.3   54.4 31.6 46.9
      6           44.8       22.3   53.8 44.8 38.4
      7           45.5       22.0   48.1 42.9 45.5
      8           46.9       21.3   44.4 46.9 46.9
     10           37.7       26.5   46.2 28.3 37.7
     14           19.8       50.5   19.8 19.2 22.9
```

注意「各轮」那一列：同一个配置三次能差 1.7 倍（4 线程 31.6–54.4），只跑一次得不出结论。
最优在 4–8 线程，14 线程只有一半 —— 这就是 `hello_gpt` 默认用一半逻辑核的依据。

**⑤ 数据竞争检查**：

```bash
bench/scripts/tsan_thread_pool.sh     # 第一次约 1 分钟
```

```
== test_thread_pool
5 个用例，0 个失败
== test_kernel_matmul
32 个用例，0 个失败
== test_kernel
27 个用例，0 个失败

ThreadSanitizer 没有报告数据竞争
```

**⑥ 正确性和端到端**：

```bash
bench/scripts/check_step.sh 3          # 131 个用例（+7），指纹不变
bench/scripts/ab_compare.sh 2 3 4
```

```
A = 2          decode 中位数  26.85 token/s（37.2 ms/token）
B = 3          decode 中位数     47 token/s（21.3 ms/token）
B / A = 1.75×
```

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

#### 复盘

- **遇到的问题**：矩阵乘到了 ~30 GB/s，剩下的串行零碎开销（拷贝 1.4 + gelu 0.8 + 注意力 0.5 +
  layernorm 0.15 ms）占 decode 的 15%。其中 KV-Cache 拷贝是 O(序列长度) 的，越生成越慢。
- **怎么思考**：
  1. 优先消灭**随序列长度增长**的开销：KV-Cache 本来就是为了避免 O(n) 重复工作，每层又整段拷一遍等于白做
  2. 不拷贝的关键是让 kernel 接受**行距（stride）**：q 在 `qkv` 里的行距是 3C，k/v 在缓存里的行距是 C
  3. 预分配工作区前先确认每个 kernel 能不能**原地算**：layernorm、gelu、add 的每个输出只依赖同位置的输入，
     所以 `out == in` 是安全的
  4. gelu 0.8 ms 的原因是每个 token 约 3.7 万次标量 `std::tanh` 调用。标准库没有 SIMD 版 → 自己实现 8 路 `exp`，
     再用 `tanh(u) = 1 - 2/(e^{2u}+1)`
- **测试程序**：
  - `kernel_gelu_simd_matches_std_tanh`：24001 个点覆盖 [-12, 12]，含原地计算。原有的 gelu 测试只有
    2–5 个元素，**根本走不到 8 个一组的 AVX2 路径**，新 kernel 等于没被测过
  - `kernel_attention_strided_matches_contiguous`：带行距版本和连续版本逐元素一致
  - 逐层对拍 + 贪心文本不变；ctest 133 个用例（第一次跑还是 131，重新配置 CMake 后新用例才注册上）
  - **git worktree A/B**：`git worktree add /tmp/li_step3 89c4702` 编出第 3 步的版本（`data/` 用软链接），
    和第 4 步**交替**各跑 5 轮，取中位数（现在一条命令：`bench/scripts/ab_compare.sh 3 4`）
  - 1 线程 vs 7 线程对照：同一时段比较串行算子的耗时
- **遇到的问题 2：测量噪声**。第一次跑 decode 总耗时在 43–85 ms 之间跳，单线程三次是 95 / 38.5 / 35 ms。
  机器发热降频，前后两次运行不可比。**解决**：把改之前的版本编出来，和改之后的交替运行，
  让两者经历同样的降频状态；只比较中位数
- **遇到的问题 3：attention 反而变慢了 0.2 ms**。原因：以前的 `ck/cv` 拷贝顺带把 KV 数据拉进了缓存，
  注意力读的是热数据；拷贝删掉后，缓存未命中算到了注意力头上。看「copy + attention」合计是 1.9 → 0.9 ms，
  仍是净收益。**教训：删掉一步时，它的副作用会转移到下一步，分项数据要合起来看**
- **遇到的问题 4：多线程时串行算子慢 2–3 倍**（7 线程下 attention 1.23 ms，1 线程 0.52 ms）。
  自旋的工作线程抢主线程的超线程资源，或者主线程被调度到 E 核。**解决**：留给第 5 步把注意力并行化
- **结果**：copy 降 40 倍、gelu 降 7 倍，串行部分 3.5 → 1.2 ms，decode 中位数 23.4 → 19.1 ms/token

#### 跟着做

**① 看这一步改了什么**（6 个文件，+260 / -58 行）：

```bash
git diff 89c4702 c98ce1b --stat
git diff 89c4702 c98ce1b -- include/inference/gpt.hpp   # GPT::Workspace：x、h、qkv、att、fc 五块缓冲区
git diff 89c4702 c98ce1b -- src/gpt.cpp                 # forward_impl 原地计算；K/V 直接写进 cache
git diff 89c4702 c98ce1b -- src/kernel.cpp              # AVX2 exp + gelu_f32
git diff 89c4702 c98ce1b -- src/kernel_attention.cpp    # attention_f32_strided：q/k/v 各自带行距
```

自己写的顺序：先把 attention 改成接受行距（删掉 `ck/cv` 拷贝，收益最大），再做工作区和原地计算，
最后做 SIMD gelu。每做完一项跑一次 `check_step.sh`，指纹变了马上就知道是哪一项。

**② 这一步要消灭的开销有多大**：

```bash
./build-release/bench/bench_serial_ops
```

```
===== 1. gelu：每 token 12 层 × 3072 个数 =====
  标量 std::tanh     257.1 µs/token
  AVX2                20.6 µs/token   （快 12.5 倍，最大误差 3.2e-07）

===== 2. KV-Cache 拷贝：第 4 步之前每层把整段缓存拷出来 =====
    已生成 拷贝 ms/token 每 token 拷贝量
        64         1.189          4.7 MB
       256         7.788         18.9 MB
       512        16.139         37.7 MB
      1024        41.336         75.5 MB

===== 3. 每个算子新分配一个张量：一个 token 约 160 次 Tensor::zeros =====
  156 次分配 + 清零: 17.7 µs/token
```

拷贝随长度线性增长：生成到第 1024 个 token 时每 token 光拷贝就 41 ms，比整个 decode 还贵。
分配本身（17.7 µs）不值得为它费劲，做工作区主要是为了顺带去掉拷贝。

**③ 正确性**：

```bash
bench/scripts/check_step.sh 4          # 133 个用例（+2），指纹不变
```

如果用例数还是 131：新测试没注册，见本节开头的「注意」。

**④ 端到端**：60 个 token 时 KV 拷贝只有 ~1 ms，降频噪声比它还大，必须交替跑：

```bash
bench/scripts/ab_compare.sh 3 4 6
```

```
A = 3          decode 中位数   46.6 token/s（21.5 ms/token）
B = 4          decode 中位数   50.7 token/s（19.7 ms/token）
B / A = 1.09×
```

单次运行里 A 在 44.4–51.5、B 在 48.3–55.0 之间跳，两个范围是重叠的。只跑一次的话，
完全可能得出「第 4 步更慢」的结论。

**⑤ 多线程时串行算子变慢**（遇到的问题 4）：

```bash
for t in 1 7; do
    echo "== $t 线程"
    INFER_NUM_THREADS=$t .worktrees/c98ce1b/build-release/examples/hello_gpt "The capital of France is" 60 \
        | sed -n '/^decode 分算子耗时/,/合计/p' | grep -E "layernorm|attention|gelu"
done
```

```
== 1 线程
  layernorm            0.124    0.3%
  attention            0.756    1.6%
  gelu                 0.095    0.2%
== 7 线程
  layernorm            0.147    0.3%
  attention            1.168    2.6%
  gelu                 0.131    0.3%
```

这三个算子在两种配置下都只在主线程上跑，代码完全一样，7 线程时却慢 1.2–1.5 倍。

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

**实测 1：注意力 kernel 本身**（[`bench/bench_attn.cpp`](../bench/bench_attn.cpp)，热缓存，每层一次，µs）：

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

冷缓存对比（[`bench/bench_kvlayout.cpp`](../bench/bench_kvlayout.cpp)：12 层各一份 K/V，总量 150 MB，远超 12 MB 的 L3，
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

#### 复盘

- **遇到的问题**：注意力成了最大的串行算子，是标量三重循环，又受第 4 步发现的线程干扰拖慢；
  上下文越长，它每个 token 要读的 K/V 越多（T=1024 时约 75 MB，和权重同一量级）。
- **怎么思考**：
  1. `head_size = 64` 正好是 8 个 `__m256`：点积、softmax、加权求和 V 都能整段向量化
  2. 每个（头, 查询）互相独立 → decode 有 12 个任务，prefill 有 12 × T 个，交给线程池
  3. 原来的标量版改名 `attention_f32_naive` **留作参考实现**，测试和 benchmark 都拿它对比
  4. 并行有同步开销，小任务串行更快 → 要一个按工作量判断的阈值，而且**阈值要实测，不要拍脑袋**
- **测试程序**：
  - `kernel_attention_fast_matches_naive`：10 个形状 × 1/4 线程，含 decode、prefill、分段 prefill、
    head_size 不是 8 的倍数、Tk = 1024
  - `bench_attn`：热缓存，同一进程里交替跑朴素版 / SIMD 1 线程 / SIMD 7 线程，扫 decode Tk 和 prefill T
  - `hello_gpt` 生成 900 个 token，看长上下文下注意力占 decode 的比例
  - `bench_kvlayout`：**冷缓存**，12 层各一份独立的 K/V（总量远超 12 MB 的 L3），
    同一进程交替测两种布局；再强制总是并行，扫 Tk 找并行的交叉点
  - ctest 134 个用例通过
- **遇到的问题 1：SIMD 只快了 ~2 倍，不是 8 倍**。短序列时每个 key 只做 4 次 FMA，之后 8 → 1 的
  水平求和有 ~10 条指令的依赖链；长序列时受内存带宽限制。没有继续优化，留作练习（一次算 8 个 key 再转置归约）
- **遇到的问题 2：并行阈值太低**。第一版 6.5 万（decode Tk ≈ 85 就开始并行），结果 Tk=256 时
  7 线程比单线程还慢：12 个头分给 7 个线程要跑两轮，每个任务才 3 µs。热缓存、冷缓存各扫一遍，
  交叉点都在 Tk 384–512，阈值定为 30 万（≈ Tk 390）
- **遇到的问题 3：microbenchmark 和真实模型对不上**。900 token 时注意力实际 4.5 ms/token，
  按 microbenchmark 推算只该 ~1.5 ms。原因是 microbenchmark 反复读同一块数据（一直热在 L3），
  真实模型每 token 读 12 层共 ~33 MB 的冷数据。旧布局 `[位置][C]` 下一个头的 64 个数夹在每行 768 个数中间，
  每隔 3 KB 跳着取 256 字节，硬件预取器跟不上。**解决**：KV-Cache 改成 `[头][位置][64]`，冷缓存下快 ~2 倍
- **遇到的问题 4：实验自己出了错**。改了 `bench_kvlayout` 的 Tk 列表却忘了重新编译，读到的还是旧数据；
  跨进程比较 1 线程和 7 线程又受降频影响。**解决**：重新编译，并改成在同一进程里切换线程数做对比
- **遇到的问题 5：降频让绝对值不可比**。**解决**：用同一次运行里注意力和 `mlp_fc` 的**耗时比值**做参照
  （0.93 → 0.55–0.59），`mlp_fc` 这一步没改，它就是「标尺」
- **结果**：长序列 decode 的注意力 kernel 快 ~2.9×，长 prompt prefill 快 7–10×；KV 布局再快 ~2×；
  900 token 时注意力占比 16.7% → ~11%；60 token 端到端 decode 15.6–21 ms/token

#### 跟着做

**① 看这一步改了什么**（8 个文件，+382 / -83 行）：

```bash
git diff c98ce1b 3600536 --stat
git diff c98ce1b 3600536 -- src/kernel_attention.cpp    # SIMD 点积 / softmax / axpy，按 (头, 查询) 并行
git show 3600536:src/simd_math.hpp                      # exp256_ps / hsum256_ps / hmax256_ps
git diff c98ce1b 3600536 -- include/inference/kernel.hpp  # HeadView{data, row_stride, head_stride}
git diff c98ce1b 3600536 -- include/inference/gpt.hpp   # KVCache 改成 [头][位置][64]
```

自己写的顺序：先把标量版改名 `attention_f32_naive` 留作参考，写 SIMD 单线程版并对拍；
再加并行（先不设阈值，看 ③ 的结果再定）；最后改 KV 布局（要改 `HeadView` 和 `KVCache` 两处）。

**② kernel 本身（热缓存）**：

```bash
./build-release/bench/bench_attn
```

```
形状                    朴素    SIMD 1线程   SIMD 多线程    总加速
decode Tk=64            25.3          11.0          10.9      2.3×
decode Tk=256          100.3          45.5          46.5      2.2×
decode Tk=512          222.1         110.3          84.2      2.6×
decode Tk=1024         574.6         303.4         236.7      2.4×
prefill T=128         4238.6        1853.6         590.5      7.2×
prefill T=512        93781.2       40635.2       11018.8      8.5×
```

**③ KV 布局和并行阈值（冷缓存）**：

```bash
./build-release/bench/bench_kvlayout          # 第 3 部分要加载模型、生成 5×2 次 200 个 token，约 1 分钟
```

```
===== 1. 两种布局，ms / token（12 层），同一进程交替跑，20 次取中位数 =====
    线程      Tk      按位置        按头      加速
       1     512       4.398       3.312     1.33×
       1    1024      10.249       6.688     1.53×
       7     512       2.190       1.417     1.55×
       7    1024       4.341       2.699     1.61×

===== 2. 并行阈值：强制总是并行，1 线程 vs 7 线程（按头布局） =====
      Tk     1 线程 ms     多线程 ms      工作量  结论
      32         0.105         0.075       24576  并行划算
     128         0.426         0.230       98304  并行划算
     384         1.839         0.990      294912  并行划算
    1024         5.657         2.738      786432  并行划算

===== 3. 端到端：真实模型生成 200 个 token，默认阈值 vs 总是并行（7 线程） =====
                 注意力 ms/token decode ms/token
默认阈值                   1.376          23.913
总是并行                   0.870          24.350
```

布局：按头存放快 1.3–1.6×（当初测的是 ~2×，这次机器比较凉，绝对值都小一些）。

> **复现时的新发现：并行阈值的结论不稳**。当初在冷缓存下扫出来交叉点在 Tk 384–512，
> 阈值定为 30 万。这次重写 `bench_kvlayout` 后，第 2 部分在**所有** Tk 上都是并行更快，
> 连 Tk=32 也是。区别在于测法：bench 里切换线程数后会先空跑一次，工作线程正在自旋，
> 测到的是并行的最好情况；当初的测法里工作线程可能已经睡着，要付唤醒的代价。
> 第 3 部分是最终裁判：真实模型里「总是并行」让注意力从 1.38 降到 0.87 ms/token，
> 但 decode 总耗时没有可见的变化（23.9 vs 24.4，在噪声范围内）—— 注意力只占 decode 的 ~5%。
> 所以库里的阈值暂时没改。
>
> 第 3 部分还踩了一个坑：第一版是「每轮先跑默认、再跑总是并行」，结果总是并行的 decode
> 慢了 6 ms（26.4 vs 32.6）。改成 ABBA 顺序（奇数轮反过来）之后差距消失。
> 总是后跑的那一方更容易赶上降频，**交替不够，还要轮换先后**。

**④ 长上下文端到端**：注意力的收益要上下文够长才看得出来：

```bash
bench/scripts/ab_compare.sh 4 5 2 900     # 2 轮，每次 900 个 token，约 2 分钟
```

```
A = 4          decode 中位数   28.7 token/s（34.8 ms/token）
B = 5          decode 中位数  38.65 token/s（25.9 ms/token）
B / A = 1.35×
```

对比 60 个 token 时第 4、5 步几乎一样快（进度表 55.8 vs 56.3 token/s）。

**⑤ 正确性**：

```bash
bench/scripts/check_step.sh 5          # 134 个用例（+1），指纹不变
```

### 第 6 步：prefill 的小 M 问题

> **状态：没有单独做**。第 2 步让 M < 4 的行走 GEMV，第 3 步让 M < 线程数时按列 N 切，
> M=5 的 prefill 已经走了「按列切 + GEMV」路径，从 224 ms 降到 ~25 ms，基本达到下面的预期。
> 剩下没做的是长 prompt（M 大）时 GEMM 的多线程切分改成按 N 切，等需要处理长 prompt 时再做。

prefill 的矩阵乘 M = prompt 长度。按 1.3 节的分析：

- **M 小（< 16 左右）**：仍然是访存密集，应该走「按 N 切的多线程 + 一次读权重服务 M 行」的路径。
  可以把第 2 步的 GEMV 扩展成一次处理 M 个 x 向量（读一次 W 的一行，和 M 个 x 做点积），
  即 **GEMM with small M**，llama.cpp 里叫 `vec_dot` 批处理
- **M 大**：走现有的 GotoBLAS GEMM，但多线程切分要改：M 小于线程数时不要按行切。
  可以按 N 切（每个线程打包 B 的不同列块，不再冗余）

另外 `lm_head` 在 prefill 时已经做了 `last_only`，M=1，修完第 2/3 步自动受益。

**预期**：5 token 的 prefill 从 219 ms 降到和一次 decode 差不多（~20 ms）。

#### 跟着做（验证不需要单独做）

```bash
bench/scripts/ab_compare.sh baseline 3 3     # 每一行都打印 prefill，看 A、B 两列
```

进度表里 prefill（5 个 token）：起点 195 ms → 第 2 步 68 ms → 第 3 步 31 ms → 第 5 步 26 ms，
和一次 decode（17.8 ms）在同一量级，已经达到上面的预期。想看长 prompt 下 GEMM 路径的问题，
可以把 prompt 换成几百个 token 的长文本，看 `prefill 分算子耗时` 里矩阵乘的有效带宽。

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

#### 已实现

**最终方案**：每层的 4 个矩阵（c_attn、attn_proj、mlp_fc、mlp_proj）用 Q8_0（权重和激活都是
int8），lm_head 用 fp16 权重 + fp32 激活，embedding 查表保持 fp32。每 token 读的权重
494 MB → 173 MB。入口是 `GPT::quantize_q8_0()`，加载后调用一次（~0.7 s）。
它等价于 `GPT::quantize(QuantOptions{})`；做实验时可以用 `QuantOptions` 单独开关每类矩阵、
给 lm_head 选 fp32 / fp16 / Q8_0，或者用 `weight_only = true` 只看权重的量化误差。
量化会释放 fp32 权重，所以每个模型对象只能量化一次。

**代码**：

- `src/kernel_quant.cpp`：
  - `quantize_row_q8_0` / `dequantize_row_q8_0`：每 32 个数一组，`d = max|x| / 127`，
    scale 用 fp32 存，所以是 36/32 = 1.125 字节/权重（llama.cpp 用 fp16 存 scale，是 1.0625）
  - `matmul_nt_q8_0`：先把 A 的每行量化成 Q8_0（一次，被 W 的全部 N 行复用），
    再按输出列切块交给线程池，块内跑 `gemv_q8_0`
  - `gemv_q8_0`：AVX2 整数点积。AVX2 没有「有符号 × 有符号」的 8 位乘加，只有
    `maddubs`（无符号 u8 × 有符号 s8，相邻两对相加成 s16）。技巧是把 x 的符号挪到 w 上：
    `x·w = |x| · (w × sign(x))`，用 `_mm256_sign_epi8` 实现。两对之和最大
    2 × 127 × 127 = 32258 < 32767，s16 不会饱和。再用 `madd(·, 1)` 加成 s32、转 float，
    乘上两边的 scale 累加。4 行一组共用 x 的加载
  - `matmul_nt_f16` / `gemv_f16`：fp16 权重用 `_mm256_cvtph_ps`（F16C）转 fp32 再 FMA，
    结构和 fp32 的 GEMV 一样（4 行 × 2 条 FMA 链）
- `include/inference/gpt.hpp`：`QMatrixQ8 { rows, cols, q, scale }`，每层多 4 个 `*_wq`，
  量化后 fp32 版本被释放；`lm_head_f16_` 是 wte 的 fp16 拷贝
- `src/gpt.cpp`：`linear_bias` 里的 `matmul_weight` 根据有没有量化版选择 kernel
- 测试：Q8_0 往返误差 ≤ scale/2、离群值只影响自己的组；`matmul_nt_q8_0` 和
  「量化再反量化后做 fp32 矩阵乘」逐元素一致（覆盖 M=1、M>1、N 非 4 的倍数、1/4 线程）；
  和原始 fp32 的相对 RMS 误差 0.54%；fp16 转换和 `matmul_nt_f16` 同样对拍
- `examples/hello_gpt.cpp` 第 7 节：量化前后在一段 83 token 的文本上测困惑度、
  对比 logits 和 top-10，再生成一次并打印分算子耗时

**实测 1：精度，以及 lm_head 的离群值问题**

第一版把 lm_head 也做成了 Q8_0，困惑度涨了 **7.8%**，远超预期。逐个矩阵排查
（只量化某一类，其余保持 fp32；括号里是只量化权重、激活保持 fp32 的结果）：

```
只量化        困惑度变化
c_attn        +0.10%  (-0.19%)
attn_proj     -0.13%  (-0.11%)
mlp_fc        -0.01%  (+0.02%)
mlp_proj      +0.27%  (+0.18%)
lm_head       +8.01%  (+2.07%)
```

层内 48 个矩阵量化几乎无损（±0.3%，在这么短的文本上就是噪声），问题全在 lm_head。
看它的输入（`ln_f` 的输出）：

```
普通通道 max|x| 的中位数     0.76
通道 496                    241.8
通道 430                    112.5
通道 36                      94.1
```

通道 496 比普通通道大 **300 倍**。按 32 个一组量化激活时，它所在那组的步长是
241.8 / 127 ≈ 1.9，同组其它 31 个 ~0.76 的值只能变成 0 或 ±1.9，信息基本丢光。
即使激活保持 fp32，这个通道也会把 wte 第 496 列的权重量化误差放大 240 倍（所以只量化权重也有 +2%）。
层内矩阵没这个问题，是因为误差会被下一个 LayerNorm 重新归一化，而 lm_head 的输出直接就是 logits。

这就是 LLM.int8() 论文说的 **激活离群值（outlier features）**，大模型里普遍存在。
常见对策：离群通道单独用高精度算（LLM.int8()）、把离群值的幅度挪到权重上（SmoothQuant），
或者像这里一样让敏感的矩阵保持高精度。llama.cpp 的 k-quant 也会给 output.weight 用更高的位宽。

```
方案                               困惑度变化   每 token 读    decode（同一轮）
全部 fp32                          —            494 MB        53 token/s
全部 Q8_0                          +7.84%       139 MB        ~110 token/s
层内 Q8_0 + lm_head fp32           +0.06%       250 MB        ~70 token/s
层内 Q8_0 + lm_head fp16（采用）    +0.08%       173 MB        ~103 token/s
```

采用的方案 top-1 和 top-10 都和 fp32 完全一致，prompt 最后位置的 logits 最大误差 0.67（量级 ±100）。

**实测 2：速度**（7 线程，交替运行 5 次，去掉第一次降频的结果）：
decode fp32 51–57 token/s → 量化后 95–108 token/s，约 **2×**；prefill 5 token ~34 → ~19 ms。

分算子（decode，每 token，同一次运行）：

```
算子          fp32 ms   量化后 ms   加速   量化后有效带宽
qkv           3.82      2.02        1.9×   11.8 GB/s
attn_proj     1.47      0.80        1.8×    9.9 GB/s
mlp_fc        4.87      2.24        2.2×   14.2 GB/s
mlp_proj      4.91      2.33        2.1×   13.7 GB/s
lm_head       5.89      3.86        1.5×   20.0 GB/s    (fp16)
合计          21.97     12.44       1.8×
```

字节数降到 1/3.6，层内矩阵却只快了 ~2×，有效带宽从 ~23 GB/s 掉到 ~12 GB/s。
当时的结论是「Q8_0 的 GEMV 每 32 个权重要走 sign → maddubs → madd → cvt → 乘 scale → FMA，
瓶颈从带宽转到了计算」。

> **更正（写 `bench_decode` 复现时发现）**：这个结论是错的。把 kernel 单独拿出来测
> （`bench_decode 7 20 gemvN,q8`，12 层独立的权重、冷缓存，含每次调用的激活量化和线程池分发）：
>
> ```
>               fp32 (gemvN)          层内 Q8_0 + lm_head fp16
> qkv           2.472 ms  34.4 GB/s   0.789 ms  30.3 GB/s
> attn_proj     0.884 ms  32.0 GB/s   0.324 ms  24.5 GB/s
> mlp_fc        3.227 ms  35.1 GB/s   1.048 ms  30.4 GB/s
> mlp_proj      3.204 ms  35.3 GB/s   1.058 ms  30.1 GB/s
> lm_head       4.338 ms  35.6 GB/s   2.171 ms  35.6 GB/s
> 合计         14.125 ms              5.390 ms   （2.6×）
> ```
>
> 单独测时 Q8_0 有 ~30 GB/s，和 fp32 差不多，**kernel 本身仍然是访存密集的**，算力没有成为瓶颈。
> 差距出在**放进模型之后**。同一时段 `hello_gpt` 的分算子表：
>
> ```
>               单独测     模型里     模型里 / 单独测
> fp32 qkv      2.47       2.68       1.08×
> Q8_0 qkv      0.79       1.17       1.48×
> Q8_0 attn_proj 0.32      0.53       1.63×
> Q8_0 mlp_fc   1.05       1.48       1.41×
> Q8_0 mlp_proj 1.06       1.60       1.51×
> ```
>
> fp32 放进模型只慢 8%，Q8_0 慢 40–60%，折合每次调用多 ~30 µs。一次 Q8_0 的 c_attn 只要 ~65 µs，
> 每次调用的固定开销（激活量化在主线程上串行做、线程池分发、两次矩阵乘之间主线程做 layernorm /
> 注意力时工作线程的状态）占比比 fp32 大得多。具体是哪一项还**没有定位**，可以作为练习：
> 在 `bench_decode` 的两次矩阵乘之间插入真实的串行工作，看单独测的数字会不会向模型里的靠拢。
>
> 教训：**下「计算受限」的结论之前，先把 kernel 单独拿出来，用冷缓存测一次有效带宽**。

可以继续做的：

- 每两组再转一次 float（把两组的 int32 先加起来，不同 scale 时不能直接加，需要换数据布局）
- 用 AVX-VNNI 的 `_mm256_dpbusd_epi32`（这颗 Core Ultra 支持）一条指令完成 maddubs + madd
- scale 改存 fp16，权重字节再少 6%
- 层内矩阵用 Q4_0，lm_head 保持 fp16

#### 复盘

- **遇到的问题**：矩阵乘已经到了 ~30 GB/s，离 38 GB/s 的天花板只差 20%。fp32 下再怎么优化 kernel
  也快不了多少，只能**每个 token 少读一些字节**。
- **怎么思考**：
  1. 先选 Q8_0：8 位精度通常几乎无损，格式简单，适合第一次做量化
  2. 只量化权重（读 int8 → 转 float → FMA）实现最简单，但每个权重要多几条转换指令；
     把激活也按组量化成 int8，就能用整数 SIMD 一次处理 32 个数（llama.cpp 的做法）
  3. AVX2 只有 `maddubs`（u8 × s8），没有 s8 × s8 → 把 x 的符号挪给 w；
     要先算清楚 s16 会不会饱和（最大 32258 < 32767）
  4. 测试**分两层**：kernel 对不对（和「量化再反量化后做 fp32 乘法」比，应该只差浮点舍入）、
     量化误差有多大（和原始 fp32 比）。两件事混在一起测，出了误差分不清是 bug 还是量化本身
  5. 精度看三个指标：一段固定文本上的**困惑度**、logits 最大误差、top-1 / top-10 是否一致
- **测试程序**：
  - `q8_0_quantize_roundtrip`：每个数的还原误差 ≤ scale/2；全 0 的组不出 NaN；离群值只影响自己那组
  - `q8_0_matmul_matches_dequantized_reference`：6 个形状 × 1/4 线程（M=1、M>1、N 不是 4 的倍数）
  - `q8_0_matmul_error_vs_fp32`：相对 RMS 误差 0.54%（< 2%）
  - `q8_0_rejects_k_not_multiple_of_32`
  - `f16_convert_roundtrip`、`f16_matmul_matches_converted_reference`
  - 用 `-mno-f16c` 编译一遍，确认软件回退路径能编译、测试能过。**复现时发现当初这一步没做对**：
    fp16 的 SIMD 路径只判断了 AVX2，没判断 F16C，`-mno-f16c` 下 `_mm256_cvtph_ps` 编译失败。
    已改成 `INFER_SIMD_AVX2 && defined(__F16C__)` 才走 SIMD（见下面「跟着做」⑦）
  - `hello_gpt` 第 7 节：量化前后的困惑度（83 token 的固定段落）、logits 误差、top-10，再生成一次看速度
  - **诊断开关**：排查时是在 kernel 里临时加的环境变量 `INFER_Q8_WONLY=1`（把 int8 权重反量化回
    fp32 走 fp32 kernel，隔离「只有权重误差」）和 `INFER_Q8_SKIP=qkv,aproj,fc,mproj,lm`（按矩阵类型跳过量化）。
    后来改成了正式的接口 `GPT::quantize(QuantOptions)`：每类矩阵一个开关、`lm_head` 选 fp32 / fp16 / Q8_0、
    `weight_only` 量化后立即反量化回 fp32。`bench_quant ppl` 用它扫完全部组合
  - **离群值探针**（`bench_quant outlier`）：调 `forward_trace()`，统计 `final_norm`（lm_head 的输入）每个通道的 max|x|
  - fp32 / 量化交替运行 5 次比速度；ctest 140 个用例通过
- **遇到的问题 1：困惑度 +7.8%**，而 Q8_0 一般 < 0.1%。排查是**逐步缩小范围**：
  1. 权重还是激活？`INFER_Q8_WONLY=1` → 只量化权重 +1.95%，所以激活贡献了大头，但权重也不干净
  2. 哪个矩阵？`INFER_Q8_SKIP` 每次只留一类 → 层内 48 个矩阵 ±0.3%（噪声），lm_head 单独 +8%
  3. 为什么是它？离群值探针 → `ln_f` 输出的第 496 通道 max|x| = 241.8，普通通道中位数 0.76
  4. 方案对比：全 Q8_0（+7.84%）、lm_head fp32（+0.06%，但 lm_head 还要读 154 MB）、
     lm_head fp16（+0.08%，77 MB）
  5. **解决**：层内 Q8_0 + lm_head fp16。新写了 `matmul_nt_f16`（F16C 的 `_mm256_cvtph_ps`）
- **遇到的问题 2：诊断开关自己有 bug**。第一轮扫描时 W8A8 和只量化权重两列数据**完全相同**。
  原因是 shell 里写了 `INFER_Q8_WONLY=`（空字符串），`std::getenv` 对空字符串也返回非空指针，
  所以每一轮都走了只量化权重的路径。改成检查值是不是 `"1"` 后重测。
  **教训：本应不同的两组数据完全一样时，先怀疑实验本身**
- **遇到的问题 3：`std::float16_t` 用不了**。项目是 C++20，它是 C++23 才有的。
  **解决**：有 `__F16C__` 时用 `_cvtss_sh` / `_cvtsh_ss` 指令，否则用手写的位操作转换（最近偶数舍入）
- **遇到的问题 4：新测试第一次没进二进制**。构建和写文件撞在一起，测试二进制没重新链接；
  ctest 显示 138 而不是 140。**解决**：重新构建 + 重新配置 CMake，看到 140 才算数
- **遇到的问题 5：速度测量又被降频干扰**。有一次量化后只有 31 token/s，层内 kernel 明明没改，
  耗时却是上一轮的 4 倍。**解决**：连跑 5 次，fp32 和量化在同一次运行里对比，去掉明显降频的那一次
- **结果**：decode ~53 → ~103 token/s（约 2×），困惑度 +0.08%，top-1 / top-10 和 fp32 一致；
  每 token 读的权重 494 → 173 MB
- **新瓶颈**：层内矩阵字节降到 1/3.6，速度只快 ~2×，有效带宽从 ~23 降到 ~12 GB/s。
  当时判断是「瓶颈从带宽转到了计算」，**复现时发现这个判断不对**：单独测 Q8_0 kernel 有 ~30 GB/s，
  和 fp32 一样是访存密集的；慢在放进模型之后每次调用多出 ~30 µs 的固定开销（见上面的「更正」）。
  也试了加线程数（4/7/10/12），噪声太大看不出规律，默认值没改

#### 跟着做

**① 看这一步改了什么**：第 7 步还没提交，和第 5 步的提交比：

```bash
git status --short                          # src/kernel_quant.cpp 是新文件（??）
git diff 3600536 --stat -- src include tests examples
cat src/kernel_quant.cpp                    # Q8_0 量化 / 整数点积 GEMV / fp16 GEMV，全在这一个文件
git diff 3600536 -- include/inference/gpt.hpp src/gpt.cpp   # QMatrixQ8、QuantOptions、quantize()
```

自己写的顺序：`quantize_row_q8_0` / `dequantize_row_q8_0` + 往返测试 → `gemv_q8_0` 先写标量版、
和「反量化后做 fp32 乘法」对拍 → AVX2 版（sign / maddubs / madd）对拍同一个测试 →
接进 `GPT`，测困惑度 → 发现 lm_head 的问题后再写 fp16。

**② kernel 对比**：

```bash
./build-release/bench/bench_decode 7 20 gemvN,q8,q8all,f16
```

```
  方案（每 token 只算矩阵乘）           合计 ms    相当于
  gemvN  fp32（第 3 步）                14.125     71 token/s
  q8     层内 Q8_0 + lm_head fp16       5.390     186 token/s
  q8all  全部 Q8_0（第一版）             4.676     214 token/s
  f16    全部 fp16（参考）               7.073     141 token/s
```

（上面是把四段输出的「合计」行摘出来的；每段里的分矩阵有效带宽见上面的「更正」。）

**③ 找出 lm_head 的离群值**：

```bash
./build-release/bench/bench_quant outlier
```

```
  83 个位置 × 768 个通道，每个通道取所有位置上的 max|x|
  普通通道（中位数）: 1.17
  通道 496:   248.6   （是中位数的 212 倍）
  通道 430:   131.1   （是中位数的 112 倍）
  通道  36:   103.4   （是中位数的 88 倍）
  ...
  最后一个位置按 Q8_0 量化：通道 496 所在的第 15 组，步长 1.810
  同组其它 31 个数里有 28 个被量化成了 0
  整行相对 RMS 误差只有 0.94% —— 误差被大值「平均」掉了，看不出小值全毁了
```

（上文的 0.76 是当初的临时探针程序统计的，文本和统计方式略有不同，所以中位数不一样；结论一样。）

**④ 逐个矩阵排查困惑度**（每行重新加载一次模型，约 45 s）：

```bash
./build-release/bench/bench_quant ppl
```

```
  83 个 token 的文本，fp32 困惑度 8.917

  配置                              W8A8      只量化权重
  只量化 c_attn                   +0.03%          -0.19%
  只量化 attn_proj                -0.14%          -0.11%
  只量化 mlp_fc                   -0.02%          +0.02%
  只量化 mlp_proj                 +0.28%          +0.18%
  只量化 lm_head                  +8.00%          +2.07%
  层内全部，lm_head Q8_0          +7.84%          +1.95%

  层内全部，lm_head fp32          +0.06%
  层内全部，lm_head fp16          +0.08%   ← 采用
```

**⑤ 各方案的端到端速度**：

```bash
./build-release/bench/bench_quant speed
```

```
  方案                           每 token 读        decode
  fp32                                494 MB     43.7 tok/s
  全部 Q8_0                           139 MB    105.4 tok/s
  层内 Q8_0 + lm_head fp32            250 MB     71.1 tok/s
  层内 Q8_0 + lm_head fp16            173 MB     90.6 tok/s
```

**⑥ 采用方案的精度**（`hello_gpt` 第 7 节）：

```bash
./build-release/examples/hello_gpt "The capital of France is" 60 | sed -n '/7. 量化/,/量化后再生成/p'
```

```
量化耗时 663 ms（只在加载时做一次）
每 token 读的权重: fp32 494 MB → 量化后 173 MB
...
精度对比（83 个 token 的文本）:
  困惑度   fp32 8.917   量化 8.924   （+0.08%）
  prompt 最后位置 logits 最大误差 0.669（logits 量级约 ±100）
  top-1 是否一致: 是，top-10 重合 10/10
```

**⑦ 没有 F16C 时的回退路径**：

```bash
cmake -S . -B build-nof16c -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-mno-f16c
cmake --build build-nof16c -j --target test_kernel_matmul
./build-nof16c/tests/test_kernel_matmul | grep -E "f16|q8_0|个用例"    # 32 个用例，0 个失败
```

`-mno-f16c` 放在 `-march=native` 前面也有效：GCC 里显式的 `-mno-xxx` 优先于 `-march` 带来的默认值
（`g++ -mno-f16c -march=native -dM -E -x c++ /dev/null | grep F16C` 没有输出）。

**⑧ 正确性和最终的进度表**：

```bash
bench/scripts/check_step.sh .                    # 140 个用例（+6），fp32 那次生成的指纹不变
DEBUG=1 bench/scripts/reproduce_progress.sh      # 所有步骤重跑一遍，第一次约 8 分钟（要编 6 个旧版本）
```

```
commit     prefill(5 tok) ms   decode tok/s   ms/token   步骤
f142a75                 1431            1.2      833.3   起点-Debug
f142a75                  195            7.6      131.6   起点-Release
3227974                  185            8.5      117.6   0-测量工具
5624734                   68           24.9       40.2   2-GEMV
89c4702                   31           50.9       19.6   3-线程池
c98ce1b                   28           55.8       17.9   4-去拷贝
3600536                   26           56.3       17.8   5-注意力
.                         28           57.0       17.5   7-fp32(同一程序)
.                         22          111.2        9.0   7-量化
```

### 第 8 步：进阶方向（了解即可）

- **fp16 / bf16 权重**：用 F16C 指令 `_mm256_cvtph_ps` 读 fp16 转 fp32 再算，带宽减半、精度几乎无损
  （第 7 步已经给 lm_head 用上了，见 `matmul_nt_f16`）
- **算子融合**：layernorm + GEMV、GEMV + GELU 合并成一个 kernel，少一次读写激活值
- **Sampler 优化**：现在每个 token 对 50257 个 logits 做 `partial_sort`，约几百微秒；
  用 `nth_element` 先选出 top-k 再排序更快
- **投机解码（speculative decoding）**：用一个小模型先猜几个 token，大模型一次 forward 验证多个。
  把访存密集的 decode 变成 M>1 的「小 prefill」，突破带宽墙
- **批处理（batching）**：服务端同时处理多个请求，M = batch size，提高算术强度。
  这是 vLLM 等推理服务的核心思路

---

## 3. 附录：测试程序清单

### 3.1 项目里的程序（长期保留）

| 程序 | 用在哪几步 | 作用 |
|------|-----------|------|
| `ctest --test-dir build-release` | 每一步 | 全部单元测试，见下表 |
| `examples/hello_gpt` 第 3 节 | 每一步 | 和 numpy 参考实现逐层对拍 |
| `examples/hello_gpt` 第 6 节 | 0 起每一步 | 生成 + prefill / decode 分算子耗时表，矩阵乘行显示有效带宽 |
| `examples/hello_gpt` 第 7 节 | 7 | 量化前后的困惑度、logits 误差、top-10，以及量化后的分算子表 |
| `examples/hello_perf` | 3 | 大矩阵 GEMM 的 GFLOPS，确认换线程池后大矩阵没有退化 |
| 环境变量 `INFER_NUM_THREADS` | 3、7 | 覆盖线程数，扫线程数时用 |

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release   # 新加 TEST 后必须重跑这一步
cmake --build build-release -j
ctest --test-dir build-release
./build-release/examples/hello_gpt "The capital of France is" 60          # 温度 0，可复现
INFER_NUM_THREADS=14 ./build-release/examples/hello_gpt "The capital of France is" 60
```

每一步新增的单元测试：

| 步骤 | 测试 | 检查什么 |
|------|------|----------|
| 2 | `gemv_nt_matches_reference` | 12 个形状（含尾部）和朴素实现一致，输出末尾哨兵不被改写 |
| 2 | `gemv_nt_equals_matmul_nt_row` | GEMV 和整块矩阵乘的同一行一致 |
| 3 | `thread_pool_runs_every_task_once` | 每个任务恰好执行一次 |
| 3 | `thread_pool_many_rounds` | 5000 批连续发，抓批次之间的竞态 |
| 3 | `thread_pool_results_visible_after_return` | 返回后能看到所有任务写的结果（内存序） |
| 3 | `thread_pool_nested_call_runs_serially` | 任务里再调线程池不会死锁 |
| 3 | `thread_pool_sleeping_workers_wake_up` | 工作线程睡着之后能被唤醒 |
| 3 | `matmul_nt_parallel_small_m_splits_columns` | 按列切的路径 × 4 种线程数 |
| 4 | `kernel_gelu_simd_matches_std_tanh` | 24001 个点、原地计算，和 `std::tanh` 版误差 < 1e-6 |
| 4、5 | `kernel_attention_strided_matches_contiguous` | 带行距 / 按头存放的视图和连续版本一致 |
| 5 | `kernel_attention_fast_matches_naive` | SIMD + 并行版和标量参考版一致（10 个形状 × 1/4 线程） |
| 7 | `q8_0_quantize_roundtrip` | 还原误差 ≤ scale/2，全 0 组、离群值 |
| 7 | `q8_0_matmul_matches_dequantized_reference` | 整数 SIMD 和「反量化后做 fp32 乘法」一致 |
| 7 | `q8_0_matmul_error_vs_fp32` | 相对 fp32 的量化误差 < 2%（实测 0.54%） |
| 7 | `q8_0_rejects_k_not_multiple_of_32` | 非法形状抛异常 |
| 7 | `f16_convert_roundtrip`、`f16_matmul_matches_converted_reference` | fp16 转换和 fp16 矩阵乘 |

### 3.2 测量程序 `bench/`

当初这些测量是 `/tmp` 下的一次性程序，WSL 重启后被清掉了。现在按同样的方法重写进了
[`bench/`](../bench)，作为 CMake 目标和库一起编译（`cmake --build build-release -j --target bench_all`），
可执行文件在 `build-release/bench/`。每个文件开头的注释写了用法和测量方法。

| 程序 | 步骤 | 参数 | 测什么、怎么测 |
|------|------|------|----------------|
| `bench_bw` | 0 | `[MB=496] [线程列表]` | 每个线程用 AVX2 + **4 个独立累加器**顺序读自己那一段（只用 1 个累加器测到的是加法延迟）；每个线程数取 7 次里最快的 |
| `bench_decode` | 0、2、3、7 | `[线程] [次数=20] [模式列表]` | 12 层 × 4 + lm_head 各一份**独立**的权重（fp32 494 MB，冷缓存），按 decode 顺序各算一次 = 一个 token。模式 `gemm`（第 0 步路径）`gemv1` `gemvN` `q8` `q8all` `f16` |
| `bench_transpose` | 2 | — | 48 个 Conv1D 权重（340 MB）用朴素 / 32×32 分块 / `Tensor::transpose().contiguous()` 转置，校验结果一致 |
| `bench_pool` | 3 | `[线程列表]` | 文件里实现了第一版 `BarrierPool`（等每个线程报到），和库里的 `ThreadPool` 同进程交替：空任务的调用开销（中位数 / p99）、模拟 decode 的 49 次 GEMV |
| `bench_serial_ops` | 4 | — | 标量 `std::tanh` gelu vs `kernel::gelu_f32`；第 4 步之前的 KV 整段拷贝随长度的增长；156 次 `Tensor::zeros` |
| `bench_attn` | 5 | `[线程]` | 热缓存：`attention_f32_naive` / SIMD 1 线程 / SIMD 多线程，decode Tk 64–1024、prefill T 128/512 |
| `bench_kvlayout` | 5 | `[线程] [token 数=200] [轮数=5]` | 冷缓存：12 层 × 2 种布局（150 MB）交替；`set_attention_parallel_min_work(0)` 强制并行扫交叉点；加载真实模型 ABBA 对比阈值 |
| `bench_quant` | 7 | `[all\|outlier\|ppl\|speed]` | `forward_trace` 统计 lm_head 输入的离群通道；用 `QuantOptions` 逐类矩阵扫困惑度（W8A8 / 只量化权重）；4 种方案交替生成比速度 |

为了让 bench 能做实验，库里加了两个接口（默认行为不变）：

- `GPT::quantize(const QuantOptions &)`：每类矩阵一个开关、lm_head 选 fp32 / fp16 / Q8_0、
  `weight_only` 只量化权重。代替了当初临时加在 kernel 里的环境变量开关
- `kernel::set_attention_parallel_min_work(work)`：改注意力的并行阈值，0 = 总是并行，负数 = 恢复默认，返回旧值

### 3.3 脚本 `bench/scripts/`

所有脚本的「版本」参数都可以是步骤名（`baseline 0 2 3 4 5 7`）、提交号、`.`（当前工作区）或一个源码目录。
旧版本检出到 `.worktrees/<提交>/`，`data/` 自动软链过去。

| 脚本 | 用法 | 做什么 |
|------|------|--------|
| `build_commit.sh` | `<版本> [Release\|Debug]` | 编出该版本的 `hello_gpt`，最后一行输出路径 |
| `check_step.sh` | `<版本> [token 数=60]` | 重新配置 + 全部编译 → ctest → 逐层对拍 → 贪心文本和指纹 |
| `ab_compare.sh` | `<A> <B> [轮数=5] [token 数=60]` | ABBA 交替跑 `hello_gpt`，打印每次和中位数；`WHICH=2` 取第 7 节量化后那次生成 |
| `sweep_threads.sh` | `[版本=.] [线程列表] [轮数=3] [token 数=60]` | 每轮把所有线程数跑一遍，取中位数 |
| `tsan_thread_pool.sh` | — | `build-tsan/`（RelWithDebInfo + `-fsanitize=thread`）里跑 `test_thread_pool`、`test_kernel_matmul`、`test_kernel` |
| `reproduce_progress.sh` | `[轮数=3] [token 数=60]`，`DEBUG=1` 加跑 Debug 起点 | 编出每一步，轮流跑，重建第 5 节的进度表 |

用完旧版本的 worktree 可以删掉：`git worktree remove .worktrees/89c4702`（或者全部 `rm -rf .worktrees && git worktree prune`）。

### 3.4 脚本背后的手工做法

脚本只是把下面这些命令包了一层，想理解或者脚本出问题时可以手工做：

- **git worktree A/B**：

  ```bash
  git worktree add --detach .worktrees/89c4702 89c4702   # 第 3 步
  ln -s $PWD/data .worktrees/89c4702/data                 # 模型文件不在 git 里
  cmake -S .worktrees/89c4702 -B .worktrees/89c4702/build-release -DCMAKE_BUILD_TYPE=Release
  cmake --build .worktrees/89c4702/build-release -j --target hello_gpt
  for i in 1 2 3; do       # 交替，并且每轮换先后（ABBA）
      .worktrees/89c4702/build-release/examples/hello_gpt "The capital of France is" 60 | grep '^decode [0-9]'
      ./build-release/examples/hello_gpt "The capital of France is" 60 | grep '^decode [0-9]'
      ./build-release/examples/hello_gpt "The capital of France is" 60 | grep '^decode [0-9]'
      .worktrees/89c4702/build-release/examples/hello_gpt "The capital of France is" 60 | grep '^decode [0-9]'
  done
  ```

  注意 grep 要写 `'^decode [0-9]'`：只写 `decode` 会把「decode 分算子耗时」那一行也匹配进来。
- **ThreadSanitizer**：TSan 不能和 Debug 构建里的 ASan 同时开，要单独一个构建目录：

  ```bash
  cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS=-fsanitize=thread \
      -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=thread
  cmake --build build-tsan -j --target test_thread_pool
  setarch $(uname -m) -R ./build-tsan/tests/test_thread_pool   # 关掉地址随机化，否则新内核上 TSan 可能起不来
  ```
- **诊断开关的教训**（第 7 步）：当初用环境变量做开关时，`INFER_Q8_WONLY=`（空字符串）让
  `std::getenv` 返回了非空指针，两组本该不同的实验跑成了同一组。现在换成 `QuantOptions` 结构体，
  从源头避免这类问题；如果还要用环境变量，记得判断值而不只是判断「有没有」。

## 4. 跨步骤的经验总结

**测量方法**

1. **先算天花板再优化**。实测带宽 + 每 token 字节数，算出理论下限，就知道还有多少空间、该在哪里挖
2. **用有效带宽而不是毫秒**判断访存密集算子的好坏，它能直接和上限比
3. **笔记本的测量噪声能到 2 倍**。前后两次运行不可比：要么同一进程里交替跑，要么用 git worktree
   编出旧版本交替跑；看中位数；或者用同一次运行里没改动的算子（比如 `mlp_fc`）做标尺，比耗时比值。
   交替时还要**轮换先后**（ABBA）：固定「A 先 B 后」时，后跑的一方系统性地更容易赶上降频（第 5 步复现时
   这样测出了 6 ms 的假差距）
4. **microbenchmark 的数据一直热在缓存里**，会掩盖真实场景的访存问题（第 5 步的 KV 布局）。
   测访存相关的优化，数据量要超过 L3
5. **实验本身也会错**：忘了重新编译、环境变量空字符串也算「已设置」、CMake 没重新配置导致新测试没注册。
   结果反常（比如两组本应不同的数据完全一样、测试总数没涨）时，先检查实验

**分析方法**

6. **大的正常、小的变慢 → 固定开销**（第 3 步的线程池）；**字节少了、速度没按比例涨 → 瓶颈换了**，
   但换到了哪里要单独测才知道：第 7 步当初以为是「转到了计算」，把 kernel 单独拿出来测发现仍然是
   ~30 GB/s 的访存密集，慢的是放进模型之后每次调用的固定开销 —— 又回到了第一条规律
7. **删掉一步时，它的副作用会转移到下一步**（第 4 步删掉拷贝后注意力变慢），分项要合起来看
8. **排查精度问题要逐步缩小范围**：权重还是激活 → 哪个矩阵 → 看那个矩阵输入的数值分布（第 7 步）

**工程做法**

9. **快速版 kernel 旁边永远留一个朴素参考版**，测试拿它对拍；测试要覆盖到 SIMD 主循环、尾部、多线程
   （第 4 步发现原 gelu 测试只有 5 个元素，根本没走到 AVX2 路径）
10. **线程不是越多越好**：带宽 6–8 线程就饱和，E 核掉队、自旋抢超线程。线程数应由应用按场景选
11. **阈值要扫出来**：并行阈值、分块大小都在冷 / 热两种场景下实测交叉点，不要拍脑袋。
    而且扫的条件（工作线程是醒着还是睡着）要和真实场景一致，最后用端到端验证（第 5 步复现时
    microbenchmark 的交叉点变了，端到端却看不出差别）
12. **并发代码要专门测**：除了功能测试，还要用 ThreadSanitizer，以及「连续发几千批」这类压力测试
13. **测量程序要进仓库**：放在 `/tmp` 的 benchmark 丢了，结论就没法复查。重写成 `bench/` 并逐步复现时，
    查出了两处当初的错误结论（第 7 步的「计算受限」、`-mno-f16c` 编译不过）和一处不稳的阈值（第 5 步）。
    能一条命令复现的结论才可信
14. **每个结论配一个指纹**：贪心文本的 md5 一行就能确认「优化没改变模型的行为」，比肉眼对比文本可靠

## 5. 进度记录

| 步骤 | prefill (5 tok) | decode (ms/token) | decode token/s | 备注 |
|------|-----------------|-------------------|----------------|------|
| 起点 Debug | 1376 ms | 751 | 1.3 | `-O0` + ASan + UBSan |
| 1 Release | 219 ms | 131 | 7.6 | |
| 0 加计时后复测 | 224 ms | 126 | 8.1 | 矩阵乘占 decode 98%，有效带宽 ~4 GB/s |
| 2 GEMV + 权重转成 [out, in] | 78–107 ms | 44.9 | 22.6 | 有效带宽 ~12 GB/s，接近单线程上限 |
| 3 多线程 + 线程池 | 35–51 ms | 19.4–22 | ~45 | 7 线程；矩阵乘 ~30 GB/s；14 线程反而更慢 |
| 4 去拷贝 + 工作区 + SIMD gelu | 35–50 ms | 19.1（中位） | ~52 | 与第 3 步交替 A/B：23.4 → 19.1；串行部分 3.5 → 1.2 ms |
| 5 注意力 SIMD + 并行 + KV 按头存放 | 25–28 ms | 15.6–21 | 48–65 | 900 token 时注意力占比 16.7% → ~11% |
| 6 prefill 小 M | — | — | — | 未单独做：第 2/3 步已让 M=5 走按列切的 GEMV |
| 7 量化：层内 Q8_0 + lm_head fp16 | ~19 ms | 9.3–10.5 | 95–108 | 同轮 fp32 51–57 token/s；困惑度 +0.08%；每 token 读 494 → 173 MB |

上表是每一步做完时记下的数字，各行不是同一时段测的。用 `DEBUG=1 bench/scripts/reproduce_progress.sh`
把所有版本编出来、**同一时段轮流跑**（3 轮取中位数，60 个 token，7 线程）的结果：

```
commit     prefill(5 tok) ms   decode tok/s   ms/token   步骤
f142a75                 1431            1.2      833.3   起点-Debug
f142a75                  195            7.6      131.6   起点-Release
3227974                  185            8.5      117.6   0-测量工具
5624734                   68           24.9       40.2   2-GEMV
89c4702                   31           50.9       19.6   3-线程池
c98ce1b                   28           55.8       17.9   4-去拷贝
3600536                   26           56.3       17.8   5-注意力
.                         28           57.0       17.5   7-fp32(同一程序)
.                         22          111.2        9.0   7-量化
```

从起点 1.2 token/s 到 111 token/s，约 **93 倍**；同为 Release 时 7.6 → 111，约 15 倍。
fp32 的 17.5 ms/token 离 13 ms 的带宽天花板还有 ~35%，其中矩阵乘本身只占 14 ms（`bench_decode gemvN`），
剩下的是注意力、layernorm、采样和每次调用的固定开销。
