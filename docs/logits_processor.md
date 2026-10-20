# Logits Processor：Presence Penalty 与 Frequency Penalty

> 对应代码：[`include/inference/logits_processor.hpp`](../include/inference/logits_processor.hpp)、
> [`src/logits_processor.cpp`](../src/logits_processor.cpp)、[`src/gpt.cpp`](../src/gpt.cpp)（`GPT::generate_with`）
>
> 示例 / 测试：[`examples/hello_penalty.cpp`](../examples/hello_penalty.cpp)（贪心，最小示例）、
> [`examples/hello_processor.cpp`](../examples/hello_processor.cpp)（贪心 + 采样的完整对比，第 5 节）、
> [`tests/test_logits_processor.cpp`](../tests/test_logits_processor.cpp)
>
> 前置阅读：[`gpt.hpp`](../include/inference/gpt.hpp) 里的 `Sampler`（temperature / top-k / top-p）
>
> 图解版：[logits_processor.html](logits_processor.html)

---

## 0. 一页看懂

模型每一步输出 50257 个 logits，`Sampler` 从中挑一个 token。**Logits Processor** 插在这两者之间，
在采样之前原地改写 logits：

```
GPT::forward ──→ logits ──→ [processor.process] ──→ Sampler::sample ──→ token
                                  ▲                                     │
                                  └──────── processor.accept ◄──────────┘
```

`Sampler` 只看「这一步」的分布，processor 能记住「之前生成过什么」。所以惩罚重复、禁止某些词、
强制输出格式这类和历史有关的规则，都放在 processor 里。

本次实现的 `PenaltyProcessor` 就是 OpenAI API 的 `presence_penalty` / `frequency_penalty`。
设 \(c_j\) 为 token \(j\) 已经出现的次数：

\[
\text{logit}'_j = \text{logit}_j - \alpha_{\text{freq}} \cdot c_j - \alpha_{\text{pres}} \cdot \mathbb{1}[c_j > 0]
\]

| 参数 | 含义 | 惩罚随次数增长？ |
|------|------|------------------|
| `presence_penalty` | 出现过就扣一次，和次数无关 | 否，有上限 |
| `frequency_penalty` | 每出现一次扣一次 | 是，线性增长 |

正值抑制重复，负值鼓励重复。OpenAI 允许的范围是 \([-2, 2]\)。

用法：

```cpp
#include "inference/gpt.hpp"   // 已经包含 logits_processor.hpp

inference::Sampler sampler{inference::SamplerConfig{.temperature = 0.8F, .top_k = 40}};
inference::PenaltyProcessor penalty{inference::PenaltyConfig{
    .presence_penalty = 0.6F,
    .frequency_penalty = 0.4F,
}};
auto out = model.generate(prompt_ids, 100, sampler, /*stats=*/nullptr, /*on_token=*/{}, &penalty);
```

`generate` 和 `generate_graph` 都接受最后这个 `LogitsProcessor *` 参数，不传时行为和以前完全一样。

一个反直觉的实测结论（第 4 节）：在 GPT-2 small 上做贪心解码时，**常用的「温和」参数（0.5~1.0）
几乎拦不住复读**，因为复读越久，模型对下一个复读 token 越有把握，领先幅度从 ~2 涨到 ~10。
随机采样时则相反：同样的温和参数就能把 4-gram 重复率降到 0（第 5 节）。

---

## 1. 为什么需要：贪心解码会复读

`hello_penalty` 用 prompt `"The meaning of life is"`，贪心解码 60 个 token，不加任何惩罚：

```
 not the same as the meaning of death.

The meaning of life is not the same as the meaning of death.

The meaning of life is not the same as the meaning of death.

The meaning of life is not the same as the meaning of death.

The
```

60 个 token 里只有 20% 是不同的词，`' the'` 出现了 8 次。这是小模型 + 贪心解码的经典病：
一旦某句话出现过一次，注意力里的「归纳头」（induction head）就会学着照抄前文，
抄得越多越笃定，形成正反馈。

解决思路有两类：

- **改采样**：加温度、top-k、top-p，让模型有机会走出循环。但它们不知道历史，循环里的 token
  概率可能高达 99%，随机性也救不回来。
- **改 logits**：直接给「已经说过的词」扣分。这就是 presence / frequency penalty。

---

## 2. 两种惩罚的区别

举个例子。生成到某一步时，历史里 `' the'` 出现了 4 次，`' cat'` 出现了 1 次，`' dog'` 没出现过：

| token | 次数 \(c\) | presence=1.0 扣 | frequency=0.5 扣 | 两者都开扣 |
|-------|-----------|-----------------|------------------|------------|
| `' the'` | 4 | 1.0 | 2.0 | 3.0 |
| `' cat'` | 1 | 1.0 | 0.5 | 1.5 |
| `' dog'` | 0 | 0 | 0 | 0 |

- **presence**：「这个话题聊过了，换一个」。对出现过的词一视同仁，惩罚有上限，不会无限累积。
  适合鼓励模型引入新话题、新词汇。
- **frequency**：「这个词你用太多了」。用得越多扣得越狠。适合抑制同一个词的高频复读。
  代价是长文本里 `' the'`、`','`、`'.'` 这类高频功能词也会被越扣越多，生成越长语法越容易坏。

### 2.1 和 Repetition Penalty 的区别

还有一个经常被混淆的参数 `repetition_penalty`（CTRL 论文提出，HuggingFace / llama.cpp 都有），
它是**乘法**而不是减法：

\[
\text{logit}'_j = \begin{cases} \text{logit}_j / \theta & \text{logit}_j > 0 \\ \text{logit}_j \cdot \theta & \text{logit}_j \le 0 \end{cases}
\quad (\text{如果 } j \text{ 出现过，} \theta > 1)
\]

| | presence / frequency | repetition |
|---|---|---|
| 运算 | 减法 | 乘除 |
| 是否看次数 | frequency 看 | 不看 |
| 默认统计范围 | 只看生成的 token（OpenAI / vLLM） | 通常包括 prompt |
| 中性值 | 0 | 1 |

乘法的问题是效果依赖 logit 的绝对大小（logit 是 0.1 还是 100，除以 1.2 的效果差很多），
而 logit 整体平移不改变 softmax 结果，所以乘法惩罚在数学上不太「干净」。本项目只实现了减法版本，
接口留着，需要的话再加一个 `RepetitionProcessor` 即可。

---

## 3. 实现

### 3.1 接口：三个钩子

```cpp
class LogitsProcessor
{
  public:
    virtual ~LogitsProcessor() = default;
    virtual void begin(std::span<const int> prompt);     // 每次 generate 开始调用一次
    virtual void process(std::span<float> logits) = 0;   // 原地改写下一个位置的 logits
    virtual void accept(int token);                      // 采样出 token 之后调用
};
```

为什么拆成三步而不是一个 `process(logits, history)`：

- 有状态的 processor 不用每步重新扫一遍完整历史，`accept` 时增量更新即可；
- `begin` 负责清空上一次生成的状态，同一个 processor 对象可以反复用于多次 `generate`。

### 3.2 PenaltyProcessor：只存出现过的 token

```cpp
std::unordered_map<int, int> counts_;   // token id → 出现次数
```

计数没有开成 50257 维的数组：生成几百个 token 只涉及几百个不同的词，`process` 只遍历这几百项，
每步的开销是 \(O(\text{distinct})\) 而不是 \(O(V)\)。实测 55 个词被惩罚时 `process()` 每次 0.28 µs，
贪心 `sample()`（在 50257 个数里找最大值）每次约 170 µs，一次 forward 约 10 ms，processor 的开销可以忽略。

两个参数都是 0 时 `process` 直接返回，不碰 logits。

### 3.3 include_prompt：prompt 算不算「出现过」

```cpp
struct PenaltyConfig
{
    float presence_penalty = 0.0F;
    float frequency_penalty = 0.0F;
    bool include_prompt = false;
};
```

不同框架做法不一样：

| 框架 | presence / frequency 统计范围 |
|------|-------------------------------|
| OpenAI API、vLLM | 只统计生成的 token |
| llama.cpp | 最近 `penalty_last_n`（默认 64）个 token，包括 prompt |

默认值跟 OpenAI 一致。如果希望模型不要照抄 prompt 里的话，设 `include_prompt = true`，
`begin(prompt)` 时会把 prompt 里的 token 也计入。

### 3.4 接进生成循环

`generate` / `generate_graph` 共用 `GPT::generate_with`，改动只在采样那一步：

```cpp
if (processor)
    processor->begin(prompt);
auto timed_sample = [&](Tensor &lg, OpProfile *prof)
{
    ScopedTimer tm(prof ? &prof->sample : nullptr);
    if (!processor)
        return sampler.sample(lg.data());
    processor->process(lg.data());
    const int id = sampler.sample(lg.data());
    processor->accept(id);
    return id;
};
```

几个细节：

- **原地改写**：每一步的 logits 张量都是 forward 刚返回的新结果，用完就丢，直接改不会影响别的东西，
  省掉一次 50257 个 float 的拷贝。
- **耗时计入 `sample`**：processor 的开销会出现在分算子耗时表的 `sample` 一行。
- **惩罚在温度之前**：processor 改的是原始 logits，`Sampler` 之后才除以温度、做 top-k / top-p。
  这和 OpenAI / vLLM / HuggingFace 的顺序一致。

### 3.5 惩罚在概率空间里是什么意思

softmax 对 logits 做减法等于对概率做乘法。带温度 \(T\) 时：

\[
p'_j \propto p_j \cdot e^{-\Delta_j / T}
\]

其中 \(\Delta_j\) 是 token \(j\) 被扣掉的分数。所以：

- 扣 1 分 ≈ 概率乘以 \(e^{-1} \approx 0.37\)；扣 2 分 ≈ 乘以 0.14；扣 3 分 ≈ 乘以 0.05。
- 温度越低，同样的惩罚效果越强（\(T = 0.5\) 时扣 1 分相当于乘以 0.14）。
- 贪心（\(T \to 0\)）时惩罚变成「是 / 否」：只有扣的分数**超过**它和第二名的差距，才会换词。

最后一条正是下一节实验里「温和参数没效果」的原因。

---

## 4. 实验：GPT-2 small，贪心解码

设置：prompt `"The meaning of life is"`，贪心生成 60 个 token，层内 Q8_0 + lm_head fp16。
两个指标：

- **distinct**：不同 token 数 / 总 token 数，越高越不重复；
- **max_rep**：出现最多的那个 token 出现了几次。

| presence | frequency | distinct | max_rep | 是否跳出循环 |
|----------|-----------|----------|---------|--------------|
| 0 | 0 | 0.20 | 8 | 否 |
| 1.0 | 0 | 0.22 | 9 | 否 |
| 2.0 | 0 | 0.23 | 9 | 否 |
| **3.0** | 0 | **0.80** | 4 | 是 |
| 0 | 0.5 | 0.22 | 9 | 否 |
| 0 | 1.0 | 0.38 | 6 | 部分 |
| 0 | 1.5 | 0.62 | 4 | 是 |
| 0 | 2.0 | 0.72 | 4 | 是 |
| 1.0 | 0.5 | 0.22 | 9 | 否 |
| **2.0** | **1.0** | **0.92** | **2** | 是 |

presence=2.0 + frequency=1.0 的输出：

```
 not the same as that of death.

The meaning of life is not a matter for us to decide, but rather an act which we can do in
order to live and die on our own terms. We are all born with this idea: "I am alive." The only
way out from
```

### 4.1 为什么 presence=1.0 不但没用，max_rep 还从 8 涨到了 9？

看 presence=1.0 的输出：

```
 not the same as the meaning of death.

The meaning of life is not the same as the meaning of death. The meaning of life is not the
same as the meaning of death. The meaning of life is not the
```

循环还在，只是句子之间的 `\n\n` 变成了空格。惩罚**最先干掉的是领先幅度最小的 token**，
在这里恰好是换行（它领先第二名只有 0.2~0.7），而真正的循环内容一个都没拦住。
句子变短，60 个 token 里塞进了更多遍复读，max_rep 反而上升。

### 4.2 复读越久，模型越笃定

为了弄清楚要多大的惩罚才够，测了一下**不加惩罚**时，贪心每一步第一名和第二名的 logit 差
（gap 越大，越难被惩罚扳倒）：

| 第几遍复读 | `' same'` | `' as'` | `' of'` | `'\n'`（第二个） |
|-----------|-----------|---------|---------|------------------|
| 第 1 遍（step 2~10） | 2.10 | 1.75 | 6.09 | 6.03 |
| 第 2 遍（step 18~26） | 6.39 | 6.72 | 9.23 | 10.62 |
| 第 3 遍（step 34~42） | 10.57 | 9.24 | 8.56 | 10.64 |
| 第 4 遍（step 50~58） | 10.39 | 8.39 | 8.27 | 10.38 |

第一遍时 gap 只有 2 左右，第二遍就涨到 6~10。也就是说，**惩罚必须在复读的头一两遍就足够大**，
否则等循环形成，gap 已经大到 frequency penalty 线性累积也追不上了。这解释了实验表里的阈值：

- presence 是一次性的，需要 > 2 才能在第二遍复读时压过 `' same'` / `' as'` 那 ~6 的 gap
  （第二名往往也是出现过的词，同样被扣分，所以实际需要的值比 gap 小）；
- frequency=1.0 时第二遍才扣 1 分、第三遍扣 2 分，赶不上 gap 的增长速度；
  frequency=1.5~2.0 才能在第二、三遍时追上。

### 4.3 惩罚过大的副作用

presence=3.0 的输出：

```
The meaning of life is not a matter of time, but rather an experience which can be experienced
in one's lifetime and will continue to exist for many years after it has been passed on from
generation to generation. The meaning of life is not something you
```

第一句就和不加惩罚时不一样了：`"as the meaning of death"` 变成了 `"as that of death"`，
因为 `' the'` 只在三个 token 之前出现过一次，就被扣了 3 分。这里碰巧有 `' that'` 可以替换，
句子仍然通顺；但如果用过的词没有好的替代品，模型就只能选别扭的表达。

另外注意结尾又出现了 `"The meaning of life is"`：惩罚只是让复读更难，并不能保证不复读。
gap 足够大时（`' of'` 在第一遍就有 6 以上），扣 3 分也拦不住。

frequency 在长文本里问题更明显：`' the'`、`','` 这种词出现几十次之后会被扣到几十分，
相当于被禁用，语法会逐渐崩坏。

---

## 5. 全面对比：hello_processor

第 4 节只测了贪心解码和一个指标。[`hello_processor`](../examples/hello_processor.cpp) 用 8 组设置做 4 个实验，
补上随机采样和 `include_prompt`。设置覆盖了负值、单独 presence、单独 frequency 和两者组合：

| presence | 0 | −1.0 | 1.0 | 3.0 | 0 | 0 | 1.0 | 2.0 |
|----------|---|------|-----|-----|---|---|-----|-----|
| frequency | 0 | 0 | 0 | 0 | 0.5 | 1.5 | 0.5 | 1.0 |

指标都只看生成部分：

| 指标 | 含义 | 越…越好 |
|------|------|---------|
| dist-1 / dist-2 | 不同的 1-gram / 2-gram 占比 | 高 |
| rep-4 | 已经出现过的 4-gram 占比，整段复读时接近 1 | 低 |
| maxrep | 出现最多的 token 出现了几次 | 低 |
| ppl | **不加惩罚**的原模型给这段文本打的困惑度 | 见下文 |

ppl 衡量的是「生成结果偏离模型自己的分布有多远」。它不是越低越好：复读的文本每个词都被前文「预言」了，
ppl 会低到 1.5 左右，这恰恰是退化的标志；惩罚过大时 ppl 会一路升高，文本开始不通顺。
合适的区间要和「正常随机采样」的 ppl 对比着看。

### 5.1 单步观察：惩罚怎么改写 top-5

先不加惩罚贪心生成，找到第一个「已经生成过 2 次、还要再选一次」的位置，把这一步的 logits
分别交给不同的 processor：

```
上下文（生成部分的最后 12 个 token）: … meaning of death.\n\nThe meaning of life is not▌
不加惩罚时贪心会选 ' the'，它在生成部分已经出现了 2 次

  pres  0.0  freq  0.0 │  ' the'  -21.53(c=2)  ' a'  -24.51  ' that' -24.83  ' death' -25.30(c=1)  ' life' -25.32(c=1)
  pres  1.0  freq  0.0 │  ' the'  -22.53(c=2)  ' a'  -24.51  ' that' -24.83  ' what'  -25.39       ' to'   -25.61
  pres  3.0  freq  0.0 │  ' a'    -24.51       ' the' -24.53(c=2)  ' that' -24.83  ' what' -25.39  ' to'   -25.61
  pres  0.0  freq  1.5 │  ' a'    -24.51       ' the' -24.53(c=2)  ' that' -24.83  ' what' -25.39  ' to'   -25.61
```

（GPT-2 的 logits 整体偏负，绝对值没有意义，只看差值。）

- `' the'` 领先第二名 `' a'` 2.98。presence 1.0 只扣 1 分，`' the'` 仍领先 1.98，**选择不变**。
- presence 3.0 扣 3 分、frequency 1.5 × 2 次也是扣 3 分，`' the'` 以 0.02 之差落到第二，选择变成 `' a'`。
  这正是第 3.5 节说的贪心阈值：扣分必须超过领先幅度。
- 只出现过 1 次的 `' death'`、`' life'` 在 presence 1.0 下就掉出了 top-5，被没出现过的 `' what'`、`' to'` 取代。
  惩罚对排名的影响不只在第一名，后面候选的重新洗牌对随机采样更重要。

### 5.2 贪心解码

```
  setting                         dist-1  dist-2   rep-4  maxrep     ppl
  pres  0.0  freq  0.0              0.20    0.25    0.72     8.0    1.49
  pres -1.0  freq  0.0              0.20    0.27    0.70    12.0    1.55
  pres  1.0  freq  0.0              0.22    0.29    0.65     9.0    1.55
  pres  3.0  freq  0.0              0.80    0.93    0.04     4.0    5.74
  pres  0.0  freq  0.5              0.22    0.29    0.65     9.0    1.55
  pres  0.0  freq  1.5              0.62    0.83    0.05     4.0    4.39
  pres  1.0  freq  0.5              0.22    0.29    0.65     9.0    1.55
  pres  2.0  freq  1.0              0.92    1.00    0.00     2.0    8.03
```

- **三组温和设置结果完全相同**：(1.0, 0)、(0, 0.5)、(1.0, 0.5) 生成的是同一段文本，都只是把句间的 `\n\n`
  换成了空格（第 4.1 节），72% → 65% 的 4-gram 仍是复读。
- **负值换了个循环**：presence −1.0 生成 `"The meaning of life is not the same as the meaning of the world."` 复读 4 遍，
  maxrep 从 8 涨到 12（`' the'` 在每遍里出现 3 次）。
- **跳出循环后 ppl 从 1.5 升到 4~8**。对比 5.3 节不加惩罚的随机采样 ppl 是 6.8，说明惩罚后的贪心文本和
  正常采样的文本「同样可能」：ppl 的上升是离开了退化的复读区，而不是变得不通顺。

### 5.3 随机采样：temperature 0.8、top-k 40、top-p 0.95

每组设置跑 5 个随机种子取平均：

```
  setting                         dist-1  dist-2   rep-4  maxrep     ppl
  pres  0.0  freq  0.0              0.55    0.75    0.15     6.4    6.82
  pres -1.0  freq  0.0              0.30    0.45    0.44     8.8    4.02
  pres  1.0  freq  0.0              0.79    0.96    0.00     4.4   10.16
  pres  3.0  freq  0.0              0.92    1.00    0.00     2.8   12.65
  pres  0.0  freq  0.5              0.74    0.95    0.00     3.8    9.42
  pres  0.0  freq  1.5              0.85    1.00    0.00     2.6   10.46
  pres  1.0  freq  0.5              0.86    0.99    0.00     2.8   10.86
  pres  2.0  freq  1.0              0.93    1.00    0.00     2.4   13.22
```

- **采样本身就打破了大部分循环**：不加惩罚时 rep-4 是 0.15，贪心是 0.72。但仍有 15% 的 4-gram 是重复的。
- **温和设置在采样下就够了**：贪心时毫无作用的 presence 1.0、frequency 0.5，在采样下都把 rep-4 降到 0，
  dist-1 从 0.55 升到 0.74~0.79。原因见第 3.5 节：采样时惩罚是「连续地」压低概率（扣 1 分 ≈ 概率乘
  \(e^{-1/0.8} \approx 0.29\)），没有贪心那种必须越过领先幅度的阈值。
- **负值会把采样拖回循环**：presence −1.0 的种子 1 输出是
  `"not so simple as it might seem. It's not so simple as it might seem. It's not so simple as it might seem."`，
  rep-4 从 0.15 涨到 0.44。
- **多样性的边际收益递减，代价持续上升**：从 (1.0, 0.5) 加到 (2.0, 1.0)，dist-1 只从 0.86 涨到 0.93，
  ppl 却从 10.9 涨到 13.2。采样时 0.5~1.0 是性价比最高的区间，这和 OpenAI 推荐的范围一致。

各组的种子 1 文本开头都是 `"not so simple as it might seem. It"`：同一个种子下，惩罚只有在出现重复词之后才开始起作用，
在此之前所有设置的分布完全相同。

### 5.4 include_prompt

frequency 1.5，贪心，只切换 prompt 是否计入：

```
── pres  0.0  freq  1.5
 not the same as the meaning of death.

The meaning of life is not a matter of what you do, but rather what you do with your life. The meaning of
life is that which comes from God and that which comes from God's will. It is this sense in which we live

── pres  0.0  freq  1.5  +prompt
 not the same as the meaning of death.

In this way, we can see that there are two kinds of people: those who live in a world where they have no
choice but to die and those who live in a world where they have no choice but to die. The former are living

  setting                         dist-1  dist-2   rep-4  maxrep     ppl
  pres  0.0  freq  1.5              0.62    0.83    0.05     4.0    4.39
  pres  0.0  freq  1.5  +prompt     0.68    0.78    0.19     2.0    3.98
```

- 不计入 prompt 时，生成部分两次重述了 prompt 原话 `"The meaning of life is"`；计入后模型改口
  `"In this way, ..."`，不再照抄 prompt。
- 但计入 prompt 的那段又复读了 `"those who live in a world where they have no choice but to die"`：
  这句话里每个词之前都只出现过一次，每个只扣 1.5 分，压不过归纳头照抄的把握，rep-4 反而更高。
- 这只是一条贪心样本，说明的是机制（prompt 里的词被惩罚后模型会避开它们），不能据此判断哪种设置更好。

---

## 6. 调参建议

1. **先确认 logits 的尺度**。OpenAI 推荐的 0.1~1.0 是针对他们的模型和默认温度（1.0）调的。
   GPT-2 small 贪心时，logit gap 动辄 5~10，同样的值效果小得多。换模型要重新测。
2. **采样时比贪心时温和**。温度 0.8 + top-k 时，扣 1 分就让概率变成原来的 \(e^{-1/0.8} \approx 0.29\)，
   已经很明显了，一般 0.5~1.0 就够。实测（第 5.3 节）presence 1.0 就把 4-gram 重复率从 0.15 降到 0，
   再往上加多样性涨得很少，困惑度却持续上升。
3. **长文本优先用 presence**。它有上限，不会把功能词扣到禁用；frequency 适合短回复。
4. **想让模型别照抄 prompt**（摘要、改写类任务）就打开 `include_prompt`；
   想让模型复用 prompt 里的术语（问答、代码补全）就关掉。
5. **负值可以鼓励复读**，比如希望模型反复使用 prompt 里给定的关键词。

---

## 7. 跟着做

```bash
# 单元测试（不需要模型文件）
cmake --build build --target test_logits_processor && ./build/tests/test_logits_processor

# Release 构建
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target hello_penalty hello_processor

# 最小示例：贪心，默认跑 5 组参数（第 1、4 节）
./build-release/examples/hello_penalty
./build-release/examples/hello_penalty "Once upon a time" 80 0.5 0.3   # 自定义 prompt、长度、presence、frequency

# 完整对比：单步观察 + 贪心 + 采样 + include_prompt（第 5 节），约 1 分钟
./build-release/examples/hello_processor
./build-release/examples/hello_processor "Once upon a time" 80 8       # prompt、生成长度、采样种子数
```

预期：`hello_penalty` 第一组（不加惩罚）的 distinct 在 0.2 左右，最后一组（2.0 / 1.0）在 0.9 以上；
`hello_processor` 的表格和第 5 节一致（不同机器上浮点求和顺序可能让个别贪心结果略有不同）。

---

## 8. 还能加什么 processor

`LogitsProcessor` 接口是通用的，下面这些都可以用同样的三个钩子实现：

| processor | 做什么 | 需要的状态 |
|-----------|--------|------------|
| `RepetitionProcessor` | CTRL 式乘法惩罚（第 2.1 节） | 出现过的 token 集合 |
| `NoRepeatNGramProcessor` | 禁止任何 n-gram 出现第二次（HuggingFace `no_repeat_ngram_size`） | 历史 token 序列 |
| `LogitBiasProcessor` | 给指定 token 加减固定分数（OpenAI `logit_bias`） | 无 |
| 滑动窗口惩罚 | 只惩罚最近 N 个 token（llama.cpp `penalty_last_n`） | 环形缓冲区 |

要同时用多个，可以写一个 `ProcessorList : LogitsProcessor`，在三个钩子里依次转发给每个成员。
