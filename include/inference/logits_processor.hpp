#pragma once
#include <span>
#include <unordered_map>

namespace inference
{
// ---------------------------------------------------------------------------
// Logits Processor：模型输出 logits 之后、Sampler 挑 token 之前的那一步
//
//   GPT::forward ──→ logits ──→ [processor.process] ──→ Sampler::sample ──→ token
//                                      ▲                                     │
//                                      └──────── processor.accept ◄──────────┘
//
// Sampler 只看「这一步」的分布；processor 能记住「之前生成过什么」，
// 所以惩罚重复、禁止某些词、强制格式这类和历史有关的规则都放在这里。
// ---------------------------------------------------------------------------
class LogitsProcessor
{
  public:
    virtual ~LogitsProcessor() = default;
    // 每次 generate 开始时调用一次，清掉上一次生成留下的状态
    virtual void begin(std::span<const int> prompt)
    {
        (void)prompt;
    }
    // 原地改写下一个位置的 logits [n_vocab]
    virtual void process(std::span<float> logits) = 0;
    // 采样出 token 之后调用
    virtual void accept(int token)
    {
        (void)token;
    }
};

// Presence / Frequency Penalty（OpenAI API 的同名参数，vLLM / llama.cpp 也都有）。
// 设 c[j] 是 token j 已经出现的次数：
//
//   logit[j] -= frequency_penalty * c[j]            每多出现一次，多扣一次
//             + presence_penalty  * (c[j] > 0)      出现过就扣一次，和次数无关
//
// 正值抑制重复，负值鼓励重复；OpenAI 允许的范围是 [-2, 2]。
struct PenaltyConfig
{
    float presence_penalty = 0.0F;
    float frequency_penalty = 0.0F;
    // false：只统计新生成的 token（OpenAI / vLLM 的做法）
    // true：prompt 里的 token 也算「出现过」
    bool include_prompt = false;
};

class PenaltyProcessor final : public LogitsProcessor
{
  public:
    explicit PenaltyProcessor(const PenaltyConfig &cfg) : cfg_(cfg) {}

    void begin(std::span<const int> prompt) override;
    void process(std::span<float> logits) override;
    void accept(int token) override;

    // token 目前被统计到的出现次数
    int count(int token) const;

  private:
    PenaltyConfig cfg_;
    // 只存出现过的 token：生成几百个 token 只涉及几百个词，
    // process 只需遍历这几百项，不用扫一遍 50257 维的词表
    std::unordered_map<int, int> counts_;
};

} // namespace inference
