#include "inference/logits_processor.hpp"
#include <format>
#include <stdexcept>
#include <string>

namespace inference
{

namespace
{

[[noreturn]] void fail(const std::string &msg)
{
    throw std::runtime_error("[infer::PenaltyProcessor] " + msg);
}

} // namespace

void PenaltyProcessor::begin(std::span<const int> prompt)
{
    counts_.clear();
    if (cfg_.include_prompt)
    {
        for (int t : prompt)
            accept(t);
    }
}

void PenaltyProcessor::process(std::span<float> logits)
{
    if (cfg_.presence_penalty == 0.0F && cfg_.frequency_penalty == 0.0F)
        return;
    for (const auto &[token, c] : counts_)
    {
        if (static_cast<std::size_t>(token) >= logits.size())
            fail(std::format("token {} 超出词表大小 {}", token, logits.size()));
        logits[static_cast<std::size_t>(token)] -=
            cfg_.frequency_penalty * static_cast<float>(c) + cfg_.presence_penalty;
    }
}

void PenaltyProcessor::accept(int token)
{
    if (token < 0)
        fail(std::format("token id 不能为负：{}", token));
    ++counts_[token];
}

int PenaltyProcessor::count(int token) const
{
    const auto it = counts_.find(token);
    return it == counts_.end() ? 0 : it->second;
}

} // namespace inference
