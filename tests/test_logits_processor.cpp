#include <vector>

#include "inference/gpt.hpp"
#include "inference/logits_processor.hpp"
#include "minitest.hpp"

using inference::PenaltyConfig;
using inference::PenaltyProcessor;

TEST(penalty_zero_is_noop)
{
    PenaltyProcessor p{PenaltyConfig{}};
    p.begin({});
    p.accept(1);
    p.accept(1);
    std::vector<float> logits{1.0F, 2.0F, 3.0F};
    p.process(logits);
    CHECK_NEAR(logits[0], 1.0, 1e-6);
    CHECK_NEAR(logits[1], 2.0, 1e-6);
    CHECK_NEAR(logits[2], 3.0, 1e-6);
}

TEST(presence_penalty_ignores_count)
{
    PenaltyProcessor p{PenaltyConfig{.presence_penalty = 0.5F}};
    p.begin({});
    p.accept(0);
    p.accept(2);
    p.accept(2);
    p.accept(2);
    std::vector<float> logits{1.0F, 1.0F, 1.0F, 1.0F};
    p.process(logits);
    CHECK_NEAR(logits[0], 0.5, 1e-6);
    CHECK_NEAR(logits[1], 1.0, 1e-6);
    CHECK_NEAR(logits[2], 0.5, 1e-6);
    CHECK_NEAR(logits[3], 1.0, 1e-6);
}

TEST(frequency_penalty_scales_with_count)
{
    PenaltyProcessor p{PenaltyConfig{.frequency_penalty = 0.25F}};
    p.begin({});
    p.accept(0);
    p.accept(2);
    p.accept(2);
    p.accept(2);
    std::vector<float> logits{1.0F, 1.0F, 1.0F};
    p.process(logits);
    CHECK_NEAR(logits[0], 0.75, 1e-6);
    CHECK_NEAR(logits[1], 1.0, 1e-6);
    CHECK_NEAR(logits[2], 0.25, 1e-6);
}

TEST(presence_and_frequency_combine)
{
    PenaltyProcessor p{PenaltyConfig{.presence_penalty = 1.0F, .frequency_penalty = 0.5F}};
    p.begin({});
    p.accept(1);
    p.accept(1);
    std::vector<float> logits{0.0F, 5.0F};
    p.process(logits);
    // 5 - (0.5 * 2 + 1.0)
    CHECK_NEAR(logits[1], 3.0, 1e-6);
    CHECK_NEAR(logits[0], 0.0, 1e-6);
}

TEST(negative_penalty_encourages_repeat)
{
    PenaltyProcessor p{PenaltyConfig{.presence_penalty = -1.0F}};
    p.begin({});
    p.accept(0);
    std::vector<float> logits{0.0F, 0.0F};
    p.process(logits);
    CHECK_NEAR(logits[0], 1.0, 1e-6);
}

TEST(begin_resets_and_respects_include_prompt)
{
    const std::vector<int> prompt{3, 3, 4};

    PenaltyProcessor out_only{PenaltyConfig{.frequency_penalty = 1.0F}};
    out_only.accept(7);
    out_only.begin(prompt);
    CHECK(out_only.count(7) == 0);
    CHECK(out_only.count(3) == 0);

    PenaltyProcessor with_prompt{
        PenaltyConfig{.frequency_penalty = 1.0F, .include_prompt = true}};
    with_prompt.begin(prompt);
    CHECK(with_prompt.count(3) == 2);
    CHECK(with_prompt.count(4) == 1);
    std::vector<float> logits(5, 0.0F);
    with_prompt.process(logits);
    CHECK_NEAR(logits[3], -2.0, 1e-6);
    CHECK_NEAR(logits[4], -1.0, 1e-6);
}

TEST(penalty_rejects_bad_token)
{
    PenaltyProcessor p{PenaltyConfig{.presence_penalty = 1.0F}};
    CHECK_THROWS(p.accept(-1));
    p.accept(10);
    std::vector<float> logits(4, 0.0F);
    CHECK_THROWS(p.process(logits));
}

TEST(penalty_changes_greedy_choice)
{
    // 贪心本来永远选 0；token 0 用过一次之后被扣掉 2，第二步改选 1
    inference::Sampler greedy{inference::SamplerConfig{.temperature = 0.0F}};
    PenaltyProcessor p{PenaltyConfig{.presence_penalty = 2.0F}};
    p.begin({});

    std::vector<float> step1{3.0F, 2.0F, 0.0F};
    p.process(step1);
    const int a = greedy.sample(step1);
    p.accept(a);
    CHECK(a == 0);

    std::vector<float> step2{3.0F, 2.0F, 0.0F};
    p.process(step2);
    CHECK(greedy.sample(step2) == 1);
}
