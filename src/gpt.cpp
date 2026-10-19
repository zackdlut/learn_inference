#include "inference/gpt.hpp"
#include "inference/kernel.hpp"
#include "inference/model_file.hpp"
#include "inference/ops.hpp"
#include "inference/tensor.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <numeric>
#include <stdexcept>

namespace inference
{

namespace
{

[[noreturn]] void fail(const std::string &msg)
{
    throw std::runtime_error("[infer::GPT] " + msg);
}

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// 作用域结束时把耗时加到 *acc 上。acc 为空时什么都不做，不测量就几乎零开销。
class ScopedTimer
{
  public:
    explicit ScopedTimer(double *acc) : acc_(acc)
    {
        if (acc_)
            t0_ = Clock::now();
    }
    ~ScopedTimer()
    {
        if (acc_)
            *acc_ += ms_since(t0_);
    }
    ScopedTimer(const ScopedTimer &) = delete;
    ScopedTimer &operator=(const ScopedTimer &) = delete;

  private:
    double *acc_;
    Clock::time_point t0_{};
};

} // namespace

// ---------------------------------------------------------------------------
// KVCache
// ---------------------------------------------------------------------------

KVCache::KVCache(const GPTConfig &cfg, int max_len)
    : cfg_(cfg), max_len_(std::min(max_len, cfg.n_ctx))
{
    k_.reserve(static_cast<std::size_t>(cfg.n_layer));
    v_.reserve(static_cast<std::size_t>(cfg.n_layer));
    for (int i = 0; i < cfg.n_layer; ++i)
    {
        k_.push_back(Tensor::zeros({max_len_, cfg.n_embd}));
        v_.push_back(Tensor::zeros({max_len_, cfg.n_embd}));
    }
}

void KVCache::append(int layer, int pos, std::span<const float> k, std::span<const float> v)
{
    const auto C = static_cast<std::size_t>(cfg_.n_embd);
    if (k.size() != v.size() || k.size() % C != 0)
    {
        fail("KVCache::append: K/V 长度必须相等且是 n_embd 的整数倍");
    }
    const auto n_new = static_cast<int>(k.size() / C);
    if (pos < 0 || pos + n_new > max_len_)
    {
        fail(std::format("KVCache 装不下了：想写到位置 {}，容量只有 {}。"
                         "GPT-2 的上下文上限是 {} 个 token。",
                         pos + n_new, max_len_, cfg_.n_ctx));
    }
    const auto off = static_cast<std::ptrdiff_t>(pos) * static_cast<std::ptrdiff_t>(C);
    std::ranges::copy(k, k_[static_cast<std::size_t>(layer)].data().begin() + off);
    std::ranges::copy(v, v_[static_cast<std::size_t>(layer)].data().begin() + off);
    len_ = std::max(len_, pos + n_new);
}
std::span<const float> KVCache::keys(int layer) const
{
    return k_[static_cast<std::size_t>(layer)].data().subspan(
        0, static_cast<std::size_t>(len_) * static_cast<std::size_t>(cfg_.n_embd));
}
std::span<const float> KVCache::values(int layer) const
{
    return v_[static_cast<std::size_t>(layer)].data().subspan(
        0, static_cast<std::size_t>(len_) * static_cast<std::size_t>(cfg_.n_embd));
}

std::size_t KVCache::bytes() const noexcept
{
    return static_cast<std::size_t>(cfg_.n_layer) * 2U * static_cast<std::size_t>(max_len_) *
           static_cast<std::size_t>(cfg_.n_embd) * sizeof(float);
}

// ---------------------------------------------------------------------------
// Sampler
//
// 模型输出的是 50257 个 logits，不是「答案」。怎么从分布里挑一个词，
// 直接决定了生成文本的风格：
//
//   temperature=0   永远选概率最大的            → 确定、重复、无聊
//   temperature=0.8 压一压分布再随机采样         → 常用默认值
//   temperature=1.5 拉平分布                    → 天马行空、容易胡说
//
//   top_k=40        只在最可能的 40 个里选       → 砍掉长尾的怪词
//   top_p=0.9       只在累计概率 90% 的集合里选  → 自适应的 top_k
// ---------------------------------------------------------------------------
float Sampler::next_uniform()
{
    // xorshift64*，够用且不依赖 <random> 的实现差异（换机器结果一致，方便复现）
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    const std::uint64_t x = state_ * 0x2545F4914F6CDD1DULL;
    return static_cast<float>((x >> 11) * (1.0 / 9007199254740992.0));
}

int Sampler::sample(std::span<const float> logits)
{
    const auto n = static_cast<int>(logits.size());
    if (n == 0)
        fail("Sampler: logits 是空的");

    // 贪心：温度为 0 就是「只选最大的」，不用算 softmax
    if (cfg_.temperature <= 0.0F)
    {
        return static_cast<int>(std::ranges::max_element(logits) - logits.begin());
    }

    // 按 logit 从大到小排序，后面 top-k / top-p 都在这个顺序上做
    std::vector<int> idx(static_cast<std::size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    int keep = n;
    if (cfg_.top_k > 0 && cfg_.top_k < n)
        keep = cfg_.top_k;

    std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(static_cast<std::size_t>(keep));

    // 带温度的 softmax。温度就是除在 logit 上的那个数：
    // 除以小于 1 的数 = 放大差距 = 更确定；除以大于 1 = 缩小差距 = 更随机。
    const float maxv = logits[idx[0]];
    std::vector<float> prob(idx.size());
    float sum = 0.0F;
    for (std::size_t i = 0; i < idx.size(); ++i)
    {
        prob[i] = std::exp((logits[idx[i]] - maxv) / cfg_.temperature);
        sum += prob[i];
    }
    for (float &p : prob)
        p /= sum;

    // top-p（核采样）：从最大的开始累加概率，够 p 了就把后面的全扔掉
    if (cfg_.top_p < 1.0F)
    {
        float acc = 0.0F;
        std::size_t cut = prob.size();
        for (std::size_t i = 0; i < prob.size(); ++i)
        {
            acc += prob[i];
            if (acc >= cfg_.top_p)
            {
                cut = i + 1;
                break;
            }
        }
        prob.resize(cut);
        idx.resize(cut);
        const float s = std::accumulate(prob.begin(), prob.end(), 0.0F);
        for (float &p : prob)
            p /= s;
    }

    // 轮盘赌
    const float r = next_uniform();
    float acc = 0.0F;
    for (std::size_t i = 0; i < prob.size(); ++i)
    {
        acc += prob[i];
        if (r < acc)
            return idx[i];
    }
    return idx.back();
}

// ---------------------------------------------------------------------------
// GPT 加载
// ---------------------------------------------------------------------------
GPT GPT::load(const std::string &model_dir)
{
    GPT m;
    const ModelFile cfg_file = ModelFile::load(model_dir + "/gpt2_config.inference");
    auto read_int = [&](const char *name)
    { return static_cast<int>(std::lround(cfg_file.get_tensor(name).data()[0])); };
    m.cfg_.n_layer = read_int("n_layer");
    m.cfg_.n_head = read_int("n_head");
    m.cfg_.n_embd = read_int("n_embd");
    m.cfg_.n_vocab = read_int("n_vocab");
    m.cfg_.n_ctx = read_int("n_ctx");

    // ModelFile 内部持有整个文件的字节。get_tensor() 会把张量拷出来，
    // 所以这里用个作用域，让 ModelFile 在取完权重后立刻释放 —— 否则 500 MB 白占。
    {
        const ModelFile w = ModelFile::load(model_dir + "/gpt2.inference");
        m.wte_ = w.get_tensor("wte.weight");
        m.wpe_ = w.get_tensor("wpe.weight");
        m.lnf_w_ = w.get_tensor("ln_f.weight");
        m.lnf_b_ = w.get_tensor("ln_f.bias");

        m.layers_.reserve(static_cast<std::size_t>(m.cfg_.n_layer));
        for (int i = 0; i < m.cfg_.n_layer; ++i)
        {
            const std::string p = std::format("h.{}.", i);
            Layer L;
            L.ln1_w = w.get_tensor(p + "ln_1.weight");
            L.ln1_b = w.get_tensor(p + "ln_1.bias");
            L.ln2_w = w.get_tensor(p + "ln_2.weight");
            L.ln2_b = w.get_tensor(p + "ln_2.bias");
            L.attn_w = w.get_tensor(p + "attn.c_attn.weight");
            L.attn_b = w.get_tensor(p + "attn.c_attn.bias");
            L.attn_proj_w = w.get_tensor(p + "attn.c_proj.weight");
            L.attn_proj_b = w.get_tensor(p + "attn.c_proj.bias");
            L.fc_w = w.get_tensor(p + "mlp.c_fc.weight");
            L.fc_b = w.get_tensor(p + "mlp.c_fc.bias");
            L.proj_w = w.get_tensor(p + "mlp.c_proj.weight");
            L.proj_b = w.get_tensor(p + "mlp.c_proj.bias");
            m.layers_.push_back(std::move(L));
        }
    }

    // 形状自检。权重文件对不上时，与其让后面某个算子抛个看不懂的错，
    // 不如在这里说清楚哪里不对。
    const int C = m.cfg_.n_embd;
    if (m.wte_.dim(1) != C || m.wte_.dim(0) != m.cfg_.n_vocab)
    {
        fail(std::format("wte 形状应为 ({}, {})，实际 ({}, {})", m.cfg_.n_vocab, C, m.wte_.dim(0),
                         m.wte_.dim(1)));
    }
    if (m.layers_[0].attn_w.dim(0) != C || m.layers_[0].attn_w.dim(1) != 3 * C)
    {
        fail(std::format("c_attn 形状应为 ({}, {})，实际 ({}, {})。"
                         "注意 HuggingFace 的 Conv1D 是 (in, out) 布局，别转置。",
                         C, 3 * C, m.layers_[0].attn_w.dim(0), m.layers_[0].attn_w.dim(1)));
    }
    return m;
}

// ---------------------------------------------------------------------------
// 前向
// ---------------------------------------------------------------------------

Tensor GPT::forward_impl(std::span<const int> tokens, int start_pos, KVCache *cache, Trace *trace,
                         bool last_only)
{
    const auto T = static_cast<int64>(tokens.size());
    if (T == 0)
        fail("forward: token 序列是空的");
    const int64 C = cfg_.n_embd;
    if (start_pos + T > cfg_.n_ctx)
    {
        fail(std::format("序列长度 {} 超过 GPT-2 的上下文上限 {}", start_pos + T, cfg_.n_ctx));
    }

    // --- 1. 词嵌入 + 位置嵌入 ---
    //
    // GPT-2 用的是**学出来的**位置嵌入：wpe 是一张 [1024, 768] 的表，
    // 第 p 行就是「第 p 个位置」的向量，直接加到词向量上。
    // 这也是 GPT-2 最多只能处理 1024 个 token 的原因 —— 表就这么大。
    // （后来的模型改用 RoPE 等相对位置编码，就没这个硬上限了。）
    OpProfile *const prof = profile_;
    if (prof)
        ++prof->forwards;
    auto slot = [prof](double OpProfile::*field) { return prof ? &(prof->*field) : nullptr; };

    Tensor x;
    {
        ScopedTimer tm(slot(&OpProfile::embed));
        x = ops::embedding(wte_, tokens);
        std::span<float> xv = x.data();
        std::span<const float> pe = wpe_.data();
        for (int64 t = 0; t < T; ++t)
        {
            const auto src = static_cast<std::size_t>(start_pos + t) * static_cast<std::size_t>(C);
            const auto dst = static_cast<std::size_t>(t) * static_cast<std::size_t>(C);
            for (int64 j = 0; j < C; ++j)
                xv[dst + static_cast<std::size_t>(j)] += pe[src + static_cast<std::size_t>(j)];
        }
    }
    if (trace)
        trace->embed = x;

    // --- 2. 12 个 Transformer Block ---
    for (int li = 0; li < cfg_.n_layer; ++li)
    {
        const Layer &L = layers_[static_cast<std::size_t>(li)];

        // ---- 注意力子层 ----
        // 注意 LayerNorm 在**进入**子层前做（pre-norm）。原版 Transformer 是
        // 做完子层再 norm（post-norm），GPT-2 改成了 pre-norm，训练更稳。
        Tensor h;
        {
            ScopedTimer tm(slot(&OpProfile::layernorm));
            h = ops::layernorm(x, L.ln1_w, L.ln1_b);
        }

        // 一次矩阵乘同时算出 Q、K、V：权重 [C, 3C] 是三个 [C, C] 拼起来的。
        // 合并成一次比分三次快 —— 大矩阵乘的访存效率更高。
        Tensor qkv;
        {
            ScopedTimer tm(slot(&OpProfile::qkv));
            qkv = ops::affine(h, L.attn_w, L.attn_b); // [T, 3C]
        }

        Tensor q, k, v;
        {
            ScopedTimer tm(slot(&OpProfile::copy));
            q = qkv.slice(1, 0, C).contiguous();
            k = qkv.slice(1, C, 2 * C).contiguous();
            v = qkv.slice(1, 2 * C, 3 * C).contiguous();
        }

        Tensor y;
        if (cache)
        {
            // 新算出的 K/V 存进缓存，然后拿**全部历史** K/V 做注意力。
            // q_offset = start_pos 告诉 kernel：我这批 query 的绝对位置从哪开始。
            Tensor ck, cv;
            {
                ScopedTimer tm(slot(&OpProfile::copy));
                cache->append(li, start_pos, k.data(), v.data());
                ck = Tensor::zeros({cache->length(), C});
                cv = Tensor::zeros({cache->length(), C});
                std::ranges::copy(cache->keys(li), ck.data().begin());
                std::ranges::copy(cache->values(li), cv.data().begin());
            }
            ScopedTimer tm(slot(&OpProfile::attention));
            y = ops::attention(q, ck, cv, cfg_.n_head, start_pos);
        }
        else
        {
            ScopedTimer tm(slot(&OpProfile::attention));
            y = ops::attention(q, k, v, cfg_.n_head, start_pos);
        }

        {
            ScopedTimer tm(slot(&OpProfile::attn_proj));
            y = ops::affine(y, L.attn_proj_w, L.attn_proj_b);
        }
        {
            ScopedTimer tm(slot(&OpProfile::residual));
            x = ops::add(x, y); // 残差连接
        }
        if (trace && li == 0)
            trace->l0_attn_out = x;

        // ---- MLP 子层 ----
        // 先升维到 4C（3072）再降回来。这一段占了模型 2/3 的计算量，
        // 一般认为它是模型「存事实知识」的地方。
        {
            ScopedTimer tm(slot(&OpProfile::layernorm));
            h = ops::layernorm(x, L.ln2_w, L.ln2_b);
        }
        {
            ScopedTimer tm(slot(&OpProfile::mlp_fc));
            h = ops::affine(h, L.fc_w, L.fc_b);
        }
        {
            ScopedTimer tm(slot(&OpProfile::gelu));
            h = ops::gelu(h);
        }
        {
            ScopedTimer tm(slot(&OpProfile::mlp_proj));
            h = ops::affine(h, L.proj_w, L.proj_b);
        }
        {
            ScopedTimer tm(slot(&OpProfile::residual));
            x = ops::add(x, h); // 残差连接
        }
        if (trace && li == 0)
            trace->l0_out = x;
    }

    // --- 3. 最后的 LayerNorm + 输出投影 ---
    {
        ScopedTimer tm(slot(&OpProfile::layernorm));
        x = ops::layernorm(x, lnf_w_, lnf_b_);
    }
    if (trace)
        trace->final_norm = x;

    ScopedTimer tm(slot(&OpProfile::lm_head));
    // 只要最后一个位置的 logits 时，别把 T×50257 全算出来。
    // T=5 时这一步能省 80% 的输出层计算；prompt 越长省得越多。
    if (last_only && T > 1)
        x = x.slice(0, T - 1, T).contiguous();

    // 输出层复用词嵌入矩阵（weight tying）：logits = x @ wteᵀ。
    // wte 是 [n_vocab, C]，正好是 ops::linear 要的 (out, in) 布局。
    Tensor logits = ops::linear(x, wte_);
    if (trace)
        trace->logits = logits;
    return logits;
}

Tensor GPT::forward(std::span<const int> tokens)
{
    return forward_impl(tokens, 0, nullptr, nullptr, /*last_only=*/false);
}

Tensor GPT::forward_one(int token, int pos, KVCache &cache)
{
    const int one[1] = {token};
    Tensor logits = forward_impl(one, pos, &cache, nullptr, /*last_only=*/true);
    return logits.reshape({cfg_.n_vocab});
}

GPT::Trace GPT::forward_trace(std::span<const int> tokens)
{
    Trace t;
    forward_impl(tokens, 0, nullptr, &t, /*last_only=*/false);
    return t;
}

// ---------------------------------------------------------------------------
// 生成
//
//   prompt: [The][capital][of][France][is]
//              │
//              ▼  prefill：一次矩阵乘处理 5 个 token，顺便填满 KV-Cache
//          logits ──sample──→ [ Paris]
//              │
//              ▼  decode：每次只算 1 个 token，读 KV-Cache
//          logits ──sample──→ [.]  ──→ ...
//
// prefill 是计算密集的（大矩阵乘，吃满 CPU），
// decode 是访存密集的（每生成一个词都要把 124M 权重完整读一遍）。
// 这就是为什么 LLM 的「首字延迟」和「每字延迟」要分开看。
// ---------------------------------------------------------------------------
std::vector<int> GPT::generate(std::span<const int> prompt, int max_new_tokens, Sampler &sampler,
                               GenStats *stats, const std::function<void(int)> &on_token)
{
    if (prompt.empty())
        fail("generate: prompt 不能为空");

    const int budget = std::min(max_new_tokens, cfg_.n_ctx - static_cast<int>(prompt.size()));
    KVCache cache(cfg_, static_cast<int>(prompt.size()) + std::max(budget, 0));

    struct ProfileReset
    {
        OpProfile *&p;
        ~ProfileReset() { p = nullptr; }
    } profile_reset{profile_};
    OpProfile *const prefill_prof = stats ? &stats->prefill_ops : nullptr;
    OpProfile *const decode_prof = stats ? &stats->decode_ops : nullptr;
    auto timed_sample = [&](const Tensor &lg, OpProfile *prof)
    {
        ScopedTimer tm(prof ? &prof->sample : nullptr);
        return sampler.sample(lg.data());
    };

    // --- prefill ---
    profile_ = prefill_prof;
    const auto t0 = Clock::now();
    Tensor logits = forward_impl(prompt, 0, &cache, nullptr, /*last_only=*/true);
    const double prefill_ms = ms_since(t0);

    std::vector<int> out(prompt.begin(), prompt.end());
    int next = timed_sample(logits, prefill_prof);

    // --- decode ---
    profile_ = decode_prof;
    const auto t1 = Clock::now();
    int generated = 0;
    for (int i = 0; i < budget; ++i)
    {
        out.push_back(next);
        ++generated;
        if (on_token)
            on_token(next);
        if (i + 1 == budget)
            break;
        const int pos = static_cast<int>(out.size()) - 1;
        logits = forward_one(next, pos, cache);
        next = timed_sample(logits, decode_prof);
    }

    if (stats)
    {
        stats->prefill_ms = prefill_ms;
        stats->decode_ms = ms_since(t1);
        stats->generated = generated;
    }
    return out;
}
} // namespace inference