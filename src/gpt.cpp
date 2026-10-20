#include "inference/gpt.hpp"
#include "inference/kernel.hpp"
#include "inference/model_file.hpp"
#include "inference/ops.hpp"
#include "inference/tensor.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
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

// HuggingFace 的 Conv1D 把权重存成 [in, out]，这里转成 PyTorch Linear 的 [out, in]。
// 这样每个输出对应 W 的一行连续内存，整个模型只需要 matmul_nt / gemv_nt 一套内核。
// 只在加载时做一次，不走 Tensor::contiguous() 的通用多维下标循环 —— 那个对 8500 万个元素太慢。
// 按 32×32 分块转置：朴素写法每写一个元素就跨一整行，几乎每次都 cache miss；
// 分块后读写两边都落在 32 条 cache line 以内。
Tensor to_out_in(const Tensor &w)
{
    constexpr int64 kBlock = 32;
    const int64 rows = w.dim(0), cols = w.dim(1);
    const Tensor src = w.contiguous();
    Tensor dst = Tensor::zeros({cols, rows});
    const float *s = src.data().data();
    float *d = dst.data().data();
    for (int64 r0 = 0; r0 < rows; r0 += kBlock)
        for (int64 c0 = 0; c0 < cols; c0 += kBlock)
        {
            const int64 r1 = std::min(r0 + kBlock, rows), c1 = std::min(c0 + kBlock, cols);
            for (int64 r = r0; r < r1; ++r)
                for (int64 c = c0; c < c1; ++c)
                    d[c * rows + r] = s[r * cols + c];
        }
    return dst;
}

// ---- forward_impl 用的「无分配」算子 ----
// ops:: 里的算子每次都返回一个新 Tensor（分配 + 清零）。前向里每层要调十几次，
// decode 每个 token 就是几百次堆分配。这里直接在调用者给的缓冲区上算。

// out[r] = layernorm(x[r])，x 和 out 可以是同一块内存
void layernorm_rows(const float *x, const Tensor &w, const Tensor &b, float *out, int64 rows,
                    int64 cols)
{
    const auto n = static_cast<std::size_t>(cols);
    for (int64 r = 0; r < rows; ++r)
    {
        kernel::layernorm_f32({x + r * cols, n}, w.data(), b.data(), {out + r * cols, n}, 1e-5F);
    }
}

// out[M, N] = a[M, K] × wᵀ（w 是 [N, K]）。有量化版就用量化版，否则用 fp32
void matmul_weight(const float *a, const Tensor &w, const QMatrixQ8 &wq, float *out, int64 M,
                   int64 N, int64 K)
{
    const std::span<const float> as{a, static_cast<std::size_t>(M * K)};
    const std::span<float> os{out, static_cast<std::size_t>(M * N)};
    if (!wq.empty())
        kernel::matmul_nt_q8_0(as, wq.q.data_as<std::int8_t>().data(), wq.scale.data().data(), os,
                               M, N, K);
    else
        kernel::matmul_nt_f32(as, w.data(), os, M, N, K);
}

// out[M, N] = a[M, K] × wᵀ + bias
void linear_bias(const float *a, const Tensor &w, const QMatrixQ8 &wq, const Tensor &bias,
                 float *out, int64 M, int64 N, int64 K)
{
    matmul_weight(a, w, wq, out, M, N, K);
    const float *b = bias.data().data();
    for (int64 r = 0; r < M; ++r)
    {
        float *o = out + r * N;
        for (int64 j = 0; j < N; ++j)
            o[j] += b[j];
    }
}

// x += y（残差连接），原地
void add_inplace(float *x, const float *y, int64 n)
{
    for (int64 i = 0; i < n; ++i)
        x[i] += y[i];
}

// 把缓冲区里的一段拷成 Tensor，只给 Trace（对拍）用
Tensor to_tensor(const float *src, int64 rows, int64 cols)
{
    Tensor t = Tensor::zeros({rows, cols});
    std::copy_n(src, rows * cols, t.data().begin());
    return t;
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
        k_.push_back(Tensor::zeros({cfg.n_head, max_len_, cfg.head_size()}));
        v_.push_back(Tensor::zeros({cfg.n_head, max_len_, cfg.head_size()}));
    }
}

void KVCache::append(int layer, int pos, std::span<const float> k, std::span<const float> v)
{
    const auto C = static_cast<std::size_t>(cfg_.n_embd);
    if (k.size() != v.size() || k.size() % C != 0)
    {
        fail("KVCache::append: K/V 长度必须相等且是 n_embd 的整数倍");
    }
    append_strided(layer, pos, k.data(), v.data(), static_cast<int>(k.size() / C), cfg_.n_embd);
}

void KVCache::append_strided(int layer, int pos, const float *k, const float *v, int n_rows,
                             std::int64_t row_stride)
{
    if (pos < 0 || pos + n_rows > max_len_)
    {
        fail(std::format("KVCache 装不下了：想写到位置 {}，容量只有 {}。"
                         "GPT-2 的上下文上限是 {} 个 token。",
                         pos + n_rows, max_len_, cfg_.n_ctx));
    }
    // 输入是按位置排的 [n_rows][n_embd]，缓存是按头排的 [n_head][capacity][head_size]：
    // 每行切成 n_head 段，各自放到对应头的第 pos + r 个位置
    const std::int64_t hs = cfg_.head_size();
    const std::int64_t hstride = head_stride();
    float *kd = k_[static_cast<std::size_t>(layer)].data().data();
    float *vd = v_[static_cast<std::size_t>(layer)].data().data();
    for (int r = 0; r < n_rows; ++r)
    {
        const float *krow = k + r * row_stride;
        const float *vrow = v + r * row_stride;
        for (int h = 0; h < cfg_.n_head; ++h)
        {
            const std::int64_t dst = h * hstride + (pos + r) * hs;
            std::copy_n(krow + h * hs, hs, kd + dst);
            std::copy_n(vrow + h * hs, hs, vd + dst);
        }
    }
    len_ = std::max(len_, pos + n_rows);
}

std::span<const float> KVCache::keys(int layer) const
{
    return k_[static_cast<std::size_t>(layer)].data();
}
std::span<const float> KVCache::values(int layer) const
{
    return v_[static_cast<std::size_t>(layer)].data();
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
            L.attn_w = to_out_in(w.get_tensor(p + "attn.c_attn.weight"));
            L.attn_b = w.get_tensor(p + "attn.c_attn.bias");
            L.attn_proj_w = to_out_in(w.get_tensor(p + "attn.c_proj.weight"));
            L.attn_proj_b = w.get_tensor(p + "attn.c_proj.bias");
            L.fc_w = to_out_in(w.get_tensor(p + "mlp.c_fc.weight"));
            L.fc_b = w.get_tensor(p + "mlp.c_fc.bias");
            L.proj_w = to_out_in(w.get_tensor(p + "mlp.c_proj.weight"));
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
    if (m.layers_[0].attn_w.dim(0) != 3 * C || m.layers_[0].attn_w.dim(1) != C)
    {
        fail(std::format("c_attn 转置后形状应为 ({}, {})，实际 ({}, {})。"
                         "权重文件里应该是 HuggingFace Conv1D 的 (in, out) 布局，加载时才转成 (out, in)。",
                         3 * C, C, m.layers_[0].attn_w.dim(0), m.layers_[0].attn_w.dim(1)));
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

    // 所有中间结果都放在复用的工作区里。resize 只在 T 变大时才真正分配内存，
    // decode 阶段（T 恒为 1）每个 token 都是零分配。
    const int64 C3 = 3 * C, C4 = 4 * C;
    const int64 H = cfg_.n_head, hs = cfg_.head_size();
    ws_.x.resize(static_cast<std::size_t>(T * C));
    ws_.h.resize(static_cast<std::size_t>(T * C));
    ws_.qkv.resize(static_cast<std::size_t>(T * C3));
    ws_.att.resize(static_cast<std::size_t>(T * C));
    ws_.fc.resize(static_cast<std::size_t>(T * C4));
    float *const x = ws_.x.data();
    float *const h = ws_.h.data();
    float *const qkv = ws_.qkv.data();
    float *const att = ws_.att.data();
    float *const fc = ws_.fc.data();

    {
        ScopedTimer tm(slot(&OpProfile::embed));
        const float *te = wte_.data().data();
        const float *pe = wpe_.data().data();
        for (int64 t = 0; t < T; ++t)
        {
            const int id = tokens[static_cast<std::size_t>(t)];
            if (id < 0 || id >= cfg_.n_vocab)
                fail(std::format("token id {} 越界，词表只有 {} 项", id, cfg_.n_vocab));
            const float *tok_row = te + static_cast<int64>(id) * C;
            const float *pos_row = pe + (start_pos + t) * C;
            for (int64 j = 0; j < C; ++j)
                x[t * C + j] = tok_row[j] + pos_row[j];
        }
    }
    if (trace)
        trace->embed = to_tensor(x, T, C);

    // --- 2. 12 个 Transformer Block ---
    for (int li = 0; li < cfg_.n_layer; ++li)
    {
        const Layer &L = layers_[static_cast<std::size_t>(li)];

        // ---- 注意力子层 ----
        // 注意 LayerNorm 在**进入**子层前做（pre-norm）。原版 Transformer 是
        // 做完子层再 norm（post-norm），GPT-2 改成了 pre-norm，训练更稳。
        {
            ScopedTimer tm(slot(&OpProfile::layernorm));
            layernorm_rows(x, L.ln1_w, L.ln1_b, h, T, C);
        }

        // 一次矩阵乘同时算出 Q、K、V：权重 [3C, C] 是三个 [C, C] 上下拼起来的。
        // 合并成一次比分三次快 —— 大矩阵乘的访存效率更高。
        // 输出 [T, 3C] 的每一行是 [q | k | v]，后面直接按行距 3C 读，不拆成三个张量。
        {
            ScopedTimer tm(slot(&OpProfile::qkv));
            linear_bias(h, L.attn_w, L.attn_wq, L.attn_b, qkv, T, C3, C);
        }

        // k/v 从哪读：不用 cache 时直接读 qkv 里的 k/v 段（行距 3C），
        // 用 cache 时读缓存里的全部历史（按头存放，见 KVCache）
        kernel::HeadView kv_k{qkv + C, C3, hs};
        kernel::HeadView kv_v{qkv + 2 * C, C3, hs};
        int64 Tk = T;
        if (cache)
        {
            // 新算出的 K/V 写进缓存，然后拿**全部历史** K/V 做注意力。
            // 注意力直接读缓存本身 —— 以前这里每层都把整段缓存拷出来一份，
            // 生成第 n 个 token 时要拷 n 行，KV-Cache 想省掉的 O(n) 又被加了回来。
            ScopedTimer tm(slot(&OpProfile::copy));
            cache->append_strided(li, start_pos, qkv + C, qkv + 2 * C, static_cast<int>(T), C3);
            kv_k = {cache->keys(li).data(), hs, cache->head_stride()};
            kv_v = {cache->values(li).data(), hs, cache->head_stride()};
            Tk = cache->length();
        }
        {
            // q_offset = start_pos 告诉 kernel：这批 query 的绝对位置从哪开始
            ScopedTimer tm(slot(&OpProfile::attention));
            kernel::attention_f32_strided({qkv, C3, hs}, kv_k, kv_v, att, C, T, Tk, H, hs,
                                          start_pos);
        }

        {
            ScopedTimer tm(slot(&OpProfile::attn_proj));
            linear_bias(att, L.attn_proj_w, L.attn_proj_wq, L.attn_proj_b, h, T, C, C);
        }
        {
            ScopedTimer tm(slot(&OpProfile::residual));
            add_inplace(x, h, T * C); // 残差连接
        }
        if (trace && li == 0)
            trace->l0_attn_out = to_tensor(x, T, C);

        // ---- MLP 子层 ----
        // 先升维到 4C（3072）再降回来。这一段占了模型 2/3 的计算量，
        // 一般认为它是模型「存事实知识」的地方。
        {
            ScopedTimer tm(slot(&OpProfile::layernorm));
            layernorm_rows(x, L.ln2_w, L.ln2_b, h, T, C);
        }
        {
            ScopedTimer tm(slot(&OpProfile::mlp_fc));
            linear_bias(h, L.fc_w, L.fc_wq, L.fc_b, fc, T, C4, C);
        }
        {
            ScopedTimer tm(slot(&OpProfile::gelu));
            const auto n = static_cast<std::size_t>(T * C4);
            kernel::gelu_f32({fc, n}, {fc, n}); // 原地
        }
        {
            ScopedTimer tm(slot(&OpProfile::mlp_proj));
            linear_bias(fc, L.proj_w, L.proj_wq, L.proj_b, h, T, C, C4);
        }
        {
            ScopedTimer tm(slot(&OpProfile::residual));
            add_inplace(x, h, T * C); // 残差连接
        }
        if (trace && li == 0)
            trace->l0_out = to_tensor(x, T, C);
    }

    // --- 3. 最后的 LayerNorm + 输出投影 ---
    {
        ScopedTimer tm(slot(&OpProfile::layernorm));
        layernorm_rows(x, lnf_w_, lnf_b_, h, T, C);
    }
    if (trace)
        trace->final_norm = to_tensor(h, T, C);

    ScopedTimer tm(slot(&OpProfile::lm_head));
    // 只要最后一个位置的 logits 时，别把 T×50257 全算出来。
    // T=5 时这一步能省 80% 的输出层计算；prompt 越长省得越多。
    const int64 rows = last_only ? 1 : T;
    const float *last = h + (T - rows) * C;

    // 输出层复用词嵌入矩阵（weight tying）：logits = x @ wteᵀ。
    // wte 是 [n_vocab, C]，正好是 matmul_nt 要的 (out, in) 布局。
    // logits 是返回值，只有它每次新分配。
    Tensor logits = Tensor::zeros({rows, cfg_.n_vocab});
    const std::span<const float> last_span{last, static_cast<std::size_t>(rows * C)};
    if (!lm_head_f16_.empty())
        kernel::matmul_nt_f16(last_span, lm_head_f16_.data_as<std::uint16_t>().data(), logits.data(),
                              rows, cfg_.n_vocab, C);
    else
        matmul_weight(last, lm_head_f32_.numel() ? lm_head_f32_ : wte_, lm_head_q_,
                      logits.data().data(), rows, cfg_.n_vocab, C);
    if (trace)
        trace->logits = logits;
    return logits;
}

// ---------------------------------------------------------------------------
// 量化
// ---------------------------------------------------------------------------
QMatrixQ8 QMatrixQ8::quantize(const Tensor &w)
{
    if (w.ndim() != 2 || w.dim(1) % kernel::kQ8Block != 0)
        fail(std::format("Q8_0 量化要求二维权重且列数是 {} 的倍数，拿到 {}", kernel::kQ8Block,
                         to_string(w.shape())));
    QMatrixQ8 m;
    m.rows = w.dim(0);
    m.cols = w.dim(1);
    const int64 nb = m.cols / kernel::kQ8Block;
    m.q = Tensor::zeros({m.rows, m.cols}, DType::INT8);
    m.scale = Tensor::zeros({m.rows, nb});
    const Tensor src = w.contiguous();
    const float *s = src.data().data();
    std::int8_t *q = m.q.data_as<std::int8_t>().data();
    float *sc = m.scale.data().data();
    for (int64 r = 0; r < m.rows; ++r)
        kernel::quantize_row_q8_0(s + r * m.cols, q + r * m.cols, sc + r * nb, m.cols);
    return m;
}

namespace
{
Tensor dequantize(const QMatrixQ8 &m)
{
    Tensor t = Tensor::zeros({m.rows, m.cols});
    kernel::dequantize_row_q8_0(m.q.data_as<std::int8_t>().data(), m.scale.data().data(),
                                t.data().data(), m.rows * m.cols);
    return t;
}
} // namespace

void GPT::quantize(const QuantOptions &opt)
{
    if (quantized_)
        fail("GPT::quantize 只能调用一次（fp32 权重在第一次量化时已经释放）");

    // weight_only：存「量化再反量化」的 fp32 权重，forward 仍走 fp32 kernel，激活不量化
    auto apply = [&](bool enabled, Tensor &w, QMatrixQ8 &wq)
    {
        if (!enabled)
            return;
        QMatrixQ8 q = QMatrixQ8::quantize(w);
        if (opt.weight_only)
        {
            w = dequantize(q);
            return;
        }
        wq = std::move(q);
        w = Tensor{}; // fp32 版本不再需要，释放掉：量化的意义之一就是省内存
    };
    for (Layer &L : layers_)
    {
        apply(opt.qkv, L.attn_w, L.attn_wq);
        apply(opt.attn_proj, L.attn_proj_w, L.attn_proj_wq);
        apply(opt.mlp_fc, L.fc_w, L.fc_wq);
        apply(opt.mlp_proj, L.proj_w, L.proj_wq);
    }

    // wte 既是词嵌入表（查表只读一行，不值得压缩），也是 lm_head（每 token 整张读一遍）。
    // 保留 fp32 的 wte 查表，另存一份压缩版专门给 lm_head 用
    switch (opt.lm_head)
    {
    case QuantOptions::LmHead::fp32:
        break;
    case QuantOptions::LmHead::fp16:
    {
        const Tensor wte = wte_.contiguous();
        lm_head_f16_ = Tensor::zeros(wte.shape(), DType::FP16);
        kernel::convert_f32_to_f16(wte.data().data(), lm_head_f16_.data_as<std::uint16_t>().data(),
                                   lm_head_f16_.numel());
        break;
    }
    case QuantOptions::LmHead::q8_0:
        if (opt.weight_only)
            lm_head_f32_ = dequantize(QMatrixQ8::quantize(wte_));
        else
            lm_head_q_ = QMatrixQ8::quantize(wte_);
        break;
    }
    quantized_ = true;
    // 之前建好的图引用的是 fp32 权重：既会算错版本，又会让刚释放的 fp32 权重继续占着内存
    graphs_.clear();
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
                               GenStats *stats, const std::function<void(int)> &on_token,
                               LogitsProcessor *processor)
{
    return generate_with([this](std::span<const int> tokens, int start_pos, KVCache &cache)
                         { return forward_impl(tokens, start_pos, &cache, nullptr, /*last_only=*/true); },
                         prompt, max_new_tokens, sampler, stats, on_token, processor);
}

std::vector<int> GPT::generate_with(const StepFn &step, std::span<const int> prompt,
                                    int max_new_tokens, Sampler &sampler, GenStats *stats,
                                    const std::function<void(int)> &on_token,
                                    LogitsProcessor *processor)
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

    // --- prefill ---
    profile_ = prefill_prof;
    const auto t0 = Clock::now();
    Tensor logits = step(prompt, 0, cache);
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
        const int one[1] = {next};
        logits = step(one, pos, cache);
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