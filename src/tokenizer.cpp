#include "inference/tokenizer.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace inference
{
std::array<std::string, 256> make_byte_encoder()
{
    std::array<int, 256> codepoint{};
    std::array<bool, 256> keep{};
    auto mark = [&](int lo, int hi)
    {
        for (int b = lo; b <= hi; ++b)
        {
            keep[b] = true;
            codepoint[b] = b;
        }
    };
    // 保留所有 ASCII 字符和拉丁文扩展字符
    mark('!', '~');
    mark(0xA1, 0xAC);
    mark(0xAE, 0xFF);

    // 剩下的字节用一个未定义的码位表示，并添加到词表末尾
    int extra = 0;
    for (int b = 0; b < 256; ++b)
    {
        if (!keep[b])
            codepoint[b] = 256 + extra++;
    }
    // 把码位编成 UTF-8。这里码位最大 323，只会用到 1 或 2 字节。
    std::array<std::string, 256> table;
    for (int b = 0; b < 256; ++b)
    {
        const int cp = codepoint[b];

        if (cp < 0x80)
        {
            // 1 字节 UTF-8 直接转字符串
            table[b] = std::string(1, static_cast<char>(cp));
        }
        else
        {
            // 2 字节 UTF-8 需要转换
            table[b] = {static_cast<char>(0xC0 | (cp >> 6)), static_cast<char>(0x80 | (cp & 0x3F))};
        }
    }
    return table;
}

const std::array<std::string, 256> &byte_encoder()
{
    static const auto table = make_byte_encoder();
    return table;
}

// 反向表：UTF-8 字符串 → 原始字节
const std::unordered_map<std::string, unsigned char> &byte_decoder()
{
    static const auto table = []
    {
        std::unordered_map<std::string, unsigned char> m;
        const auto &enc = byte_encoder();
        for (int b = 0; b < 256; ++b)
            m.emplace(enc[b], static_cast<unsigned char>(b));
        return m;
    }();
    return table;
}

bool is_letter(unsigned char c)
{
    // 非 ASCII 字节一律当字母处理。GPT-2 原版用的是 Unicode 属性 \p{L}，
    // 需要完整的 Unicode 表。这个近似对英文完全正确，对中文等会切得偏碎
    // （多出几个 token），但不影响正确性。
    return std::isalpha(c) != 0 || c >= 0x80;
}

bool is_digit(unsigned char c)
{
    return c >= '0' && c <= '9';
}
bool is_space(unsigned char c)
{
    return std::isspace(c) != 0;
}

// 读定长小端整数
template <class T> T read_le(std::ifstream &f)
{
    T v{};
    f.read(reinterpret_cast<char *>(&v), sizeof(T));
    if (!f)
        throw std::runtime_error("文件提前结束");
    return v;
}

std::string read_sized_string(std::ifstream &f)
{
    const auto n = read_le<std::uint32_t>(f);
    std::string s(n, '\0');
    if (n > 0)
        f.read(s.data(), n);
    if (!f)
        throw std::runtime_error("读字符串时文件提前结束");
    return s;
}

// ---------------------------------------------------------------------------
// 预分词
//
// BPE 不是直接作用在整段文本上，而是先按规则把文本切成「词块」，
// 每块单独做 BPE。这样能保证合并不会跨越单词边界。
//
// GPT-2 的原始规则是一个 Unicode 正则：
//     's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// std::regex 不支持 \p{L} 这种 Unicode 属性，所以这里手写等价逻辑。
// 注意规则里的「可选前导空格」—— 这就是为什么 GPT-2 的 token 大多长这样：
// " the"、" capital"，空格是跟在单词**前面**的。
// ---------------------------------------------------------------------------
std::vector<std::string> Tokenizer::pre_tokenize(std::string_view text)
{
    std::vector<std::string> out;
    std::size_t i = 0;
    const std::size_t n = text.size();

    auto starts_with = [&](std::size_t pos, std::string_view s)
    { return pos + s.size() <= n && text.compare(pos, s.size(), s) == 0; };
    while (i < n)
    {
        // 1) 英语缩写：'s 't 're 've 'm 'll 'd
        if (text[i] == '\'')
        {
            bool matched = false;
            for (std::string_view suffix : {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"})
            {
                if (starts_with(i, suffix))
                {
                    out.emplace_back(suffix);
                    i += suffix.size();
                    matched = true;
                    break;
                }
            }
            if (matched)
                continue;
        }
        const std::size_t start = i;
        // 2) 可选的一个前导空格
        const bool lead_space = (text[i] == ' ');
        const std::size_t body = lead_space ? i + 1 : i;
        if (body < n && is_letter(static_cast<unsigned char>(text[body])))
        {
            i = body;
            while (i < n && is_letter(static_cast<unsigned char>(text[i])))
                ++i;
            out.emplace_back(text.substr(start, i - start));
        }
        else if (body < n && is_digit(static_cast<unsigned char>(text[body])))
        {
            i = body;
            while (i < n && is_digit(static_cast<unsigned char>(text[i])))
                ++i;
            out.emplace_back(text.substr(start, i - start));
        }
        else if (body < n && !is_space(static_cast<unsigned char>(text[body])))
        {
            // 标点等符号
            i = body;
            while (i < n && !is_space(static_cast<unsigned char>(text[i])) &&
                   !is_letter(static_cast<unsigned char>(text[i])) &&
                   !is_digit(static_cast<unsigned char>(text[i])))
            {
                ++i;
            }
            out.emplace_back(text.substr(start, i - start));
        }
        else
        {
            // 3) 连续空白。最后一个空白如果后面还跟着非空白，要留给下一块当前导空格
            i = start;
            while (i < n && is_space(static_cast<unsigned char>(text[i])))
                ++i;
            std::size_t end = i;
            if (i < n && end > start + 1)
                --end; // 把最后一个空格让出去
            if (end > start)
                out.emplace_back(text.substr(start, end - start));
            i = end;
            if (i == start)
                ++i; // 保险：绝不原地打转
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// BPE 主体
// ---------------------------------------------------------------------------
std::vector<std::string> Tokenizer::bpe(const std::string &word) const
{
    // 先拆成单个「字符」（这里的字符是 byte_encoder 映射后的 UTF-8 字符，1~2 字节）
    std::vector<std::string> parts;
    for (std::size_t i = 0; i < word.size();)
    {
        // UTF-8 首字节决定这个字符占几个字节
        const auto c = static_cast<unsigned char>(word[i]);
        const std::size_t len = (c < 0x80) ? 1 : (c < 0xE0 ? 2 : (c < 0xF0 ? 3 : 4));
        parts.push_back(word.substr(i, len));
        i += len;
    }
    if (parts.size() < 2)
        return parts;

    // 反复找「排名最靠前的相邻对」并合并，直到没有可合并的为止。
    //
    // 这是 O(n²) 的朴素实现。真实的分词器会用优先队列或链表优化，
    // 但词块通常只有几个字符长，实际影响不大。
    while (parts.size() > 1)
    {
        int best_rank = std::numeric_limits<int>::max();
        std::size_t best_idx = 0;

        for (std::size_t i = 0; i + 1 < parts.size(); ++i)
        {
            const std::string key = parts[i] + '\x01' + parts[i + 1];
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank)
            {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_rank == std::numeric_limits<int>::max())
            break; // 没有规则可用了

        // 合并这一对**所有**出现的位置（不只是 best_idx）。
        // 这是 GPT-2 原始实现的行为，漏掉会导致结果和官方对不上。
        const std::string left = parts[best_idx];
        const std::string right = parts[best_idx + 1];
        std::vector<std::string> merged;
        merged.reserve(parts.size());
        for (std::size_t i = 0; i < parts.size();)
        {
            if (i + 1 < parts.size() && parts[i] == left && parts[i + 1] == right)
            {
                merged.push_back(left + right);
                i += 2;
            }
            else
            {
                merged.push_back(parts[i]);
                ++i;
            }
        }
        parts.swap(merged);
    }
    return parts;
}

Tokenizer Tokenizer::load(const std::string &vocab_path, const std::string &merges_path)
{
    Tokenizer t;

    {
        std::ifstream f(vocab_path, std::ios::binary);
        if (!f)
            throw std::runtime_error("打不开词表: " + vocab_path +
                                     "\n  先运行: .venv/bin/python tools/convert_gpt2.py");
        char magic[4];
        f.read(magic, 4);
        if (std::memcmp(magic, "GPTV", 4) != 0)
            throw std::runtime_error("词表文件头不对: " + vocab_path);

        const auto count = read_le<std::uint32_t>(f);
        t.id_to_token_.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            std::string tok = read_sized_string(f);
            t.token_to_id_.emplace(tok, static_cast<int>(i));
            t.id_to_token_.push_back(std::move(tok));
        }
    }
    {
        std::ifstream f(merges_path, std::ios::binary);
        if (!f)
            throw std::runtime_error("打不开合并规则: " + merges_path);
        char magic[4];
        f.read(magic, 4);
        if (std::memcmp(magic, "GPTM", 4) != 0)
            throw std::runtime_error("合并规则文件头不对: " + merges_path);

        const auto count = read_le<std::uint32_t>(f);
        t.merge_rank_.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            const std::string a = read_sized_string(f);
            const std::string b = read_sized_string(f);
            // 排名就是在文件里的行号：越靠前，训练时出现得越频繁，越优先合并
            t.merge_rank_.emplace(a + '\x01' + b, static_cast<int>(i));
        }
    }
    return t;
}

std::vector<int> Tokenizer::encode(std::string_view text) const
{
    std::vector<int> ids;
    const auto &enc = byte_encoder();

    for (const std::string &chunk : pre_tokenize(text))
    {
        // 每个字节先过 byte→unicode 映射
        std::string mapped;
        mapped.reserve(chunk.size() * 2);
        for (char ch : chunk)
            mapped += enc[static_cast<unsigned char>(ch)];

        for (const std::string &piece : bpe(mapped))
        {
            auto it = token_to_id_.find(piece);
            if (it == token_to_id_.end())
            {
                // byte-level BPE 理论上不会遇到未知 token：单个字节一定在词表里。
                // 真到了这一步，说明词表或合并规则加载错了。
                throw std::runtime_error("词表里没有 '" + piece + "'，词表文件可能不匹配");
            }
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::string Tokenizer::decode_one(int id) const
{
    if (id < 0 || id >= static_cast<int>(id_to_token_.size()))
    {
        throw std::runtime_error(
            std::format("token id {} 超出词表范围 [0, {})", id, id_to_token_.size()));
    }
    // 反向走一遍 unicode→byte
    const std::string &tok = id_to_token_[static_cast<std::size_t>(id)];
    const auto &dec = byte_decoder();
    std::string out;
    out.reserve(tok.size());
    for (std::size_t i = 0; i < tok.size();)
    {
        const auto c = static_cast<unsigned char>(tok[i]);
        const std::size_t len = (c < 0x80) ? 1 : (c < 0xE0 ? 2 : (c < 0xF0 ? 3 : 4));
        auto it = dec.find(tok.substr(i, len));
        if (it != dec.end())
            out += static_cast<char>(it->second);
        i += len;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int> &ids) const
{
    std::string out;
    for (int id : ids)
        out += decode_one(id);
    return out;
}

} // namespace inference