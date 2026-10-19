#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace inference
{
// ---------------------------------------------------------------------------
// GPT-2 的 byte-level BPE 分词器
//
// 模型只认整数（token id），不认文字。分词器负责两边的翻译：
//
//     "The capital of France is"
//              │ encode
//              ▼
//     [464, 3139, 286, 4881, 318]
//              │ 模型推理，产生新 token
//              ▼
//     [464, 3139, 286, 4881, 318, 6342]
//              │ decode
//              ▼
//     "The capital of France is Paris"
//
// 为什么叫 byte-level？因为它先把文本看成**字节**流，而不是字符流。
// 好处是任何输入都能编码（不会遇到「未知字符」），词表也不用塞进所有 Unicode 字符。
//
// BPE（Byte Pair Encoding）的思想：从单个字节出发，反复把「最常一起出现的
// 相邻两块」合并成一块。合并规则是训练时统计出来的，存在 merges.txt 里，
// 按频次从高到低排序。
//
//     "lower"  →  l o w e r
//                 └┬┘              第 1 条规则合并 "lo"
//                 lo w e r
//                    └┬┘           后面某条规则合并 "we"
//                 lo we r
//                    ...
// ---------------------------------------------------------------------------
class Tokenizer
{
  public:
    // vocab_path: gpt2_vocab.bin，merges_path: gpt2_merges.bin
    // 都由 tools/convert_gpt2.py 生成
    static Tokenizer load(const std::string &vocab_path, const std::string &merges_path);
    std::vector<int> encode(std::string_view text) const;
    std::string decode(const std::vector<int> &ids) const;
    std::string decode_one(int id) const;
    std::size_t vocab_size() const noexcept
    {
        return id_to_token_.size();
    }

    static std::vector<std::string> pre_tokenize(std::string_view text);

  private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> merge_rank_;
    std::vector<std::string> bpe(const std::string &word) const;
};

} // namespace inference