#include "inference/mlp.hpp"
#include "inference/ops.hpp"
#include "inference/tensor.hpp"
#include <cmath>
#include <format>
#include <stdexcept>

namespace inference
{
MLP MLP::load(const std::string &path)
{
    const ModelFile model_file = ModelFile::load(path);
    if (model_file.version() != 1)
    {
        throw std::runtime_error("Unsupported INFR version");
    }
    MLP model;
    model.w1_ = model_file.get_tensor("fc1.weight");
    model.b1_ = model_file.get_tensor("fc1.bias");
    model.w2_ = model_file.get_tensor("fc2.weight");
    model.b2_ = model_file.get_tensor("fc2.bias");

    if (model.w1_.ndim() != 2 || model.w2_.ndim() != 2)
    {
        throw std::runtime_error("Invalid weight tensor dimensions");
    }
    if (model.b1_.ndim() != 1 || model.b2_.ndim() != 1)
    {
        throw std::runtime_error("Invalid bias tensor dimensions");
    }
    if (model.w1_.dim(0) != model.w2_.dim(1))
    {
        throw std::runtime_error("Invalid weight tensor dimensions");
    }

    return model;
}

Tensor MLP::forward(const Tensor &input) const
{
    Tensor h = ops::linear(input, w1_, b1_);
    h = ops::relu(h);
    Tensor prob = ops::linear(h, w2_, b2_);
    prob = ops::softmax(prob);
    return prob;
}

MLP::Trace MLP::forward_trace(const Tensor &input) const
{
    const Tensor z1 = ops::linear(input, w1_, b1_);
    const Tensor h1 = ops::relu(z1);
    const Tensor z2 = ops::linear(h1, w2_, b2_);
    const Tensor prob = ops::softmax(z2);
    return {z1, h1, z2, prob};
}

int64 MLP::predict(const Tensor &input) const
{
    const Tensor image = input.ndim() == 1 ? input.reshape({1, input.numel()}) : input;
    const Tensor prob = forward(image);
    return ops::argmax(prob);
}

std::string MLP::info() const
{
    if (w1_.numel() == 0 || w2_.numel() == 0)
    {
        return "MLP (weights not loaded)";
    }
    const int64 in = w1_.dim(1);
    const int64 hidden = w1_.dim(0);
    const int64 out = w2_.dim(0);
    // 特征维：input[in] → Linear → ReLU → Linear → softmax → argmax（每样本 1 个类别下标）
    return std::format(
        "input({}) → linear({}, {}) → ReLU({}) → linear({}, {}) → softmax({}) → argmax(1)",
        in, in, hidden, hidden, hidden, out, out);
}
} // namespace inference