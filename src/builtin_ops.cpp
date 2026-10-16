#include "inference/op.hpp"
#include "inference/ops.hpp"
#include "inference/tensor.hpp"

namespace inference
{
class UnaryElementwiseOp : public Op
{
  public:
    int num_inputs() const override
    {
        return 1;
    }
    Shape inference_shape(std::span<const Shape> input_shapes,
                          const Attributes &attributes) const override
    {
        return input_shapes[0];
    }
};
class ReluOp final : public UnaryElementwiseOp
{
  public:
    std::string type() const override
    {
        return "relu";
    }
    Tensor compute(const OpContext *context) const override
    {
        return ops::relu(*context->inputs[0]);
    }
};

class GeluOp final : public UnaryElementwiseOp
{
  public:
    std::string type() const override
    {
        return "gelu";
    }
    Tensor compute(const OpContext *context) const override
    {
        return ops::gelu(*context->inputs[0]);
    }
};

class SoftmaxOp final : public UnaryElementwiseOp
{
  public:
    std::string type() const override
    {
        return "softmax";
    }
    Tensor compute(const OpContext *context) const override
    {
        return ops::softmax(*context->inputs[0]);
    }
};

class AddOp final : public Op
{
  public:
    std::string type() const override
    {
        return "add";
    }
    int num_inputs() const override
    {
        return 2;
    }
    Shape inference_shape(std::span<const Shape> input_shapes,
                          const Attributes &attributes) const override
    {
        if (input_shapes[0] != input_shapes[1])
        {
            throw std::runtime_error("Input shapes must be the same");
        }
        return input_shapes[0];
    }
    Tensor compute(const OpContext *context) const override
    {
        return ops::add(*context->inputs[0], *context->inputs[1]);
    }
};

class MulOp final : public Op
{
  public:
    std::string type() const override
    {
        return "mul";
    }
    int num_inputs() const override
    {
        return 2;
    }
    Shape inference_shape(std::span<const Shape> input_shapes,
                          const Attributes &attributes) const override
    {
        if (input_shapes[0] != input_shapes[1])
        {
            throw std::runtime_error("Input shapes must be the same");
        }
        return input_shapes[0];
    }
    Tensor compute(const OpContext *context) const override
    {
        return ops::mul(*context->inputs[0], *context->inputs[1]);
    }
};

class MatmulOp final : public Op
{
  public:
    std::string type() const override
    {
        return "matmul";
    }
    int num_inputs() const override
    {
        return 2;
    }
    Shape inference_shape(std::span<const Shape> input_shapes,
                          const Attributes &attributes) const override
    {
        if (input_shapes[0].size() != 2 || input_shapes[1].size() != 2)
        {
            throw std::runtime_error("Input shapes must be 2D");
        }
        if (input_shapes[0][1] != input_shapes[1][0])
        {
            throw std::runtime_error("Input shapes must be compatible for matmul");
        }
        return Shape{input_shapes[0][0], input_shapes[1][1]};
    }
    Tensor compute(const OpContext *context) const override
    {
        return ops::matmul(*context->inputs[0], *context->inputs[1]);
    }
};

class LinearOp final : public Op
{
  public:
    std::string type() const override
    {
        return "linear";
    }
    int num_inputs() const override
    {
        return -1;
    }
    Shape inference_shape(std::span<const Shape> input_shapes,
                          const Attributes &attributes) const override
    {
        if (input_shapes.size() != 2 && input_shapes.size() != 3)
        {
            throw std::runtime_error("Linear Input must be 2 or 3");
        }
        const Shape &x = input_shapes[0];
        const Shape &w = input_shapes[1];
        if (x.size() != 2 || w.size() != 2)
        {
            throw std::runtime_error("Linear Input and weight must be 2D");
        }
        if (x[1] != w[1])
        {
            throw std::runtime_error("Linear feature number must be the same");
        }
        // bias is included
        if (input_shapes.size() == 3 && (input_shapes[2].size() != 1 || input_shapes[2][0] != w[0]))
        {
            throw std::runtime_error(
                "Linear bias must be 1D and have the same number of elements as the weight");
        }
        return Shape{x[0], w[0]};
    }
    Tensor compute(const OpContext *context) const override
    {
        if (context->inputs.size() == 2)
        {
            return ops::linear(*context->inputs[0], *context->inputs[1]);
        }
        else
        {
            return ops::linear(*context->inputs[0], *context->inputs[1], *context->inputs[2]);
        }
    }
};

class LayerNormOp final : public Op
{
  public:
    std::string type() const override
    {
        return "layer_norm";
    }
    int num_inputs() const override
    {
        return 3;
    }
    Shape inference_shape(std::span<const Shape> input_shapes,
                          const Attributes &attributes) const override
    {
        if (input_shapes[0].empty())
        {
            throw std::runtime_error("LayerNorm Input must be non-empty");
        }
        const auto d = input_shapes[0].back();
        if (input_shapes[1] != Shape{d} || input_shapes[2] != Shape{d})
        {
            throw std::runtime_error("LayerNorm weight and bias must be 1D and have the same "
                                     "number of elements as the input dimension");
        }
        return input_shapes[0];
    }
    Tensor compute(const OpContext *context) const override
    {
        const float eps = context->attributes ? context->attributes->get<float>("eps") : 1e-5;
        return ops::layernorm(*context->inputs[0], *context->inputs[1], *context->inputs[2], eps);
    }
};

REGISTER_OP(ReluOp)
REGISTER_OP(GeluOp)
REGISTER_OP(SoftmaxOp)
REGISTER_OP(AddOp)
REGISTER_OP(MulOp)
REGISTER_OP(MatmulOp)
REGISTER_OP(LinearOp)
REGISTER_OP(LayerNormOp)

void register_builtin_ops() {}

} // namespace inference