#pragma once
#include "inference/model_file.hpp"
#include "inference/tensor.hpp"
#include <string>

namespace inference
{
class MLP
{
  public:
    static MLP load(const std::string &path);
    Tensor forward(const Tensor &input) const;
    int64 predict(const Tensor &input) const;
    

    struct Trace
    {
        Tensor z1;
        Tensor h1;
        Tensor z2;
        Tensor prob;
    };
    Trace forward_trace(const Tensor &input) const;

    const Tensor& w1() {return w1_;}
    const Tensor& b1() {return b1_;}
    const Tensor& w2() {return w2_;}
    const Tensor& b2() {return b2_;}

    std::string info() const;


  private:
    Tensor w1_;
    Tensor b1_;
    Tensor w2_;
    Tensor b2_;
};
} // namespace inference