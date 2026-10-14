#include <cstdlib>
#include <format>
#include <inference/mlp.hpp>
#include <inference/ops.hpp>
#include <inference/tensor.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace
{
std::string format_indices(const std::vector<inference::int64> &indices)
{
    std::string out = "[";
    for (std::size_t j = 0; j < indices.size(); ++j)
    {
        if (j > 0)
        {
            out += ", ";
        }
        out += std::format("{}", indices[j]);
    }
    out += "]";
    return out;
}
} // namespace

#ifndef INFER_MODEL_DIR
#define INFER_MODEL_DIR "data/models"
#endif

int main(int argc, char *argv[])
{
    const char *infer_model_dir = std::getenv("INFER_MODEL_DIR");
    if (infer_model_dir == nullptr)
    {
        infer_model_dir = INFER_MODEL_DIR;
    }
    // load model file
    const std::string model_path = std::string(infer_model_dir) + "/mnist_mlp.inference";
    const inference::ModelFile model_file = inference::ModelFile::load(model_path);
    std::cout << "Model file info: " << model_file.info() << std::endl;
    // create model from model file
    const inference::MLP model = inference::MLP::load(model_path);
    std::cout << "Model info: " << model.info() << std::endl;
    // load sample file
    const std::string sample_path = std::string(infer_model_dir) + "/mnist_samples.inference";
    const inference::ModelFile sample_file = inference::ModelFile::load(sample_path);
    const inference::Tensor images = sample_file.get_tensor("images");
    const inference::Tensor labels = sample_file.get_tensor("labels");

    // test model
    const inference::int64 n = images.dim(0);
    std::vector<inference::int64> incorrect_indices;
    for (inference::int64 i = 0; i < n; ++i)
    {
        const inference::Tensor image = images.slice(0, i, i + 1);
        const inference::Tensor label = labels.slice(0, i, i + 1);
        const inference::Tensor prob = model.forward(image);
        const inference::int64 pred = model.predict(image);
        std::cout << std::format("i: {} prediction: {} label: {:.0f}\n", i, pred, label(0));
        if (label(0) != static_cast<float>(pred))
        {
            incorrect_indices.push_back(i);
        }
    }
    if (!incorrect_indices.empty())
    {
        std::cout << std::format("Incorrect indices: {}\n", format_indices(incorrect_indices));
    }
    else
    {
        std::cout << "All predictions are correct\n";
    }

    // 对拍
    // load activation file
    const std::string activation_path =
        std::string(infer_model_dir) + "/mnist_activation.inference";
    const inference::ModelFile activation_file = inference::ModelFile::load(activation_path);
    const inference::Tensor input_activation = activation_file.get_tensor("input");
    const inference::Tensor z1_activation = activation_file.get_tensor("z1");
    const inference::Tensor h1_activation = activation_file.get_tensor("h1");
    const inference::Tensor z2_activation = activation_file.get_tensor("z2");
    const inference::Tensor prob_activation = activation_file.get_tensor("prob");
    const inference::int64 n_activation = input_activation.dim(0);
    // compare activation with model
    inference::MLP::Trace trace = model.forward_trace(input_activation);
    std::cout << std::format("Z1 difference: {}\n",
                             inference::ops::max_abs_diff(trace.z1, z1_activation));
    std::cout << std::format("H1 difference: {}\n",
                             inference::ops::max_abs_diff(trace.h1, h1_activation));
    std::cout << std::format("Z2 difference: {}\n",
                             inference::ops::max_abs_diff(trace.z2, z2_activation));
    std::cout << std::format("Prob difference: {}\n",
                             inference::ops::max_abs_diff(trace.prob, prob_activation));
    return 0;
}