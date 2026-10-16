#include "inference/model_file.hpp"
#include <algorithm>
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

void draw(const inference::Tensor &tensor)
{
    static const char *kRamp = " .:-=+*#%@";
    inference::Tensor reshaped = tensor.reshape({28, 28});
    for (inference::int64 i = 0; i < 28; ++i)
    {
        std::string line = "    ";
        for (inference::int64 j = 0; j < 28; ++j)
        {
            int level = static_cast<inference::int64>(reshaped(i, j) * 9.99F);
            line += kRamp[level < 0 ? 0 : (level > 9 ? 9 : level)];
            line += kRamp[level < 0 ? 0 : (level > 9 ? 9 : level)];
        }
        std::cout << line << std::endl;
    }
    std::cout << std::endl;
}

void draw_prob(const inference::Tensor &tensor)
{
    inference::Tensor reshaped = tensor.reshape({10});
    for (inference::int64 i = 0; i < 10; ++i)
    {
        std::string line = " ";
        const int num = std::clamp(static_cast<int>(reshaped(i) * 30.0F), 0, 30);
        line += std::format("{}: {}", i, std::string(num, '*'));
        std::cout << line << std::endl;
    }
}

#ifndef INFER_MODEL_DIR
#define INFER_MODEL_DIR "data/models"
#endif

#ifndef INFER_EXAMPLES_DIR
#define INFER_EXAMPLES_DIR "examples"
#endif

namespace
{

const char *resolve_dir(const char *env_name, const char *compiled_default)
{
    const char *from_env = std::getenv(env_name);
    return from_env != nullptr ? from_env : compiled_default;
}

std::string join_path(const std::string &dir, const std::string &filename)
{
    if (dir.empty())
    {
        return filename;
    }
    if (dir.back() == '/')
    {
        return dir + filename;
    }
    return dir + "/" + filename;
}

} // namespace

int main(int argc, char *argv[])
{
    const char *infer_model_dir = resolve_dir("INFER_MODEL_DIR", INFER_MODEL_DIR);
    const char *infer_examples_dir = resolve_dir("INFER_EXAMPLES_DIR", INFER_EXAMPLES_DIR);
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

    const std::string digit3_path = join_path(infer_examples_dir, "3.inference");
    const std::string digit4_path = join_path(infer_examples_dir, "4.inference");
    const std::string digit5_path = join_path(infer_examples_dir, "5.inference");

    std::cout << "================================================" << std::endl;
    inference::ModelFile mf = inference::ModelFile::load(digit3_path);
    inference::Tensor image = mf.get_tensor("image"); // [1, 784]
    const inference::int64 digit = model.predict(image);
    draw(image);
    std::cout << "Predict digit = " << digit << " (" << digit3_path << ")\n";
    std::cout << "================================================" << std::endl;
    std::cout << "================================================" << std::endl;
    inference::ModelFile mf2 = inference::ModelFile::load(digit4_path);
    inference::Tensor image2 = mf2.get_tensor("image"); // [1, 784]
    const inference::int64 digit2 = model.predict(image2);
    draw(image2);
    std::cout << "Predict digit = " << digit2 << " (" << digit4_path << ")\n";
    std::cout << "================================================" << std::endl;
    inference::ModelFile mf3 = inference::ModelFile::load(digit5_path);
    inference::Tensor image3 = mf3.get_tensor("image"); // [1, 784]
    const inference::int64 digit3 = model.predict(image3);
    draw(image3);
    std::cout << "Predict digit = " << digit3 << " (" << digit5_path << ")\n";
    std::cout << "================================================" << std::endl;
    // test model
    const inference::int64 n = images.dim(0);
    std::vector<inference::int64> incorrect_indices;
    for (inference::int64 i = 0; i < n; ++i)
    {
        const inference::Tensor image = images.slice(0, i, i + 1);
        const inference::Tensor label = labels.slice(0, i, i + 1);
        const inference::Tensor prob = model.forward(image);
        const inference::int64 pred = model.predict(image);
        std::cout << "================================================" << std::endl;
        draw(image);
        std::cout << "================================================" << std::endl;
        draw_prob(prob);
        std::cout << std::format("Prediction: {} label: {:.0f}\n", pred, label(0));
        if (label(0) != static_cast<float>(pred))
        {
            incorrect_indices.push_back(i);
        }
        std::cout << "================================================" << std::endl;
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