#include "inference/mlp.hpp"
#include "inference/model_file.hpp"
#include "inference/ops.hpp"
#include "infr_test_utils.hpp"
#include "minitest.hpp"
#include <cmath>
#include <string>
#include <vector>

using namespace inference;
namespace ops = inference::ops;

namespace
{

std::vector<infr_test::TensorSpec> tiny_mlp_specs()
{
    // input=2, hidden=2, output=2；两层均为单位权重、零偏置
    return {
        {"fc1.weight", {2, 2}, {1, 0, 0, 1}},
        {"fc1.bias", {2}, {0, 0}},
        {"fc2.weight", {2, 2}, {1, 0, 0, 1}},
        {"fc2.bias", {2}, {0, 0}},
    };
}

std::string write_tiny_mlp(const char *tag)
{
    return infr_test::write_temp_file(infr_test::pack_infr(1, tiny_mlp_specs()), tag);
}

} // namespace

TEST(mlp_load_info_and_weights)
{
    const MLP model = MLP::load(write_tiny_mlp("mlp_tiny.inference"));

    CHECK(model.w1().shape() == Shape({2, 2}));
    CHECK(model.b1().shape() == Shape({2}));
    CHECK(model.w2().shape() == Shape({2, 2}));
    CHECK(model.b2().shape() == Shape({2}));

    const std::string s = model.to_string();
    CHECK(s.find("input(2)") != std::string::npos);
    CHECK(s.find("linear(2, 2)") != std::string::npos);
    CHECK(s.find("softmax(2)") != std::string::npos);
}

TEST(mlp_forward_matches_manual_ops)
{
    const MLP model = MLP::load(write_tiny_mlp("mlp_forward.inference"));
    const Tensor x = Tensor::from({1, 2}, {1, 0});

    const Tensor prob = model.forward(x);

    Tensor h = ops::relu(ops::linear(x, model.w1(), model.b1()));
    const Tensor expected = ops::softmax(ops::linear(h, model.w2(), model.b2()));

    CHECK(prob.shape() == Shape({1, 2}));
    CHECK(ops::max_abs_diff(prob, expected) < 1e-6F);

    const float e = std::exp(1.0F);
    const float p0 = e / (e + 1.0F);
    CHECK_NEAR(prob(0, 0), p0, 1e-5);
    CHECK_NEAR(prob(0, 0) + prob(0, 1), 1.0, 1e-6);
}

TEST(mlp_forward_trace)
{
    const MLP model = MLP::load(write_tiny_mlp("mlp_trace.inference"));
    const Tensor x = Tensor::from({1, 2}, {1, 0});

    const MLP::Trace trace = model.forward_trace(x);
    const Tensor z1 = ops::linear(x, model.w1(), model.b1());
    const Tensor h1 = ops::relu(z1);
    const Tensor z2 = ops::linear(h1, model.w2(), model.b2());
    const Tensor prob = ops::softmax(z2);

    CHECK(ops::max_abs_diff(trace.z1, z1) < 1e-6F);
    CHECK(ops::max_abs_diff(trace.h1, h1) < 1e-6F);
    CHECK(ops::max_abs_diff(trace.z2, z2) < 1e-6F);
    CHECK(ops::max_abs_diff(trace.prob, prob) < 1e-6F);
}

TEST(mlp_predict_1d_and_2d)
{
    const MLP model = MLP::load(write_tiny_mlp("mlp_predict.inference"));

    const Tensor row = Tensor::from({1, 2}, {1, 0});
    const Tensor vec = Tensor::from({2}, {1, 0});

    CHECK(model.predict(row) == 0);
    CHECK(model.predict(vec) == 0);

    const Tensor skewed = Tensor::from({1, 2}, {0, 5});
    CHECK(model.predict(skewed) == 1);
}

TEST(mlp_load_errors)
{
    CHECK_THROWS(MLP::load(infr_test::model_dir() + "/no_such_mlp.inference"));

    const auto bad_ver = infr_test::write_temp_file(infr_test::pack_infr(2, tiny_mlp_specs()),
                                                    "mlp_bad_ver.inference");
    CHECK_THROWS(MLP::load(bad_ver));

    std::vector<infr_test::TensorSpec> missing_fc2 = tiny_mlp_specs();
    missing_fc2.pop_back();
    const auto missing_path = infr_test::write_temp_file(infr_test::pack_infr(1, missing_fc2),
                                                         "mlp_missing_fc2.inference");
    CHECK_THROWS(MLP::load(missing_path));

    std::vector<infr_test::TensorSpec> bad_w1 = tiny_mlp_specs();
    bad_w1[0] = {"fc1.weight", {4}, {1, 0, 0, 1}};
    const auto bad_w1_path =
        infr_test::write_temp_file(infr_test::pack_infr(1, bad_w1), "mlp_bad_w1.inference");
    CHECK_THROWS(MLP::load(bad_w1_path));

    std::vector<infr_test::TensorSpec> hidden_mismatch = tiny_mlp_specs();
    hidden_mismatch[2] = {"fc2.weight", {2, 3}, {1, 0, 0, 1, 0, 0}};
    const auto mismatch_path = infr_test::write_temp_file(infr_test::pack_infr(1, hidden_mismatch),
                                                          "mlp_hidden_mismatch.inference");
    CHECK_THROWS(MLP::load(mismatch_path));
}

TEST(mlp_mnist_matches_activation_file)
{
    const std::string weights = infr_test::model_dir() + "/mnist_mlp.inference";
    const std::string golden = infr_test::model_dir() + "/mnist_activation.inference";
    if (!infr_test::file_exists(weights) || !infr_test::file_exists(golden))
    {
        PRINT("跳过: 未找到 {} 或 {}（先运行 tools/train_mnist_numpy.py）", weights, golden);
        return;
    }

    const MLP model = MLP::load(weights);
    ModelFile ref = ModelFile::load(golden);
    const Tensor input = ref.get_tensor("input");

    const MLP::Trace trace = model.forward_trace(input);
    CHECK(ops::max_abs_diff(trace.z1, ref.get_tensor("z1")) < 1e-4F);
    CHECK(ops::max_abs_diff(trace.h1, ref.get_tensor("h1")) < 1e-4F);
    CHECK(ops::max_abs_diff(trace.z2, ref.get_tensor("z2")) < 1e-4F);
    CHECK(ops::max_abs_diff(trace.prob, ref.get_tensor("prob")) < 1e-4F);

    CHECK(model.predict(input) == ops::argmax(ref.get_tensor("prob")));
}
