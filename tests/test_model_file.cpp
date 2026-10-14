#include "inference/model_file.hpp"
#include "infr_test_utils.hpp"
#include "minitest.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace inference;
using infr_test::TensorSpec;
using infr_test::append_le;
using infr_test::file_exists;
using infr_test::model_dir;
using infr_test::pack_infr;
using infr_test::write_temp_file;

TEST(model_file_load_one_tensor)
{
    const std::vector<TensorSpec> specs = {
        {"bias", {2}, {1.0F, 2.0F}},
    };
    const auto bytes = pack_infr(1, specs);
    const auto path = write_temp_file(bytes, "one_tensor.inference");

    ModelFile mf = ModelFile::load(path);
    CHECK(mf.version() == 1U);
    CHECK(mf.contains("bias"));
    CHECK(!mf.contains("missing"));

    const auto &infos = mf.get_tensor_infos();
    CHECK(infos.size() == 1U);
    CHECK(infos[0].name == "bias");
    CHECK(infos[0].shape == Shape({2}));
    CHECK(infos[0].dtype == DType::FP32);

    Tensor bias = mf.get_tensor("bias");
    CHECK(bias.shape() == Shape({2}));
    CHECK_NEAR(bias(0), 1.0, 1e-6);
    CHECK_NEAR(bias(1), 2.0, 1e-6);
}

TEST(model_file_load_two_tensors)
{
    const std::vector<TensorSpec> specs = {
        {"a", {2, 2}, {1, 2, 3, 4}},
        {"b", {}, {42}},
    };
    const auto path = write_temp_file(pack_infr(1, specs), "two_tensors.inference");
    ModelFile mf = ModelFile::load(path);

    CHECK(mf.get_tensor_infos().size() == 2U);
    CHECK(mf.contains("a"));
    CHECK(mf.contains("b"));

    Tensor a = mf.get_tensor("a");
    CHECK_NEAR(a(0, 0), 1.0, 1e-6);
    CHECK_NEAR(a(1, 1), 4.0, 1e-6);

    Tensor b = mf.get_tensor("b");
    CHECK(b.ndim() == 0);
    CHECK_NEAR(b.data()[0], 42.0, 1e-6);
}

TEST(model_file_info)
{
    const std::vector<TensorSpec> specs = {{"x", {1}, {0}}};
    const auto path = write_temp_file(pack_infr(1, specs), "info.inference");
    ModelFile mf = ModelFile::load(path);
    const std::string s = mf.info();
    CHECK(s.find("INFR version: 1") != std::string::npos);
    CHECK(s.find("Number of tensors: 1") != std::string::npos);
}

TEST(model_file_load_errors)
{
    CHECK_THROWS(ModelFile::load(model_dir() + "/this_file_does_not_exist.inference"));

    std::vector<std::byte> bad_magic = {std::byte{'B'}, std::byte{'A'}, std::byte{'D'},
                                        std::byte{'!'}};
    bad_magic.resize(16);
    const auto bad_path = write_temp_file(bad_magic, "bad_magic.inference");
    CHECK_THROWS(ModelFile::load(bad_path));

    const std::vector<TensorSpec> ok = {{"t", {1}, {0}}};
    const auto wrong_ver = write_temp_file(pack_infr(2, ok), "bad_version.inference");
    CHECK_THROWS(ModelFile::load(wrong_ver));

    const char magic[] = "INFR";
    std::vector<std::byte> no_tensors;
    append_le(no_tensors, magic, 4);
    append_le(no_tensors, std::uint32_t{1});
    append_le(no_tensors, std::uint32_t{0});
    const auto zero_tensors_path = write_temp_file(no_tensors, "zero_tensors.inference");
    CHECK_THROWS(ModelFile::load(zero_tensors_path));

    ModelFile mf = ModelFile::load(write_temp_file(pack_infr(1, ok), "get_tensor_ok.inference"));
    CHECK_THROWS(mf.get_tensor("nope"));
}

TEST(model_file_invalid_tensor_length)
{
    std::vector<std::byte> header;
    const char magic[] = "INFR";
    append_le(header, magic, 4);
    append_le(header, std::uint32_t{1});
    append_le(header, std::uint32_t{1});
    const std::string name = "w";
    append_le(header, std::uint32_t{1});
    append_le(header, name.data(), name.size());
    append_le(header, std::uint32_t{0});
    append_le(header, std::uint32_t{1});
    append_le(header, std::int64_t{2});
    append_le(header, std::uint64_t{0});
    append_le(header, std::uint64_t{99}); // wrong: expect 8 bytes for shape [2]
    append_le(header, std::uint64_t{header.size() + 8});
    const auto path = write_temp_file(header, "bad_length.inference");
    CHECK_THROWS(ModelFile::load(path));
}

TEST(model_file_mnist_activation)
{
    const std::string path = model_dir() + "/mnist_activation.inference";
    if (!file_exists(path))
    {
        PRINT("跳过: 未找到 {}（先运行 tools/train_mnist_numpy.py）", path);
        return;
    }

    ModelFile mf = ModelFile::load(path);
    CHECK(mf.version() == 1U);
    CHECK(mf.contains("input"));
    CHECK(mf.contains("z1"));
    CHECK(mf.contains("h1"));
    CHECK(mf.contains("z2"));
    CHECK(mf.contains("prob"));

    Tensor input = mf.get_tensor("input");
    CHECK(input.shape() == Shape({1, 784}));
    Tensor prob = mf.get_tensor("prob");
    CHECK(prob.shape() == Shape({1, 10}));

    float sum = 0;
    for (int64 j = 0; j < prob.dim(1); ++j)
    {
        sum += prob(0, j);
    }
    CHECK_NEAR(sum, 1.0, 1e-4);
}
