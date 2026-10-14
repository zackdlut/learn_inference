#include "inference/model_file.hpp"
#include "minitest.hpp"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace inference;

namespace
{

void append_le(std::vector<std::byte> &buf, const void *src, std::size_t n)
{
    const auto *p = static_cast<const std::byte *>(src);
    buf.insert(buf.end(), p, p + n);
}

template <typename T> void append_le(std::vector<std::byte> &buf, T v)
{
    append_le(buf, &v, sizeof(T));
}

struct TensorSpec
{
    std::string name;
    Shape shape;
    std::vector<float> data;
};

std::vector<std::byte> pack_infr(std::uint32_t version, const std::vector<TensorSpec> &tensors)
{
    std::vector<std::byte> header;
    const char magic[] = "INFR";
    append_le(header, magic, 4);
    append_le(header, version);
    append_le(header, static_cast<std::uint32_t>(tensors.size()));

    std::vector<std::byte> body;
    std::uint64_t cursor = 0;
    for (const auto &t : tensors)
    {
        const auto name_bytes = t.name;
        append_le(header, static_cast<std::uint32_t>(name_bytes.size()));
        append_le(header, name_bytes.data(), name_bytes.size());
        append_le(header, static_cast<std::uint32_t>(0)); // FP32
        append_le(header, static_cast<std::uint32_t>(t.shape.size()));
        for (const auto dim : t.shape)
        {
            append_le(header, dim);
        }
        const auto nbytes =
            static_cast<std::uint64_t>(static_cast<std::size_t>(numel_of(t.shape)) * sizeof(float));
        append_le(header, cursor);
        append_le(header, nbytes);
        append_le(body, t.data.data(), t.data.size() * sizeof(float));
        cursor += nbytes;
    }

    const std::uint64_t data_start =
        static_cast<std::uint64_t>(header.size() + sizeof(std::uint64_t));
    append_le(header, data_start);
    header.insert(header.end(), body.begin(), body.end());
    return header;
}

std::string write_temp_file(const std::vector<std::byte> &bytes, const char *tag)
{
    const auto path =
        std::filesystem::temp_directory_path() / std::filesystem::path("learn_inference_") / tag;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out.good())
    {
        throw std::runtime_error("failed to write temp INFR file");
    }
    return path.string();
}

std::string model_dir()
{
#ifdef INFERENCE_SOURCE_DIR
    return std::string(INFERENCE_SOURCE_DIR) + "/data/models";
#else
    return "data/models";
#endif
}

bool file_exists(const std::string &path)
{
    return std::filesystem::exists(path);
}

} // namespace

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
