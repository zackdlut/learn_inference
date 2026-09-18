#include "inference/tensor.hpp"
#include "minitest.hpp"

using namespace inference;

TEST(tensor_zeros) {
    Tensor tensor = Tensor::zeros({2, 3});
    CHECK(tensor.shape() == Shape({2, 3}));
    CHECK(tensor.dtype() == DType::FP32);
    CHECK(tensor.numel() == 6);
}

TEST(tensor_ones) {
    Tensor tensor = Tensor::ones({2, 3});
    CHECK(tensor.shape() == Shape({2, 3}));
    CHECK(tensor.dtype() == DType::FP32);
    CHECK(tensor.numel() == 6);
    CHECK_NEAR(tensor(0, 0), 1.0, 1e-6);
    CHECK_NEAR(tensor(0, 1), 1.0, 1e-6);
    CHECK_NEAR(tensor(0, 2), 1.0, 1e-6);
}

TEST(shape_and_stride) {
    Tensor t = Tensor::zeros({2, 3, 4});
    CHECK(t.ndim() == 3);
    CHECK(t.numel() == 24);
    CHECK(t.shape() == Shape({2, 3, 4}));
    CHECK(t.strides() == Shape({12, 4, 1}));
    CHECK(t.is_contiguous());
}

TEST(scalar_tensor) {
    Tensor t = Tensor::zeros({});
    CHECK(t.ndim() == 0);
    CHECK(t.numel() == 1);
    CHECK(t.shape() == Shape({}));
    CHECK(t.strides() == Shape({}));
    CHECK(t.is_contiguous());
}

TEST(indexing) {
    Tensor t = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK_NEAR(t(0, 0), 1.0, 1e-6);
    CHECK_NEAR(t(0, 1), 2.0, 1e-6);
    CHECK_NEAR(t(0, 2), 3.0, 1e-6);
    CHECK_NEAR(t(1, 0), 4.0, 1e-6);
    CHECK_NEAR(t(1, 1), 5.0, 1e-6);
    CHECK_NEAR(t(1, 2), 6.0, 1e-6);
}

TEST(indexing_out_of_range) {
    Tensor t = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK_THROWS(t(0, 3));
    CHECK_THROWS(t(2, 0));
    CHECK_THROWS(t(2, 3));
}

TEST(reshape) {
    Tensor t = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b = t.reshape({3, 2});
    CHECK(b.numel() == 6);

    b(0, 1) = 99.0F;
    CHECK_NEAR(t(0, 1), 99.0, 1e-6);
}

TEST(reshape_infer_minus_one) {
    Tensor t = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});

    Tensor a = t.reshape({2, -1});
    CHECK(a.shape() == Shape({2, 3}));
    CHECK(a.strides() == Shape({3, 1}));
    CHECK_NEAR(a(1, 2), 6.0, 1e-6);

    Tensor b = t.reshape({-1, 2});
    CHECK(b.shape() == Shape({3, 2}));
    CHECK_NEAR(b(2, 1), 6.0, 1e-6);

    Tensor c = t.reshape({-1});
    CHECK(c.shape() == Shape({6}));
    CHECK_NEAR(c(5), 6.0, 1e-6);

    Tensor d = t.reshape({2, 3, -1});
    CHECK(d.shape() == Shape({2, 3, 1}));
}

TEST(reshape_infer_minus_one_throws) {
    Tensor t = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK_THROWS(t.reshape({2, -1, -1})); // 多于一个 -1
    CHECK_THROWS(t.reshape({5, -1}));     // 6 不能被 5 整除
    CHECK_THROWS(t.reshape({-2, 3}));     // 非法负数
    CHECK_THROWS(t.reshape({0, -1}));     // 已知乘积为 0，无法推断
}

TEST(transpose) {
    Tensor a = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor at = a.transpose(0, 1);
    CHECK(at.numel() == 6);
    CHECK(at.shape() == Shape({3, 2}));
    CHECK(at.strides() == Shape({1, 3}));
    CHECK(at.is_contiguous() == false);
    CHECK_THROWS(at.data()); // 非连续，禁止直接取裸内存
    CHECK_THROWS(at.reshape({6}));
}

TEST(contiguous_materializes) {
    Tensor a = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor c = a.transpose(0, 1).contiguous();

    CHECK(c.is_contiguous());
    // 转置后按行优先重新排布应为 1 4 2 5 3 6
    const Shape expect_shape{3, 2};
    CHECK(c.shape() == expect_shape);
    const float want[] = {1, 4, 2, 5, 3, 6};
    auto got = c.data();
    for (std::size_t i = 0; i < 6; ++i) {
        CHECK_NEAR(got[i], want[i], 1e-6);
    }
}

TEST(slice_moves_offset) {
    Tensor a = Tensor::arange(12).reshape({3, 4});
    Tensor row = a.slice(0, 1, 2); // 取第 1 行，形状 (1, 4)

    CHECK((row.shape() == Shape{1, 4}));
    CHECK_NEAR(row(0, 0), 4.0, 1e-6);
    CHECK_NEAR(row(0, 3), 7.0, 1e-6);

    Tensor col = a.slice(1, 2, 4); // 取第 2~3 列，形状 (3, 2)
    CHECK((col.shape() == Shape{3, 2}));
    CHECK_NEAR(col(0, 0), 2.0, 1e-6);
    CHECK_NEAR(col(2, 1), 11.0, 1e-6);
    CHECK(!col.is_contiguous()); // 列切片跳着取，不连续
    Tensor col_contiguous = col.contiguous();
    CHECK(col_contiguous.is_contiguous());
    CHECK((col_contiguous.shape() == Shape{3, 2}));
    CHECK_NEAR(col_contiguous(0, 0), 2.0, 1e-6);
    CHECK_NEAR(col_contiguous(2, 1), 11.0, 1e-6);
}

TEST(bad_construction_throws) {
    CHECK_THROWS(Tensor::from({2, 3}, {1, 2, 3})); // 数据只有 3 个，形状要 6 个
    CHECK_THROWS(Tensor::zeros({2, 3}).reshape({5}));
}