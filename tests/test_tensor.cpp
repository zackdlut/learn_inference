#include "inference/tensor.hpp"
#include "minitest.hpp"

using namespace inference;

TEST(tensor_zeros) {
    Tensor tensor = Tensor::zeros({2, 3});
    CHECK(tensor.shape() == Shape({2, 3}));
    CHECK(tensor.dtype() == DType::FP32);
    CHECK(tensor.numel() == 6);
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

    b(0, 1) = 99.0F;
    CHECK_NEAR(t(0, 0), 99.0, 1e-6);
}