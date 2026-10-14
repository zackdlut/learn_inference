# Ops API

头文件：[`include/inference/ops.hpp`](../../include/inference/ops.hpp)  
命名空间：`inference::ops`

算子层负责形状校验、输出分配，并调用 `kernel` 做数值计算。形状错误时抛 `std::invalid_argument`，前缀 `[inference::ops]`。

```cpp
#include "inference/ops.hpp"
namespace ops = inference::ops;
```

---

## 逐元素（二元输入形状必须完全一致）

| 函数 | 签名 | 说明 |
|------|------|------|
| `add` | `Tensor add(a, b)` | `a + b` |
| `mul` | `Tensor mul(a, b)` | `a * b`（逐元素） |
| `add_scalar` | `Tensor add_scalar(a, s)` | `a + s` |
| `relu` | `Tensor relu(x)` | `max(0, x)` |
| `gelu` | `Tensor gelu(x)` | GELU 激活 |

不支持广播；`a.shape() != b.shape()` 会失败。

---

## 归约（沿最后一维）

把前缀维展平为「行」，对**最后一维**做归一化或归约。

| 函数 | 签名 | 默认 | 说明 |
|------|------|------|------|
| `softmax` | `Tensor softmax(x)` | — | 每行 softmax；**标量**非法 |
| `layernorm` | `Tensor layernorm(x, w, b, eps)` | `eps=1e-5` | `w`,`b` 长度 = 最后一维 |
| `rmsnorm` | `Tensor rmsnorm(x, w, eps)` | `eps=1e-5` | `w` 长度 = 最后一维 |

例：`x` 为 `(2, 3, 4)` 时，按 6 行、每行 4 列处理。

---

## 矩阵乘与全连接

| 函数 | 形状约定 | 输出 |
|------|----------|------|
| `matmul(a, b)` | `a`: (M, K)，`b`: (K, N)，**均为 2D** | (M, N) |
| `linear(x, w)` | `x`: (M, in)，`w`: (out, in) | (M, out) |
| `linear(x, w, bias)` | 同上，`bias`: (out,) | (M, out) |

`linear` 计算 **y = x @ Wᵀ + b**，与 PyTorch `nn.Linear` 权重布局一致（`w` 每行对应一个输出神经元）。

常见错误信息：

- `matmul`: 内维 K 不一致，或非 2D
- `linear`: `x` 最后一维须等于 `w` 的第二维

---

## 杂项

| 函数 | 签名 | 说明 |
|------|------|------|
| `argmax` | `int64 argmax(x)` | 全张量最大元素的下标（扁平）；空张量失败 |
| `allclose` | `bool allclose(a, b, atol=1e-5)` | 逐元素 \|a−b\| ≤ atol |
| `max_abs_diff` | `float max_abs_diff(a, b)` | 最大绝对误差，对拍用 |

---

## 与 MLP / MNIST 的对应关系

```
z1 = linear(x, W1, b1)     # W1: (128, 784)
h1 = relu(z1)
z2 = linear(h1, W2, b2)    # W2: (10, 128)
p  = softmax(z2)
pred = argmax(p)           # 与 argmax(z2) 等价（单调变换）
```

---

## 最小示例

```cpp
#include "inference/ops.hpp"
#include "inference/tensor.hpp"

using inference::Tensor;
namespace ops = inference::ops;

Tensor x = Tensor::zeros({4, 784});
Tensor w = Tensor::zeros({128, 784});
Tensor b = Tensor::zeros({128});
Tensor h = ops::relu(ops::linear(x, w, b));  // (4, 128)
```

更多用例见 [`examples/hello_ops.cpp`](../../examples/hello_ops.cpp)。
