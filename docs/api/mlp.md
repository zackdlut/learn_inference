# MLP API

头文件：[`include/inference/mlp.hpp`](../../include/inference/mlp.hpp)  
命名空间：`inference`

两层全连接 + ReLU + Softmax，用于 MNIST（784 → 128 → 10）。权重从 [INFR 文件](../infr_format.md) 加载，张量名须与 Python 导出一致。

---

## 类 `MLP`

### 静态加载

| 方法 | 说明 |
|------|------|
| `static MLP load(const std::string& path)` | 读 `.inference`，取 `fc1.weight`、`fc1.bias`、`fc2.weight`、`fc2.bias` |

**期望权重形状**

| 成员 | 形状 | 含义 |
|------|------|------|
| `w1` | (hidden, 784) | 第一层权重 |
| `b1` | (hidden,) | 第一层偏置 |
| `w2` | (10, hidden) | 第二层权重 |
| `b2` | (10,) | 第二层偏置 |

且 `w1.dim(0) == w2.dim(1)`（隐藏维一致）。

**异常**：`ModelFile` 加载失败、`version != 1`、维数或隐藏维不匹配 → `std::runtime_error`。

---

### 推理

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `forward(input)` | `x` 形状 (M, 784) 或可被 `linear` 接受的 2D | (M, 10) **概率** | `linear → relu → linear → softmax` |
| `predict(input)` | 同上；若为 1D 则先 `reshape({1, numel})` | `int64` 类别下标 | 内部 `forward` + `ops::argmax` |
| `forward_trace(input)` | 同上 | `Trace` 结构体 | 返回中间量，便于对拍 |

**`Trace` 字段**

| 字段 | 典型形状 (M=1) | 含义 |
|------|----------------|------|
| `z1` | (M, 128) | 第一层线性输出 |
| `h1` | (M, 128) | ReLU 后 |
| `z2` | (M, 10) | logits |
| `prob` | (M, 10) | softmax 后 |

---

### 权重访问与描述

| 方法 | 说明 |
|------|------|
| `w1()`, `b1()`, `w2()`, `b2()` | 非 const 引用返回内部 `Tensor`（当前 API 未提供 const 重载） |
| `to_string()` | 单行计算图字符串，如 `input(784) → linear(784, 128) → … → argmax(1)` |

---

## 计算图（与 `forward` 一致）

```text
input [M, 784]
  → linear(w1, b1)  →  [M, 128]
  → ReLU
  → linear(w2, b2)  →  [M, 10]
  → softmax         →  prob [M, 10]
```

`predict` 在 `prob` 上做 `argmax`（全张量扁平下标；batch=1 时即类别 0–9）。

---

## 最小示例

```cpp
#include "inference/mlp.hpp"
#include "inference/tensor.hpp"

inference::MLP model = inference::MLP::load("data/models/mnist_mlp.inference");
inference::Tensor x = /* [1, 784] 或从 ModelFile 取 images 的一行 */;

inference::Tensor prob = model.forward(x);
inference::int64 label = model.predict(x);

std::cout << model.to_string() << '\n';
```

完整流程见 [`examples/hello_mlp.cpp`](../../examples/hello_mlp.cpp)。

---

## 依赖

- [`ModelFile`](../../include/inference/model_file.hpp)：读 INFR
- [`ops::linear`](ops.md)、[`ops::relu`](ops.md)、[`ops::softmax`](ops.md)、[`ops::argmax`](ops.md)

训练与导出在 Python：[`tools/train_mnist_numpy.py`](../../tools/train_mnist_numpy.py)。
