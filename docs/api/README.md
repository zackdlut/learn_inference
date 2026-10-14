# Inference C++ API 索引

头文件位于 [`include/inference/`](../../include/inference/)。命名空间：`inference`；算子在 `inference::ops`。

| 模块 | 文档 | 头文件 | 示例 / 测试 |
|------|------|--------|-------------|
| **Tensor** | [tensor.md](tensor.md) | [`tensor.hpp`](../../include/inference/tensor.hpp) | [`hello_tensor.cpp`](../../examples/hello_tensor.cpp)、[`test_tensor.cpp`](../../tests/test_tensor.cpp) |
| **Ops** | [ops.md](ops.md) | [`ops.hpp`](../../include/inference/ops.hpp) | [`hello_ops.cpp`](../../examples/hello_ops.cpp)、[`test_ops.cpp`](../../tests/test_ops.cpp) |
| **MLP** | [mlp.md](mlp.md) | [`mlp.hpp`](../../include/inference/mlp.hpp) | [`hello_mlp.cpp`](../../examples/hello_mlp.cpp) |

相关：

- 数据类型：[`dtype.hpp`](../../include/inference/dtype.hpp)
- 模型文件：[`infr_format.md`](../infr_format.md)、[`model_file.hpp`](../../include/inference/model_file.hpp)

```cpp
#include "inference/tensor.hpp"
#include "inference/ops.hpp"
#include "inference/mlp.hpp"

using inference::Tensor;
namespace ops = inference::ops;
```
