# Tensor API

头文件：[`include/inference/tensor.hpp`](../../include/inference/tensor.hpp)  
命名空间：`inference`

张量 = 共享堆内存 + `shape` / `strides` / `offset`。`reshape`、`transpose`、`permute`、`slice` 多为**零拷贝视图**；`contiguous()` 会在必要时拷贝成紧凑布局。

---

## 类型别名与工具函数

| 名称 | 说明 |
|------|------|
| `int64` | `std::int64_t` |
| `Shape` | `std::vector<int64>`，如 `{2, 3}` 表示 2×3；**标量**用空 shape `{}` |
| `to_string(shape)` | 形状转可读字符串 |
| `contiguous_strides(shape)` | 行主序连续步长 |
| `numel_of(shape)` | 元素个数 ∏dims |

---

## 构造与工厂

| 方法 | 签名 | 说明 |
|------|------|------|
| 默认构造 | `Tensor()` | 空张量，`numel() == 0` |
| `zeros` | `static Tensor zeros(Shape shape, DType dtype = FP32)` | 全 0 |
| `ones` | `static Tensor ones(Shape shape, DType dtype = FP32)` | 全 1 |
| `full` | `static Tensor full(Shape shape, float value)` | 全为 `value`（fp32） |
| `arange` | `static Tensor arange(int64 n)` | `[0, 1, …, n-1]`，shape `{n}` |
| `from` | `static Tensor from(Shape, span<const float>)` | 从缓冲区拷贝 |
| `from` | `static Tensor from(Shape, initializer_list<float>)` | 从列表拷贝 |

当前实现以 **FP32** 存储；`dtype()` 反映类型，但 `data()` 始终暴露 `float` 视图。

---

## 元信息

| 方法 | 返回 | 说明 |
|------|------|------|
| `shape()` | `const Shape&` | 各维长度 |
| `strides()` | `const Shape&` | 各维步长（元素单位） |
| `dim(i)` | `int64` | 第 `i` 维长度，越界抛 `std::out_of_range` |
| `ndim()` | `int64` | 维数 |
| `numel()` | `int64` | 逻辑元素个数 |
| `dtype()` | `DType` | 数据类型枚举 |
| `empty()` | `bool` | `numel() == 0` |
| `is_contiguous()` | `bool` | 是否 C 连续且 `offset==0` 可线性遍历 |

---

## 数据访问

| 方法 | 说明 |
|------|------|
| `data()` / `data() const` | `span<float>`，**要求** `is_contiguous()`，否则抛 `std::invalid_argument` |
| `operator()(i, j, …)` | 多维下标，下标个数须等于 `ndim()`；标量用 `data()[0]`，不要写 `t(0)` |

---

## 形状变换（视图）

| 方法 | 说明 |
|------|------|
| `reshape(new_shape)` | 新 shape，`numel` 不变；**至多一个** `-1` 自动推断 |
| `transpose(a, b)` | 交换轴 `a` 与 `b` |
| `permute(new_order)` | 轴重排，`new_order` 为 `0..ndim-1` 的排列 |
| `slice(axis, start, end)` | 沿 `axis` 取 `[start, end)`，该维长度变为 `end - start` |
| `contiguous()` | 非连续时拷贝为连续张量；已连续则返回共享同一存储的视图 |

---

## 字符串与格式化

| 方法 | 说明 |
|------|------|
| `to_string()` | 含数据的完整字符串（大张量慎用） |
| `meta_string()` | 仅 shape / strides / dtype 等元信息 |

`std::format` 支持：

```cpp
std::format("{}", t);      // 同 to_string()
std::format("{:m}", t);    // 同 meta_string()
```

---

## 使用注意

1. **视图与别名**：`reshape` / `slice` 等与源张量共享内存；改一处会影响另一处。
2. **算子输入**：`ops::*` 内部常对非连续输入调用 `contiguous()`；你自己写 kernel 时要先检查 `is_contiguous()`。
3. **标量**：`ndim()==0`，`numel()==1`，索引用 `data()[0]`。

---

## 最小示例

```cpp
#include "inference/tensor.hpp"
using inference::Tensor;

Tensor a = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});
Tensor b = a.reshape({3, 2});           // 视图，共享内存
Tensor row = a.slice(0, 1, 2);          // 第 1 行，shape (1, 3)
float x = a(0, 1);                      // 2
```
