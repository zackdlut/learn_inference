// 第 1 阶段的演示程序：亲眼看看 shape / stride / view 是怎么回事。
//
//   cmake -S . -B build && cmake --build build -j
//   ./build/examples/01_hello_tensor

#include <exception>
#include <format>
#include <iostream>

#include "inference/tensor.hpp"

using inference::Tensor;

void section(const char *title) {
    std::cout << std::format("\n\033[36m===== {} =====\033[0m\n", title);
}

int main() {
    section("1. 造一个 2x3 的张量");
    Tensor a = Tensor::from({2, 3}, {1, 2, 3, 4, 5, 6});
    std::cout << a.to_string();
    std::cout << "内存里实际的排列顺序（行优先）：\n";
    for (float v : a.data())
        std::cout << std::format("{} ", v);
    std::cout << '\n';

    section("2. reshape：说明书变了，内存没动");
    Tensor b = a.reshape({3, 2});
    std::cout << b.to_string();
    std::cout << "改 b(0,0)=99 之后，a 也会变 —— 它们共享同一块内存：\n";
    b(0, 0) = 99.0F;
    std::cout << std::format("a(0,0) = {}\n", a(0, 0));
    a(0, 0) = 1.0F; // 改回去

    section("3. transpose：只交换 shape 和 stride");
    Tensor at = a.transpose(0, 1);
    std::cout << std::format("原张量 : {}\n", a.meta_string());
    std::cout << std::format("转置后 : {}\n", at.meta_string());
    std::cout << at.to_string();
    std::cout << "注意 contiguous=false：逻辑上是 3x2，但内存顺序还是原来的 1 2 3 4 5 6。\n";

    section("4. contiguous：把 view 落实成真实内存（唯一发生拷贝的地方）");
    Tensor atc = at.contiguous();
    std::cout << std::format("{}\n", atc.meta_string());
    std::cout << "内存顺序现在真的变了：";
    for (float v : atc.data())
        std::cout << std::format("{} ", v);
    std::cout << '\n';

    section("5. slice：把起点往后挪，再改一下这一维的长度");
    Tensor m = Tensor::arange(12).reshape({3, 4});
    std::cout << m.to_string();
    Tensor col = m.slice(1, 1, 3); // 取第 1~2 列
    std::cout << std::format("取第 1~2 列 -> {}\n", col.meta_string());
    std::cout << col.to_string();

    section("6. 越界和形状错误会立刻抛异常，而不是悄悄读脏数据");
    try {
        (void)m(5, 0);
    } catch (const std::exception &e) {
        std::cout << std::format("捕获到：{}\n", e.what());
    }

    std::cout << '\n';
    return 0;
}
