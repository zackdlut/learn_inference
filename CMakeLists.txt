cmake_minimum_required(VERSION 3.20)
project(Learn_Inference LANGUAGES CXX VERSION 0.1.0)

# 使用C++20标准语法编译
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# CMAKE_CXX_EXTENSIONS OFF 表示用 -std=c++20 而不是 -std=gnu++20
# 强制我们只用标准里的东西，避免不小心依赖 GCC 私有扩展。
set(CMAKE_CXX_EXTENSIONS OFF)

# 导出 compile_commands.json 文件，方便 IDE 使用
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 添加编译选项 Release 和 Debug 的编译选项
# Release 选项: -O3 优化级别
# Debug 选项: -g 调试信息, -fsanitize=address,undefined 地址和未定义行为检查
add_library(inference_flags  INTERFACE)
target_compile_options(inference_flags INTERFACE
    # 添加编译警告选项
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g -fsanitize=address,undefined>
)
target_link_options(inference_flags INTERFACE
    # 添加链接地址和未定义行为检查选项
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

# 为本机 CPU 开启全部指令集(-march=native)，提高性能，降低可移植性
option(INFERENCE_NATIVE_API "为本机 CPU 开启全部指令集(-march=native)" ON)
if(INFERENCE_NATIVE_API)
    target_compile_options(inference_flags INTERFACE -march=native)
endif()

# STEP 1. 核心库: 张量计算
add_library(inference
    src/tensor.cpp
)
# 给inference libary 添加公共头文件目录
target_include_directories(inference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# 给inference libary 添加编译选项
target_link_libraries(inference PUBLIC inference_flags)

# 代码示例
add_subdirectory(examples)
add_subdirectory(misc)

# 测试用例
enable_testing()
add_subdirectory(tests)