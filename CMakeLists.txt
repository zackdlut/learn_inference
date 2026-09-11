cmake_minimum_required(VERSION 3.20)
project(LearnInference LANGUAGE CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)



find_package(Torch REQUIRED)

add_executable(LearnInference main.cpp)

target_include_directories(LearnInference PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(LearnInference PRIVATE ${CMAKE_SOURCE_DIR}/lib/libtorch.so)
target_link_libraries(LearnInference PRIVATE ${CMAKE_SOURCE_DIR}/lib/libtorch_cpu.so)
target_link_libraries(LearnInference PRIVATE ${CMAKE_SOURCE_DIR}/lib/libtorch_cuda.so)
target_link_libraries(LearnInference PRIVATE ${CMAKE_SOURCE_DIR}/lib/libtorch_cuda_cu.so)
target_link_libraries(LearnInference PRIVATE ${CMAKE_SOURCE_DIR}/lib/libtorch_cuda_cu.so)