# 性能实验程序：docs/gpt_inference_optimization.md 里每一步用到的测量都在这里。
# 和 examples/ 的区别：examples 演示怎么用库，bench 回答「快不快、为什么」。
#
#   cmake --build build-release -j --target bench_all
#   ./build-release/bench/bench_bw
function(add_bench name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE inference)
  target_compile_definitions(${name} PRIVATE
    INFER_MODEL_GPT_DIR="${CMAKE_SOURCE_DIR}/data/gpt"
  )
  list(APPEND BENCH_TARGETS ${name})
  set(BENCH_TARGETS ${BENCH_TARGETS} PARENT_SCOPE)
endfunction()

add_bench(bench_bw)          # 第 0 步：内存带宽上限
add_bench(bench_decode)      # 第 0/2/3/7 步：decode 形状的矩阵乘，各版本 kernel 对比
add_bench(bench_transpose)   # 第 2 步：加载时转置权重，朴素 vs 分块
add_bench(bench_pool)        # 第 3 步：「等每个线程报到」vs「只等任务完成」的线程池
add_bench(bench_serial_ops)  # 第 4 步：gelu 标量 vs SIMD、KV-Cache 拷贝随长度增长
add_bench(bench_attn)        # 第 5 步：注意力 kernel，热缓存
add_bench(bench_kvlayout)    # 第 5 步：KV-Cache 布局 + 并行阈值，冷缓存
add_bench(bench_quant)       # 第 7 步：离群值、逐矩阵精度排查、各量化方案的速度

add_custom_target(bench_all DEPENDS ${BENCH_TARGETS})
