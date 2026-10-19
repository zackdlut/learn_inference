function(add_example name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE inference)
  # 模型目录的绝对路径，这样从任何工作目录运行都能找到权重文件
  target_compile_definitions(${name} PRIVATE
    INFER_MODEL_DIR="${CMAKE_SOURCE_DIR}/data/models"
    INFER_MODEL_GPT_DIR="${CMAKE_SOURCE_DIR}/data/gpt"
  )
endfunction()

add_example(hello_tensor)
add_example(hello_ops)
add_example(hello_mlp)
add_example(hello_graph)
add_example(hello_perf)
add_example(hello_gpt)

target_compile_definitions(hello_mlp PRIVATE
  INFER_EXAMPLES_DIR="${CMAKE_SOURCE_DIR}/examples"
)