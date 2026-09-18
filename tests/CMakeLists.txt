add_executable(run_tests main.cpp test_tensor.cpp)

target_include_directories(run_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(run_tests PRIVATE inference)

add_test(NAME run_tests COMMAND run_tests)