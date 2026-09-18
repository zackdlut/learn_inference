# 自动收集 tests/test_*.cpp：每份源文件一个可执行文件，并把其中的 TEST() 注册到 CTest。
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")
list(SORT TEST_SOURCES)

if(NOT TEST_SOURCES)
    message(WARNING "tests/ 下没有找到 test_*.cpp")
endif()

foreach(src IN LISTS TEST_SOURCES)
    get_filename_component(test_exe "${src}" NAME_WE)

    add_executable(${test_exe} main.cpp "${src}")
    target_include_directories(${test_exe} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${test_exe} PRIVATE inference)

    # 整文件一条 ctest：ctest -R '^test_tensor$'
    add_test(NAME ${test_exe} COMMAND ${test_exe})
    set_tests_properties(${test_exe} PROPERTIES LABELS "file")

    # 每个 TEST(name) 一条 ctest：ctest -R tensor_zeros
    file(READ "${src}" _test_src)
    string(REGEX MATCHALL "TEST[ \t]*\\([ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]*\\)"
           _found_tests "${_test_src}")

    foreach(_macro IN LISTS _found_tests)
        string(REGEX REPLACE ".*TEST[ \t]*\\([ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\\).*"
               "\\1" case_name "${_macro}")
        add_test(NAME ${test_exe}.${case_name} COMMAND ${test_exe} "${case_name}")
        set_tests_properties(${test_exe}.${case_name} PROPERTIES LABELS "case")
    endforeach()
endforeach()
