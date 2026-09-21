# 自动收集 tests/test_*.cpp：每份源文件一个可执行文件，并把其中的 TEST() 注册到 CTest。
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")
list(SORT TEST_SOURCES)

if(NOT TEST_SOURCES)
    message(WARNING "tests/ 下没有找到 test_*.cpp")
endif()

set(_run_visible "${CMAKE_CURRENT_SOURCE_DIR}/run_visible.cmake")

# CTest 会截走测试 stdout。用 cmake -P 包一层，把输出再打到 /dev/tty。
function(add_visible_test name exe)
    cmake_parse_arguments(_AVT "" "" "ARGS" ${ARGN})
    set(_cmd
        ${CMAKE_COMMAND}
        "-DEXE=$<TARGET_FILE:${exe}>"
        "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
        -P "${_run_visible}"
    )
    if(DEFINED _AVT_ARGS AND NOT _AVT_ARGS STREQUAL "")
        list(INSERT _cmd 2 "-DARGS=${_AVT_ARGS}")
    endif()
    add_test(NAME ${name} COMMAND ${_cmd})
endfunction()

foreach(src IN LISTS TEST_SOURCES)
    get_filename_component(test_exe "${src}" NAME_WE)

    add_executable(${test_exe} main.cpp "${src}")
    target_include_directories(${test_exe} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${test_exe} PRIVATE inference)

    # 整文件一条 ctest：ctest -R '^test_tensor$'
    add_visible_test(${test_exe} ${test_exe})
    set_tests_properties(${test_exe} PROPERTIES LABELS "file")

    # 每个 TEST(name) 一条 ctest：ctest -R tensor_zeros
    file(READ "${src}" _test_src)
    string(REGEX MATCHALL "TEST[ \t]*\\([ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]*\\)"
           _found_tests "${_test_src}")

    foreach(_macro IN LISTS _found_tests)
        string(REGEX REPLACE ".*TEST[ \t]*\\([ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\\).*"
               "\\1" case_name "${_macro}")
        add_visible_test(${test_exe}.${case_name} ${test_exe} ARGS "${case_name}")
        set_tests_properties(${test_exe}.${case_name} PROPERTIES LABELS "case")
    endforeach()
endforeach()
