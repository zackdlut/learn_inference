# CTest 默认吞掉测试进程的 stdout。这个脚本把输出再写到 /dev/tty，
# 这样直接运行 `ctest`（不必带 -V）也能看到 CHECK/PASS 行。
if(NOT DEFINED EXE)
    message(FATAL_ERROR "run_visible.cmake: EXE is required")
endif()

set(_cmd "${EXE}")
if(DEFINED ARGS AND NOT ARGS STREQUAL "")
    list(APPEND _cmd ${ARGS})
endif()

execute_process(
    COMMAND ${_cmd}
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
)

set(_text "${_out}")
if(NOT _err STREQUAL "")
    string(APPEND _text "${_err}")
endif()

if(NOT _text STREQUAL "")
    string(RANDOM LENGTH 16 _id)
    if(DEFINED BUILD_DIR AND NOT BUILD_DIR STREQUAL "")
        set(_tmp "${BUILD_DIR}/.ctest_out_${_id}.txt")
    else()
        set(_tmp "/tmp/ctest_out_${_id}.txt")
    endif()
    file(WRITE "${_tmp}" "${_text}")

    # 写控制终端：CTest 只重定向了 stdout/stderr，没有断开 tty。
    if(EXISTS "/dev/tty")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E cat "${_tmp}"
            OUTPUT_FILE "/dev/tty"
            ERROR_QUIET
        )
    endif()
    # 再写回 CTest 捕获的管道。CTEST_FULL_OUTPUT 避免通过用例被截成 1KiB。
    execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "CTEST_FULL_OUTPUT")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E cat "${_tmp}")
    file(REMOVE "${_tmp}")
endif()

if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "Test exited with code ${_rc}")
endif()
