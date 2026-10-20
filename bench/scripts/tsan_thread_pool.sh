#!/usr/bin/env bash
# 用 ThreadSanitizer 检查线程池和所有用到线程池的 kernel（第 3、5 步）。
#
#   bench/scripts/tsan_thread_pool.sh
#
# 单独建一个 build-tsan 目录：TSan 不能和 Debug 构建里的 ASan 同时开。
# 没有 "WARNING: ThreadSanitizer" 输出、退出码为 0 才算通过。
source "$(dirname "$0")/common.sh"

BUILD="$ROOT/build-tsan"
TESTS=(test_thread_pool test_kernel_matmul test_kernel) # test_kernel 里有注意力的并行测试

mkdir -p "$BUILD"
log "[build] $BUILD（第一次约 1 分钟）"
quiet "$BUILD/configure.log" cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="-fsanitize=thread" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
    -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=thread"
quiet "$BUILD/build.log" cmake --build "$BUILD" -j --target "${TESTS[@]}"

# 新内核的 mmap 随机化位数太高时 TSan 会报 "unexpected memory mapping"，
# 用 setarch -R 关掉这个进程的地址随机化即可
export TSAN_OPTIONS="halt_on_error=1 second_deadlock_stack=1"
for t in "${TESTS[@]}"; do
    echo "== $t"
    if ! setarch "$(uname -m)" -R "$BUILD/tests/$t" >"$BUILD/$t.log" 2>&1; then
        cat "$BUILD/$t.log"
        echo "$t 失败（上面是完整输出）"
        exit 1
    fi
    tail -n 1 "$BUILD/$t.log"
done
echo
echo "ThreadSanitizer 没有报告数据竞争"
