#!/usr/bin/env bash
# 某一步的三道正确性关卡：单元测试、和 numpy 逐层对拍、贪心生成的文本。
#
#   bench/scripts/check_step.sh <步骤|提交|.> [token 数=60]
#
#   bench/scripts/check_step.sh 2
#   bench/scripts/check_step.sh .
#
# 输出最后一行是贪心文本的指纹（md5 前 12 位）。fp32 的各步之间指纹应该相同：
# 优化只改了计算方式，不该改变模型的选择。
source "$(dirname "$0")/common.sh"

if [[ $# -lt 1 ]]; then
    sed -n '2,10p' "$0"
    exit 1
fi
REV="$1" N="${2:-60}"
SRC="$(source_dir "$REV")"
BUILD="$SRC/build-release"

mkdir -p "$BUILD"
log "[build] $REV 全部目标 → $BUILD"
# 必须重新配置：测试用例是配置阶段扫源文件注册的
quiet "$BUILD/configure.log" cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
quiet "$BUILD/build.log" cmake --build "$BUILD" -j

echo "== 1. 单元测试"
if ctest --test-dir "$BUILD" >"$BUILD/ctest.log" 2>&1; then
    grep -E 'tests passed' "$BUILD/ctest.log"
else
    grep -E 'tests passed|Failed|\*\*\*' "$BUILD/ctest.log"
    exit 1
fi

echo "== 2. 和 numpy 参考实现逐层对拍（hello_gpt 第 3 节）"
out="$("$BUILD/examples/hello_gpt" "$PROMPT" "$N" 2>&1)"
sed -n '/张量 /,/逐层对齐/p' <<<"$out"

echo "== 3. 贪心生成 $N 个 token（hello_gpt 第 6 节，温度 0）"
# 生成的文本以黄色的 prompt 开头（ESC[33m<prompt>ESC[0m<生成的部分>），可能有多行，
# 到第一行 "prefill N 个 token" 为止
text="$(awk '/^\033\[33m/ { on = 1 } /^prefill [0-9]+ 个 token/ { exit } on' <<<"$out" |
    sed -E $'s/\e\\[[0-9;]*m//g')"
echo "$text"
echo "指纹 $(printf '%s' "$text" | md5sum | cut -c1-12)"
