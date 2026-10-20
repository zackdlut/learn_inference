#!/usr/bin/env bash
# 两个版本交替跑 hello_gpt，比 decode 速度的中位数。
#
#   bench/scripts/ab_compare.sh <A> <B> [轮数=5] [token 数=60]
#
#   bench/scripts/ab_compare.sh 3 4            # 第 3 步 vs 第 4 步
#   bench/scripts/ab_compare.sh 5 .            # 第 5 步 vs 当前工作区
#   WHICH=2 bench/scripts/ab_compare.sh . .    # 第 7 节（量化后）那次生成
#
# 为什么要交替、要 ABBA：笔记本的降频能让同一个程序前后两次差 2 倍。
# A、B 轮流跑，并且每轮换一下先后（ABBA），降频对两边的影响才一样。
# 环境变量 INFER_NUM_THREADS 会原样传给 hello_gpt（第 3 步起才认）。
source "$(dirname "$0")/common.sh"

if [[ $# -lt 2 ]]; then
    sed -n '2,12p' "$0"
    exit 1
fi
A="$1" B="$2" ROUNDS="${3:-5}" N="${4:-60}" WHICH="${WHICH:-1}"

BIN_A="$(build_hello "$A")"
BIN_B="$(build_hello "$B")"

ta=() tb=()
run() { # run <标签> <可执行文件> <数组名>
    local r
    r=($(run_hello "$2" "$N" "$WHICH"))
    printf "  %-3s prefill %6s ms   decode %6s ms   %6s token/s\n" "$1" "${r[0]}" "${r[1]}" "${r[2]}"
    eval "$3+=(${r[2]})"
}

log "交替跑 $ROUNDS 轮，每次生成 $N 个 token，prompt: \"$PROMPT\""
for ((i = 0; i < ROUNDS; ++i)); do
    echo "第 $((i + 1)) 轮"
    if ((i % 2 == 0)); then
        run A "$BIN_A" ta
        run B "$BIN_B" tb
    else
        run B "$BIN_B" tb
        run A "$BIN_A" ta
    fi
done

ma="$(printf '%s\n' "${ta[@]}" | median)"
mb="$(printf '%s\n' "${tb[@]}" | median)"
echo
printf "A = %-10s decode 中位数 %6s token/s（%.1f ms/token）\n" "$A" "$ma" "$(awk "BEGIN{print 1000/$ma}")"
printf "B = %-10s decode 中位数 %6s token/s（%.1f ms/token）\n" "$B" "$mb" "$(awk "BEGIN{print 1000/$mb}")"
printf "B / A = %.2f×\n" "$(awk "BEGIN{print $mb/$ma}")"
