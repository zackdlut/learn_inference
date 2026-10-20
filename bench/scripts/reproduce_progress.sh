#!/usr/bin/env bash
# 重建文档第 5 节的「进度记录」表：把每一步的提交都编出来，轮流跑 hello_gpt。
#
#   bench/scripts/reproduce_progress.sh [轮数=3] [token 数=60]
#   DEBUG=1 bench/scripts/reproduce_progress.sh      # 额外跑一次优化前的 Debug 构建（慢，约 1 分钟）
#
# 第一次运行要编 6 个旧版本，几分钟；之后只是增量编译。
# 每一轮把所有版本都跑一遍再进入下一轮，最后取中位数。
source "$(dirname "$0")/common.sh"

ROUNDS="${1:-3}" N="${2:-60}"

# 行：标签 版本 构建类型 第几次生成
ROWS=(
    "起点-Release baseline Release 1"
    "0-测量工具 0 Release 1"
    "2-GEMV 2 Release 1"
    "3-线程池 3 Release 1"
    "4-去拷贝 4 Release 1"
    "5-注意力 5 Release 1"
    "7-fp32(同一程序) 7 Release 1"
    "7-量化 7 Release 2"
)
[[ "${DEBUG:-0}" == 1 ]] && ROWS=("起点-Debug baseline Debug 1" "${ROWS[@]}")

declare -A BIN PRE TPS
for row in "${ROWS[@]}"; do
    read -r label rev type which <<<"$row"
    BIN[$label]="$(build_hello "$rev" "$type")"
done

for ((i = 0; i < ROUNDS; ++i)); do
    for row in "${ROWS[@]}"; do
        read -r label rev type which <<<"$row"
        # Debug 版太慢，只跑一轮
        [[ "$type" == Debug && $i -gt 0 ]] && continue
        log "第 $((i + 1))/$ROUNDS 轮：$label"
        r=($(run_hello "${BIN[$label]}" "$N" "$which"))
        PRE[$label]+="${r[0]} "
        TPS[$label]+="${r[2]} "
    done
done

# printf 按字节对齐，中文会错位，所以中文标签放在最后一列
echo
echo "commit     prefill(5 tok) ms   decode tok/s   ms/token   步骤"
for row in "${ROWS[@]}"; do
    read -r label rev type which <<<"$row"
    p="$(tr ' ' '\n' <<<"${PRE[$label]}" | grep . | median)"
    t="$(tr ' ' '\n' <<<"${TPS[$label]}" | grep . | median)"
    printf "%-10s %17s %14s %10.1f   %s\n" "$(resolve_rev "$rev")" "$p" "$t" "$(awk "BEGIN{print 1000/$t}")" "$label"
done
echo
echo "线程数：第 3 步起默认用一半逻辑核（$(($(nproc) / 2))），可用 INFER_NUM_THREADS 覆盖"
