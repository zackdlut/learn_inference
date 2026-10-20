#!/usr/bin/env bash
# 扫线程数：同一个版本用不同的 INFER_NUM_THREADS 跑 hello_gpt。
#
#   bench/scripts/sweep_threads.sh [步骤|提交|.=.] [线程列表="1 2 4 6 7 8 10 14"] [轮数=3] [token 数=60]
#
#   bench/scripts/sweep_threads.sh 3                 # 第 3 步：复现「14 线程反而更慢」
#   bench/scripts/sweep_threads.sh . "4 7 14"
#   WHICH=2 bench/scripts/sweep_threads.sh .         # 量化后
#
# 每一轮把所有线程数都跑一遍再进入下一轮，降频对每个线程数的影响差不多。
# 第 3 步之前的版本是单线程的，不认 INFER_NUM_THREADS。
source "$(dirname "$0")/common.sh"

REV="${1:-.}" THREADS="${2:-1 2 4 6 7 8 10 14}" ROUNDS="${3:-3}" N="${4:-60}" WHICH="${WHICH:-1}"
BIN="$(build_hello "$REV")"

declare -A res
for ((i = 0; i < ROUNDS; ++i)); do
    log "第 $((i + 1))/$ROUNDS 轮"
    for t in $THREADS; do
        r=($(INFER_NUM_THREADS=$t run_hello "$BIN" "$N" "$WHICH"))
        res[$t]+="${r[2]} "
    done
done

echo
echo "threads   decode tok/s   ms/token   各轮 tok/s"
for t in $THREADS; do
    m="$(tr ' ' '\n' <<<"${res[$t]}" | grep . | median)"
    printf "%7s %14s %10.1f   %s\n" "$t" "$m" "$(awk "BEGIN{print 1000/$m}")" "${res[$t]}"
done
