# 被其它脚本 source 的公共部分，不单独运行。
#
#   ROOT      仓库根目录
#   WT_DIR    旧版本的 git worktree 放在哪（默认 $ROOT/.worktrees，已在 .gitignore 里）
#   PROMPT    hello_gpt 用的 prompt，所有测量都用同一个，结果才能比
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WT_DIR="${WT_DIR:-$ROOT/.worktrees}"
PROMPT="${PROMPT:-The capital of France is}"

# 文档里每一步对应的提交。"." 表示当前工作区（第 7 步还没提交时就用它）。
# 所有脚本里的「版本」参数都可以是：步骤名、提交号、"."，或者一个源码目录（比如你自己的 worktree）
declare -A STEP_COMMIT=(
    [baseline]=f142a75 # 优化前
    [0]=3227974        # 测量工具
    [2]=5624734        # GEMV + 权重转成 [out, in]
    [3]=89c4702        # 线程池
    [4]=c98ce1b        # 去拷贝 + 工作区 + SIMD gelu
    [5]=3600536        # 注意力 SIMD + 并行 + KV 按头存放
    [7]=.              # 量化
)

log() { echo "$@" >&2; }

# 执行命令，输出写进日志文件；失败时把日志末尾打出来
#   quiet <日志文件> <命令...>
quiet() {
    local logf="$1"
    shift
    if ! "$@" >"$logf" 2>&1; then
        log "失败：$*（完整日志 $logf）"
        tail -n 30 "$logf" >&2
        return 1
    fi
}

# 把 "3" / "baseline" 这样的步骤名翻译成提交；本来就是提交或 "." 的原样返回
resolve_rev() {
    local r="$1"
    if [[ -n "${STEP_COMMIT[$r]+x}" ]]; then echo "${STEP_COMMIT[$r]}"; else echo "$r"; fi
}

# 源码目录：当前工作区、你自己的源码目录，或该提交的 worktree（没有就创建）
source_dir() {
    local rev
    rev="$(resolve_rev "$1")"
    if [[ "$rev" == "." ]]; then
        echo "$ROOT"
        return
    fi
    if [[ -f "$rev/CMakeLists.txt" ]]; then
        [[ -e "$rev/data" ]] || ln -s "$ROOT/data" "$rev/data"
        (cd "$rev" && pwd)
        return
    fi
    local hash
    hash="$(git -C "$ROOT" rev-parse --short "$rev")"
    local wt="$WT_DIR/$hash"
    if [[ ! -d "$wt" ]]; then
        log "[worktree] 检出 $hash → $wt"
        mkdir -p "$WT_DIR"
        git -C "$ROOT" worktree add --detach "$wt" "$hash" >&2
    fi
    # 模型文件不在 git 里，链接过去；hello_gpt 在编译时把 data/gpt 的绝对路径写死
    [[ -e "$wt/data" ]] || ln -s "$ROOT/data" "$wt/data"
    echo "$wt"
}

# 编译某个版本的 hello_gpt，stdout 只输出可执行文件路径
#   build_hello <步骤|提交|.> [Release|Debug]
build_hello() {
    local src type build
    src="$(source_dir "$1")"
    type="${2:-Release}"
    build="$src/build-${type,,}"
    log "[build] $1 ($type) → $build"
    mkdir -p "$build"
    quiet "$build/configure.log" cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE="$type"
    quiet "$build/build.log" cmake --build "$build" -j --target hello_gpt
    echo "$build/examples/hello_gpt"
}

# 跑一次 hello_gpt，输出 "prefill_ms decode_ms decode_tok_per_s"
#   run_hello <可执行文件> <生成 token 数> [第几次生成：1 = 第 6 节 fp32，2 = 第 7 节量化后]
run_hello() {
    local bin="$1" n="$2" which="${3:-1}" out
    out="$("$bin" "$PROMPT" "$n" 2>&1)"
    local pre dec
    pre="$(grep -E '^prefill [0-9]+ 个 token' <<<"$out" | sed -n "${which}p")"
    dec="$(grep -E '^decode [0-9]+ 个 token' <<<"$out" | sed -n "${which}p")"
    if [[ -z "$dec" ]]; then
        log "没找到第 $which 行 decode 输出（这个版本可能没有第 7 节）"
        echo "nan nan nan"
        return
    fi
    # 格式：decode 60 个 token:  1124 ms  （53.4 token/s）
    local pre_ms dec_ms tps
    pre_ms="$(sed -E 's/.*: +([0-9.]+) ms.*/\1/' <<<"$pre")"
    dec_ms="$(sed -E 's/.*: +([0-9.]+) ms.*/\1/' <<<"$dec")"
    tps="$(sed -E 's/.*（([0-9.]+) token\/s）.*/\1/' <<<"$dec")"
    echo "$pre_ms $dec_ms $tps"
}

# 从 stdin 读一列数字，输出中位数
median() {
    sort -g | awk '{ a[NR] = $1 } END {
        if (NR == 0) { print "nan"; exit }
        print (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2 }'
}
