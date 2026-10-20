#!/usr/bin/env bash
# 把文档里某一步的代码编出来（用 git worktree，不动你当前的工作区）。
#
#   bench/scripts/build_commit.sh <步骤|提交|.> [Release|Debug]
#
#   bench/scripts/build_commit.sh baseline Debug   # 优化前、Debug 构建（文档的 1.3 token/s）
#   bench/scripts/build_commit.sh 3                # 第 3 步，Release
#   bench/scripts/build_commit.sh 89c4702          # 直接给提交也行
#
# 步骤名：baseline 0 2 3 4 5 7（7 = 当前工作区）。最后一行输出 hello_gpt 的路径，
# worktree 在 .worktrees/<提交>/，用完可以 git worktree remove .worktrees/<提交>。
source "$(dirname "$0")/common.sh"

if [[ $# -lt 1 ]]; then
    sed -n '2,11p' "$0"
    exit 1
fi
build_hello "$1" "${2:-Release}"
