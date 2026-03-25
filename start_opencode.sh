#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/opencoderun.log"
RESTART_DELAY_SECONDS=5

cd "$SCRIPT_DIR"

while true; do
  {
    printf '\n[%s] starting opencode session\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    OPENCODE_PERMISSION='{"*":"allow"}' \
    opencode run --title "locomo-prompt-loop" \
      "查看 program.md，按流程执行。默认当前目录就是仓库根目录。每次会话只完成一个 step：如果没有历史状态就跑 baseline，否则按 resume 流程完成一轮新的 prompt 改进实验。完成归档和记录后正常退出，由外层脚本再次启动。"
    status=$?
    printf '[%s] opencode session exited with status %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status"
  } >> "$LOG_FILE" 2>&1

  sleep "$RESTART_DELAY_SECONDS"
done
