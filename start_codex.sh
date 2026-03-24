#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/codexrun.log"
RESTART_DELAY_SECONDS=5

cd "$SCRIPT_DIR"

while true; do
  {
    printf '\n[%s] starting codex session\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    codex exec \
      --sandbox danger-full-access \
      --skip-git-repo-check \
      "查看program.md，按流程执行"
    status=$?
    printf '[%s] codex session exited with status %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status"
  } >> "$LOG_FILE" 2>&1

  sleep "$RESTART_DELAY_SECONDS"
done

# 连续跑14~15小时，额度从75降到了13，花费了62额度