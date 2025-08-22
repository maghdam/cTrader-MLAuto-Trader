#!/usr/bin/env bash
set -e

# Only watch the source tree; ignore logs/db mounts.
if [ "${DEV_RELOAD}" = "1" ]; then
  echo "[dev] Auto-reload enabled"
  exec python -m watchfiles "python -u src/live_trader.py" src
else
  exec python -u src/live_trader.py
fi
