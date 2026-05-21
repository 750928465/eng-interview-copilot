#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN"
  echo "Install Python or run with: PYTHON_BIN=/path/to/python ./start.sh"
  exit 1
fi

echo "Starting English Interview Copilot..."
echo "Project: $APP_DIR"
echo "Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"

exec "$PYTHON_BIN" main.py
