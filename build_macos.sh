#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p mac/build
swiftc \
  -parse-as-library \
  mac/SystemAudioCapture.swift \
  -o mac/build/SystemAudioCapture \
  -module-cache-path mac/build/module-cache \
  -framework ScreenCaptureKit \
  -framework AVFoundation

"$PYTHON_BIN" -m PyInstaller EnglishInterviewCopilot.spec

echo "Built dist/English Interview Copilot.app"
