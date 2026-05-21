"""
Audio sources that are not regular microphone devices.
"""
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class MacSystemAudioSource:
    """Read macOS system audio from a small ScreenCaptureKit helper."""

    def __init__(self, sample_rate: int = 16000, blocksize: int = 1024):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.process: Optional[subprocess.Popen] = None
        self.project_root = Path(config.project_root)
        self.source_path = self.project_root / "mac" / "SystemAudioCapture.swift"
        self.binary_path = self.project_root / "mac" / "build" / "SystemAudioCapture"
        self.last_error = ""

    def start(self):
        if sys.platform != "darwin":
            raise RuntimeError("macOS 系统音频捕获仅支持 macOS")

        self._ensure_helper()
        self.process = subprocess.Popen(
            [str(self.binary_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def read_chunk(self) -> np.ndarray:
        if not self.process or not self.process.stdout:
            raise RuntimeError("系统音频捕获尚未启动")

        byte_count = self.blocksize * np.dtype(np.float32).itemsize
        data = self._read_exact(byte_count)
        if not data:
            code = self.process.poll()
            if code is not None:
                detail = f": {self.last_error}" if self.last_error else ""
                raise RuntimeError(f"系统音频捕获已退出，退出码: {code}{detail}")
            return np.zeros((0, 1), dtype=np.float32)

        audio = np.frombuffer(data, dtype=np.float32)
        return audio.reshape(-1, 1)

    def _read_exact(self, byte_count: int) -> bytes:
        chunks = []
        remaining = byte_count
        while remaining > 0:
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def stop(self):
        if not self.process:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def _ensure_helper(self):
        if not self.source_path.exists():
            raise RuntimeError(f"系统音频 helper 不存在: {self.source_path}")

        needs_build = not self.binary_path.exists()
        if not needs_build:
            needs_build = self.source_path.stat().st_mtime > self.binary_path.stat().st_mtime
        if not needs_build:
            return

        self.binary_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "swiftc",
            "-parse-as-library",
            str(self.source_path),
            "-o",
            str(self.binary_path),
            "-module-cache-path",
            str(self.binary_path.parent / "module-cache"),
            "-framework",
            "ScreenCaptureKit",
            "-framework",
            "AVFoundation",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"系统音频 helper 编译失败: {detail}")

    def _drain_stderr(self):
        if not self.process or not self.process.stderr:
            return
        for line in iter(self.process.stderr.readline, b""):
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                self.last_error = text
                print(text, flush=True)
