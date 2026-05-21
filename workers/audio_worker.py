"""
音频录制工作线程
"""
from PyQt5.QtCore import QThread, pyqtSignal
from typing import Optional
import traceback

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asr.recognizer import create_asr, BaseASR
from config import config


class AudioWorker(QThread):
    """音频录制和识别线程"""

    text_recognized = pyqtSignal(str)       # 识别到一段文本
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    volume_changed = pyqtSignal(float)
    asr_timing = pyqtSignal(float)

    def __init__(
        self,
        asr_provider: str = None,
        silence_gap: int = None,
        mode: str = "auto",
        audio_device_index: int = None,
        audio_device_name: str = "",
        audio_gain: float = None,
        audio_source: str = None
    ):
        super().__init__()
        self.asr_provider = asr_provider or config.asr_provider
        self.silence_gap = silence_gap if silence_gap is not None else config.silence_gap
        self.mode = mode
        self.audio_source = audio_source or config.audio_source
        self.audio_device_index = (
            audio_device_index if audio_device_index is not None else config.audio_device_index
        )
        self.audio_device_name = audio_device_name or config.audio_device_name
        self.audio_gain = audio_gain if audio_gain is not None else config.audio_gain
        self.asr: Optional[BaseASR] = None
        self._is_running = False

    def run(self):
        """线程主循环"""
        print(f"[AudioWorker] 线程启动, 模式: {self.mode}")
        try:
            self.status_changed.emit("初始化麦克风...")

            self.asr = create_asr(
                self.asr_provider,
                model_size=config.asr_model_size,
                silence_gap=self.silence_gap,
                audio_device_index=self.audio_device_index,
                audio_device_name=self.audio_device_name,
                audio_gain=self.audio_gain,
                audio_source=self.audio_source,
                volume_callback=self._on_volume_changed,
                timing_callback=self._on_asr_timing,
                error_callback=self._on_capture_error
            )
            self._is_running = True

            self.status_changed.emit("正在监听...")

            if self.audio_source == "mac_system":
                self.asr.listen_system_audio(self._on_text_recognized)
            elif self.mode == "manual":
                # 手动模式：持续录音，分段识别
                self.asr.record_until_stop(self._on_text_recognized)
            else:
                # 自动模式：静音间隔自动识别
                self.asr.listen_microphone(self._on_text_recognized)

        except Exception as e:
            error_msg = f"录音初始化失败: {e}"
            print(f"[AudioWorker] 错误: {error_msg}")
            traceback.print_exc()
            self.error_occurred.emit(error_msg)
        finally:
            self._is_running = False
            print("[AudioWorker] 线程结束")

    def _on_text_recognized(self, text: str):
        """识别结果回调"""
        print(f"[AudioWorker] 收到识别结果: {text}")
        if self._is_running and text:
            self.text_recognized.emit(text)

    def _on_volume_changed(self, level: float):
        if self._is_running:
            self.volume_changed.emit(level)

    def _on_asr_timing(self, elapsed_ms: float):
        if self._is_running:
            self.asr_timing.emit(elapsed_ms)

    def _on_capture_error(self, message: str):
        self.error_occurred.emit(message)

    def stop(self):
        """停止录音"""
        print("[AudioWorker] 停止请求")
        if self.asr:
            self.asr.stop_listening()
        self.status_changed.emit("已停止监听")
