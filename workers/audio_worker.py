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

    text_recognized = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self, asr_provider: str = None):
        super().__init__()
        self.asr_provider = asr_provider or config.asr_provider
        self.asr: Optional[BaseASR] = None
        self._is_running = False

    def run(self):
        """线程主循环"""
        print("[AudioWorker] 线程启动")
        try:
            self.status_changed.emit("初始化麦克风...")
            print(f"[AudioWorker] 创建 ASR: {self.asr_provider}")

            self.asr = create_asr(self.asr_provider)
            self._is_running = True

            self.status_changed.emit("正在监听...")
            print("[AudioWorker] 开始监听...")

            # 开始监听麦克风（阻塞调用）
            self.asr.listen_microphone(self._on_text_recognized)

            print("[AudioWorker] 监听结束")

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
            print(f"[AudioWorker] 已发送信号: {text}")

    def stop(self):
        """停止录音"""
        print("[AudioWorker] 停止请求")
        self._is_running = False
        if self.asr:
            self.asr.stop_listening()
        self.status_changed.emit("已停止监听")