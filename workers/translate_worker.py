"""
翻译工作线程
默认使用本地 OPUS-MT 模型做粗略英译中
"""
from PyQt5.QtCore import QThread, pyqtSignal
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from translation.local_opus import get_local_opus_translator


class TranslateWorker(QThread):
    """翻译线程"""

    translate_completed = pyqtSignal(str, float)
    error_occurred = pyqtSignal(str)
    _translate_lock = None

    def __init__(self, text: str = ""):
        super().__init__()
        self.text = text

    def set_text(self, text: str):
        """设置待翻译文本"""
        self.text = text

    def run(self):
        """执行翻译"""
        if not self.text:
            return

        try:
            print(f"[Translate] 翻译: {self.text[:50]}...", flush=True)
            started = time.perf_counter()

            if config.translation_provider != "local_opus":
                raise RuntimeError(f"不支持的翻译 provider: {config.translation_provider}")

            translator = get_local_opus_translator(
                config.translation_model_name,
                config.translation_local_files_only,
            )
            lock = self._get_translate_lock()
            with lock:
                translated = translator.translate(self.text)
            elapsed_ms = (time.perf_counter() - started) * 1000

            print(f"[Translate] 结果: {translated}", flush=True)
            self.translate_completed.emit(translated, elapsed_ms)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Translate] 错误: {e}", flush=True)
            self.translate_completed.emit(f"(翻译失败)", 0.0)
            self.error_occurred.emit(f"翻译失败: {str(e)[:50]}")

    @classmethod
    def _get_translate_lock(cls):
        if cls._translate_lock is None:
            import threading
            cls._translate_lock = threading.Lock()
        return cls._translate_lock
