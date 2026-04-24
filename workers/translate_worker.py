"""
翻译工作线程
使用 Google 翻译 API（通过 deep-translator）
"""
from PyQt5.QtCore import QThread, pyqtSignal


class TranslateWorker(QThread):
    """翻译线程"""

    translate_completed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.text = ""

    def set_text(self, text: str):
        """设置待翻译文本"""
        self.text = text

    def run(self):
        """执行翻译"""
        if not self.text:
            return

        try:
            print(f"[Translate] 翻译: {self.text[:50]}...", flush=True)

            from deep_translator import GoogleTranslator

            # 使用 Google 翻译
            translator = GoogleTranslator(source='en', target='zh-CN')
            translated = translator.translate(self.text)

            print(f"[Translate] 结果: {translated}", flush=True)
            self.translate_completed.emit(translated)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Translate] 错误: {e}", flush=True)
            self.translate_completed.emit(f"(翻译失败)")
            self.error_occurred.emit(f"翻译失败: {str(e)[:50]}")