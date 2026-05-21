"""
LLM 流式生成工作线程
"""
from PyQt5.QtCore import QThread, pyqtSignal
from typing import Dict, List, Optional
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.engine import get_llm_engine, update_llm_config


class LLMWorker(QThread):
    """LLM 流式生成线程"""

    # 信号定义
    token_received = pyqtSignal(str)    # 收到单个 token
    first_token_latency = pyqtSignal(float)
    generation_complete = pyqtSignal(float)  # 生成完成，总耗时 ms
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        question: str = "",
        context: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ):
        super().__init__()
        self.question = question
        self.context = context
        self.conversation_history = conversation_history or []
        self._is_running = False
        self._started_at = 0.0
        self._first_token_seen = False

    def set_input(
        self,
        question: str,
        context: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ):
        """设置输入参数"""
        self.question = question
        self.context = context
        self.conversation_history = conversation_history or []

    def run(self):
        """执行流式生成"""
        self._is_running = True
        self._started_at = time.perf_counter()
        self._first_token_seen = False
        try:
            engine = get_llm_engine()

            for token in engine.generate(
                question=self.question,
                context=self.context,
                conversation_history=self.conversation_history,
                callback=self._on_token
            ):
                if not self._is_running:
                    break

            self.generation_complete.emit((time.perf_counter() - self._started_at) * 1000)

        except Exception as e:
            self.error_occurred.emit(f"LLM 生成失败: {e}")

    def _on_token(self, token: str):
        """Token 回调"""
        if self._is_running:
            if not self._first_token_seen:
                self._first_token_seen = True
                self.first_token_latency.emit((time.perf_counter() - self._started_at) * 1000)
            self.token_received.emit(token)

    def stop(self):
        """停止生成"""
        self._is_running = False
