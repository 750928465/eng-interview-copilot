"""
RAG 检索工作线程
"""
from PyQt5.QtCore import QThread, pyqtSignal
from typing import List
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from rag.vector_store import get_vector_store


class RagWorker(QThread):
    """向量检索线程"""

    # 信号定义
    search_completed = pyqtSignal(list, float)  # 检索结果 (List[str]), 耗时 ms
    error_occurred = pyqtSignal(str)

    def __init__(self, query: str = "", top_k: int = None):
        super().__init__()
        self.query = query
        self.top_k = top_k or config.top_k

    def set_query(self, query: str, top_k: int = None):
        """设置查询参数"""
        self.query = query
        self.top_k = top_k or config.top_k

    def run(self):
        """执行检索"""
        try:
            started = time.perf_counter()
            store = get_vector_store()
            results: List[str] = store.search(self.query, self.top_k)
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.search_completed.emit(results, elapsed_ms)
        except Exception as e:
            self.error_occurred.emit(f"检索失败: {e}")
