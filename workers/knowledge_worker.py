"""
知识库初始化工作线程
"""
from PyQt5.QtCore import QThread, pyqtSignal

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.vector_store import get_vector_store


class KnowledgeWorker(QThread):
    """知识库加载线程"""

    load_completed = pyqtSignal(int)   # 加载完成，参数为文档数量
    error_occurred = pyqtSignal(str)

    def run(self):
        try:
            store = get_vector_store()
            count = store.load_knowledge()
            self.load_completed.emit(count)
        except Exception as e:
            self.error_occurred.emit(str(e))
