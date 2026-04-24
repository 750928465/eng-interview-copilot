"""
主界面模块
PyQt5 实现的 Windows 桌面界面
"""
import os
import sys

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QLineEdit,
    QGroupBox, QFormLayout, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from rag.vector_store import get_vector_store
from llm.engine import update_llm_config
from workers.audio_worker import AudioWorker
from workers.rag_worker import RagWorker
from workers.llm_worker import LLMWorker
from workers.translate_worker import TranslateWorker


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()

        self.audio_worker: AudioWorker = None
        self.rag_worker = RagWorker()
        self.llm_worker = LLMWorker()
        self.translate_worker = TranslateWorker()

        self.is_recording = False
        self.current_question = ""
        self.retrieved_context = ""

        self._init_ui()
        self._connect_signals()
        self._init_knowledge_base()

    def _init_ui(self):
        self.setWindowTitle("English Interview Copilot")
        self.setMinimumSize(1000, 750)
        self.resize(1100, 850)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 1. 配置区域
        config_group = QGroupBox("LLM 配置 (自动保存)")
        config_layout = QFormLayout()

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("输入 API Key (如: sk-xxx)")
        self.api_key_input.setText(config.api_key)

        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText("API 地址")
        self.base_url_input.setText(config.base_url)

        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("模型名称")
        self.model_input.setText(config.model_name)

        self.save_config_cb = QCheckBox("记住配置")
        self.save_config_cb.setChecked(True)

        config_layout.addRow("API Key:", self.api_key_input)
        config_layout.addRow("Base URL:", self.base_url_input)
        config_layout.addRow("Model:", self.model_input)
        config_layout.addRow("", self.save_config_cb)
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        # 2. 控制按钮
        button_layout = QHBoxLayout()

        self.record_btn = QPushButton("🎤 开始录音")
        self.record_btn.setMinimumHeight(50)
        self.record_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px; font-weight: bold;
                background-color: #4CAF50; color: white; border-radius: 8px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        button_layout.addWidget(self.record_btn)

        self.clear_btn = QPushButton("清空")
        self.clear_btn.setMinimumHeight(50)
        self.clear_btn.setMaximumWidth(100)
        button_layout.addWidget(self.clear_btn)

        main_layout.addLayout(button_layout)

        # 3. 状态标签
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        main_layout.addWidget(self.status_label)

        # 4. 文本展示区域
        texts_layout = QVBoxLayout()

        # ===== 面试官问题（左右分栏）=====
        question_group = QGroupBox("🎤 面试官问题 (ASR)")
        question_h_layout = QHBoxLayout()

        # 左边：英文原文
        left_layout = QVBoxLayout()
        left_label = QLabel("英文原文")
        left_label.setStyleSheet("color: #1976D2; font-weight: bold;")
        left_layout.addWidget(left_label)

        self.question_text = QTextEdit()
        self.question_text.setReadOnly(True)
        self.question_text.setFont(QFont("Arial", 12))
        self.question_text.setMinimumHeight(100)
        self.question_text.setPlaceholderText("识别到的英文问题...")
        left_layout.addWidget(self.question_text)

        # 右边：中文翻译
        right_layout = QVBoxLayout()
        right_label = QLabel("中文翻译")
        right_label.setStyleSheet("color: #388E3C; font-weight: bold;")
        right_layout.addWidget(right_label)

        self.translate_text = QTextEdit()
        self.translate_text.setReadOnly(True)
        self.translate_text.setFont(QFont("Arial", 12))
        self.translate_text.setMinimumHeight(100)
        self.translate_text.setPlaceholderText("翻译结果...")
        right_layout.addWidget(self.translate_text)

        question_h_layout.addLayout(left_layout)
        question_h_layout.addLayout(right_layout)
        question_group.setLayout(question_h_layout)
        texts_layout.addWidget(question_group)

        # 背景信息
        context_group = QGroupBox("📚 检索到的背景信息")
        context_layout = QVBoxLayout()
        self.context_text = QTextEdit()
        self.context_text.setReadOnly(True)
        self.context_text.setFont(QFont("Arial", 11))
        self.context_text.setMaximumHeight(80)
        self.context_text.setPlaceholderText("相关背景信息...")
        context_layout.addWidget(self.context_text)
        context_group.setLayout(context_layout)
        texts_layout.addWidget(context_group)

        # AI 回答
        answer_group = QGroupBox("💡 英文回答 (LLM)")
        answer_layout = QVBoxLayout()
        self.answer_text = QTextEdit()
        self.answer_text.setReadOnly(True)
        self.answer_text.setFont(QFont("Arial", 13))
        self.answer_text.setMinimumHeight(200)
        self.answer_text.setPlaceholderText("AI 生成的英文回答将流式显示在这里...")
        answer_layout.addWidget(self.answer_text)
        answer_group.setLayout(answer_layout)
        texts_layout.addWidget(answer_group)

        main_layout.addLayout(texts_layout)

        # 按钮事件
        self.record_btn.clicked.connect(self._toggle_recording)
        self.clear_btn.clicked.connect(self._clear_texts)

    def _connect_signals(self):
        self.rag_worker.search_completed.connect(self._on_search_completed)
        self.rag_worker.error_occurred.connect(self._on_error)
        self.llm_worker.token_received.connect(self._on_token_received)
        self.llm_worker.generation_complete.connect(self._on_generation_complete)
        self.llm_worker.error_occurred.connect(self._on_error)
        self.translate_worker.translate_completed.connect(self._on_translate_completed)
        self.translate_worker.error_occurred.connect(self._on_error)

    def _init_knowledge_base(self):
        try:
            store = get_vector_store()
            count = store.load_knowledge()
            if count > 0:
                self.status_label.setText(f"状态: 知识库已加载 ({count} 条)")
            else:
                self.status_label.setText("状态: 知识库为空")
        except Exception as e:
            self.status_label.setText(f"状态: 知识库初始化失败 - {e}")

    def _toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _save_config(self):
        config.api_key = self.api_key_input.text().strip()
        config.base_url = self.base_url_input.text().strip()
        config.model_name = self.model_input.text().strip()
        if self.save_config_cb.isChecked():
            config.save()

    def _start_recording(self):
        api_key = self.api_key_input.text().strip()
        base_url = self.base_url_input.text().strip()
        model_name = self.model_input.text().strip()

        if not api_key:
            QMessageBox.warning(self, "配置错误", "请输入 API Key")
            return

        self._save_config()
        update_llm_config(api_key, base_url, model_name)

        self.is_recording = True
        self.record_btn.setText("⏹ 停止录音")
        self.record_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px; font-weight: bold;
                background-color: #f44336; color: white; border-radius: 8px;
            }
            QPushButton:hover { background-color: #da190b; }
        """)
        self.status_label.setText("状态: 正在监听...")

        self.audio_worker = AudioWorker()
        self.audio_worker.text_recognized.connect(self._on_text_recognized)
        self.audio_worker.error_occurred.connect(self._on_error)
        self.audio_worker.status_changed.connect(self._on_status_changed)
        self.audio_worker.start()

    def _stop_recording(self):
        self.is_recording = False
        self.record_btn.setText("🎤 开始录音")
        self.record_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px; font-weight: bold;
                background-color: #4CAF50; color: white; border-radius: 8px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.status_label.setText("状态: 已停止监听")

        if self.audio_worker:
            self.audio_worker.stop()
            self.audio_worker.quit()
            self.audio_worker.wait()
            self.audio_worker = None

    def _on_text_recognized(self, text: str):
        """识别到文本后的处理"""
        self.current_question = text

        # 显示英文原文
        self.question_text.setText(text)

        # 启动翻译（异步）
        self.translate_worker.set_text(text)
        self.translate_worker.start()

        # 清空之前的回答
        self.answer_text.clear()
        self.context_text.clear()

        # 启动 RAG 检索
        self.rag_worker.set_query(text)
        self.rag_worker.start()

    def _on_translate_completed(self, translated: str):
        """翻译完成回调"""
        self.translate_text.setText(translated)

    def _on_search_completed(self, results: list):
        if results:
            self.retrieved_context = "\n\n---\n\n".join(results)
            self.context_text.setText(self.retrieved_context)
        else:
            self.retrieved_context = ""
            self.context_text.setText("(未找到相关背景信息)")
        self.llm_worker.set_input(self.current_question, self.retrieved_context)
        self.llm_worker.start()

    def _on_token_received(self, token: str):
        cursor = self.answer_text.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(token)
        self.answer_text.ensureCursorVisible()

    def _on_generation_complete(self):
        self.status_label.setText("状态: 回答完成")

    def _on_status_changed(self, status: str):
        self.status_label.setText(f"状态: {status}")

    def _on_error(self, error_msg: str):
        self.status_label.setText(f"错误: {error_msg}")
        QMessageBox.warning(self, "错误", error_msg)

    def _clear_texts(self):
        self.question_text.clear()
        self.translate_text.clear()
        self.context_text.clear()
        self.answer_text.clear()
        self.current_question = ""
        self.retrieved_context = ""
        self.status_label.setText("状态: 已清空")

    def closeEvent(self, event):
        if self.save_config_cb.isChecked():
            self._save_config()

        if self.audio_worker:
            self.audio_worker.stop()
            self.audio_worker.quit()
            self.audio_worker.wait()

        self.rag_worker.quit()
        self.rag_worker.wait()
        self.llm_worker.stop()
        self.llm_worker.quit()
        self.llm_worker.wait()
        self.translate_worker.quit()
        self.translate_worker.wait()

        event.accept()