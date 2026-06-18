"""
主界面模块
PyQt5 实现 - 双模式 + 双窗口对话列表
配色：黑橙红
"""
import os
import re
import sys
import uuid

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QLineEdit,
    QGroupBox, QFormLayout, QMessageBox, QCheckBox,
    QSlider, QScrollArea, QFrame, QTabWidget, QComboBox, QFileDialog,
    QProgressBar
)
from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from llm.engine import update_llm_config
from translation.history import (
    append_translation_history,
    clear_translation_history,
    load_translation_requests,
)
from workers.audio_worker import AudioWorker
from workers.rag_worker import RagWorker
from workers.llm_worker import LLMWorker
from workers.translate_worker import TranslateWorker
from workers.knowledge_worker import KnowledgeWorker


DEFAULT_RAG_REWRITE_PROMPT = """You are preparing a commercial RAG knowledge base for an English interview copilot.

Rewrite the uploaded source into concise Markdown that is optimized for semantic retrieval.
The goal is to help interview questions quickly retrieve the user's core information and the correct project details.

Requirements:
1. Start with a "User Core Profile" section. Extract the user's identity, background, target role or program, research direction, main skills, strengths, and key achievements.
2. Give every project an explicit stable number and title, such as "Project 1: ...", "Project 2: ...", "Research Direction 1: ...". Keep these numbers consistent throughout the rewrite.
3. For each project or research direction, include aliases and retrieval keywords, including phrases like "first project", "project one", "Project 1", and the project name if available.
4. For each project, structure the content with: background, goal, method, personal contribution, technical details, results or metrics, challenges, limitations, and interview talking points.
5. Preserve concrete facts, names, dates, metrics, tools, methods, datasets, responsibilities, and achievements from the source.
6. Do not invent information. If a field is missing, omit it or write "Not specified" only when useful.
7. Prefer short headings and bullet points. Keep wording clear enough for direct use as RAG context.

Source:
{source_text}
"""

# ============================================
# 配色方案
# ============================================
STYLESHEET = """
QMainWindow {
    background-color: #1a1a1a;
}
QWidget {
    background-color: #1a1a1a;
    color: #e0e0e0;
    font-family: -apple-system, 'Segoe UI', sans-serif;
}
QGroupBox {
    border: 1px solid #333;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #ff6b35;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QLineEdit {
    background-color: #2a2a2a;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 6px 10px;
    color: #e0e0e0;
    selection-background-color: #ff6b35;
}
QLineEdit:focus {
    border: 1px solid #ff6b35;
}
QTextEdit {
    background-color: #222;
    border: 1px solid #333;
    border-radius: 4px;
    color: #e0e0e0;
    padding: 4px;
}
QSlider::groove:horizontal {
    height: 4px; background: #444; border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px; height: 14px; margin: -5px 0;
    background: #ff6b35; border-radius: 7px;
}
QProgressBar {
    background-color: #242424;
    border: 1px solid #444;
    border-radius: 4px;
    color: #aaa;
    height: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #ff6b35;
    border-radius: 3px;
}
QScrollBar:vertical {
    background: #1a1a1a; width: 8px; border: none;
}
QScrollBar::handle:vertical {
    background: #444; border-radius: 4px; min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #ff6b35;
}
QCheckBox {
    color: #aaa; spacing: 6px;
}
QCheckBox::indicator {
    width: 14px; height: 14px; border: 1px solid #555; border-radius: 7px;
    background: #2a2a2a;
}
QCheckBox::indicator:checked {
    background: #ff6b35;
    border: 4px solid #2a2a2a;
    border-radius: 7px;
}
QLabel {
    color: #ccc;
}
QScrollArea#PanelBox {
    background-color: #202020;
    border: 1px solid #333;
    border-radius: 6px;
}
"""


class RoundCard(QFrame):
    """单轮对话卡片"""
    add_requested = pyqtSignal(str)

    def __init__(self, round_num: int, parent=None):
        super().__init__(parent)
        self.round_num = round_num
        self.is_active = True
        self.english_text = ""
        self.translation_text = ""
        self.segment_widgets = []
        self.asr_ms = 0.0
        self.history_saved = False
        self.history_request_id = ""
        self.history_request_started_at = ""
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            RoundCard {
                background-color: #f7f7f5;
                border: 1px solid #d8d8d8;
                border-radius: 8px;
                margin: 4px 0;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # 轮次标签
        self.round_label = QLabel(f"Round {round_num}")
        self.round_label.setStyleSheet("color: #ff6b35; font-size: 13px; font-weight: bold;")
        layout.addWidget(self.round_label)

        self.segment_container = QWidget()
        self.segment_container.setStyleSheet("background-color: transparent;")
        self.segment_layout = QVBoxLayout(self.segment_container)
        self.segment_layout.setContentsMargins(0, 0, 0, 0)
        self.segment_layout.setSpacing(12)
        layout.addWidget(self.segment_container)

        footer = QHBoxLayout()
        footer.setSpacing(6)
        self.metrics_label = QLabel("")
        self.metrics_label.setStyleSheet("color: #777; font-size: 11px;")
        footer.addWidget(self.metrics_label, 1)

        self.add_btn = QPushButton("加入提问")
        self.add_btn.setMinimumHeight(26)
        self.add_btn.setMaximumWidth(86)
        self.add_btn.setStyleSheet("""
            QPushButton {
                background-color: #333; color: #ddd;
                border-radius: 4px; border: 1px solid #444;
                font-size: 12px;
            }
            QPushButton:hover { border-color: #ff6b35; color: white; }
        """)
        self.add_btn.clicked.connect(lambda: self.add_requested.emit(self.english_text))
        footer.addWidget(self.add_btn)
        layout.addLayout(footer)

    def set_english(self, text: str):
        self.english_text = text
        self._render_segments()

    def set_translation(self, text: str):
        self.translation_text = text
        self._render_segments()

    def set_metrics(self, text: str):
        self.metrics_label.setText(text)

    def _split_text(self, text: str, is_chinese: bool = False) -> list:
        text = (text or "").strip()
        if not text:
            return []
        if "\n" in text:
            return [part.strip() for part in text.splitlines() if part.strip()]
        if is_chinese:
            parts = re.split(r"(?<=[。！？!?])\s*", text)
        else:
            parts = re.split(r"(?<=[.!?])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def _clear_segments(self):
        for widget in self.segment_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self.segment_widgets = []

    def _render_segments(self):
        self._clear_segments()
        english_parts = self._split_text(self.english_text)
        chinese_parts = self._split_text(self.translation_text, is_chinese=True)

        if not english_parts and self.english_text:
            english_parts = [self.english_text.strip()]

        for index, english in enumerate(english_parts):
            block = QWidget()
            block.setStyleSheet("background-color: transparent;")
            block_layout = QVBoxLayout(block)
            block_layout.setContentsMargins(0, 0, 0, 0)
            block_layout.setSpacing(4)

            en_label = QLabel(english)
            en_label.setWordWrap(True)
            en_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            en_label.setStyleSheet("color: #111; font-size: 15px; line-height: 1.45;")
            block_layout.addWidget(en_label)

            if chinese_parts:
                chinese = chinese_parts[index] if index < len(chinese_parts) else ""
                if chinese:
                    zh_label = QLabel(chinese)
                    zh_label.setWordWrap(True)
                    zh_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                    zh_label.setStyleSheet("color: #9a9a9a; font-size: 13px; line-height: 1.45;")
                    block_layout.addWidget(zh_label)

            self.segment_layout.addWidget(block)
            self.segment_widgets.append(block)


class AnswerCard(QFrame):
    """单轮 AI 回答卡片"""
    def __init__(self, round_num: int, parent=None):
        super().__init__(parent)
        self.round_num = round_num
        self.is_active = True
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            AnswerCard {
                background-color: #222;
                border: 1px solid #333;
                border-radius: 8px;
                margin: 2px 0;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)

        self.round_label = QLabel(f"Round {round_num}")
        self.round_label.setStyleSheet("color: #e63946; font-size: 11px; font-weight: bold;")
        self.round_label.setMaximumHeight(16)
        layout.addWidget(self.round_label)

        self.answer_text = QTextEdit()
        self.answer_text.setReadOnly(True)
        self.answer_text.setFont(QFont("Arial", 13))
        self.answer_text.setMinimumHeight(260)
        self.answer_text.setMaximumHeight(660)
        self.answer_text.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
                color: #e0e0e0;
                font-size: 13px;
                padding: 0;
                line-height: 1.35;
            }
        """)
        layout.addWidget(self.answer_text)

        self.metrics_label = QLabel("")
        self.metrics_label.setStyleSheet("color: #777; font-size: 11px;")
        layout.addWidget(self.metrics_label)

    def append_token(self, token: str):
        cursor = self.answer_text.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(token)
        self.answer_text.ensureCursorVisible()

    def set_answer(self, text: str):
        self.answer_text.setText(text)

    def get_answer(self) -> str:
        return self.answer_text.toPlainText()

    def set_metrics(self, text: str):
        self.metrics_label.setText(text)


class HistoryRequestCard(QFrame):
    """历史记录中的一次完整录音请求"""

    ROUND_STEP = 10

    def __init__(self, request_data: dict, visible_rounds: int, parent=None):
        super().__init__(parent)
        self.request_data = request_data
        self.visible_rounds = visible_rounds
        self.expand_requested = None
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            HistoryRequestCard {
                background-color: #222;
                border: 1px solid #333;
                border-radius: 8px;
                margin: 4px 0;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        rounds = request_data.get("rounds", [])
        started_at = request_data.get("started_at", "")
        title = QLabel(f"{started_at} · {len(rounds)} rounds")
        title.setStyleSheet("color: #ff6b35; font-size: 13px; font-weight: bold;")
        layout.addWidget(title)

        for index, item in enumerate(rounds[:visible_rounds], start=1):
            round_num = item.get("round_num") or index
            source = item.get("source_text", "")
            translated = item.get("translated_text", "")
            row = QLabel(f"Round {round_num}\n原文: {source}\n译文: {translated}")
            row.setWordWrap(True)
            row.setTextInteractionFlags(Qt.TextSelectableByMouse)
            row.setStyleSheet("""
                QLabel {
                    color: #ddd;
                    font-size: 12px;
                    line-height: 1.4;
                    padding: 7px;
                    background-color: #1b1b1b;
                    border: 1px solid #333;
                    border-radius: 5px;
                }
            """)
            layout.addWidget(row)

        if len(rounds) > self.ROUND_STEP:
            self.toggle_btn = QPushButton("展开更多")
            if visible_rounds >= len(rounds):
                self.toggle_btn.setText("收起")
            else:
                remaining = len(rounds) - visible_rounds
                self.toggle_btn.setText(f"展开更多（剩余 {remaining}）")
            self.toggle_btn.setMinimumHeight(30)
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #333; color: #ddd;
                    border-radius: 4px; border: 1px solid #444;
                }
                QPushButton:hover { border-color: #ff6b35; color: white; }
            """)
            self.toggle_btn.clicked.connect(self._on_toggle_clicked)
            layout.addWidget(self.toggle_btn)

    def _on_toggle_clicked(self):
        if self.expand_requested:
            self.expand_requested(self.request_data)


class KnowledgeImportWorker(QThread):
    """解析上传文件，并可用当前 LLM 配置改写成适合 RAG 的 Markdown。"""

    import_completed = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, file_path: str, rewrite_with_llm: bool = True, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.rewrite_with_llm = rewrite_with_llm

    def run(self):
        try:
            raw_text = self._extract_text(self.file_path).strip()
            if not raw_text:
                raise ValueError("文件没有解析出可用文本")

            if self.rewrite_with_llm and config.api_key:
                rewritten = self._rewrite_text(raw_text).strip()
                if rewritten and not rewritten.startswith("["):
                    self.import_completed.emit(rewritten, "已解析并使用 LLM 改写")
                    return

            self.import_completed.emit(raw_text, "已解析文件内容")
        except Exception as e:
            self.error_occurred.emit(f"导入失败: {e}")

    def _extract_text(self, file_path: str) -> str:
        suffix = os.path.splitext(file_path)[1].lower()
        if suffix in {".txt", ".md"}:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        if suffix == ".pdf":
            return self._extract_pdf_text(file_path)
        raise ValueError("当前仅支持 .txt、.md 和 .pdf 文件")

    def _extract_pdf_text(self, file_path: str) -> str:
        try:
            from pypdf import PdfReader
        except Exception as e:
            raise RuntimeError("缺少 PDF 解析依赖 pypdf，请先安装 requirements.txt") from e

        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)

    def _rewrite_text(self, raw_text: str) -> str:
        client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        prompt = DEFAULT_RAG_REWRITE_PROMPT.format(source_text=raw_text[:24000])
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite uploaded source material into retrieval-friendly Markdown for a RAG knowledge base.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=3000,
        )
        return response.choices[0].message.content or ""


class AudioDebugWorker(QThread):
    """只监听音频电平，不执行 ASR。"""

    volume_changed = pyqtSignal(float)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        audio_source: str,
        audio_device_index: int = None,
        audio_device_name: str = "",
        audio_gain: float = 1.0,
        parent=None,
    ):
        super().__init__(parent)
        self.audio_source = audio_source
        self.audio_device_index = audio_device_index
        self.audio_device_name = audio_device_name
        self.audio_gain = max(0.1, float(audio_gain or 1.0))
        self._is_running = False
        self._system_source = None

    def run(self):
        self._is_running = True
        try:
            if self.audio_source == "mac_system":
                self._run_system_audio()
            else:
                self._run_microphone()
        except Exception as e:
            self.error_occurred.emit(f"音频调试失败: {e}")
        finally:
            self._is_running = False
            if self._system_source:
                self._system_source.stop()
                self._system_source = None
            self.status_changed.emit("音频调试已停止")

    def _run_system_audio(self):
        import numpy as np
        from asr.audio_sources import MacSystemAudioSource

        self.status_changed.emit("正在监听系统音频...")
        self._system_source = MacSystemAudioSource(
            sample_rate=config.mac_system_audio_sample_rate,
            blocksize=1024,
        )
        self._system_source.start()
        while self._is_running:
            data = self._system_source.read_chunk()
            if data.size:
                self._emit_volume(np.clip(data * self.audio_gain, -1.0, 1.0))

    def _run_microphone(self):
        import numpy as np
        import sounddevice as sd
        from asr.recognizer import find_working_microphone

        device_index, sample_rate = find_working_microphone(
            preferred_index=self.audio_device_index,
            preferred_name=self.audio_device_name,
        )
        device_info = sd.query_devices(device_index)
        self.status_changed.emit(f"正在监听: {device_info['name']}")

        def audio_callback(indata, frames, time, status):
            if self._is_running:
                self._emit_volume(np.clip(indata * self.audio_gain, -1.0, 1.0))

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device_index,
            callback=audio_callback,
            blocksize=1024,
        ):
            while self._is_running:
                sd.sleep(100)

    def _emit_volume(self, data):
        import numpy as np

        level = float(np.abs(data).mean())
        self.volume_changed.emit(level)

    def stop(self):
        self._is_running = False
        if self._system_source:
            self._system_source.stop()


class MainWindow(QMainWindow):
    """主窗口"""

    MAX_ROUNDS = 5

    def __init__(self):
        super().__init__()

        self.audio_worker: AudioWorker = None
        self.audio_debug_worker = None
        self.knowledge_worker = None
        self.import_worker = None
        self.active_workers = []

        self.is_recording = False
        self.is_audio_debugging = False
        self.config_dirty = False
        self.current_mode = config.mode  # auto / manual
        self.current_question = ""
        self.retrieved_context = ""
        self.last_asr_ms = 0.0
        self.current_rag_ms = 0.0
        self.current_llm_ttft_ms = 0.0
        self.screen_capture_permission_prompted = False
        self.conversation_history = []
        self.current_translation_request_id = ""
        self.current_translation_request_started_at = ""
        self.history_page = 0
        self.history_expanded_rounds = {}
        self.history_request_widgets = []

        # 对话轮次管理
        self.round_num = 0
        self.answer_round_num = 0
        self.translate_cards = []  # RoundCard 列表
        self.answer_cards = []     # AnswerCard 列表

        self._init_ui()
        self._connect_signals()
        self._init_knowledge_base()

    def _init_ui(self):
        self.setWindowTitle("English Interview Copilot")
        self.setMinimumSize(1100, 800)
        self.resize(1200, 900)
        self.setStyleSheet(STYLESHEET)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setSpacing(0)
        root_layout.setContentsMargins(12, 12, 12, 12)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; border-radius: 6px; }
            QTabBar::tab {
                background: #242424; color: #aaa;
                padding: 8px 18px; border: 1px solid #333;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: #ff6b35; color: white; font-weight: bold;
            }
        """)
        root_layout.addWidget(self.tabs)

        conversation_tab = QWidget()
        rag_tab = QWidget()
        config_tab = QWidget()
        history_tab = QWidget()
        self.tabs.addTab(conversation_tab, "首页")
        self.tabs.addTab(rag_tab, "RAG知识库")
        self.tabs.addTab(config_tab, "配置界面")
        self.tabs.addTab(history_tab, "历史记录")

        main_layout = QVBoxLayout(conversation_tab)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)

        rag_layout = QVBoxLayout(rag_tab)
        rag_layout.setSpacing(8)
        rag_layout.setContentsMargins(10, 10, 10, 10)

        config_page_layout = QVBoxLayout(config_tab)
        config_page_layout.setSpacing(8)
        config_page_layout.setContentsMargins(10, 10, 10, 10)

        history_layout = QVBoxLayout(history_tab)
        history_layout.setSpacing(8)
        history_layout.setContentsMargins(10, 10, 10, 10)

        # ====== 1. 配置区域（折叠式） ======
        config_group = QGroupBox("LLM 配置")
        config_body_layout = QHBoxLayout()
        config_body_layout.setSpacing(16)
        config_layout = QFormLayout()
        config_layout.setSpacing(6)
        config_layout.setLabelAlignment(Qt.AlignRight)
        config_layout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        self.api_key_input = QLineEdit()
        self.api_key_input.setFixedWidth(420)
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("API Key")
        self.api_key_input.setText(config.api_key)

        self.base_url_input = QLineEdit()
        self.base_url_input.setFixedWidth(420)
        self.base_url_input.setPlaceholderText("API Base URL")
        self.base_url_input.setText(config.base_url)

        self.model_input = QLineEdit()
        self.model_input.setFixedWidth(420)
        self.model_input.setPlaceholderText("Model")
        self.model_input.setText(config.model_name)

        self.audio_source_combo = QComboBox()
        self.audio_source_combo.setMinimumHeight(30)
        self.audio_source_combo.setFixedWidth(260)
        self.audio_source_combo.addItem("macOS 系统音频", "mac_system")
        self.audio_source_combo.addItem("麦克风 / 虚拟声卡", "microphone")
        source_index = self.audio_source_combo.findData(config.audio_source)
        self.audio_source_combo.setCurrentIndex(max(0, source_index))

        self.audio_device_combo = QComboBox()
        self.audio_device_combo.setMinimumHeight(30)
        self.audio_device_combo.setFixedWidth(320)
        self._load_audio_devices()

        self.audio_device_row = QWidget()
        self.audio_device_row.setStyleSheet("background-color: transparent;")
        audio_device_row_layout = QHBoxLayout(self.audio_device_row)
        audio_device_row_layout.setContentsMargins(0, 0, 0, 0)
        audio_device_row_layout.addWidget(self.audio_device_combo)
        audio_device_row_layout.addStretch(1)

        self.system_prompt_input = QTextEdit()
        self.system_prompt_input.setFont(QFont("Arial", 11))
        self.system_prompt_input.setMinimumHeight(210)
        self.system_prompt_input.setPlaceholderText(
            "Describe the assistant role, candidate identity, answer style, and interview context..."
        )
        self.system_prompt_input.setText(config.system_prompt)

        self.save_config_cb = QCheckBox("记住配置")
        self.save_config_cb.setChecked(True)

        config_layout.addRow("API Key:", self.api_key_input)
        config_layout.addRow("Base URL:", self.base_url_input)
        config_layout.addRow("Model:", self.model_input)
        config_layout.addRow("音频来源:", self.audio_source_combo)
        self.audio_device_label = QLabel("音频输入:")
        config_layout.addRow(self.audio_device_label, self.audio_device_row)
        config_layout.addRow("", self.save_config_cb)

        left_config_layout = QVBoxLayout()
        left_config_layout.setSpacing(10)
        left_config_layout.addLayout(config_layout)

        prompt_layout = QVBoxLayout()
        prompt_layout.setSpacing(6)
        prompt_label = QLabel("系统提示词")
        prompt_label.setStyleSheet("color: #ff6b35; font-size: 12px; font-weight: bold;")
        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.system_prompt_input, 1)

        # ====== 2. 准备页：模式切换 ======
        mode_group = QGroupBox("对话模式")
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(10)

        # 模式切换 Tab
        self.btn_auto = QPushButton("自动模式")
        self.btn_auto.setCheckable(True)
        self.btn_auto.setChecked(self.current_mode == "auto")
        self.btn_auto.setMinimumHeight(44)
        self.btn_auto.setMinimumWidth(100)
        self._apply_mode_btn_style(self.btn_auto, self.current_mode == "auto")

        self.btn_manual = QPushButton("手动模式")
        self.btn_manual.setCheckable(True)
        self.btn_manual.setChecked(self.current_mode == "manual")
        self.btn_manual.setMinimumHeight(44)
        self.btn_manual.setMinimumWidth(100)
        self._apply_mode_btn_style(self.btn_manual, self.current_mode == "manual")

        self.btn_auto.clicked.connect(lambda: self._switch_mode("auto"))
        self.btn_manual.clicked.connect(lambda: self._switch_mode("manual"))

        mode_layout.addWidget(self.btn_auto)
        mode_layout.addWidget(self.btn_manual)
        mode_layout.addStretch(1)
        mode_group.setLayout(mode_layout)
        left_config_layout.addWidget(mode_group)

        audio_debug_widget = QWidget()
        audio_debug_widget.setStyleSheet("background-color: transparent;")
        audio_debug_outer_layout = QVBoxLayout(audio_debug_widget)
        audio_debug_outer_layout.setContentsMargins(0, 0, 0, 0)
        audio_debug_outer_layout.setSpacing(4)
        audio_debug_layout = QHBoxLayout()
        audio_debug_layout.setSpacing(8)
        self.audio_debug_btn = QPushButton("音频调试")
        self.audio_debug_btn.setMinimumHeight(34)
        self.audio_debug_btn.setMaximumWidth(110)
        self.audio_debug_btn.setStyleSheet("""
            QPushButton {
                background-color: #333; color: #ddd;
                border-radius: 5px; border: 1px solid #444;
                font-weight: bold;
            }
            QPushButton:hover { border-color: #ff6b35; color: white; }
        """)
        self.audio_debug_bar = QProgressBar()
        self.audio_debug_bar.setRange(0, 100)
        self.audio_debug_bar.setValue(0)
        self.audio_debug_bar.setTextVisible(False)
        self.audio_debug_value_label = QLabel("0%")
        self.audio_debug_value_label.setMinimumWidth(36)
        self.audio_debug_value_label.setStyleSheet("color: #888; font-size: 11px;")
        self.audio_debug_status_label = QLabel("选择音频来源后点击调试")
        self.audio_debug_status_label.setWordWrap(True)
        self.audio_debug_status_label.setStyleSheet("color: #888; font-size: 11px;")
        audio_debug_layout.addWidget(self.audio_debug_btn)
        audio_debug_layout.addWidget(self.audio_debug_bar, 1)
        audio_debug_layout.addWidget(self.audio_debug_value_label)
        audio_debug_outer_layout.addLayout(audio_debug_layout)
        audio_debug_outer_layout.addWidget(self.audio_debug_status_label)
        left_config_layout.addWidget(audio_debug_widget)

        # ====== 3. 对话页：录音控制 ======
        record_layout = QHBoxLayout()
        record_layout.setSpacing(10)
        self.record_btn = QPushButton("🎤 开始录音")
        self.record_btn.setMinimumHeight(44)
        self.record_btn.setMinimumWidth(160)
        self.record_btn.setStyleSheet("""
            QPushButton {
                font-size: 15px; font-weight: bold;
                background-color: #ff6b35; color: white;
                border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #ff8c5a; }
        """)
        record_layout.addWidget(self.record_btn, 1)

        # 清空按钮
        self.clear_btn = QPushButton("清空")
        self.clear_btn.setMinimumHeight(44)
        self.clear_btn.setMaximumWidth(70)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #333; color: #aaa;
                border-radius: 6px; border: 1px solid #444;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #444; color: #e0e0e0; }
        """)
        record_layout.addWidget(self.clear_btn)

        main_layout.addLayout(record_layout)

        level_layout = QHBoxLayout()
        level_layout.setSpacing(8)
        level_label = QLabel("输入音量")
        level_label.setStyleSheet("color: #888; font-size: 11px;")
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        self.volume_bar.setValue(0)
        self.volume_bar.setTextVisible(False)
        self.volume_value_label = QLabel("0%")
        self.volume_value_label.setMinimumWidth(36)
        self.volume_value_label.setStyleSheet("color: #888; font-size: 11px;")
        level_layout.addWidget(level_label)
        level_layout.addWidget(self.volume_bar, 1)
        level_layout.addWidget(self.volume_value_label)
        main_layout.addLayout(level_layout)

        # ====== 4. 准备页：静音间隔（仅自动模式显示） ======
        gap_layout = QHBoxLayout()
        gap_label = QLabel("静音间隔:")
        gap_label.setStyleSheet("color: #888; font-size: 11px;")
        gap_layout.addWidget(gap_label)

        self.gap_slider = QSlider(Qt.Horizontal)
        self.gap_slider.setMinimum(10)
        self.gap_slider.setMaximum(100)
        self.gap_slider.setSingleStep(5)
        self.gap_slider.setValue(config.silence_gap)
        gap_layout.addWidget(self.gap_slider, 1)

        self.gap_value_label = QLabel(f"{config.silence_gap // 10}s")
        self.gap_value_label.setMinimumWidth(30)
        self.gap_value_label.setStyleSheet("color: #ff6b35; font-weight: bold; font-size: 11px;")
        gap_layout.addWidget(self.gap_value_label)

        self.gap_widget = QWidget()
        gap_widget_layout = QVBoxLayout(self.gap_widget)
        gap_widget_layout.setContentsMargins(0, 0, 0, 0)
        gap_widget_layout.addLayout(gap_layout)
        self.gap_widget.setVisible(self.current_mode == "auto")
        left_config_layout.addWidget(self.gap_widget)

        gain_layout = QHBoxLayout()
        gain_label = QLabel("音频增益:")
        gain_label.setStyleSheet("color: #888; font-size: 11px;")
        gain_layout.addWidget(gain_label)

        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setMinimum(10)
        self.gain_slider.setMaximum(800)
        self.gain_slider.setSingleStep(10)
        self.gain_slider.setValue(int(config.audio_gain * 100))
        gain_layout.addWidget(self.gain_slider, 1)

        self.gain_value_label = QLabel(f"{config.audio_gain:.1f}x")
        self.gain_value_label.setMinimumWidth(44)
        self.gain_value_label.setStyleSheet("color: #ff6b35; font-weight: bold; font-size: 11px;")
        gain_layout.addWidget(self.gain_value_label)

        self.gain_widget = QWidget()
        gain_widget_layout = QVBoxLayout(self.gain_widget)
        gain_widget_layout.setContentsMargins(0, 0, 0, 0)
        gain_widget_layout.addLayout(gain_layout)
        left_config_layout.addWidget(self.gain_widget)

        left_config_layout.addStretch(1)
        config_body_layout.addLayout(left_config_layout, 0)
        config_body_layout.addLayout(prompt_layout, 1)
        config_group.setLayout(config_body_layout)
        self.config_group = config_group
        config_page_layout.addWidget(config_group, 1)

        # ====== 5. 准备页：QA 对编辑 ======
        qa_group = QGroupBox("RAG 与 QA 知识库")
        qa_layout = QVBoxLayout()
        qa_layout.setSpacing(8)

        import_actions = QHBoxLayout()
        import_actions.setSpacing(8)
        self.import_text_btn = QPushButton("上传文本/Markdown")
        self.import_pdf_btn = QPushButton("上传 PDF")
        self.rewrite_import_cb = QCheckBox("用当前 LLM 改写后填充")
        self.rewrite_import_cb.setChecked(True)
        for btn in (self.import_text_btn, self.import_pdf_btn):
            btn.setMinimumHeight(34)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #333; color: #ddd;
                    border-radius: 4px; border: 1px solid #444;
                    padding: 4px 12px;
                }
                QPushButton:hover { border-color: #ff6b35; color: white; }
                QPushButton:disabled { color: #555; border-color: #333; }
            """)
        import_actions.addWidget(self.import_text_btn)
        import_actions.addWidget(self.import_pdf_btn)
        import_actions.addWidget(self.rewrite_import_cb)
        import_actions.addStretch(1)
        qa_layout.addLayout(import_actions)

        import_tip = QLabel("可直接填写你的主要信息，或上传文本/PDF；勾选 LLM 改写后，会整理成更适合检索的 RAG 内容。")
        import_tip.setWordWrap(True)
        import_tip.setStyleSheet("color: #888; font-size: 11px;")
        qa_layout.addWidget(import_tip)

        knowledge_split_layout = QHBoxLayout()
        knowledge_split_layout.setSpacing(10)

        qa_column = QVBoxLayout()
        qa_column.setSpacing(6)
        qa_label = QLabel("当前 QA")
        qa_label.setStyleSheet("color: #ff6b35; font-size: 12px; font-weight: bold;")
        qa_column.addWidget(qa_label)

        self.qa_editor = QTextEdit()
        self.qa_editor.setFont(QFont("Arial", 12))
        self.qa_editor.setPlaceholderText("Q: Tell me about a challenging project.\nA: One challenging project I worked on was...")
        self.qa_editor.setText(self._load_qa_text())
        qa_column.addWidget(self.qa_editor, 1)
        knowledge_split_layout.addLayout(qa_column, 1)

        preview_column = QVBoxLayout()
        preview_column.setSpacing(6)
        preview_label = QLabel("RAG 知识库")
        preview_label.setStyleSheet("color: #ff6b35; font-size: 12px; font-weight: bold;")
        preview_column.addWidget(preview_label)

        self.knowledge_editor = QTextEdit()
        self.knowledge_editor.setFont(QFont("Arial", 12))
        self.knowledge_editor.setPlaceholderText(
            "将你的主要信息填写进来，例如简历、项目经历、研究方向、个人优势、动机和关键成果。\n"
            "配置好 LLM 后，可以通过上传文本/PDF让大模型进行改写，提升 RAG 检索效果。"
        )
        self.knowledge_editor.setText(self._load_knowledge_text())
        preview_column.addWidget(self.knowledge_editor, 1)
        knowledge_split_layout.addLayout(preview_column, 1)

        qa_layout.addLayout(knowledge_split_layout, 1)

        terms_label = QLabel("ASR 领域词库（每行一个术语）")
        terms_label.setStyleSheet("color: #ff6b35; font-size: 12px; font-weight: bold;")
        qa_layout.addWidget(terms_label)

        self.terms_editor = QTextEdit()
        self.terms_editor.setFont(QFont("Arial", 11))
        self.terms_editor.setMaximumHeight(110)
        self.terms_editor.setPlaceholderText("large language model\nbioinformatics\nPhD proposal")
        self.terms_editor.setText(self._load_terms_text())
        qa_layout.addWidget(self.terms_editor)

        self.prep_status_label = QLabel("")
        self.prep_status_label.setStyleSheet("color: #888; font-size: 11px;")
        qa_layout.addWidget(self.prep_status_label)

        qa_group.setLayout(qa_layout)
        rag_layout.addWidget(qa_group, 1)

        self.apply_config_btn = QPushButton("应用更新")
        self.apply_config_btn.setMinimumHeight(40)
        config_page_layout.addWidget(self.apply_config_btn)

        self.apply_rag_btn = QPushButton("应用更新")
        self.apply_rag_btn.setMinimumHeight(40)
        rag_layout.addWidget(self.apply_rag_btn)

        history_actions = QHBoxLayout()
        history_actions.setSpacing(8)
        history_title = QLabel("只保留翻译原文和译文，不记录知识库问答")
        history_title.setStyleSheet("color: #ff6b35; font-size: 13px; font-weight: bold;")
        history_actions.addWidget(history_title, 1)

        self.refresh_history_btn = QPushButton("刷新")
        self.refresh_history_btn.setMinimumHeight(32)
        self.clear_history_btn = QPushButton("清空历史")
        self.clear_history_btn.setMinimumHeight(32)
        for btn in (self.refresh_history_btn, self.clear_history_btn):
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #333; color: #ddd;
                    border-radius: 4px; border: 1px solid #444;
                    padding: 4px 12px;
                }
                QPushButton:hover { border-color: #ff6b35; color: white; }
            """)
        history_actions.addWidget(self.refresh_history_btn)
        history_actions.addWidget(self.clear_history_btn)
        history_layout.addLayout(history_actions)

        self.history_scroll = QScrollArea()
        self.history_scroll.setObjectName("PanelBox")
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history_container = QWidget()
        self.history_list_layout = QVBoxLayout(self.history_container)
        self.history_list_layout.setAlignment(Qt.AlignTop)
        self.history_list_layout.setSpacing(8)
        self.history_list_layout.setContentsMargins(8, 8, 8, 8)
        self.history_empty_label = QLabel("暂无翻译历史")
        self.history_empty_label.setAlignment(Qt.AlignCenter)
        self.history_empty_label.setWordWrap(True)
        self.history_empty_label.setMinimumHeight(160)
        self.history_empty_label.setStyleSheet("color: #666; font-size: 15px;")
        self.history_list_layout.addWidget(self.history_empty_label)
        self.history_scroll.setWidget(self.history_container)
        history_layout.addWidget(self.history_scroll, 1)

        history_page_actions = QHBoxLayout()
        history_page_actions.setSpacing(8)
        self.prev_history_page_btn = QPushButton("上一页")
        self.next_history_page_btn = QPushButton("下一页")
        self.history_page_label = QLabel("")
        self.history_page_label.setAlignment(Qt.AlignCenter)
        self.history_page_label.setStyleSheet("color: #888; font-size: 12px;")
        for btn in (self.prev_history_page_btn, self.next_history_page_btn):
            btn.setMinimumHeight(32)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #333; color: #ddd;
                    border-radius: 4px; border: 1px solid #444;
                    padding: 4px 12px;
                }
                QPushButton:hover { border-color: #ff6b35; color: white; }
                QPushButton:disabled { color: #555; border-color: #333; }
            """)
        history_page_actions.addWidget(self.prev_history_page_btn)
        history_page_actions.addWidget(self.history_page_label, 1)
        history_page_actions.addWidget(self.next_history_page_btn)
        history_layout.addLayout(history_page_actions)

        # ====== 3. 状态栏 ======
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("color: #888; font-size: 11px; padding: 2px 0;")
        main_layout.addWidget(self.status_label)

        # ====== 4. 双窗口对话区域 ======
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(8)

        # 左侧：翻译窗口
        left_panel = QVBoxLayout()
        left_panel.setSpacing(4)
        left_title = QLabel("翻译")
        left_title.setStyleSheet("color: #ff6b35; font-size: 15px; font-weight: bold; padding: 4px 0;")
        left_panel.addWidget(left_title)

        self.translate_scroll = QScrollArea()
        self.translate_scroll.setObjectName("PanelBox")
        self.translate_scroll.setWidgetResizable(True)
        self.translate_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.translate_container = QWidget()
        self.translate_list_layout = QVBoxLayout(self.translate_container)
        self.translate_list_layout.setAlignment(Qt.AlignTop)
        self.translate_list_layout.setSpacing(6)
        self.translate_list_layout.setContentsMargins(8, 8, 8, 8)
        self.translate_empty_label = QLabel("这里会实时显示对话内容")
        self.translate_empty_label.setAlignment(Qt.AlignCenter)
        self.translate_empty_label.setWordWrap(True)
        self.translate_empty_label.setMinimumHeight(160)
        self.translate_empty_label.setStyleSheet("color: #666; font-size: 15px;")
        self.translate_list_layout.addWidget(self.translate_empty_label)
        self.translate_scroll.setWidget(self.translate_container)
        left_panel.addWidget(self.translate_scroll, 1)
        panels_layout.addLayout(left_panel, 1)

        # 右侧：AI 回答窗口
        right_panel = QVBoxLayout()
        right_panel.setSpacing(4)
        right_title = QLabel("AI 回答")
        right_title.setStyleSheet("color: #e63946; font-size: 15px; font-weight: bold; padding: 4px 0;")
        right_panel.addWidget(right_title)

        self.answer_scroll = QScrollArea()
        self.answer_scroll.setObjectName("PanelBox")
        self.answer_scroll.setWidgetResizable(True)
        self.answer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.answer_container = QWidget()
        self.answer_list_layout = QVBoxLayout(self.answer_container)
        self.answer_list_layout.setAlignment(Qt.AlignTop)
        self.answer_list_layout.setSpacing(6)
        self.answer_list_layout.setContentsMargins(8, 8, 8, 8)
        self.answer_empty_label = QLabel("这里会输出 AI 的回复")
        self.answer_empty_label.setAlignment(Qt.AlignCenter)
        self.answer_empty_label.setWordWrap(True)
        self.answer_empty_label.setMinimumHeight(160)
        self.answer_empty_label.setStyleSheet("color: #666; font-size: 15px;")
        self.answer_list_layout.addWidget(self.answer_empty_label)
        self.answer_scroll.setWidget(self.answer_container)
        right_panel.addWidget(self.answer_scroll, 1)

        question_box = QGroupBox("LLM 提问框")
        question_layout = QVBoxLayout()
        question_layout.setSpacing(6)
        self.question_input = QTextEdit()
        self.question_input.setFont(QFont("Arial", 12))
        self.question_input.setMaximumHeight(110)
        self.question_input.setPlaceholderText("点击左侧字幕的“加入提问”，或在这里手动编辑后触发检索和回答。")
        question_layout.addWidget(self.question_input)

        question_actions = QHBoxLayout()
        question_actions.setSpacing(8)
        self.ask_llm_btn = QPushButton("检索并回答")
        self.ask_llm_btn.setMinimumHeight(34)
        self.ask_llm_btn.setStyleSheet("""
            QPushButton {
                background-color: #e63946; color: white;
                border-radius: 6px; border: none;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #ff4d5a; }
        """)
        self.clear_question_btn = QPushButton("清空")
        self.clear_question_btn.setMinimumHeight(34)
        self.clear_question_btn.setStyleSheet("""
            QPushButton {
                background-color: #333; color: #aaa;
                border-radius: 6px; border: 1px solid #444;
            }
            QPushButton:hover { background-color: #444; color: #e0e0e0; }
        """)
        question_actions.addWidget(self.ask_llm_btn, 1)
        question_actions.addWidget(self.clear_question_btn)
        question_layout.addLayout(question_actions)
        question_box.setLayout(question_layout)
        right_panel.addWidget(question_box)
        panels_layout.addLayout(right_panel, 1)

        main_layout.addLayout(panels_layout, 1)

        # 按钮事件
        self.record_btn.clicked.connect(self._toggle_recording)
        self.clear_btn.clicked.connect(self._clear_all)
        self.audio_debug_btn.clicked.connect(self._toggle_audio_debug)
        self.gap_slider.valueChanged.connect(self._on_gap_changed)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)
        self.apply_config_btn.clicked.connect(self._apply_preparation_updates)
        self.apply_rag_btn.clicked.connect(self._apply_preparation_updates)
        self.import_text_btn.clicked.connect(lambda: self._choose_knowledge_file("text"))
        self.import_pdf_btn.clicked.connect(lambda: self._choose_knowledge_file("pdf"))
        self.ask_llm_btn.clicked.connect(self._ask_from_question_box)
        self.clear_question_btn.clicked.connect(self.question_input.clear)
        self.refresh_history_btn.clicked.connect(self._refresh_translation_history)
        self.clear_history_btn.clicked.connect(self._clear_translation_history)
        self.prev_history_page_btn.clicked.connect(lambda: self._change_history_page(-1))
        self.next_history_page_btn.clicked.connect(lambda: self._change_history_page(1))
        self._connect_dirty_signals()
        self._set_config_dirty(False)
        self._refresh_translation_history()

    # ============================================
    # 样式辅助
    # ============================================

    def _apply_mode_btn_style(self, btn: QPushButton, active: bool):
        if active:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #ff6b35; color: white;
                    border-radius: 6px; border: none;
                    font-weight: bold; font-size: 13px;
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2a2a2a; color: #888;
                    border-radius: 6px; border: 1px solid #444;
                    font-size: 13px;
                }
                QPushButton:hover { color: #e0e0e0; border-color: #ff6b35; }
            """)

    # ============================================
    # 模式切换
    # ============================================

    def _switch_mode(self, mode: str):
        if self.is_recording:
            return  # 录音中不允许切换
        self.current_mode = mode
        self.btn_auto.setChecked(mode == "auto")
        self.btn_manual.setChecked(mode == "manual")
        self._apply_mode_btn_style(self.btn_auto, mode == "auto")
        self._apply_mode_btn_style(self.btn_manual, mode == "manual")
        self.gap_widget.setVisible(mode == "auto")
        self._set_config_dirty(True)

    # ============================================
    # 录音控制
    # ============================================

    def _toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _load_qa_text(self) -> str:
        if not os.path.exists(config.qa_path):
            return ""
        try:
            with open(config.qa_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[QA] 加载失败: {e}")
            return ""

    def _load_terms_text(self) -> str:
        if not os.path.exists(config.asr_hotwords_path):
            return ""
        try:
            with open(config.asr_hotwords_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[ASR] 领域词加载失败: {e}")
            return ""

    def _load_knowledge_text(self) -> str:
        if not os.path.exists(config.knowledge_path):
            return ""
        try:
            with open(config.knowledge_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[Knowledge] 预览加载失败: {e}")
            return ""

    def _connect_dirty_signals(self):
        self.api_key_input.textChanged.connect(lambda: self._set_config_dirty(True))
        self.base_url_input.textChanged.connect(lambda: self._set_config_dirty(True))
        self.model_input.textChanged.connect(lambda: self._set_config_dirty(True))
        self.audio_source_combo.currentIndexChanged.connect(lambda _: self._set_config_dirty(True))
        self.audio_source_combo.currentIndexChanged.connect(self._on_audio_source_changed)
        self.audio_device_combo.currentIndexChanged.connect(lambda _: self._set_config_dirty(True))
        self.system_prompt_input.textChanged.connect(lambda: self._set_config_dirty(True))
        self.qa_editor.textChanged.connect(lambda: self._set_config_dirty(True))
        self.knowledge_editor.textChanged.connect(lambda: self._set_config_dirty(True))
        self.terms_editor.textChanged.connect(lambda: self._set_config_dirty(True))
        self.gap_slider.valueChanged.connect(lambda _: self._set_config_dirty(True))
        self.gain_slider.valueChanged.connect(lambda _: self._set_config_dirty(True))
        self.save_config_cb.stateChanged.connect(lambda _: self._set_config_dirty(True))

    def _button_style(self, dirty: bool) -> str:
        if dirty:
            return """
                QPushButton {
                    background-color: #ff6b35; color: white;
                    border-radius: 6px; border: none;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #ff8c5a; }
            """
        return """
            QPushButton {
                background-color: #333; color: #aaa;
                border-radius: 6px; border: 1px solid #444;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #444; color: #e0e0e0; }
        """

    def _set_config_dirty(self, dirty: bool):
        self.config_dirty = dirty
        text = "应用更新 *" if dirty else "应用更新"
        for button_name in ("apply_config_btn", "apply_rag_btn"):
            if hasattr(self, button_name):
                button = getattr(self, button_name)
                button.setText(text)
                button.setStyleSheet(self._button_style(dirty))
        if dirty:
            self.prep_status_label.setText("有未应用的更新")
        elif self.prep_status_label.text() == "有未应用的更新":
            self.prep_status_label.setText("")

    def _write_qa_text(self):
        with open(config.qa_path, "w", encoding="utf-8") as f:
            f.write(self.qa_editor.toPlainText().strip() + "\n")

    def _write_knowledge_text(self):
        with open(config.knowledge_path, "w", encoding="utf-8") as f:
            f.write(self.knowledge_editor.toPlainText().strip() + "\n")

    def _write_terms_text(self):
        with open(config.asr_hotwords_path, "w", encoding="utf-8") as f:
            f.write(self.terms_editor.toPlainText().strip() + "\n")

    def _load_audio_devices(self):
        self.audio_device_combo.clear()
        self.audio_device_combo.addItem("自动选择（优先 BlackHole / Loopback）", None)
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            selected_index = 0

            for index, device in enumerate(devices):
                if device["max_input_channels"] <= 0:
                    continue

                label = f"{device['name']}  ({device['max_input_channels']}ch)"
                self.audio_device_combo.addItem(label, index)
                if config.audio_device_index == index:
                    selected_index = self.audio_device_combo.count() - 1
                elif (
                    config.audio_device_index is None
                    and config.audio_device_name
                    and config.audio_device_name.lower() in device["name"].lower()
                ):
                    selected_index = self.audio_device_combo.count() - 1

            self.audio_device_combo.setCurrentIndex(selected_index)
        except Exception as e:
            self.audio_device_combo.addItem(f"音频设备加载失败: {e}", None)

        self._on_audio_source_changed()

    def _on_audio_source_changed(self):
        is_microphone = self.audio_source_combo.currentData() == "microphone"
        if not hasattr(self, "audio_device_row"):
            return
        self.audio_device_row.setVisible(is_microphone)
        self.audio_device_label.setVisible(is_microphone)
        self.audio_device_combo.setEnabled(is_microphone)

    def _toggle_audio_debug(self):
        if self.is_audio_debugging:
            self._stop_audio_debug()
        else:
            self._start_audio_debug()

    def _start_audio_debug(self):
        if self.is_recording:
            QMessageBox.warning(self, "正在录音", "录音中不能开启音频调试。")
            return
        if self.audio_debug_worker and self.audio_debug_worker.isRunning():
            return

        self._save_config()
        self.is_audio_debugging = True
        self.audio_debug_btn.setText("停止调试")
        self.audio_debug_status_label.setText("正在初始化音频...")
        self.audio_debug_bar.setValue(0)
        self.audio_debug_value_label.setText("0%")

        self.audio_debug_worker = AudioDebugWorker(
            audio_source=config.audio_source,
            audio_device_index=config.audio_device_index,
            audio_device_name=config.audio_device_name,
            audio_gain=config.audio_gain,
        )
        self.audio_debug_worker.volume_changed.connect(self._on_audio_debug_volume)
        self.audio_debug_worker.status_changed.connect(self._on_audio_debug_status)
        self.audio_debug_worker.error_occurred.connect(self._on_audio_debug_error)
        self.audio_debug_worker.finished.connect(self._on_audio_debug_finished)
        self.audio_debug_worker.start()

    def _stop_audio_debug(self):
        self.is_audio_debugging = False
        if self.audio_debug_worker:
            self.audio_debug_worker.stop()
            self.audio_debug_worker.quit()
            self.audio_debug_worker.wait()
            self.audio_debug_worker = None
        if hasattr(self, "audio_debug_btn"):
            self.audio_debug_btn.setText("音频调试")
            self.audio_debug_bar.setValue(0)
            self.audio_debug_value_label.setText("0%")
            self.audio_debug_status_label.setText("音频调试已停止")

    def _on_audio_debug_volume(self, level: float):
        value = max(0, min(100, int(level * 700)))
        self.audio_debug_bar.setValue(value)
        self.audio_debug_value_label.setText(f"{value}%")

    def _on_audio_debug_status(self, status: str):
        if hasattr(self, "audio_debug_status_label"):
            self.audio_debug_status_label.setText(status)

    def _on_audio_debug_error(self, error_msg: str):
        self.audio_debug_status_label.setText(error_msg)
        self.status_label.setText(f"错误: {error_msg}")
        if self._is_screen_capture_permission_error(error_msg):
            self._show_screen_capture_permission_dialog(error_msg)

    def _on_audio_debug_finished(self):
        self.is_audio_debugging = False
        self.audio_debug_worker = None
        if hasattr(self, "audio_debug_btn"):
            self.audio_debug_btn.setText("音频调试")

    def _choose_knowledge_file(self, file_type: str):
        if file_type == "pdf":
            file_filter = "PDF 文件 (*.pdf)"
        else:
            file_filter = "文本文件 (*.txt *.md);;所有文件 (*)"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择要导入的知识库文件",
            "",
            file_filter,
        )
        if not file_path:
            return

        rewrite_with_llm = self.rewrite_import_cb.isChecked()
        if rewrite_with_llm:
            self._save_config()
            if not config.api_key:
                QMessageBox.information(
                    self,
                    "未配置 API Key",
                    "当前会先解析原文并填充到 RAG；配置 API Key 后可启用 LLM 改写。",
                )

        self._set_import_buttons_enabled(False)
        self.prep_status_label.setText("正在解析上传文件...")
        self.import_worker = self._track_worker(KnowledgeImportWorker(file_path, rewrite_with_llm))
        self.import_worker.import_completed.connect(self._on_knowledge_import_completed)
        self.import_worker.error_occurred.connect(self._on_knowledge_import_error)
        self.import_worker.start()

    def _set_import_buttons_enabled(self, enabled: bool):
        if hasattr(self, "import_text_btn"):
            self.import_text_btn.setEnabled(enabled)
            self.import_pdf_btn.setEnabled(enabled)
            self.rewrite_import_cb.setEnabled(enabled)

    def _on_knowledge_import_completed(self, text: str, message: str):
        existing = self.knowledge_editor.toPlainText().strip()
        imported = text.strip()
        if existing:
            combined = f"{existing}\n\n---\n\n{imported}\n"
        else:
            combined = f"{imported}\n"
        self.knowledge_editor.setText(combined)
        self.knowledge_editor.moveCursor(self.knowledge_editor.textCursor().End)
        self._set_import_buttons_enabled(True)
        self._set_config_dirty(True)
        self.prep_status_label.setText(f"{message}，请点击应用更新刷新索引")

    def _on_knowledge_import_error(self, error_msg: str):
        self._set_import_buttons_enabled(True)
        self._on_error(error_msg)

    def _apply_preparation_updates(self):
        try:
            self._save_config()
            self._write_qa_text()
            self._write_knowledge_text()
            self._write_terms_text()
            update_llm_config(
                config.api_key,
                config.base_url,
                config.model_name,
                config.system_prompt
            )
            status = "配置、RAG 知识库和 QA 已应用，正在刷新索引..."
            self.prep_status_label.setText(status)
            self.status_label.setText(f"状态: {status}")
            self._set_config_dirty(False)
            self._init_knowledge_base()
        except Exception as e:
            self._on_error(f"应用更新失败: {e}")

    def _save_config(self):
        config.api_key = self.api_key_input.text().strip()
        config.base_url = self.base_url_input.text().strip()
        config.model_name = self.model_input.text().strip()
        config.system_prompt = self.system_prompt_input.toPlainText().strip()
        config.audio_source = self.audio_source_combo.currentData() or "mac_system"
        config.audio_device_index = self.audio_device_combo.currentData()
        if config.audio_device_index is None:
            config.audio_device_name = ""
        else:
            config.audio_device_name = self.audio_device_combo.currentText().split("  (", 1)[0]
        config.silence_gap = self.gap_slider.value()
        config.audio_gain = self.gain_slider.value() / 100.0
        config.mode = self.current_mode
        if self.save_config_cb.isChecked():
            config.save()

    def _start_recording(self):
        if self.config_dirty:
            QMessageBox.warning(self, "有未应用的更新", "请先在配置界面或 RAG 知识库页点击“应用更新”。")
            self.tabs.setCurrentIndex(2)
            return

        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "配置错误", "请输入 API Key")
            return

        if self.is_audio_debugging:
            self._stop_audio_debug()

        self._save_config()
        update_llm_config(
            api_key,
            self.base_url_input.text().strip(),
            self.model_input.text().strip(),
            self.system_prompt_input.toPlainText().strip()
        )

        self.is_recording = True
        self.current_translation_request_id = uuid.uuid4().hex
        self.current_translation_request_started_at = self._now_iso()
        self.record_btn.setText("⏹ 停止录音")
        self.record_btn.setStyleSheet("""
            QPushButton {
                font-size: 15px; font-weight: bold;
                background-color: #e63946; color: white;
                border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #ff4d5a; }
        """)
        self.status_label.setText(f"状态: 录音中 ({self.current_mode}模式)...")
        self.btn_auto.setEnabled(False)
        self.btn_manual.setEnabled(False)
        self.audio_debug_btn.setEnabled(False)

        self.audio_worker = AudioWorker(
            silence_gap=self.gap_slider.value(),
            mode=self.current_mode,
            audio_device_index=config.audio_device_index,
            audio_device_name=config.audio_device_name,
            audio_gain=config.audio_gain,
            audio_source=config.audio_source
        )
        self.audio_worker.text_recognized.connect(self._on_text_recognized)
        self.audio_worker.error_occurred.connect(self._on_error)
        self.audio_worker.status_changed.connect(self._on_status_changed)
        self.audio_worker.volume_changed.connect(self._on_volume_changed)
        self.audio_worker.asr_timing.connect(self._on_asr_timing)
        self.audio_worker.start()

    def _stop_recording(self):
        self.is_recording = False
        self.record_btn.setText("🎤 开始录音")
        self.record_btn.setStyleSheet("""
            QPushButton {
                font-size: 15px; font-weight: bold;
                background-color: #ff6b35; color: white;
                border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #ff8c5a; }
        """)
        self.btn_auto.setEnabled(True)
        self.btn_manual.setEnabled(True)
        self.audio_debug_btn.setEnabled(True)

        if self.audio_worker:
            self.audio_worker.stop()
            self.audio_worker.quit()
            self.audio_worker.wait()
            self.audio_worker = None
        self._on_volume_changed(0.0)

        QApplication.processEvents()

    # ============================================
    # 对话轮次管理
    # ============================================

    def _add_round(self, num: int):
        """新增一轮对话卡片"""
        self._add_subtitle_card(num)
        self._add_answer_card(num)

    def _add_subtitle_card(self, num: int) -> RoundCard:
        self.translate_empty_label.setVisible(False)
        t_card = RoundCard(num)
        t_card.add_requested.connect(self._append_question_text)
        self.translate_cards.insert(0, t_card)
        self.translate_list_layout.insertWidget(0, t_card)

        while len(self.translate_cards) > self.MAX_ROUNDS:
            old_t = self.translate_cards.pop()
            old_t.is_active = False
            old_t.setParent(None)
            old_t.deleteLater()

        self.translate_scroll.verticalScrollBar().setValue(0)
        return t_card

    def _add_answer_card(self, num: int) -> AnswerCard:
        self.answer_empty_label.setVisible(False)
        a_card = AnswerCard(num)
        self.answer_cards.insert(0, a_card)
        self.answer_list_layout.insertWidget(0, a_card)

        while len(self.answer_cards) > self.MAX_ROUNDS:
            old_a = self.answer_cards.pop()
            old_a.is_active = False
            old_a.setParent(None)
            old_a.deleteLater()

        self.answer_scroll.verticalScrollBar().setValue(0)
        return a_card

    def _current_translate_card(self) -> RoundCard:
        return self.translate_cards[0] if self.translate_cards else None

    def _current_answer_card(self) -> AnswerCard:
        return self.answer_cards[0] if self.answer_cards else None

    # ============================================
    # 信号回调
    # ============================================

    def _connect_signals(self):
        pass

    def _init_knowledge_base(self):
        if self.knowledge_worker and self.knowledge_worker.isRunning():
            self.knowledge_worker.quit()
            self.knowledge_worker.wait()

        self.status_label.setText("状态: 正在加载知识库...")
        self.knowledge_worker = KnowledgeWorker()
        self.knowledge_worker.load_completed.connect(self._on_knowledge_loaded)
        self.knowledge_worker.error_occurred.connect(
            lambda e: self.status_label.setText(f"状态: 知识库初始化失败 - {e}")
        )
        self.knowledge_worker.start()

    def _on_knowledge_loaded(self, count: int):
        if count > 0:
            self.status_label.setText(f"状态: 知识库已加载 ({count} 条)")
            if hasattr(self, "prep_status_label"):
                self.prep_status_label.setText(f"知识库已加载 ({count} 条)")
        else:
            self.status_label.setText("状态: 知识库为空")
            if hasattr(self, "prep_status_label"):
                self.prep_status_label.setText("知识库为空")

    def _on_text_recognized(self, text: str):
        """识别到文本"""
        print(f"[UI] 识别结果: {text}")
        self.current_question = text
        if not self.current_translation_request_id:
            self.current_translation_request_id = uuid.uuid4().hex
            self.current_translation_request_started_at = self._now_iso()

        self.round_num += 1
        t_card = self._add_subtitle_card(self.round_num)
        t_card.history_request_id = self.current_translation_request_id
        t_card.history_request_started_at = self.current_translation_request_started_at
        t_card.set_english(text)
        if self.last_asr_ms:
            t_card.asr_ms = self.last_asr_ms
            t_card.set_metrics(f"ASR {self.last_asr_ms:.0f}ms")

        # 翻译
        self._start_translate_worker(text, t_card)

    def _track_worker(self, worker):
        self.active_workers.append(worker)
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))
        return worker

    def _cleanup_worker(self, worker):
        if worker in self.active_workers:
            self.active_workers.remove(worker)
        worker.deleteLater()

    def _start_translate_worker(self, text: str, card: RoundCard):
        worker = self._track_worker(TranslateWorker(text))
        worker.translate_completed.connect(
            lambda translated, elapsed_ms, c=card: self._on_translate_completed(translated, elapsed_ms, c)
        )
        worker.error_occurred.connect(self._on_error)
        worker.start()

    def _trigger_rag_and_llm(self, question: str = None, card: AnswerCard = None):
        """触发 RAG 检索和 LLM 生成"""
        question = question or self.current_question
        card = card or self._current_answer_card()
        if not question:
            return
        worker = self._track_worker(RagWorker(question))
        worker.search_completed.connect(
            lambda results, elapsed_ms, q=question, c=card: self._on_search_completed(results, elapsed_ms, q, c)
        )
        worker.error_occurred.connect(self._on_error)
        worker.start()

    def _on_translate_completed(self, translated: str, elapsed_ms: float, card: RoundCard):
        if card and not getattr(card, "history_saved", False):
            source_text = getattr(card, "english_text", "")
            if source_text.strip() and translated.strip() and not translated.startswith("(翻译失败"):
                try:
                    append_translation_history(
                        source_text,
                        translated,
                        getattr(card, "asr_ms", 0.0),
                        elapsed_ms,
                        getattr(card, "history_request_id", ""),
                        getattr(card, "history_request_started_at", ""),
                        getattr(card, "round_num", 0),
                    )
                    card.history_saved = True
                    self._refresh_translation_history()
                except Exception as e:
                    self._on_error(f"保存翻译历史失败: {e}")

        if card and getattr(card, "is_active", False):
            try:
                card.set_translation(translated)
                metric = f"ASR {card.asr_ms:.0f}ms" if getattr(card, "asr_ms", 0) else ""
                if elapsed_ms:
                    metric = f"{metric} | 翻译 {elapsed_ms:.0f}ms" if metric else f"翻译 {elapsed_ms:.0f}ms"
                card.set_metrics(metric)
            except RuntimeError:
                pass

    def _on_search_completed(self, results: list, elapsed_ms: float, question: str, card: AnswerCard):
        if card and not getattr(card, "is_active", False):
            return

        self.current_rag_ms = elapsed_ms
        if results:
            self.retrieved_context = "\n\n---\n\n".join(results)
        else:
            self.retrieved_context = ""
        history = self._recent_conversation_history()
        worker = self._track_worker(LLMWorker(question, self.retrieved_context, history))
        worker.token_received.connect(lambda token, c=card: self._on_token_received(token, c))
        worker.first_token_latency.connect(lambda elapsed, c=card: self._on_first_token_latency(elapsed, c))
        worker.generation_complete.connect(
            lambda elapsed, q=question, c=card: self._on_generation_complete(elapsed, q, c)
        )
        worker.error_occurred.connect(self._on_error)
        worker.start()

    def _on_token_received(self, token: str, card: AnswerCard):
        if card and getattr(card, "is_active", False):
            try:
                card.append_token(token)
            except RuntimeError:
                pass

    def _on_first_token_latency(self, elapsed_ms: float, card: AnswerCard):
        self.current_llm_ttft_ms = elapsed_ms
        if card and getattr(card, "is_active", False):
            card.set_metrics(f"RAG {self.current_rag_ms:.0f}ms | LLM 首字 {elapsed_ms:.0f}ms")

    def _on_generation_complete(self, elapsed_ms: float = 0.0, question: str = "", card: AnswerCard = None):
        if card and getattr(card, "is_active", False):
            answer = card.get_answer()
            if answer.strip() and self.current_llm_ttft_ms > 0:
                llm_metric = f"LLM 首字 {self.current_llm_ttft_ms:.0f}ms / 总计 {elapsed_ms:.0f}ms"
            elif answer.strip():
                llm_metric = f"LLM 总计 {elapsed_ms:.0f}ms"
            else:
                llm_metric = f"LLM 未返回内容 / 总计 {elapsed_ms:.0f}ms"
            card.set_metrics(f"RAG {self.current_rag_ms:.0f}ms | {llm_metric}")
            self._remember_conversation_turn(question, card.get_answer())
        self.status_label.setText("状态: 回答完成")

    def _recent_conversation_history(self):
        max_rounds = max(0, int(getattr(config, "conversation_history_rounds", 3)))
        if max_rounds == 0:
            return []
        return self.conversation_history[-max_rounds * 2:]

    def _remember_conversation_turn(self, question: str, answer: str):
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            return
        self.conversation_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        max_messages = max(0, int(getattr(config, "conversation_history_rounds", 3))) * 2
        if max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    def _on_status_changed(self, status: str):
        self.status_label.setText(f"状态: {status}")

    def _on_volume_changed(self, level: float):
        value = max(0, min(100, int(level * 700)))
        self.volume_bar.setValue(value)
        self.volume_value_label.setText(f"{value}%")

    def _on_asr_timing(self, elapsed_ms: float):
        self.last_asr_ms = elapsed_ms

    def _now_iso(self) -> str:
        from datetime import datetime
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _refresh_translation_history(self):
        if not hasattr(self, "history_list_layout"):
            return

        for widget in self.history_request_widgets:
            self.history_list_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self.history_request_widgets = []

        requests = load_translation_requests()
        total_pages = max(1, (len(requests) + 4) // 5)
        if self.history_page >= total_pages:
            self.history_page = total_pages - 1
        if self.history_page < 0:
            self.history_page = 0

        if not requests:
            self.history_empty_label.setVisible(True)
            self.history_page_label.setText("第 0 / 0 页")
            self.prev_history_page_btn.setEnabled(False)
            self.next_history_page_btn.setEnabled(False)
            return

        self.history_empty_label.setVisible(False)
        start = self.history_page * 5
        page_requests = requests[start:start + 5]

        for request_data in page_requests:
            request_id = request_data.get("request_id", "")
            round_count = len(request_data.get("rounds", []))
            visible_rounds = self.history_expanded_rounds.get(request_id, HistoryRequestCard.ROUND_STEP)
            visible_rounds = min(max(HistoryRequestCard.ROUND_STEP, visible_rounds), max(round_count, 1))
            card = HistoryRequestCard(request_data, visible_rounds)
            card.expand_requested = self._toggle_history_request
            self.history_request_widgets.append(card)
            self.history_list_layout.insertWidget(
                self.history_list_layout.count() - 1,
                card,
            )

        self.history_page_label.setText(f"第 {self.history_page + 1} / {total_pages} 页")
        self.prev_history_page_btn.setEnabled(self.history_page > 0)
        self.next_history_page_btn.setEnabled(self.history_page < total_pages - 1)
        self.history_scroll.verticalScrollBar().setValue(0)

    def _change_history_page(self, delta: int):
        self.history_page = max(0, self.history_page + delta)
        self._refresh_translation_history()

    def _toggle_history_request(self, request_data: dict):
        request_id = request_data.get("request_id", "")
        rounds = request_data.get("rounds", [])
        if not request_id or not rounds:
            return

        current = self.history_expanded_rounds.get(request_id, HistoryRequestCard.ROUND_STEP)
        if current >= len(rounds):
            self.history_expanded_rounds[request_id] = HistoryRequestCard.ROUND_STEP
        else:
            self.history_expanded_rounds[request_id] = min(
                len(rounds),
                current + HistoryRequestCard.ROUND_STEP,
            )
        self._refresh_translation_history()

    def _clear_translation_history(self):
        reply = QMessageBox.question(
            self,
            "清空翻译历史",
            "确定要清空已留存的翻译历史吗？知识库和 QA 不会受影响。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            clear_translation_history()
            self._refresh_translation_history()
            self.status_label.setText("状态: 翻译历史已清空")
        except Exception as e:
            self._on_error(f"清空翻译历史失败: {e}")

    def _append_question_text(self, text: str):
        text = text.strip()
        if not text:
            return
        existing = self.question_input.toPlainText().strip()
        joined = f"{existing} {text}" if existing else text
        self.question_input.setText(joined.strip())
        self.question_input.moveCursor(self.question_input.textCursor().End)

    def _ask_from_question_box(self):
        question = self.question_input.toPlainText().strip()
        if not question:
            QMessageBox.warning(self, "提问为空", "请先添加或输入要回答的问题。")
            return

        self.current_question = question
        self.answer_round_num += 1
        card = self._add_answer_card(self.answer_round_num)
        card.set_answer("")
        card.set_metrics("RAG 检索中...")
        self.status_label.setText("状态: 正在检索并生成回答...")
        self.current_rag_ms = 0.0
        self.current_llm_ttft_ms = 0.0
        self._trigger_rag_and_llm(question, card)

    def _on_error(self, error_msg: str):
        self.status_label.setText(f"错误: {error_msg}")
        if hasattr(self, "prep_status_label"):
            self.prep_status_label.setText(f"错误: {error_msg}")
        if self._is_screen_capture_permission_error(error_msg):
            self._show_screen_capture_permission_dialog(error_msg)

    def _on_gap_changed(self, value):
        self.gap_value_label.setText(f"{value // 10}s")

    def _on_gain_changed(self, value):
        self.gain_value_label.setText(f"{value / 100.0:.1f}x")

    def _is_screen_capture_permission_error(self, error_msg: str) -> bool:
        markers = [
            "TCC",
            "用户拒绝",
            "ScreenCaptureKit",
            "SCStreamErrorDomain",
            "屏幕",
            "系统音频",
        ]
        return any(marker in error_msg for marker in markers)

    def _show_screen_capture_permission_dialog(self, error_msg: str):
        if self.screen_capture_permission_prompted:
            return
        self.screen_capture_permission_prompted = True

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Warning)
        dialog.setWindowTitle("需要屏幕与系统音频录制权限")
        dialog.setText("无法捕获 macOS 系统音频。")
        dialog.setInformativeText(
            "请在系统设置里允许当前启动程序使用“屏幕与系统音频录制”权限，"
            "授权后需要重启本应用再开始录音。\n\n"
            f"错误信息：{error_msg}"
        )
        open_btn = dialog.addButton("打开系统设置", QMessageBox.AcceptRole)
        dialog.addButton("稍后处理", QMessageBox.RejectRole)
        dialog.exec_()

        if dialog.clickedButton() == open_btn:
            QDesktopServices.openUrl(
                QUrl("x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")
            )

    # ============================================
    # 清空
    # ============================================

    def _clear_all(self):
        for worker in list(self.active_workers):
            if hasattr(worker, "stop"):
                worker.stop()

        for card in self.translate_cards:
            card.is_active = False
            card.setParent(None)
            card.deleteLater()
        for card in self.answer_cards:
            card.is_active = False
            card.setParent(None)
            card.deleteLater()
        self.translate_cards.clear()
        self.answer_cards.clear()
        self.translate_empty_label.setVisible(True)
        self.answer_empty_label.setVisible(True)
        self.round_num = 0
        self.answer_round_num = 0
        self.conversation_history.clear()
        self.current_question = ""
        self.retrieved_context = ""
        self.question_input.clear()
        self._on_volume_changed(0.0)
        self.status_label.setText("状态: 已清空")

    # ============================================
    # 关闭
    # ============================================

    def closeEvent(self, event):
        if self.save_config_cb.isChecked():
            self._save_config()

        if self.audio_debug_worker:
            self.audio_debug_worker.stop()
            self.audio_debug_worker.quit()
            self.audio_debug_worker.wait()

        if self.audio_worker:
            self.audio_worker.stop()
            self.audio_worker.quit()
            self.audio_worker.wait()

        for worker in list(self.active_workers):
            if hasattr(worker, "stop"):
                worker.stop()
            worker.quit()
            worker.wait()

        if hasattr(self, 'knowledge_worker') and self.knowledge_worker.isRunning():
            self.knowledge_worker.quit()
            self.knowledge_worker.wait()

        event.accept()
