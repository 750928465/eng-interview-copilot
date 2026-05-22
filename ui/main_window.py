"""
主界面模块
PyQt5 实现 - 双模式 + 双窗口对话列表
配色：黑橙红
"""
import os
import re
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QLineEdit,
    QGroupBox, QFormLayout, QMessageBox, QCheckBox,
    QSlider, QScrollArea, QFrame, QTabWidget, QComboBox,
    QProgressBar
)
from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from llm.engine import update_llm_config
from workers.audio_worker import AudioWorker
from workers.rag_worker import RagWorker
from workers.llm_worker import LLMWorker
from workers.translate_worker import TranslateWorker
from workers.knowledge_worker import KnowledgeWorker

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


class MainWindow(QMainWindow):
    """主窗口"""

    MAX_ROUNDS = 5

    def __init__(self):
        super().__init__()

        self.audio_worker: AudioWorker = None
        self.knowledge_worker = None
        self.active_workers = []

        self.is_recording = False
        self.config_dirty = False
        self.current_mode = config.mode  # auto / manual
        self.current_question = ""
        self.retrieved_context = ""
        self.last_asr_ms = 0.0
        self.current_rag_ms = 0.0
        self.current_llm_ttft_ms = 0.0
        self.screen_capture_permission_prompted = False
        self.conversation_history = []

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
        prep_tab = QWidget()
        self.tabs.addTab(conversation_tab, "对话输出")
        self.tabs.addTab(prep_tab, "面试准备")

        main_layout = QVBoxLayout(conversation_tab)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)

        prep_layout = QVBoxLayout(prep_tab)
        prep_layout.setSpacing(8)
        prep_layout.setContentsMargins(10, 10, 10, 10)

        # ====== 1. 配置区域（折叠式） ======
        config_group = QGroupBox("LLM 配置")
        config_layout = QFormLayout()
        config_layout.setSpacing(6)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("API Key")
        self.api_key_input.setText(config.api_key)

        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText("API Base URL")
        self.base_url_input.setText(config.base_url)

        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Model")
        self.model_input.setText(config.model_name)

        self.audio_source_combo = QComboBox()
        self.audio_source_combo.setMinimumHeight(30)
        self.audio_source_combo.addItem("macOS 系统音频", "mac_system")
        self.audio_source_combo.addItem("麦克风 / 虚拟声卡", "microphone")
        source_index = self.audio_source_combo.findData(config.audio_source)
        self.audio_source_combo.setCurrentIndex(max(0, source_index))

        self.audio_device_combo = QComboBox()
        self.audio_device_combo.setMinimumHeight(30)
        self._load_audio_devices()

        self.audio_device_row = QWidget()
        self.audio_device_row.setStyleSheet("background-color: transparent;")
        audio_device_row_layout = QHBoxLayout(self.audio_device_row)
        audio_device_row_layout.setContentsMargins(0, 0, 0, 0)
        audio_device_row_layout.addWidget(self.audio_device_combo)

        self.system_prompt_input = QTextEdit()
        self.system_prompt_input.setFont(QFont("Arial", 11))
        self.system_prompt_input.setMaximumHeight(130)
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
        config_layout.addRow("系统提示词:", self.system_prompt_input)
        config_layout.addRow("", self.save_config_cb)
        config_group.setLayout(config_layout)
        self.config_group = config_group
        prep_layout.addWidget(config_group)

        # 折叠配置区域的按钮
        self.toggle_config_btn = QPushButton("▲ 收起配置")
        self.toggle_config_btn.setMinimumHeight(28)
        self.toggle_config_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a; color: #888;
                border-radius: 4px; border: 1px solid #444;
                font-size: 12px;
            }
            QPushButton:hover { color: #e0e0e0; border-color: #ff6b35; }
        """)
        self.toggle_config_btn.clicked.connect(self._toggle_config)
        self.config_collapsed = False
        prep_layout.addWidget(self.toggle_config_btn)

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
        prep_layout.addWidget(mode_group)

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
        prep_layout.addWidget(self.gap_widget)

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
        prep_layout.addWidget(self.gain_widget)

        # ====== 5. 准备页：QA 对编辑 ======
        qa_group = QGroupBox("QA 知识库")
        qa_layout = QVBoxLayout()
        qa_layout.setSpacing(8)

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
        self.knowledge_editor.setPlaceholderText("在这里维护 knowledge.md，例如简历、项目、研究经历、动机和可检索背景材料。")
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
        prep_layout.addWidget(qa_group, 1)

        self.apply_config_btn = QPushButton("应用更新")
        self.apply_config_btn.setMinimumHeight(40)
        prep_layout.addWidget(self.apply_config_btn)

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
        self.gap_slider.valueChanged.connect(self._on_gap_changed)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)
        self.apply_config_btn.clicked.connect(self._apply_preparation_updates)
        self.ask_llm_btn.clicked.connect(self._ask_from_question_box)
        self.clear_question_btn.clicked.connect(self.question_input.clear)
        self._connect_dirty_signals()
        self._set_config_dirty(False)

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
    # 配置折叠
    # ============================================

    def _toggle_config(self):
        self.config_collapsed = not self.config_collapsed
        self.config_group.setVisible(not self.config_collapsed)
        if self.config_collapsed:
            self.toggle_config_btn.setText("▼ 展开配置")
        else:
            self.toggle_config_btn.setText("▲ 收起配置")

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
        self.apply_config_btn.setText(text)
        self.apply_config_btn.setStyleSheet(self._button_style(dirty))
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
            QMessageBox.warning(self, "有未应用的更新", "请先在面试准备页点击“应用更新”。")
            self.tabs.setCurrentIndex(1)
            return

        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "配置错误", "请输入 API Key")
            return

        self._save_config()
        update_llm_config(
            api_key,
            self.base_url_input.text().strip(),
            self.model_input.text().strip(),
            self.system_prompt_input.toPlainText().strip()
        )

        self.is_recording = True
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

        self.round_num += 1
        t_card = self._add_subtitle_card(self.round_num)
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
