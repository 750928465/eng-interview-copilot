"""
配置管理模块
支持从 settings.json 加载和保存配置
"""
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from app_paths import ensure_user_data_files, resource_root, user_data_path


SETTINGS_FILE = "settings.json"
ensure_user_data_files()


@dataclass
class Config:
    """应用配置"""

    # LLM 配置
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o"
    system_prompt: str = (
        "You are helping the candidate answer in first person during an English interview. "
        "Adopt the candidate's identity, background, research proposal, and projects from the retrieved context."
    )
    conversation_history_rounds: int = 3

    # RAG 配置
    knowledge_file: str = "knowledge.md"
    qa_file: str = "qa.md"
    collection_name: str = "interview_knowledge"
    top_k: int = 5
    candidate_top_k: int = 12
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ASR 配置
    asr_provider: str = "faster_whisper"
    asr_model_size: str = "small"
    asr_hotwords_file: str = "terms.txt"
    audio_source: str = "mac_system"  # mac_system / microphone
    audio_device_index: Optional[int] = None
    audio_device_name: str = ""
    audio_gain: float = 1.0
    mac_system_audio_sample_rate: int = 16000
    mac_system_audio_channels: int = 1
    silence_gap: int = 50  # 静音间隔帧数（1帧=100ms，50=5秒）
    mode: str = "auto"     # auto / manual

    # 翻译配置
    translation_provider: str = "local_opus"
    translation_model_name: str = "Helsinki-NLP/opus-mt-en-zh"
    translation_local_files_only: bool = False
    translation_history_file: str = "translation_history.jsonl"

    # 项目根目录
    project_root: str = field(default_factory=lambda: str(resource_root()))

    @property
    def knowledge_path(self) -> str:
        return str(user_data_path(self.knowledge_file))

    @property
    def qa_path(self) -> str:
        return str(user_data_path(self.qa_file))

    @property
    def asr_hotwords_path(self) -> str:
        return str(user_data_path(self.asr_hotwords_file))

    @property
    def translation_history_path(self) -> str:
        return str(user_data_path(self.translation_history_file))

    @property
    def chroma_persist_dir(self) -> str:
        return str(user_data_path("chroma_db"))

    @property
    def settings_path(self) -> str:
        return str(user_data_path(SETTINGS_FILE))

    def load(self) -> bool:
        """从文件加载配置"""
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 只更新有效字段
                for key, value in data.items():
                    if hasattr(self, key) and key != "project_root":
                        setattr(self, key, value)
                print(f"[Config] 已加载配置: {self.settings_path}")
                return True
            except Exception as e:
                print(f"[Config] 加载失败: {e}")
        return False

    def save(self) -> bool:
        """保存配置到文件"""
        try:
            data = asdict(self)
            # 不保存 project_root
            data.pop("project_root", None)
            os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[Config] 已保存配置: {self.settings_path}")
            return True
        except Exception as e:
            print(f"[Config] 保存失败: {e}")
            return False


# 全局配置实例
config = Config()
config.load()  # 启动时自动加载
