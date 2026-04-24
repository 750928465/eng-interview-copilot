"""
配置管理模块
支持从 settings.json 加载和保存配置
"""
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


SETTINGS_FILE = "settings.json"


@dataclass
class Config:
    """应用配置"""

    # LLM 配置
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o"

    # RAG 配置
    knowledge_file: str = "knowledge.md"
    collection_name: str = "interview_knowledge"
    top_k: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ASR 配置
    asr_provider: str = "sounddevice"

    # 项目根目录
    project_root: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))

    @property
    def knowledge_path(self) -> str:
        return os.path.join(self.project_root, self.knowledge_file)

    @property
    def chroma_persist_dir(self) -> str:
        return os.path.join(self.project_root, "chroma_db")

    @property
    def settings_path(self) -> str:
        return os.path.join(self.project_root, SETTINGS_FILE)

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