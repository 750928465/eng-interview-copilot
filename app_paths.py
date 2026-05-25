"""
Application path helpers for source and PyInstaller macOS builds.
"""
import shutil
import sys
from pathlib import Path


APP_NAME = "English Interview Copilot"
SOURCE_ROOT = Path(__file__).resolve().parent
USER_DATA_FILES = (
    "knowledge.md",
    "qa.md",
    "terms.txt",
    "translation_history.jsonl",
)


def is_packaged() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    if is_packaged():
        return Path(getattr(sys, "_MEIPASS", SOURCE_ROOT))
    return SOURCE_ROOT


def resource_path(*parts: str) -> Path:
    return resource_root().joinpath(*parts)


def user_data_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return SOURCE_ROOT


def user_data_path(*parts: str) -> Path:
    return user_data_dir().joinpath(*parts)


def ensure_user_data_files() -> None:
    data_dir = user_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    settings_source = SOURCE_ROOT / "settings.json"
    settings_target = data_dir / "settings.json"
    if settings_source.exists() and not settings_target.exists():
        shutil.copy2(settings_source, settings_target)

    for filename in USER_DATA_FILES:
        target = data_dir / filename
        if target.exists():
            continue

        source = SOURCE_ROOT / filename
        if source.exists():
            shutil.copy2(source, target)
            continue

        if filename == "knowledge.md":
            template = resource_path("knowledge.md.template")
            if template.exists():
                shutil.copy2(template, target)
                continue

        target.touch()

    source_chroma = SOURCE_ROOT / "chroma_db"
    target_chroma = data_dir / "chroma_db"
    if source_chroma.exists() and not target_chroma.exists():
        shutil.copytree(source_chroma, target_chroma)
