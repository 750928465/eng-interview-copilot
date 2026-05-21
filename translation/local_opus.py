"""
Lightweight local English-to-Chinese translation with OPUS-MT.
"""
import os
import re
import sys
import threading
from functools import lru_cache
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


_MODEL_LOCK = threading.Lock()


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences or [text.strip()]


class LocalOpusTranslator:
    """Lazy-loaded OPUS-MT translator for rough offline subtitles."""

    def __init__(
        self,
        model_name: str = None,
        local_files_only: bool = None,
    ):
        self.model_name = model_name or config.translation_model_name
        self.local_files_only = (
            config.translation_local_files_only
            if local_files_only is None
            else local_files_only
        )
        self._tokenizer = None
        self._model = None

    def translate(self, text: str) -> str:
        sentences = split_sentences(text)
        if not sentences:
            return ""

        self._ensure_loaded()
        encoded = self._tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        generated = self._model.generate(
            **encoded,
            max_new_tokens=160,
            num_beams=1,
            do_sample=False,
        )
        translated = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        return "\n".join(part.strip() for part in translated if part.strip())

    def _ensure_loaded(self):
        if self._tokenizer is not None and self._model is not None:
            return

        with _MODEL_LOCK:
            if self._tokenizer is not None and self._model is not None:
                return

            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "本地翻译依赖缺失，请安装 transformers、sentencepiece、sacremoses"
                ) from exc

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=self.local_files_only,
                )
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    local_files_only=self.local_files_only,
                )
            except OSError as exc:
                raise RuntimeError(
                    f"本地翻译模型不可用: {self.model_name}。请先联网运行一次以下载模型，"
                    "或把模型文件放入 Hugging Face 本地缓存。"
                ) from exc
            self._model.eval()


@lru_cache(maxsize=2)
def get_local_opus_translator(
    model_name: str = None,
    local_files_only: bool = None,
) -> LocalOpusTranslator:
    return LocalOpusTranslator(model_name, local_files_only)
