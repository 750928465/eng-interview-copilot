"""
ASR 抽象接口
"""
from abc import ABC, abstractmethod
from typing import Optional, Callable


class BaseASR(ABC):
    """ASR 抽象基类，便于替换不同的语音识别服务"""

    @abstractmethod
    def recognize(self, audio_data: bytes) -> Optional[str]:
        """
        识别音频数据并返回文本

        Args:
            audio_data: 音频数据字节

        Returns:
            识别出的文本，失败返回 None
        """
        pass

    @abstractmethod
    def listen_microphone(self, callback: Callable[[str], None]) -> None:
        """
        监听麦克风并进行实时识别

        Args:
            callback: 识别结果回调函数
        """
        pass

    @abstractmethod
    def stop_listening(self) -> None:
        """停止监听"""
        pass