"""
语音识别实现
使用本地 Whisper 模型（离线运行）
"""
import queue
import threading
import numpy as np
import sounddevice as sd
import wave
import tempfile
import os
from typing import Optional, Callable
from .base import BaseASR


def log(msg):
    print(msg, flush=True)


def find_working_microphone() -> tuple:
    devices = sd.query_devices()
    log("[ASR] 搜索可用麦克风...")

    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            name = dev['name'].lower()
            if 'macbook' in name or '内置' in name or 'built-in' in name:
                log(f"[ASR] 选中: {dev['name']}")
                return i, int(dev['default_samplerate'])

    default = sd.query_devices(kind='input')
    return default['index'], int(default['default_samplerate'])


class LocalWhisperASR(BaseASR):
    """本地 Whisper 模型"""

    def __init__(self, model_size: str = "base", language: str = "en"):
        self.model_size = model_size
        self.language = language
        self.is_listening = False
        self.target_rate = 16000
        self.model = None
        self.is_recognizing = False

        self.device_index, self.sample_rate = find_working_microphone()
        log(f"[ASR] 初始化: 采样率={self.sample_rate}")

        self._load_model()

    def _load_model(self):
        try:
            import whisper
            log(f"[ASR] 加载 Whisper {self.model_size}...")
            self.model = whisper.load_model(self.model_size)
            log("[ASR] 模型就绪")
        except ImportError:
            log("[ASR] 请安装: pip install openai-whisper")
            raise

    def recognize(self, audio_data: np.ndarray, sample_rate: int = None) -> Optional[str]:
        sr_rate = sample_rate or self.target_rate

        try:
            log(f"[ASR] 音频长度: {len(audio_data)} 样本, 采样率: {sr_rate}")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                with wave.open(f, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr_rate)
                    audio_int16 = audio_data.astype(np.int16)
                    wf.writeframes(audio_int16.tobytes())

            log("[ASR] Whisper 识别中...")
            result = self.model.transcribe(temp_path, language=self.language)
            os.unlink(temp_path)

            text = result.get("text", "").strip()
            log(f"[ASR] Whisper 原始结果: '{text}'")
            return text or None

        except Exception as e:
            log(f"[ASR] 识别错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def listen_microphone(self, callback: Callable[[str], None]) -> None:
        self.is_listening = True

        device_info = sd.query_devices(self.device_index)
        self.sample_rate = int(device_info['default_samplerate'])
        log(f"[ASR] 监听: {device_info['name']}")

        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time, status):
            if self.is_listening:
                audio_queue.put(indata.copy())

        def recognition_loop():
            buffer = []
            frame_count = 0
            voice_start = None
            last_voice = 0
            max_volume = 0

            MIN_VOICE = 25   # 最少说话 2.5 秒
            SILENCE_GAP = 50  # 静音 5 秒后识别

            log("[ASR] 就绪，请说话（说完停顿5秒）...")

            while self.is_listening:
                try:
                    data = audio_queue.get(timeout=0.3)
                    buffer.append(data)
                    frame_count += 1

                    if self.is_recognizing:
                        continue

                    volume = np.abs(data).mean()
                    max_volume = max(max_volume, volume)
                    is_speaking = volume > 0.02  # 提高阈值

                    if is_speaking:
                        if voice_start is None:
                            voice_start = frame_count
                            log(f"[ASR] >>> 检测到语音 (音量: {volume:.4f})")
                        last_voice = frame_count

                    if voice_start and (frame_count - last_voice) >= SILENCE_GAP:
                        voice_len = last_voice - voice_start
                        if voice_len >= MIN_VOICE:
                            self.is_recognizing = True
                            log(f"[ASR] >>> 识别中 (语音帧: {voice_len}, 最大音量: {max_volume:.4f})...")

                            audio_arr = np.concatenate(buffer[voice_start:last_voice+3]).flatten()
                            audio_16 = np.clip(audio_arr * 32767, -32768, 32767).astype(np.int16)

                            text = self.recognize(audio_16, self.sample_rate)
                            self.is_recognizing = False

                            if text:
                                log(f"[ASR] ✓✓✓ 结果: {text}")
                                callback(text)
                            else:
                                log("[ASR] 未识别到内容，请再试一次")

                            buffer = []
                            frame_count = 0
                            voice_start = None
                            last_voice = 0
                            max_volume = 0

                except queue.Empty:
                    continue
                except Exception as e:
                    self.is_recognizing = False
                    log(f"[ASR] 错误: {e}")

        threading.Thread(target=recognition_loop, daemon=True).start()

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                device=self.device_index,
                callback=audio_callback,
                blocksize=1024
            ):
                log("[ASR] >>> 录音中...")
                while self.is_listening:
                    sd.sleep(100)
        except Exception as e:
            log(f"[ASR] 录音错误: {e}")
            self.is_listening = False

    def stop_listening(self):
        log("[ASR] 停止")
        self.is_listening = False


SounddeviceASR = LocalWhisperASR


def create_asr(provider: str = "whisper_local", **kwargs) -> BaseASR:
    return LocalWhisperASR(**kwargs)