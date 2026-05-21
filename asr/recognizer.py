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
import time
from typing import Optional, Callable
from .base import BaseASR

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from asr.audio_sources import MacSystemAudioSource


def log(msg):
    print(msg, flush=True)


def find_working_microphone(preferred_index: int = None, preferred_name: str = "") -> tuple:
    devices = sd.query_devices()
    log("[ASR] 搜索可用麦克风...")

    if preferred_index is not None:
        try:
            dev = devices[int(preferred_index)]
            if dev["max_input_channels"] > 0:
                log(f"[ASR] 使用配置设备: {dev['name']}")
                return int(preferred_index), int(dev["default_samplerate"])
        except Exception as e:
            log(f"[ASR] 配置设备不可用，回退自动选择: {e}")

    if preferred_name:
        preferred = preferred_name.lower()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0 and preferred in dev["name"].lower():
                log(f"[ASR] 使用匹配设备: {dev['name']}")
                return i, int(dev["default_samplerate"])

    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            name = dev['name'].lower()
            if 'blackhole' in name or 'loopback' in name:
                log(f"[ASR] 选中系统音频设备: {dev['name']}")
                return i, int(dev['default_samplerate'])

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

    def __init__(
        self,
        model_size: str = "base",
        language: str = "en",
        silence_gap: int = 50,
        audio_device_index: int = None,
        audio_device_name: str = "",
        audio_gain: float = 1.0,
        audio_source: str = "microphone",
        volume_callback: Callable[[float], None] = None,
        timing_callback: Callable[[float], None] = None,
        error_callback: Callable[[str], None] = None
    ):
        self.model_size = model_size
        self.language = language
        self.is_listening = False
        self.target_rate = 16000
        self.model = None
        self.is_recognizing = False
        self.silence_gap = silence_gap
        self.audio_gain = max(0.1, float(audio_gain or 1.0))
        self.audio_source = audio_source
        self._external_audio_source = None
        self.manual_mode = False
        self._stop_requested = False
        self.volume_callback = volume_callback
        self.timing_callback = timing_callback
        self.error_callback = error_callback

        if self.audio_source == "mac_system":
            self.device_index = None
            self.sample_rate = config.mac_system_audio_sample_rate
        else:
            self.device_index, self.sample_rate = find_working_microphone(
                preferred_index=audio_device_index,
                preferred_name=audio_device_name
            )
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
            started = time.perf_counter()

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
            self._emit_timing(started)

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
        blocksize = 1024
        block_seconds = blocksize / self.sample_rate

        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time, status):
            if self.is_listening:
                data = self._apply_audio_gain(indata)
                self._emit_volume(data)
                audio_queue.put(data)

        def recognition_loop():
            buffer = []
            frame_count = 0
            voice_start = None
            last_voice = 0
            max_volume = 0

            min_voice_blocks = max(5, int(0.5 / block_seconds))
            silence_gap_blocks = max(3, int((self.silence_gap / 10) / block_seconds))
            max_chunk_blocks = max(min_voice_blocks, int(2.8 / block_seconds))
            voice_threshold = 0.006

            log(
                f"[ASR] 就绪，请说话（停顿{self.silence_gap / 10:.1f}秒或连续"
                "2.8秒会触发识别）..."
            )

            while self.is_listening:
                try:
                    data = audio_queue.get(timeout=0.3)
                    buffer.append(data)
                    frame_count += 1

                    if self.is_recognizing:
                        continue

                    volume = np.abs(data).mean()
                    max_volume = max(max_volume, volume)
                    is_speaking = volume > voice_threshold

                    if is_speaking:
                        if voice_start is None:
                            voice_start = frame_count - 1
                            log(f"[ASR] >>> 检测到语音 (音量: {volume:.4f})")
                        last_voice = frame_count - 1

                    if voice_start is not None:
                        voice_len = last_voice - voice_start + 1
                        hit_silence = (frame_count - last_voice) >= silence_gap_blocks
                        hit_max_chunk = voice_len >= max_chunk_blocks
                        if (hit_silence or hit_max_chunk) and voice_len >= min_voice_blocks:
                            self.is_recognizing = True
                            reason = "静音" if hit_silence else "短分段"
                            log(
                                f"[ASR] >>> 识别中 ({reason}, 语音帧: {voice_len}, "
                                f"最大音量: {max_volume:.4f})..."
                            )

                            end_frame = last_voice + 3 if hit_silence else frame_count
                            end_frame = min(end_frame, len(buffer))
                            audio_arr = np.concatenate(buffer[voice_start:end_frame]).flatten()
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
                blocksize=blocksize
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
        self._stop_requested = True
        if self._external_audio_source:
            self._external_audio_source.stop()

    def record_until_stop(self, callback: Callable[[str], None]) -> None:
        """手动模式：持续录音直到调用 stop_listening，自动分段识别"""
        self.is_listening = True
        self._stop_requested = False
        self.manual_mode = True

        device_info = sd.query_devices(self.device_index)
        self.sample_rate = int(device_info['default_samplerate'])
        log(f"[ASR] 手动模式监听: {device_info['name']}")
        blocksize = 1024
        block_seconds = blocksize / self.sample_rate

        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time, status):
            if self.is_listening:
                data = self._apply_audio_gain(indata)
                self._emit_volume(data)
                audio_queue.put(data)

        def recognition_loop():
            buffer = []
            frame_count = 0
            voice_start = None
            last_voice = 0
            max_volume = 0
            min_voice_blocks = max(5, int(0.4 / block_seconds))
            silence_gap_blocks = max(3, int((self.silence_gap / 10) / block_seconds))
            max_chunk_blocks = max(min_voice_blocks, int(2.8 / block_seconds))
            voice_threshold = 0.006

            log("[ASR] 手动模式就绪，请说话（连续2.8秒会自动分段）...")

            while self.is_listening or not audio_queue.empty():
                try:
                    data = audio_queue.get(timeout=0.3)
                    buffer.append(data)
                    frame_count += 1

                    if self.is_recognizing:
                        continue

                    volume = np.abs(data).mean()
                    max_volume = max(max_volume, volume)
                    is_speaking = volume > voice_threshold

                    if is_speaking:
                        if voice_start is None:
                            voice_start = frame_count - 1
                            log(f"[ASR] >>> 检测到语音 (音量: {volume:.4f})")
                        last_voice = frame_count - 1

                    if voice_start is not None:
                        voice_len = last_voice - voice_start + 1
                        hit_silence = (frame_count - last_voice) >= silence_gap_blocks
                        hit_max_chunk = voice_len >= max_chunk_blocks
                        if (hit_silence or hit_max_chunk) and voice_len >= min_voice_blocks:
                            self.is_recognizing = True
                            reason = "静音" if hit_silence else "短分段"
                            log(f"[ASR] >>> 分段识别 ({reason}, 语音帧: {voice_len})...")

                            end_frame = last_voice + 3 if hit_silence else frame_count
                            end_frame = min(end_frame, len(buffer))
                            audio_arr = np.concatenate(buffer[voice_start:end_frame]).flatten()
                            audio_16 = np.clip(audio_arr * 32767, -32768, 32767).astype(np.int16)

                            text = self.recognize(audio_16, self.sample_rate)
                            self.is_recognizing = False

                            if text:
                                log(f"[ASR] ✓ 手动模式结果: {text}")
                                callback(text)

                            buffer = []
                            frame_count = 0
                            voice_start = None
                            last_voice = 0
                            max_volume = 0

                except queue.Empty:
                    continue
                except Exception as e:
                    self.is_recognizing = False
                    log(f"[ASR] 手动模式错误: {e}")

            # 停止时处理尚未达到静音阈值的最后一段语音。
            if buffer and voice_start is not None:
                log("[ASR] 处理剩余音频...")
                end = last_voice + 3 if last_voice >= voice_start else len(buffer)
                audio_arr = np.concatenate(buffer[voice_start:end]).flatten()
                audio_16 = np.clip(audio_arr * 32767, -32768, 32767).astype(np.int16)
                text = self.recognize(audio_16, self.sample_rate)
                if text:
                    callback(text)

            log("[ASR] 手动模式结束")

        threading.Thread(target=recognition_loop, daemon=True).start()

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                device=self.device_index,
                callback=audio_callback,
                blocksize=blocksize
            ):
                log("[ASR] >>> 手动录音中...")
                while self.is_listening:
                    sd.sleep(100)
        except Exception as e:
            log(f"[ASR] 录音错误: {e}")
            self.is_listening = False

    def listen_system_audio(self, callback: Callable[[str], None]) -> None:
        """macOS 系统音频：从 ScreenCaptureKit helper 持续读取 PCM 后分段识别。"""
        self.is_listening = True
        self.sample_rate = config.mac_system_audio_sample_rate
        blocksize = 1024
        block_seconds = blocksize / self.sample_rate
        audio_queue = queue.Queue()
        source = MacSystemAudioSource(sample_rate=self.sample_rate, blocksize=blocksize)
        self._external_audio_source = source

        def capture_loop():
            try:
                source.start()
                log("[ASR] >>> 正在捕获 macOS 系统音频...")
                while self.is_listening:
                    data = source.read_chunk()
                    if data.size == 0:
                        continue
                    data = self._apply_audio_gain(data)
                    self._emit_volume(data)
                    audio_queue.put(data)
            except Exception as e:
                message = f"系统音频捕获错误: {e}"
                log(f"[ASR] {message}")
                self._emit_error(message)
                self.is_listening = False
            finally:
                source.stop()

        def recognition_loop():
            buffer = []
            frame_count = 0
            voice_start = None
            last_voice = 0
            max_volume = 0
            min_voice_blocks = max(5, int(0.5 / block_seconds))
            segment_seconds = max(1.0, self.silence_gap / 10)
            silence_gap_blocks = max(3, int(segment_seconds / block_seconds))
            max_chunk_seconds = max(3.0, segment_seconds * 2)
            max_chunk_blocks = max(min_voice_blocks, int(max_chunk_seconds / block_seconds))
            voice_threshold = 0.004

            log(
                f"[ASR] 系统音频就绪（停顿{segment_seconds:.1f}秒或连续"
                f"{max_chunk_seconds:.1f}秒会触发识别）..."
            )

            while self.is_listening or not audio_queue.empty():
                try:
                    data = audio_queue.get(timeout=0.3)
                    buffer.append(data)
                    frame_count += 1

                    if self.is_recognizing:
                        continue

                    volume = np.abs(data).mean()
                    max_volume = max(max_volume, volume)
                    is_speaking = volume > voice_threshold

                    if is_speaking:
                        if voice_start is None:
                            voice_start = frame_count - 1
                            log(f"[ASR] >>> 系统音频检测到语音 (音量: {volume:.4f})")
                        last_voice = frame_count - 1

                    if voice_start is not None:
                        voice_len = last_voice - voice_start + 1
                        hit_silence = (frame_count - last_voice) >= silence_gap_blocks
                        hit_max_chunk = voice_len >= max_chunk_blocks
                        if (hit_silence or hit_max_chunk) and voice_len >= min_voice_blocks:
                            self.is_recognizing = True
                            reason = "静音" if hit_silence else "短分段"
                            log(
                                f"[ASR] >>> 系统音频识别中 ({reason}, 语音帧: {voice_len}, "
                                f"最大音量: {max_volume:.4f})..."
                            )

                            end_frame = last_voice + 3 if hit_silence else frame_count
                            end_frame = min(end_frame, len(buffer))
                            audio_arr = np.concatenate(buffer[voice_start:end_frame]).flatten()
                            audio_16 = np.clip(audio_arr * 32767, -32768, 32767).astype(np.int16)

                            text = self.recognize(audio_16, self.sample_rate)
                            self.is_recognizing = False

                            if text:
                                log(f"[ASR] ✓ 系统音频结果: {text}")
                                callback(text)

                            buffer = []
                            frame_count = 0
                            voice_start = None
                            last_voice = 0
                            max_volume = 0

                except queue.Empty:
                    continue
                except Exception as e:
                    self.is_recognizing = False
                    log(f"[ASR] 系统音频识别错误: {e}")

            log("[ASR] 系统音频监听结束")

        threading.Thread(target=capture_loop, daemon=True).start()
        recognition_loop()

    def _emit_volume(self, data: np.ndarray):
        if not self.volume_callback:
            return
        try:
            audio = data.astype(np.float64, copy=False)
            rms = float(np.sqrt(np.mean(audio * audio)))
            self.volume_callback(rms)
        except Exception:
            pass

    def _apply_audio_gain(self, data: np.ndarray) -> np.ndarray:
        if self.audio_gain == 1.0:
            return data.copy()
        return np.clip(data * self.audio_gain, -1.0, 1.0).astype(np.float32, copy=False)

    def _emit_timing(self, started: float):
        if not self.timing_callback:
            return
        try:
            self.timing_callback((time.perf_counter() - started) * 1000)
        except Exception:
            pass

    def _emit_error(self, message: str):
        if not self.error_callback:
            return
        try:
            self.error_callback(message)
        except Exception:
            pass

    def _load_hotwords(self) -> str:
        try:
            if not os.path.exists(config.asr_hotwords_path):
                return ""
            with open(config.asr_hotwords_path, "r", encoding="utf-8") as f:
                terms = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
            return ", ".join(terms[:80])
        except Exception as e:
            log(f"[ASR] 领域词加载失败: {e}")
            return ""


class FasterWhisperASR(LocalWhisperASR):
    """faster-whisper 本地模型，优先用于低延迟转写。"""

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
            log(f"[ASR] 加载 faster-whisper {self.model_size}...")
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            log("[ASR] faster-whisper 模型就绪")
        except ImportError:
            log("[ASR] 未安装 faster-whisper，回退 openai-whisper")
            super()._load_model()
            self._fallback_openai_whisper = True
        except Exception as e:
            log(f"[ASR] faster-whisper 初始化失败，回退 openai-whisper: {e}")
            super()._load_model()
            self._fallback_openai_whisper = True

    def recognize(self, audio_data: np.ndarray, sample_rate: int = None) -> Optional[str]:
        if getattr(self, "_fallback_openai_whisper", False):
            return super().recognize(audio_data, sample_rate)

        sr_rate = sample_rate or self.target_rate
        started = time.perf_counter()
        try:
            audio = audio_data.astype(np.float32) / 32768.0
            if sr_rate != self.target_rate:
                audio = self._resample_audio(audio, sr_rate, self.target_rate)

            hotwords = self._load_hotwords()
            kwargs = {
                "language": self.language,
                "beam_size": 1,
                "vad_filter": True,
                "condition_on_previous_text": False,
            }
            if hotwords:
                kwargs["initial_prompt"] = (
                    "Common interview and domain terms: " + hotwords
                )
                kwargs["hotwords"] = hotwords

            log("[ASR] faster-whisper 识别中...")
            segments, _ = self.model.transcribe(audio, **kwargs)
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
            self._emit_timing(started)
            log(f"[ASR] faster-whisper 原始结果: '{text}'")
            return text or None
        except TypeError:
            try:
                kwargs.pop("hotwords", None)
                segments, _ = self.model.transcribe(audio, **kwargs)
                text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
                self._emit_timing(started)
                return text or None
            except Exception as e:
                log(f"[ASR] faster-whisper 识别错误: {e}")
                return None
        except Exception as e:
            log(f"[ASR] faster-whisper 识别错误: {e}")
            return None

    def _resample_audio(self, audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate or len(audio) == 0:
            return audio
        duration = len(audio) / float(source_rate)
        target_length = max(1, int(duration * target_rate))
        source_positions = np.linspace(0, len(audio) - 1, num=len(audio))
        target_positions = np.linspace(0, len(audio) - 1, num=target_length)
        return np.interp(target_positions, source_positions, audio).astype(np.float32)


SounddeviceASR = LocalWhisperASR


def create_asr(provider: str = "whisper_local", **kwargs) -> BaseASR:
    if provider in {"faster_whisper", "faster-whisper", "sounddevice", "whisper_local"}:
        return FasterWhisperASR(**kwargs)
    return LocalWhisperASR(**kwargs)
