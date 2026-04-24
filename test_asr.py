"""
独立测试 ASR 功能
直接运行此脚本测试麦克风和语音识别
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asr.recognizer import create_asr

def on_text(text):
    print(f"\n{'='*50}")
    print(f"识别结果: {text}")
    print(f"{'='*50}\n")

def main():
    print("="*50)
    print("ASR 独立测试")
    print("="*50)
    print("请说话，说完后停顿 2 秒")
    print("按 Ctrl+C 退出")
    print("="*50)

    asr = create_asr("sounddevice")

    try:
        asr.listen_microphone(on_text)
    except KeyboardInterrupt:
        print("\n已停止")
        asr.stop_listening()

if __name__ == "__main__":
    main()