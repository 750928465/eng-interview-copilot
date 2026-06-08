# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[("mac/build/SystemAudioCapture", "mac/build")],
    datas=[
        ("knowledge.md.template", "."),
        ("knowledge.md.example", "."),
        ("qa.md.template", "."),
    ],
    hiddenimports=collect_submodules("chromadb") + [
        "faster_whisper",
        "sentence_transformers",
        "sounddevice",
        "whisper",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="English Interview Copilot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="English Interview Copilot",
)
app = BUNDLE(
    coll,
    name="English Interview Copilot.app",
    icon=None,
    bundle_identifier="com.enginterview.copilot",
    info_plist={
        "NSMicrophoneUsageDescription": "English Interview Copilot records interview audio for local transcription.",
        "NSScreenCaptureUsageDescription": "English Interview Copilot captures system audio for interview transcription on macOS.",
    },
)
