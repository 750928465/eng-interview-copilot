# macOS 快速启动说明

macOS 可以用 PyInstaller 打包为可双击启动的 `.app`。当前目标是本机快速体验；正式分发前还需要补充签名、公证和 DMG。

## 直接体验

本机构建完成后，双击下面的应用即可启动：

```text
dist/English Interview Copilot.app
```

也可以在终端打开：

```bash
open "dist/English Interview Copilot.app"
```

## 用户数据目录

第一次运行会在下面的目录创建用户数据文件，避免把配置和知识库写入 app bundle：

```text
~/Library/Application Support/English Interview Copilot/
```

该目录保存：

```text
settings.json
knowledge.md
qa.md
terms.txt
translation_history.jsonl
chroma_db/
```

源码目录中已有的同名文件会在首次启动时复制过去；新安装环境会从 `knowledge.md.template` 初始化知识库。

## 本机重新构建

推荐使用项目内虚拟环境作为打包工具箱，最终用户不需要安装 Python、pip 或虚拟环境。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip pyinstaller
.venv/bin/python -m pip install -r requirements.txt
PYTHON_BIN=.venv/bin/python ./build_macos.sh
```

构建产物位于：

```text
dist/English Interview Copilot.app
```

`build_macos.sh` 会先编译 `mac/SystemAudioCapture.swift`，再把生成的 `mac/build/SystemAudioCapture` 放进 app。

## 系统音频权限

如果使用 macOS 系统音频采集，首次启动可能需要授权。请到：

```text
系统设置 -> 隐私与安全性 -> 屏幕与系统音频录制
```

为 English Interview Copilot 授权。也可以先切换到麦克风输入体验核心流程。
