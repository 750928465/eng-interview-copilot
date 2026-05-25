# English Interview Copilot

英语面试实时辅助智能体 - 桌面应用

## 项目概述

本项目是一个用于英语面试实时辅助的 Copilot 智能体，通过语音识别、向量检索和大语言模型，帮助用户快速生成专业的英文面试回答。

已提供 macOS 快速启动版，可打包为双击即用的 `.app`，详见 [macOS 快速启动说明](docs/macos.md)。

### 核心工作流

```
点击按钮 → 麦克风收音 → ASR 转文本 → 向量检索 → LLM 流式输出
```

## 项目结构

```
eng_interview/
├── main.py                    # 应用入口
├── config.py                  # 配置管理
├── requirements.txt           # 依赖文件
├── knowledge.md               # 个人简历及项目经验知识库
├── qa.md                      # 面试前准备的 QA 对
├── ui/
│   ├── __init__.py
│   └── main_window.py         # PyQt5 主界面
├── asr/
│   ├── __init__.py
│   ├── base.py                # ASR 抽象接口
│   └── recognizer.py          # 语音识别实现 (sounddevice + 本地 Whisper)
├── rag/
│   ├── __init__.py
│   └── vector_store.py        # ChromaDB 向量存储
├── llm/
│   ├── __init__.py
│   └── engine.py              # LLM 流式引擎 (OpenAI SDK 兼容)
└── workers/
    ├── __init__.py
    ├── audio_worker.py        # 录音线程 (QThread)
    ├── rag_worker.py          # 检索线程 (QThread)
    └── llm_worker.py          # LLM 请求线程 (QThread)
```

## 环境配置

### 1. Python 版本

推荐使用 Python 3.10 或 3.11

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

当前 ASR 默认使用 `sounddevice + openai-whisper` 或 `faster-whisper`，首次运行本地 Whisper 可能需要下载模型。PyAudio 不再是默认运行依赖；如果后续切回旧的 PyAudio 录音方案，再单独安装即可。

### 3. 配置 knowledge.md

编辑 `knowledge.md` 文件，填入你的个人简历、项目经验等信息。也可以在应用的“面试准备”页维护 `qa.md`，写入常见面试问题和第一人称英文回答。这些信息会在面试时一起被检索并用于生成回答。

## PyCharm 运行配置

### 方法一：直接运行

1. 打开 PyCharm，选择 `File → Open`，打开 `eng_interview` 项目目录
2. 等待 PyCharm 识别 Python 环境
3. 右键点击 `main.py`，选择 `Run 'main'`

### 方法二：创建运行配置

1. 点击 `Run → Edit Configurations`
2. 点击 `+` 号，选择 `Python`
3. 配置如下：
   - Name: `English Interview Copilot`
   - Script path: 选择 `main.py`
   - Working directory: 项目根目录
   - Interpreter: 选择你的 Python 解释器
4. 点击 `OK` 保存
5. 点击绿色运行按钮启动

### 调试配置

与运行配置相同，只是点击调试按钮（绿色虫子图标）即可进入调试模式。

## 使用说明

### 1. 启动应用

运行 `main.py` 后，会显示主界面。

### 2. 配置 LLM

在“面试准备”页填写：
- **API Key**: 你的 LLM API 密钥（支持 OpenAI、Claude、DeepSeek 等）
- **Base URL**: API 基础地址（默认 OpenAI，其他服务需修改）
- **Model**: 模型名称（如 `gpt-4o`, `claude-3-opus`, `deepseek-chat`）
- **音频输入**: 选择麦克风、BlackHole、Loopback 等输入设备。在线会议建议用 BlackHole/Loopback 接收电脑音频。
- **系统提示词**: 设置助手的角色、候选人身份、回答风格和面试场景

同一页面还可以设置自动/手动模式、静音间隔，并维护面试 QA 对。修改配置或 QA 后，“应用更新”按钮会高亮；点击后会统一保存配置、写入 `qa.md`，并刷新知识库索引。

### 3. 开始录音

点击 "开始录音" 按钮，麦克风会开始监听。说出面试官的问题，ASR 会自动识别并显示在上方文本框。

### 4. 查看回答

识别到问题后，系统会：
1. 从知识库检索相关背景信息
2. 流式生成英文回答
3. 实时显示在下方的回答区域

左侧翻译窗口的原文和译文会自动追加保存到 `translation_history.jsonl`，并可在顶部“翻译历史”页查看或清空。历史记录按一次录音请求分块展示，每页 5 个记录；单个记录超过 10 个 round 时可逐次展开。知识库问答和右侧 AI 回答不会写入这个历史文件。

### 5. 停止录音

再次点击按钮即可停止录音。

## ASR 说明

当前默认使用 `sounddevice + openai-whisper` 进行本地语音识别。

可在 `config.py` 中调整：
- `asr_provider`: 目前会映射到本地 Whisper 实现
- `silence_gap`: 自动模式下静音多少帧后触发识别（1 帧约 100ms）
- `mode`: `auto` 或 `manual`

## 常见问题

### Q: ChromaDB 初始化很慢？

首次加载 sentence-transformers 模型需要下载约 90MB，后续会使用本地缓存。

### Q: 修改 knowledge.md 后为什么没有立即生效？

应用启动时会检查 `knowledge.md` 和 `qa.md` 的内容哈希。如果文件变化，会自动重建本地 ChromaDB 索引；如果未变化，会复用现有索引。

### Q: ASR 识别效果不好？

建议使用耳机麦克风，确保环境安静。Google ASR 对英语识别效果较好。

### Q: 如何监听在线会议里的面试官声音？

macOS 通常不能把系统播放声音直接当作普通麦克风输入。建议安装 BlackHole 或 Loopback，在系统里创建包含耳机和虚拟设备的多输出设备，然后在“面试准备”页的“音频输入”里选择 BlackHole/Loopback。

对话页的“输入音量”条会显示当前输入设备的实时电平。如果播放会议声音时音量条不动，通常说明系统输出还没有路由到 BlackHole/Loopback。

### Q: 如何更换 LLM 服务？

修改 Base URL 和 Model Name 即可支持不同服务：
- OpenAI: `https://api.openai.com/v1`, `gpt-4o`
- Claude: `https://api.anthropic.com/v1`, `claude-3-opus`
- DeepSeek: `https://api.deepseek.com/v1`, `deepseek-chat`
- 本地模型: `http://localhost:8000/v1`, `your-model`

## 技术架构

### 多线程设计

使用 PyQt5 的 QThread 实现：
- **AudioWorker**: 麦克风录音和 ASR 识别
- **RagWorker**: 向量检索（不阻塞 UI）
- **LLMWorker**: 流式 LLM 请求

所有耗时操作都在后台线程执行，确保 UI 响应流畅。

### RAG 实现

- **向量数据库**: ChromaDB（本地持久化）
- **Embedding**: sentence-transformers/all-MiniLM-L6-v2（轻量级，CPU 运行）
- **检索策略**: 规则改写后召回候选 Top-12，本地重排后返回 Top-5
- **索引刷新**: 根据 `knowledge.md` 内容哈希判断是否自动重建
- **查询增强**: 使用规则改写扩展 motivation、novelty、contribution 等研究面试问题，并在本地候选召回后重排出最终上下文

### LLM 流式输出

使用 OpenAI SDK 兼容接口，支持流式生成并通过信号机制实时更新 UI。

## 许可证

MIT License
