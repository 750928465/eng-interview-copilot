# English Interview Copilot

英语面试实时辅助智能体 - Windows 桌面应用

## 项目概述

本项目是一个用于英语面试实时辅助的 Copilot 智能体，通过语音识别、向量检索和大语言模型，帮助用户快速生成专业的英文面试回答。

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
├── ui/
│   ├── __init__.py
│   └── main_window.py         # PyQt5 主界面
├── asr/
│   ├── __init__.py
│   ├── base.py                # ASR 抽象接口
│   └── recognizer.py          # 语音识别实现 (Google/Whisper)
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

**Windows 用户注意：**
- PyAudio 安装可能需要额外步骤，如果 `pip install pyaudio` 失败，请尝试：
  ```bash
  pip install pipwin
  pipwin install pyaudio
  ```
  或下载预编译的 wheel 文件：https://github.com/intxcc/pyaudio_portable/releases

### 3. 配置 knowledge.md

编辑 `knowledge.md` 文件，填入你的个人简历、项目经验等信息。这些信息会在面试时被检索并用于生成回答。

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

在界面顶部填写：
- **API Key**: 你的 LLM API 密钥（支持 OpenAI、Claude、DeepSeek 等）
- **Base URL**: API 基础地址（默认 OpenAI，其他服务需修改）
- **Model**: 模型名称（如 `gpt-4o`, `claude-3-opus`, `deepseek-chat`）

### 3. 开始录音

点击 "开始录音" 按钮，麦克风会开始监听。说出面试官的问题，ASR 会自动识别并显示在上方文本框。

### 4. 查看回答

识别到问题后，系统会：
1. 从知识库检索相关背景信息
2. 流式生成英文回答
3. 实时显示在下方的回答区域

### 5. 停止录音

再次点击按钮即可停止录音。

## ASR 提供者切换

默认使用 Google Speech Recognition（免费在线服务）。

可在 `config.py` 中修改 `asr_provider`：
- `google`: Google 在线 ASR（免费，需网络）
- `whisper_local`: 本地 Whisper 模型（离线，需下载模型）

## 常见问题

### Q: PyAudio 安装失败？

Windows 用户请使用预编译 wheel 文件或 `pipwin`。

### Q: ChromaDB 初始化很慢？

首次加载 sentence-transformers 模型需要下载约 90MB，后续会使用本地缓存。

### Q: ASR 识别效果不好？

建议使用耳机麦克风，确保环境安静。Google ASR 对英语识别效果较好。

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
- **检索策略**: 直接相似度检索 Top-3

### LLM 流式输出

使用 OpenAI SDK 兼容接口，支持流式生成并通过信号机制实时更新 UI。

## 许可证

MIT License