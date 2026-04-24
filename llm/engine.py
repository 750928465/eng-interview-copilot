"""
LLM 流式生成引擎
支持 OpenAI SDK 兼容的 API 调用
"""
from typing import Generator, Optional, Callable
from openai import OpenAI

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


# System Prompt 模板
SYSTEM_PROMPT_TEMPLATE = """You are an English interview assistant helping a candidate answer interview questions.

The interviewer asked: {question}

Retrieved candidate background information:
{context}

Please follow these rules to provide a pure English, conversational response:

1. If the question is about personal experience or specific projects, you MUST strictly rely on the retrieved background information to answer.

2. If the retrieved information is empty, OR the question is about general computer science theory, system design, or common interview questions, please use your own knowledge to provide a professional answer.

3. If you combine both sources of information, please transition naturally between them.

4. Keep your answer concise but comprehensive (2-4 sentences for simple questions, more for complex ones).

5. Use natural, spoken English suitable for a job interview setting.

6. Never mention that you are an AI or that you're using retrieved information - just answer naturally as if you are the candidate."""


class LLMEngine:
    """LLM 流式生成引擎"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model_name: str = None
    ):
        self.api_key = api_key or config.api_key
        self.base_url = base_url or config.base_url
        self.model_name = model_name or config.model_name
        self.client: Optional[OpenAI] = None

    def _init_client(self) -> None:
        """初始化 OpenAI 客户端"""
        if self.client is None:
            if not self.api_key:
                raise ValueError("API Key 未配置")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

    def generate(
        self,
        question: str,
        context: str = "",
        callback: Callable[[str], None] = None
    ) -> Generator[str, None, None]:
        """
        流式生成回答

        Args:
            question: 面试官问题
            context: 检索到的背景信息
            callback: 每个 token 的回调函数

        Yields:
            生成的文本片段
        """
        self._init_client()

        # 构建提示
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            question=question,
            context=context if context else "No relevant background information found."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=500
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    if callback:
                        callback(text)
                    yield text

        except Exception as e:
            error_msg = f"[LLM Error: {e}]"
            if callback:
                callback(error_msg)
            yield error_msg

    def generate_sync(
        self,
        question: str,
        context: str = ""
    ) -> str:
        """
        同步生成完整回答

        Args:
            question: 面试官问题
            context: 检索到的背景信息

        Returns:
            完整回答文本
        """
        self._init_client()

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            question=question,
            context=context if context else "No relevant background information found."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"[LLM Error: {e}]"


# 全局实例
llm_engine: Optional[LLMEngine] = None


def get_llm_engine() -> LLMEngine:
    """获取全局 LLM 引擎实例"""
    global llm_engine
    if llm_engine is None:
        llm_engine = LLMEngine()
    return llm_engine


def update_llm_config(api_key: str, base_url: str, model_name: str) -> None:
    """更新 LLM 配置"""
    global llm_engine
    config.api_key = api_key
    config.base_url = base_url
    config.model_name = model_name
    # 重置实例以应用新配置
    llm_engine = LLMEngine(api_key, base_url, model_name)