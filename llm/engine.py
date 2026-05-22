"""
LLM 流式生成引擎
支持 OpenAI SDK 兼容的 API 调用
"""
from typing import Callable, Dict, Generator, List, Optional
from openai import OpenAI, APIStatusError, AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


# System Prompt 模板
SYSTEM_PROMPT_TEMPLATE = """{custom_instructions}

You are an English interview assistant helping a candidate answer interview questions.

The interviewer asked: {question}

Retrieved candidate background information:
{context}

Please follow these rules to provide a pure English, conversational response:

1. If the question is about personal experience or specific projects, you MUST strictly rely on the retrieved background information to answer.

2. If the retrieved information is empty, OR the question is about general computer science theory, system design, or common interview questions, please use your own knowledge to provide a professional answer.

3. If you combine both sources of information, please transition naturally between them.

4. Keep your answer concise but comprehensive (2-4 sentences for simple questions, more for complex ones).

5. Use natural, spoken English suitable for a job interview setting.

6. Use the previous interview turns only to keep continuity, resolve follow-up questions, avoid repeating yourself, and make the response sound natural.

7. Never mention that you are an AI or that you're using retrieved information - just answer naturally as if you are the candidate."""


# 错误码映射
ERROR_CODE_MAP = {
    # HTTP 状态码
    400: "请求参数错误，请检查模型名称和参数配置",
    401: "API Key 无效或已过期，请检查 API Key",
    403: "权限不足，请检查 API Key 权限或订阅状态",
    404: "接口或模型不存在，请检查 Base URL 和模型名称",
    429: "请求频率过高或配额已用尽，请稍后重试",
    500: "服务端内部错误，请稍后重试",
    502: "服务端网关错误，请稍后重试",
    503: "服务暂时不可用，请稍后重试",
}

# 厂商特定错误码
PROVIDER_ERROR_MAP = {
    "InvalidSubscription": "订阅已过期或无效，请前往对应平台续费",
    "InsufficientQuota": "账户余额不足，请充值后再试",
    "InvalidAPIKey": "API Key 无效，请检查是否输入正确",
    "ModelNotFound": "模型不存在，请检查模型名称是否正确",
    "RateLimitExceeded": "已超出速率限制，请稍后重试",
    "QuotaExceeded": "配额已用尽，请升级套餐或等待重置",
}


def _clarify_question(question: str) -> str:
    """Normalize short spoken interview prompts before sending them to the model."""
    original = (question or "").strip()
    normalized = original.lower()
    asks_technical_detail = any(
        phrase in normalized
        for phrase in ("tech", "technical", "technology", "method", "architecture", "about")
    )
    project_aliases = [
        (
            ("first project", "project one", "project 1"),
            "first project, C2FTFNet",
            "motivation, coarse-to-fine pipeline, main architecture, and personal contribution",
        ),
        (
            ("second project", "project two", "project 2"),
            "second project, MACFNet",
            "multi-attention cross-scale fusion method, SC/DA/MSCA modules, and personal contribution",
        ),
        (
            ("third project", "project three", "project 3"),
            "third project, MCA-ViLSTM / MCA-VLSTM",
            "Vision LSTM, bidirectional scanning, multi-scale channel attention, lightweight design, and personal contribution",
        ),
        (
            ("research one", "research 1", "first research"),
            "first doctoral research direction, accurate multi-disease detection",
            "multi-label fundus disease prediction, model design, datasets, and expected contribution",
        ),
        (
            ("research two", "research 2", "second research"),
            "second doctoral research direction, cardiovascular risk prediction",
            "retinal vascular features, clinical variables, multimodal fusion, and expected contribution",
        ),
    ]

    for aliases, target, details in project_aliases:
        if any(alias in normalized for alias in aliases):
            if asks_technical_detail:
                return f"Please explain the method used in your {target}, including the {details}."
            return f"Please answer the interview question about your {target}: {original}"
    return original


def _format_error(e: Exception) -> str:
    """将 API 异常转换为用户友好的错误信息"""
    # OpenAI SDK 的 HTTP 状态错误
    if isinstance(e, APIStatusError):
        status_code = e.status_code
        base_msg = ERROR_CODE_MAP.get(status_code, f"HTTP {status_code} 错误")

        # 尝试解析响应体中的厂商错误码
        try:
            body = e.response.json()
            error_info = body.get("error", {})
            error_code = error_info.get("code", "")
            error_msg = error_info.get("message", "")

            # 匹配厂商错误码
            if error_code in PROVIDER_ERROR_MAP:
                return f"[{status_code} {error_code}] {PROVIDER_ERROR_MAP[error_code]}"

            # 提取简短错误信息（截断过长的 message）
            if error_msg:
                short_msg = error_msg[:120] + "..." if len(error_msg) > 120 else error_msg
                return f"[{status_code}] {base_msg}\n详情: {short_msg}"

        except Exception:
            pass

        return f"[{status_code}] {base_msg}"

    # 认证错误
    if isinstance(e, AuthenticationError):
        return "[认证失败] API Key 无效，请检查配置"

    # 速率限制
    if isinstance(e, RateLimitError):
        return "[频率限制] 请求过于频繁，请稍后重试"

    # 超时
    if isinstance(e, APITimeoutError):
        return "[超时] API 请求超时，请检查网络或稍后重试"

    # 连接错误
    if isinstance(e, APIConnectionError):
        return "[连接失败] 无法连接到 API 服务器，请检查 Base URL 和网络"

    # 其他未知错误
    return f"[错误] {str(e)[:200]}"


class LLMEngine:
    """LLM 流式生成引擎"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model_name: str = None,
        system_prompt: str = None
    ):
        self.api_key = api_key or config.api_key
        self.base_url = base_url or config.base_url
        self.model_name = model_name or config.model_name
        self.system_prompt = system_prompt if system_prompt is not None else config.system_prompt
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
        conversation_history: Optional[List[Dict[str, str]]] = None,
        callback: Callable[[str], None] = None
    ) -> Generator[str, None, None]:
        """
        流式生成回答

        Args:
            question: 面试官问题
            context: 检索到的背景信息
            conversation_history: 前几轮 user/assistant 对话历史
            callback: 每个 token 的回调函数

        Yields:
            生成的文本片段
        """
        self._init_client()
        clarified_question = _clarify_question(question)

        # 构建提示
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            custom_instructions=self.system_prompt.strip() or "Answer as the candidate in first person.",
            question=clarified_question,
            context=context if context else "No relevant background information found."
        )

        messages = self._build_messages(system_prompt, clarified_question, conversation_history)

        try:
            emitted_content = False
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=500
            )

            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                text = getattr(delta, "content", None) if delta is not None else None
                if text is None and isinstance(delta, dict):
                    text = delta.get("content")
                if text:
                    emitted_content = True
                    if callback:
                        callback(text)
                    yield text

            if not emitted_content:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                text = (response.choices[0].message.content or "").strip()
                if not text:
                    text = "[LLM 未返回内容] 流式和非流式请求都没有返回可显示文本，请检查模型名称或服务商兼容性。"
                if callback:
                    callback(text)
                yield text

        except Exception as e:
            error_msg = _format_error(e)
            if callback:
                callback(error_msg)
            yield error_msg

    def generate_sync(
        self,
        question: str,
        context: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        同步生成完整回答

        Args:
            question: 面试官问题
            context: 检索到的背景信息
            conversation_history: 前几轮 user/assistant 对话历史

        Returns:
            完整回答文本
        """
        self._init_client()
        clarified_question = _clarify_question(question)

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            custom_instructions=self.system_prompt.strip() or "Answer as the candidate in first person.",
            question=clarified_question,
            context=context if context else "No relevant background information found."
        )

        messages = self._build_messages(system_prompt, clarified_question, conversation_history)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content

        except Exception as e:
            return _format_error(e)

    def _build_messages(
        self,
        system_prompt: str,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        for item in conversation_history or []:
            role = item.get("role")
            content = (item.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})
        return messages


# 全局实例
llm_engine: Optional[LLMEngine] = None


def get_llm_engine() -> LLMEngine:
    """获取全局 LLM 引擎实例"""
    global llm_engine
    if llm_engine is None:
        llm_engine = LLMEngine()
    return llm_engine


def update_llm_config(api_key: str, base_url: str, model_name: str, system_prompt: str = None) -> None:
    """更新 LLM 配置"""
    global llm_engine
    config.api_key = api_key
    config.base_url = base_url
    config.model_name = model_name
    if system_prompt is not None:
        config.system_prompt = system_prompt
    # 重置实例以应用新配置
    llm_engine = LLMEngine(api_key, base_url, model_name, config.system_prompt)
