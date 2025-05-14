"""
LLM client implementations for various providers.
This module provides a unified interface for making requests to different LLM services.
"""

import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Base abstract class for all LLM clients."""

    @abstractmethod
    def generate_completion(self, prompt, max_tokens=None, temperature=None, **kwargs):
        """Generate a completion for the given prompt."""
        pass


class OpenAIClient(LLMClient):
    """Client for OpenAI API."""

    def __init__(self, api_key, model="gpt-4"):
        import openai

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate_completion(self, prompt, max_tokens=None, temperature=None, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class DeepSeekClient(LLMClient):
    """Client for DeepSeek API."""

    def __init__(self, api_key, model="deepseek-chat"):
        import openai

        self.client = openai.OpenAI(
            api_key=api_key, base_url="https://api.deepseek.com"
        )
        self.model = model

    def generate_completion(self, prompt, max_tokens=None, temperature=None, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )

        return response.choices[0].message.content


class QwenClient(LLMClient):
    """Client for Alibaba Qwen API."""

    def __init__(self, api_key, model="qwen-max"):
        import openai

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model

    def generate_completion(self, prompt, max_tokens=None, temperature=None, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
            # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
            # extra_body={"enable_thinking": False},
        )

        return response.choices[0].message.content


def get_llm_client(provider="openai"):
    """Factory function to get a LLM client based on the provider name."""
    from dotenv import load_dotenv

    load_dotenv(override=True)

    provider = provider.lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or .env file")
        return OpenAIClient(api_key, model)

    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment or .env file")
        return DeepSeekClient(api_key, model)

    elif provider == "qwen":
        api_key = os.getenv("QWEN_API_KEY")
        model = os.getenv("QWEN_MODEL", "qwen-max")
        if not api_key:
            raise ValueError("QWEN_API_KEY not found in environment or .env file")
        return QwenClient(api_key, model)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Supported providers: openai, deepseek, qwen"
        )
