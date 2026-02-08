"""
LLM Provider module.

Provides a unified interface for OpenAI API compatible LLM backends:
- OpenAI (GPT-3.5, GPT-4, GPT-5, o1, o3)
- vLLM (locally served models with OpenAI-compatible API)
- Ollama (locally served models with OpenAI-compatible API)
- LM Studio (locally served models with OpenAI-compatible API)

Usage:
    from llm import LLMProviderFactory

    # Create provider
    provider = LLMProviderFactory.create("openai", "gpt-4")

    # Or from config
    provider = LLMProviderFactory.from_config(cfg)

    # Use provider
    response = provider.chat_completion(messages)
    action = provider.select_action(prompt, candidates)
"""

from .base import LLMProvider, LLMConfig
from .factory import LLMProviderFactory
from .openai_provider import OpenAIProvider
from .vllm_provider import VLLMProvider
from .ollama_provider import OllamaProvider
from .lmstudio_provider import LMStudioProvider

__all__ = [
    'LLMProvider',
    'LLMConfig',
    'LLMProviderFactory',
    'OpenAIProvider',
    'VLLMProvider',
    'OllamaProvider',
    'LMStudioProvider',
]
