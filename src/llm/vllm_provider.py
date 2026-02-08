"""
vLLM LLM Provider implementation.
vLLM provides OpenAI-compatible API for locally served models.
"""
import os
from openai import OpenAI
from typing import List, Dict, Optional

from .base import LLMProvider, LLMConfig


class VLLMProvider(LLMProvider):
    """
    LLM Provider for vLLM-served models.

    vLLM exposes an OpenAI-compatible API endpoint.
    Typically runs at http://localhost:8000/v1
    """

    DEFAULT_API_BASE = "http://localhost:8000/v1"

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        # vLLM uses OpenAI-compatible API
        # API key can be anything for local vLLM (often "EMPTY" or "token-abc123")
        api_key = config.api_key or os.getenv('VLLM_API_KEY') or "EMPTY"

        # Set API base - required for vLLM
        api_base = config.api_base or os.getenv('VLLM_API_BASE') or self.DEFAULT_API_BASE

        # Initialize OpenAI client pointing to vLLM
        self.client = OpenAI(api_key=api_key, base_url=api_base)

        print(f"vLLM Provider initialized with endpoint: {api_base}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion using vLLM's OpenAI-compatible API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens
        )
        return response.choices[0].message.content.strip()

    def select_action(self, prompt: str, candidates: List[str]) -> str:
        """Select best action using vLLM."""
        messages = self._build_selection_messages(prompt, candidates)
        return self.chat_completion(messages, temperature=0, max_tokens=100)
