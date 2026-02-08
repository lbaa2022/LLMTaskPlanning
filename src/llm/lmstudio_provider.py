"""
LM Studio LLM Provider implementation.
LM Studio provides OpenAI-compatible API for locally served models.
"""
import os
from openai import OpenAI, APIConnectionError
from typing import List, Dict, Optional

from .base import LLMProvider, LLMConfig


class LMStudioProvider(LLMProvider):
    """
    LLM Provider for LM Studio-served models.

    LM Studio exposes an OpenAI-compatible API endpoint.
    Default: http://localhost:1234/v1

    Model name should match the model loaded in LM Studio.
    """

    DEFAULT_API_BASE = "http://localhost:1234/v1"

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        # LM Studio doesn't require API key, but OpenAI client needs one
        api_key = config.api_key or os.getenv('LMSTUDIO_API_KEY') or "lm-studio"

        # Set API base - use LM Studio's default port
        self.api_base = config.api_base or os.getenv('LMSTUDIO_API_BASE') or self.DEFAULT_API_BASE

        # Initialize OpenAI client pointing to LM Studio
        self.client = OpenAI(api_key=api_key, base_url=self.api_base)

        print(f"LM Studio Provider initialized with endpoint: {self.api_base}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion using LM Studio's OpenAI-compatible API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens
            )
            return response.choices[0].message.content.strip()
        except APIConnectionError as e:
            raise ConnectionError(
                f"Could not connect to LM Studio at {self.api_base}. "
                f"Is LM Studio running with a model loaded?\n"
                f"Original error: {e}"
            ) from e

    def select_action(self, prompt: str, candidates: List[str]) -> str:
        """Select best action using LM Studio."""
        messages = self._build_selection_messages(prompt, candidates)
        return self.chat_completion(messages, temperature=0, max_tokens=100)
