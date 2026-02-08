"""
OpenAI LLM Provider implementation.
Supports OpenAI API (GPT-3.5, GPT-4, GPT-5.2, etc.)
"""
import os
from openai import OpenAI
from typing import List, Dict, Optional

from .base import LLMProvider, LLMConfig


class OpenAIProvider(LLMProvider):
    """
    LLM Provider for OpenAI API.

    Supports: gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-5.2, etc.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        # Set API key
        api_key = config.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        # Set custom API base if provided
        api_base = config.api_base or os.getenv('OPENAI_API_BASE')

        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base

        self.client = OpenAI(**client_kwargs)

    def _is_reasoning_model(self) -> bool:
        """Check if the model is a reasoning model (o1, o3, gpt-5.x)."""
        return (self.model_name.startswith("gpt-5") or
                self.model_name.startswith("o1") or
                self.model_name.startswith("o3"))

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion using OpenAI API."""
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        # Build API kwargs
        kwargs = {
            "model": self.model_name,
            "messages": messages,
        }

        # Reasoning models (o1, o3, gpt-5.x) have different parameters
        if self._is_reasoning_model():
            # Use max_completion_tokens instead of max_tokens
            kwargs["max_completion_tokens"] = max_tok
            # Add reasoning_effort if configured (low, medium, high)
            if self.config.reasoning_effort:
                kwargs["reasoning_effort"] = self.config.reasoning_effort
            # Note: reasoning models may ignore temperature or have restrictions
        else:
            kwargs["temperature"] = temp
            kwargs["max_tokens"] = max_tok

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    def select_action(self, prompt: str, candidates: List[str]) -> str:
        """Select best action using OpenAI API."""
        messages = self._build_selection_messages(prompt, candidates)
        return self.chat_completion(messages, temperature=0, max_tokens=100)
