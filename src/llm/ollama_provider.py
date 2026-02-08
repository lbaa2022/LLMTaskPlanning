"""
Ollama LLM Provider implementation.
Ollama provides OpenAI-compatible API for locally served models.
"""
import os
from openai import OpenAI, APIConnectionError
from typing import List, Dict, Optional

from .base import LLMProvider, LLMConfig


class OllamaProvider(LLMProvider):
    """
    LLM Provider for Ollama-served models.

    Ollama exposes an OpenAI-compatible API endpoint.
    Default: http://localhost:11434/v1

    Popular models: llama2, llama3, mistral, codellama, phi, gemma
    """

    DEFAULT_API_BASE = "http://localhost:11434/v1"

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        # Ollama doesn't require API key, but OpenAI client needs one
        api_key = config.api_key or os.getenv('OLLAMA_API_KEY') or "ollama"

        # Set API base - use Ollama's default port
        self.api_base = config.api_base or os.getenv('OLLAMA_API_BASE') or self.DEFAULT_API_BASE

        # Initialize OpenAI client pointing to Ollama
        self.client = OpenAI(api_key=api_key, base_url=self.api_base)

        print(f"Ollama Provider initialized with endpoint: {self.api_base}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion using Ollama's OpenAI-compatible API."""
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
                f"Could not connect to Ollama at {self.api_base}. "
                f"Is Ollama running? Start with: ollama serve\n"
                f"Original error: {e}"
            ) from e

    def select_action(self, prompt: str, candidates: List[str]) -> str:
        """Select best action using Ollama."""
        messages = self._build_selection_messages(prompt, candidates)
        return self.chat_completion(messages, temperature=0, max_tokens=100)
