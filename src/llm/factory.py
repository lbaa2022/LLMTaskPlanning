"""
Factory for creating LLM providers.
Implements the Factory Pattern for instantiating the correct provider based on configuration.
"""
import os
from typing import Optional
from dotenv import load_dotenv

from .base import LLMProvider, LLMConfig
from .openai_provider import OpenAIProvider
from .vllm_provider import VLLMProvider
from .ollama_provider import OllamaProvider
from .lmstudio_provider import LMStudioProvider


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.

    Supported providers (OpenAI API compatible only):
    - openai: OpenAI API (GPT-3.5, GPT-4, GPT-5, o1, o3)
    - vllm: vLLM-served models (OpenAI-compatible API)

    Usage:
        provider = LLMProviderFactory.create("openai", config)
        response = provider.chat_completion(messages)
    """

    # Provider name to class mapping (OpenAI API compatible only)
    PROVIDERS = {
        "openai": OpenAIProvider,
        "vllm": VLLMProvider,
        "ollama": OllamaProvider,
        "lmstudio": LMStudioProvider,
    }

    @classmethod
    def create(
        cls,
        provider_type: str,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        reasoning_effort: Optional[str] = None
    ) -> LLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider_type: Type of provider ('openai', 'vllm')
            model_name: Name of the model to use
            api_key: API key (optional, will check environment variables)
            api_base: Custom API base URL (optional)
            temperature: Default temperature for generation
            max_tokens: Default max tokens for generation
            reasoning_effort: Reasoning effort for o1/o3/gpt-5.x models ("low", "medium", "high")

        Returns:
            LLMProvider instance

        Raises:
            ValueError: If provider_type is not supported
        """
        # Load environment variables
        load_dotenv()

        provider_type = provider_type.lower()

        if provider_type not in cls.PROVIDERS:
            supported = ', '.join(cls.PROVIDERS.keys())
            raise ValueError(f"Unknown provider type: {provider_type}. Supported: {supported}")

        # Create config
        config = LLMConfig(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort
        )

        # Instantiate provider
        provider_class = cls.PROVIDERS[provider_type]
        return provider_class(config)

    @classmethod
    def from_config(cls, cfg) -> LLMProvider:
        """
        Create an LLM provider from a Hydra/OmegaConf configuration object.

        Expected config structure:
            planner:
                provider: "openai"  # or "vllm" (OpenAI API compatible only)
                model_name: "gpt-4"
                api_key: ""  # optional, uses env vars if empty
                api_base: ""  # optional
                temperature: 0.0
                max_tokens: 500

        Args:
            cfg: Configuration object with planner settings

        Returns:
            LLMProvider instance
        """
        # Load environment variables
        load_dotenv()

        planner_cfg = cfg.planner

        # Get provider type (default to openai for backwards compatibility)
        provider_type = getattr(planner_cfg, 'provider', 'openai')

        # Handle legacy model_name format "OpenAI/gpt-4" -> provider=openai, model=gpt-4
        model_name = planner_cfg.model_name
        if '/' in model_name and provider_type == 'openai':
            parts = model_name.split('/')
            if len(parts) == 2:
                legacy_provider = parts[0].lower()
                model_name = parts[1]
                # Map legacy provider names (OpenAI API compatible only)
                if legacy_provider in ['openai', 'vllm', 'ollama', 'lmstudio']:
                    provider_type = legacy_provider

        # Get API key and base
        api_key = getattr(planner_cfg, 'api_key', '') or getattr(planner_cfg, 'openai_api_key', '')
        api_base = getattr(planner_cfg, 'api_base', '') or getattr(planner_cfg, 'openai_api_base', '')

        # Get generation settings
        temperature = getattr(planner_cfg, 'temperature', 0.0)
        max_tokens = getattr(planner_cfg, 'max_tokens', 500)

        # Get reasoning model settings (for o1, o3, gpt-5.x)
        reasoning_effort = getattr(planner_cfg, 'reasoning_effort', None)

        return cls.create(
            provider_type=provider_type,
            model_name=model_name,
            api_key=api_key if api_key else None,
            api_base=api_base if api_base else None,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort
        )

    @classmethod
    def list_providers(cls) -> list:
        """Return list of supported provider types."""
        return list(set(cls.PROVIDERS.keys()))
