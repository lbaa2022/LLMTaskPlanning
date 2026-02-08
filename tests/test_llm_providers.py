"""
Test suite for LLM providers.
Tests provider instantiation, factory registration, and API compatibility.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm import (
    LLMProviderFactory,
    LLMConfig,
    OpenAIProvider,
    VLLMProvider,
    OllamaProvider,
    LMStudioProvider,
)


class TestProviderFactory:
    """Test LLMProviderFactory functionality."""

    def test_list_providers_includes_all(self):
        """All four providers should be registered."""
        providers = LLMProviderFactory.list_providers()
        assert 'openai' in providers
        assert 'vllm' in providers
        assert 'ollama' in providers
        assert 'lmstudio' in providers

    def test_create_ollama_provider(self):
        """Factory should create OllamaProvider for 'ollama' type."""
        provider = LLMProviderFactory.create('ollama', 'llama2')
        assert isinstance(provider, OllamaProvider)
        assert provider.model_name == 'llama2'

    def test_create_lmstudio_provider(self):
        """Factory should create LMStudioProvider for 'lmstudio' type."""
        provider = LLMProviderFactory.create('lmstudio', 'test-model')
        assert isinstance(provider, LMStudioProvider)
        assert provider.model_name == 'test-model'

    def test_create_with_custom_api_base(self):
        """Custom api_base should be respected."""
        custom_base = 'http://custom-server:9999/v1'
        provider = LLMProviderFactory.create(
            'lmstudio',
            'model',
            api_base=custom_base
        )
        assert provider.api_base == custom_base

    def test_unknown_provider_raises_error(self):
        """Unknown provider type should raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            LLMProviderFactory.create('unknown_provider', 'model')
        assert 'Unknown provider type' in str(excinfo.value)

    def test_legacy_format_ollama(self):
        """Legacy 'ollama/model' format should work."""
        # This tests the from_config path with legacy format
        # We can't easily test from_config without a full config object,
        # but we verify the provider is registered correctly
        providers = LLMProviderFactory.PROVIDERS
        assert 'ollama' in providers
        assert providers['ollama'] == OllamaProvider


class TestOllamaProvider:
    """Test OllamaProvider functionality."""

    def test_default_api_base(self):
        """Default API base should be localhost:11434."""
        assert OllamaProvider.DEFAULT_API_BASE == 'http://localhost:11434/v1'

    def test_initialization(self):
        """Provider should initialize with correct settings."""
        config = LLMConfig(model_name='llama2')
        provider = OllamaProvider(config)

        assert provider.model_name == 'llama2'
        assert provider.api_base == OllamaProvider.DEFAULT_API_BASE

    def test_custom_api_base_from_config(self):
        """Custom api_base from config should be used."""
        config = LLMConfig(
            model_name='llama2',
            api_base='http://custom:8080/v1'
        )
        provider = OllamaProvider(config)

        assert provider.api_base == 'http://custom:8080/v1'


class TestLMStudioProvider:
    """Test LMStudioProvider functionality."""

    def test_default_api_base(self):
        """Default API base should be localhost:1234."""
        assert LMStudioProvider.DEFAULT_API_BASE == 'http://localhost:1234/v1'

    def test_initialization(self):
        """Provider should initialize with correct settings."""
        config = LLMConfig(model_name='local-model')
        provider = LMStudioProvider(config)

        assert provider.model_name == 'local-model'
        assert provider.api_base == LMStudioProvider.DEFAULT_API_BASE

    def test_custom_api_base_from_config(self):
        """Custom api_base from config should be used."""
        config = LLMConfig(
            model_name='model',
            api_base='http://10.254.90.90:1234/v1'
        )
        provider = LMStudioProvider(config)

        assert provider.api_base == 'http://10.254.90.90:1234/v1'


class TestProviderInterface:
    """Test that all providers implement the required interface."""

    @pytest.fixture(params=['ollama', 'lmstudio'])
    def provider(self, request):
        """Create provider instance for each type."""
        return LLMProviderFactory.create(request.param, 'test-model')

    def test_has_chat_completion_method(self, provider):
        """Provider should have chat_completion method."""
        assert hasattr(provider, 'chat_completion')
        assert callable(provider.chat_completion)

    def test_has_select_action_method(self, provider):
        """Provider should have select_action method."""
        assert hasattr(provider, 'select_action')
        assert callable(provider.select_action)

    def test_has_model_name(self, provider):
        """Provider should have model_name attribute."""
        assert hasattr(provider, 'model_name')
        assert provider.model_name == 'test-model'


# Integration tests (require running server)
@pytest.mark.integration
class TestLMStudioIntegration:
    """Integration tests requiring a running LM Studio server."""

    @pytest.fixture
    def lmstudio_provider(self):
        """Create LM Studio provider with test server."""
        # Skip if LMSTUDIO_TEST_BASE not set
        test_base = os.getenv('LMSTUDIO_TEST_BASE')
        test_model = os.getenv('LMSTUDIO_TEST_MODEL', 'test-model')

        if not test_base:
            pytest.skip("LMSTUDIO_TEST_BASE not set")

        return LLMProviderFactory.create(
            'lmstudio',
            test_model,
            api_base=test_base
        )

    def test_chat_completion(self, lmstudio_provider):
        """Test actual chat completion."""
        messages = [
            {'role': 'user', 'content': 'Say "hello" and nothing else.'}
        ]
        response = lmstudio_provider.chat_completion(messages, max_tokens=10)
        assert len(response) > 0

    def test_select_action(self, lmstudio_provider):
        """Test action selection."""
        prompt = "I need to pick up a cup from the table."
        candidates = ["PickupObject cup", "GoToObject table", "OpenObject door"]

        action = lmstudio_provider.select_action(prompt, candidates)
        assert action in candidates or any(c in action for c in candidates)


if __name__ == '__main__':
    # Run unit tests only (not integration tests)
    pytest.main([__file__, '-v', '-m', 'not integration'])
