from app.config import Settings
from app.providers.claude_provider import ClaudeProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.ollama_provider import OllamaProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.registry import ProviderNotRegisteredError, build_provider_registry
import pytest


class TestBuildProviderRegistry:
    def test_no_keys_configured_registers_ollama_only(self):
        settings = Settings(openai_api_key=None, anthropic_api_key=None, google_api_key=None)
        registry = build_provider_registry(settings)

        assert set(registry.all().keys()) == {"ollama"}
        assert isinstance(registry.get("ollama"), OllamaProvider)

    def test_all_keys_configured_registers_all_four(self):
        settings = Settings(openai_api_key="sk-x", anthropic_api_key="sk-ant-x", google_api_key="AIza-x")
        registry = build_provider_registry(settings)

        assert set(registry.all().keys()) == {"openai", "anthropic", "gemini", "ollama"}
        assert isinstance(registry.get("openai"), OpenAIProvider)
        assert isinstance(registry.get("anthropic"), ClaudeProvider)
        assert isinstance(registry.get("gemini"), GeminiProvider)

    def test_partial_keys_registers_only_those_plus_ollama(self):
        settings = Settings(openai_api_key="sk-x", anthropic_api_key=None, google_api_key=None)
        registry = build_provider_registry(settings)

        assert set(registry.all().keys()) == {"openai", "ollama"}

    def test_unregistered_provider_raises(self):
        settings = Settings(openai_api_key=None, anthropic_api_key=None, google_api_key=None)
        registry = build_provider_registry(settings)

        with pytest.raises(ProviderNotRegisteredError):
            registry.get("openai")
