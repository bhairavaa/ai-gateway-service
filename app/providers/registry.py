"""ProviderRegistry — resolves providers dynamically by name.

Built once at app startup (see app/main.py's lifespan) from Settings. A provider
with no credentials configured is simply absent from the registry rather than
an error, so the gateway runs fine with whatever subset of providers the
deployer has keys for. Business logic only ever calls `registry.get(name)`;
there is no if/elif on provider name anywhere in the codebase.
"""

from collections.abc import Callable

from app.config import Settings
from app.providers.base import BaseProvider
from app.providers.claude_provider import ClaudeProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.ollama_provider import OllamaProvider
from app.providers.openai_provider import OpenAIProvider


class ProviderNotRegisteredError(Exception):
    def __init__(self, name: str):
        super().__init__(f"Provider '{name}' is not registered (missing credentials or unknown name)")
        self.name = name


class ProviderRegistry:
    def __init__(self, providers: dict[str, BaseProvider]):
        self._providers = providers

    def get(self, name: str) -> BaseProvider:
        try:
            return self._providers[name]
        except KeyError:
            raise ProviderNotRegisteredError(name) from None

    def is_registered(self, name: str) -> bool:
        return name in self._providers

    def all(self) -> dict[str, BaseProvider]:
        return dict(self._providers)


# Each factory returns None when the provider's required config (typically an API key)
# is absent, so build_provider_registry() can skip it rather than fail startup. Adding a
# 5th provider is one new file (implementing BaseProvider) + one new entry here — nothing
# else in the codebase branches on provider name.
_PROVIDER_FACTORIES: dict[str, Callable[[Settings], BaseProvider | None]] = {
    "openai": lambda s: (
        OpenAIProvider(
            api_key=s.openai_api_key, timeout=s.provider_request_timeout_seconds, base_url=s.openai_base_url
        )
        if s.openai_api_key
        else None
    ),
    "anthropic": lambda s: (
        ClaudeProvider(api_key=s.anthropic_api_key, timeout=s.provider_request_timeout_seconds)
        if s.anthropic_api_key
        else None
    ),
    "gemini": lambda s: (
        GeminiProvider(api_key=s.google_api_key, timeout=s.provider_request_timeout_seconds)
        if s.google_api_key
        else None
    ),
    # No API key required — always registered. If the local daemon isn't reachable, that surfaces
    # via health_check()/generate() failures at call time, not at registration time.
    "ollama": lambda s: OllamaProvider(base_url=s.ollama_base_url, timeout=s.provider_request_timeout_seconds),
}


def build_provider_registry(settings: Settings) -> ProviderRegistry:
    providers: dict[str, BaseProvider] = {}
    for name, factory in _PROVIDER_FACTORIES.items():
        instance = factory(settings)
        if instance is not None:
            providers[name] = instance
    return ProviderRegistry(providers)
