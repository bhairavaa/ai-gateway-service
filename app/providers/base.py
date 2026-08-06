"""Provider layer — the one abstraction every LLM integration implements.

Business logic (app/services/) never imports a concrete provider class or
branches on a provider's name; it only calls through this interface via
ProviderRegistry.get(name). Adding a 5th provider is one new file implementing
BaseProvider + one registry entry — no other code changes.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.schemas.chat import ChatMessage


@dataclass(frozen=True)
class UsageInfo:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ProviderResult:
    content: str
    model: str
    finish_reason: str | None
    usage: UsageInfo | None  # None only if the provider truly returned nothing usable


@dataclass(frozen=True)
class StreamChunk:
    delta: str
    is_final: bool
    finish_reason: str | None = None
    usage: UsageInfo | None = None  # populated only on is_final=True, and only best-effort (see per-provider adapters)


@dataclass(frozen=True)
class ModelInfo:
    id: str
    provider: str


@dataclass(frozen=True)
class HealthStatus:
    provider: str
    healthy: bool
    detail: str | None = None


class ProviderCallError(Exception):
    """Normalizes every provider SDK's own exception hierarchy (openai.*, anthropic.*,
    google.*, ...) into one type, so business logic (fallback, error handling) only
    ever needs to catch this — never a provider-specific exception class.
    """

    def __init__(self, provider: str, message: str, *, original: Exception | None = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.original = original


class BaseProvider(ABC):
    name: str  # set by each subclass, e.g. "openai" — matches ChatRequest.provider values

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResult:
        """Non-streaming call: full text + usage in one response."""

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming call: async generator of text deltas, usage on the final chunk (best-effort)."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Curated list of models this provider supports. No network call — static/config-driven."""

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Cheap, non-billed reachability check — never a paid completion call."""
