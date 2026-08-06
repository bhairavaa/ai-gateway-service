"""Shared test doubles — a fake BaseProvider implementation used by both unit
tests (chat_service) and integration tests (chat endpoint) so neither hits a
real network/SDK.
"""

from collections.abc import AsyncIterator

from app.providers.base import (
    BaseProvider,
    HealthStatus,
    ModelInfo,
    ProviderResult,
    StreamChunk,
)
from app.schemas.chat import ChatMessage


class FakeProvider(BaseProvider):
    def __init__(
        self,
        *,
        name: str = "openai",
        result: ProviderResult | None = None,
        stream_chunks: list[StreamChunk] | None = None,
        error: Exception | None = None,
    ):
        self.name = name
        self._result = result
        self._stream_chunks = stream_chunks or []
        self._error = error

    async def generate(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResult:
        if self._error:
            raise self._error
        return self._result

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        if self._error:
            raise self._error
        for chunk in self._stream_chunks:
            yield chunk

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="fake-model", provider=self.name)]

    async def health_check(self) -> HealthStatus:
        return HealthStatus(provider=self.name, healthy=True)
