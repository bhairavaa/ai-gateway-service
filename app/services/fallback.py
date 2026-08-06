"""FallbackExecutor — walks a configurable provider/model chain until one
succeeds.

`run()` is for non-streaming calls. `run_streaming()` has necessarily different
fallback semantics: once a provider's first chunk has been yielded to the
caller (and, in the route, already sent to the client as SSE bytes), we can't
retract it — so fallback there only applies *before* the first chunk, never
mid-stream. A failure after that point is the committed provider's problem to
surface, not something this executor silently papers over.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.providers.base import ProviderCallError, ProviderResult, StreamChunk
from app.providers.registry import ProviderNotRegisteredError, ProviderRegistry
from app.schemas.chat import ChatMessage


@dataclass(frozen=True)
class FallbackAttempt:
    provider: str
    model: str
    error: str


class AllProvidersFailedError(Exception):
    def __init__(self, attempts: list[FallbackAttempt]):
        self.attempts = attempts
        summary = "; ".join(f"{a.provider}/{a.model}: {a.error}" for a in attempts)
        super().__init__(f"All providers in the fallback chain failed: {summary}")


class FallbackExecutor:
    def __init__(self, registry: ProviderRegistry):
        self._registry = registry

    async def run(
        self,
        chain: list[tuple[str, str]],
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, ProviderResult]:
        """Try each (provider, model) pair in order; returns (provider_name_that_served_it,
        result) on the first success. A provider that isn't registered counts as a failed
        attempt, exactly like an upstream call failure — both just move to the next entry.
        """
        attempts: list[FallbackAttempt] = []
        for provider_name, model in chain:
            try:
                provider = self._registry.get(provider_name)
            except ProviderNotRegisteredError as exc:
                attempts.append(FallbackAttempt(provider_name, model, str(exc)))
                continue

            try:
                result = await provider.generate(
                    messages, model, temperature=temperature, max_tokens=max_tokens
                )
                return provider_name, result
            except ProviderCallError as exc:
                attempts.append(FallbackAttempt(provider_name, model, str(exc)))
                continue

        raise AllProvidersFailedError(attempts)

    async def run_streaming(
        self,
        chain: list[tuple[str, str]],
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[tuple[str, StreamChunk]]:
        """Yields (provider_name, StreamChunk). Tries each provider's stream() in turn,
        but only *before* its first chunk is pulled successfully. Once a provider yields
        a first chunk, we're committed — any failure after that propagates as-is (the
        caller sees a ProviderCallError mid-iteration) rather than silently retrying.
        """
        attempts: list[FallbackAttempt] = []
        for provider_name, model in chain:
            try:
                provider = self._registry.get(provider_name)
            except ProviderNotRegisteredError as exc:
                attempts.append(FallbackAttempt(provider_name, model, str(exc)))
                continue

            stream_iter = provider.stream(
                messages, model, temperature=temperature, max_tokens=max_tokens
            ).__aiter__()
            try:
                first_chunk = await stream_iter.__anext__()
            except StopAsyncIteration:
                attempts.append(FallbackAttempt(provider_name, model, "stream produced no output"))
                continue
            except ProviderCallError as exc:
                attempts.append(FallbackAttempt(provider_name, model, str(exc)))
                continue

            # Committed to this provider for the rest of the request.
            yield provider_name, first_chunk
            async for chunk in stream_iter:
                yield provider_name, chunk
            return

        raise AllProvidersFailedError(attempts)
