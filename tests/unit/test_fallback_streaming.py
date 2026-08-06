import pytest

from app.providers.base import ProviderCallError, StreamChunk
from app.providers.registry import ProviderRegistry
from app.schemas.chat import ChatMessage
from app.services.fallback import AllProvidersFailedError, FallbackExecutor
from tests.fakes import FakeProvider


def _messages() -> list[ChatMessage]:
    return [ChatMessage(role="user", content="hi")]


class TestFallbackExecutorStreaming:
    async def test_streams_all_chunks_from_first_successful_provider(self):
        chunks = [
            StreamChunk(delta="Hel", is_final=False),
            StreamChunk(delta="lo", is_final=False),
            StreamChunk(delta="", is_final=True, usage=None),
        ]
        registry = ProviderRegistry({"openai": FakeProvider(name="openai", stream_chunks=chunks)})
        executor = FallbackExecutor(registry)

        collected = [item async for item in executor.run_streaming([("openai", "m1")], _messages())]

        assert [c.delta for _, c in collected] == ["Hel", "lo", ""]
        assert all(provider == "openai" for provider, _ in collected)

    async def test_falls_back_before_first_chunk_if_primary_produces_nothing(self):
        registry = ProviderRegistry(
            {
                "openai": FakeProvider(name="openai", error=ProviderCallError("openai", "down")),
                "anthropic": FakeProvider(
                    name="anthropic", stream_chunks=[StreamChunk(delta="hi", is_final=False)]
                ),
            }
        )
        executor = FallbackExecutor(registry)

        collected = [
            item async for item in executor.run_streaming([("openai", "m1"), ("anthropic", "m2")], _messages())
        ]

        assert collected[0][0] == "anthropic"

    async def test_mid_stream_failure_propagates_without_falling_back(self):
        class FlakyProvider(FakeProvider):
            async def stream(self, messages, model, *, temperature=None, max_tokens=None):
                yield StreamChunk(delta="partial", is_final=False)
                raise ProviderCallError("openai", "connection dropped mid-stream")

        registry = ProviderRegistry(
            {
                "openai": FlakyProvider(name="openai"),
                "anthropic": FakeProvider(
                    name="anthropic", stream_chunks=[StreamChunk(delta="should not be used", is_final=False)]
                ),
            }
        )
        executor = FallbackExecutor(registry)

        collected = []
        with pytest.raises(ProviderCallError):
            async for item in executor.run_streaming([("openai", "m1"), ("anthropic", "m2")], _messages()):
                collected.append(item)

        # Got the partial chunk from openai, then the error — never silently switched to anthropic.
        assert len(collected) == 1
        assert collected[0][0] == "openai"

    async def test_all_providers_failing_before_first_chunk_raises(self):
        registry = ProviderRegistry(
            {"openai": FakeProvider(name="openai", error=ProviderCallError("openai", "down"))}
        )
        executor = FallbackExecutor(registry)

        with pytest.raises(AllProvidersFailedError):
            async for _ in executor.run_streaming([("openai", "m1")], _messages()):
                pass
