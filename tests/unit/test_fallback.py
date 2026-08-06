import pytest

from app.providers.base import ProviderCallError, ProviderResult
from app.providers.registry import ProviderRegistry
from app.schemas.chat import ChatMessage
from app.services.fallback import AllProvidersFailedError, FallbackExecutor
from tests.fakes import FakeProvider


def _messages() -> list[ChatMessage]:
    return [ChatMessage(role="user", content="hi")]


class TestFallbackExecutor:
    async def test_returns_first_success_without_trying_later_entries(self):
        primary_result = ProviderResult(content="from primary", model="m1", finish_reason="stop", usage=None)
        registry = ProviderRegistry(
            {
                "openai": FakeProvider(name="openai", result=primary_result),
                "anthropic": FakeProvider(name="anthropic", result=ProviderResult(content="unused", model="m2", finish_reason="stop", usage=None)),
            }
        )
        executor = FallbackExecutor(registry)

        served_by, result = await executor.run([("openai", "m1"), ("anthropic", "m2")], _messages())

        assert served_by == "openai"
        assert result.content == "from primary"

    async def test_falls_through_to_second_entry_on_first_failure(self):
        registry = ProviderRegistry(
            {
                "openai": FakeProvider(name="openai", error=ProviderCallError("openai", "down")),
                "anthropic": FakeProvider(
                    name="anthropic",
                    result=ProviderResult(content="from anthropic", model="m2", finish_reason="stop", usage=None),
                ),
            }
        )
        executor = FallbackExecutor(registry)

        served_by, result = await executor.run([("openai", "m1"), ("anthropic", "m2")], _messages())

        assert served_by == "anthropic"
        assert result.content == "from anthropic"

    async def test_unregistered_provider_in_chain_counts_as_failed_attempt(self):
        registry = ProviderRegistry(
            {"anthropic": FakeProvider(name="anthropic", result=ProviderResult(content="ok", model="m2", finish_reason="stop", usage=None))}
        )
        executor = FallbackExecutor(registry)

        served_by, result = await executor.run([("openai", "m1"), ("anthropic", "m2")], _messages())

        assert served_by == "anthropic"

    async def test_all_entries_failing_raises_with_all_attempts_recorded(self):
        registry = ProviderRegistry(
            {
                "openai": FakeProvider(name="openai", error=ProviderCallError("openai", "down")),
                "anthropic": FakeProvider(name="anthropic", error=ProviderCallError("anthropic", "also down")),
            }
        )
        executor = FallbackExecutor(registry)

        with pytest.raises(AllProvidersFailedError) as exc_info:
            await executor.run([("openai", "m1"), ("anthropic", "m2")], _messages())

        assert len(exc_info.value.attempts) == 2
        assert {a.provider for a in exc_info.value.attempts} == {"openai", "anthropic"}

    async def test_empty_chain_raises_all_providers_failed(self):
        executor = FallbackExecutor(ProviderRegistry({}))

        with pytest.raises(AllProvidersFailedError) as exc_info:
            await executor.run([], _messages())

        assert exc_info.value.attempts == []
