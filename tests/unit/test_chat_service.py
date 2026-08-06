from types import SimpleNamespace

import pytest

from app.config import Settings
from app.providers.base import ProviderCallError, ProviderResult, UsageInfo
from app.providers.registry import ProviderRegistry
from app.schemas.chat import ChatMessage, ChatRequest, FallbackTarget
from app.services.chat_service import ChatService, ProviderRequestError
from tests.fakes import FakeProvider


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


def _request(**overrides) -> ChatRequest:
    defaults = dict(provider="openai", model="fake-model", messages=[ChatMessage(role="user", content="hi")])
    defaults.update(overrides)
    return ChatRequest(**defaults)


def _fake_http_request():
    # ChatService only ever touches request.state (set chat_outcome, read request_id) —
    # a SimpleNamespace is enough, no need for a real Starlette Request in these unit tests.
    return SimpleNamespace(state=SimpleNamespace())


class TestChatService:
    async def test_complete_returns_response_with_usage(self):
        result = ProviderResult(
            content="hello!",
            model="fake-model",
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        registry = ProviderRegistry({"openai": FakeProvider(result=result)})
        service = ChatService(registry, _settings())
        http_request = _fake_http_request()

        response = await service.complete(http_request, _request())

        assert response.content == "hello!"
        assert response.provider == "openai"
        assert response.model == "fake-model"
        assert response.usage.total_tokens == 8
        assert response.fallback_used is False

    async def test_complete_stashes_chat_outcome_on_request_state(self):
        result = ProviderResult(
            content="hello!",
            model="fake-model",
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        registry = ProviderRegistry({"openai": FakeProvider(result=result)})
        service = ChatService(registry, _settings())
        http_request = _fake_http_request()

        await service.complete(http_request, _request())

        outcome = http_request.state.chat_outcome
        assert outcome.status == "success"
        assert outcome.provider == "openai"
        assert outcome.usage.total_tokens == 8

    async def test_all_providers_unregistered_raises_502(self):
        service = ChatService(ProviderRegistry({}), _settings(default_fallback_chain=""))
        http_request = _fake_http_request()

        with pytest.raises(ProviderRequestError) as exc_info:
            await service.complete(http_request, _request())

        assert exc_info.value.status_code == 502
        assert http_request.state.chat_outcome.status == "error"

    async def test_all_providers_call_failure_raises_502(self):
        registry = ProviderRegistry({"openai": FakeProvider(error=ProviderCallError("openai", "boom"))})
        service = ChatService(registry, _settings(default_fallback_chain=""))
        http_request = _fake_http_request()

        with pytest.raises(ProviderRequestError) as exc_info:
            await service.complete(http_request, _request())

        assert exc_info.value.status_code == 502
        assert http_request.state.chat_outcome.status == "error"

    async def test_complete_with_no_usage_returns_none_usage(self):
        result = ProviderResult(content="hi there", model="fake-model", finish_reason="stop", usage=None)
        registry = ProviderRegistry({"openai": FakeProvider(result=result)})
        service = ChatService(registry, _settings(default_fallback_chain=""))

        response = await service.complete(_fake_http_request(), _request())

        assert response.usage is None

    async def test_complete_populates_cost_for_known_model(self):
        result = ProviderResult(
            content="hi",
            model="gpt-4o-mini",
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000),
        )
        registry = ProviderRegistry({"openai": FakeProvider(result=result)})
        service = ChatService(registry, _settings(default_fallback_chain=""))
        chat_request = _request(provider="openai", model="gpt-4o-mini")

        response = await service.complete(_fake_http_request(), chat_request)

        # gpt-4o-mini pricing: $0.15/1M input + $0.60/1M output (see app/services/pricing.py)
        assert response.usage.cost_usd == pytest.approx(0.75)

    async def test_explicit_fallback_used_when_primary_fails(self):
        fallback_result = ProviderResult(content="from anthropic", model="claude-haiku-4-5", finish_reason="stop", usage=None)
        registry = ProviderRegistry(
            {
                "openai": FakeProvider(name="openai", error=ProviderCallError("openai", "down")),
                "anthropic": FakeProvider(name="anthropic", result=fallback_result),
            }
        )
        service = ChatService(registry, _settings())
        chat_request = _request(
            provider="openai",
            model="gpt-4o-mini",
            fallback=[FallbackTarget(provider="anthropic", model="claude-haiku-4-5")],
        )

        response = await service.complete(_fake_http_request(), chat_request)

        assert response.provider == "anthropic"
        assert response.fallback_used is True
        assert response.content == "from anthropic"

    async def test_default_fallback_chain_used_when_request_specifies_none(self):
        fallback_result = ProviderResult(content="from gemini", model="gemini-1.5-flash", finish_reason="stop", usage=None)
        registry = ProviderRegistry(
            {
                "openai": FakeProvider(name="openai", error=ProviderCallError("openai", "down")),
                "gemini": FakeProvider(name="gemini", result=fallback_result),
            }
        )
        service = ChatService(registry, _settings(default_fallback_chain="openai,gemini", gemini_default_model="gemini-1.5-flash"))
        chat_request = _request(provider="openai", model="gpt-4o-mini")

        response = await service.complete(_fake_http_request(), chat_request)

        assert response.provider == "gemini"
        assert response.fallback_used is True

    async def test_no_fallback_configured_and_primary_fails_raises_502(self):
        registry = ProviderRegistry({"openai": FakeProvider(error=ProviderCallError("openai", "down"))})
        service = ChatService(registry, _settings(default_fallback_chain=""))
        http_request = _fake_http_request()

        with pytest.raises(ProviderRequestError) as exc_info:
            await service.complete(http_request, _request())

        assert exc_info.value.status_code == 502
