from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.providers.base import ProviderCallError
from app.providers.gemini_provider import GeminiProvider
from app.schemas.chat import ChatMessage


class TestGeminiProviderGenerate:
    async def test_generate_maps_usage_metadata_and_content(self):
        provider = GeminiProvider(api_key="test-key")
        fake_response = AIMessage(
            content="hello from gemini!",
            response_metadata={"finish_reason": "STOP"},
            usage_metadata={"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
        )

        with patch("app.providers.gemini_provider.ChatGoogleGenerativeAI") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(return_value=fake_response)

            result = await provider.generate([ChatMessage(role="user", content="hi")], "gemini-1.5-flash")

        assert result.content == "hello from gemini!"
        assert result.finish_reason == "STOP"
        assert result.usage.total_tokens == 10

    async def test_generate_with_no_usage_metadata_returns_none_usage(self):
        # Gemini's streaming/response usage reporting is best-effort — generate() must tolerate
        # a response with no usage_metadata rather than raising.
        provider = GeminiProvider(api_key="test-key")
        fake_response = AIMessage(content="hi", response_metadata={}, usage_metadata=None)

        with patch("app.providers.gemini_provider.ChatGoogleGenerativeAI") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(return_value=fake_response)

            result = await provider.generate([ChatMessage(role="user", content="hi")], "gemini-1.5-flash")

        assert result.usage is None

    async def test_generate_wraps_sdk_exception_as_provider_call_error(self):
        provider = GeminiProvider(api_key="test-key")

        with patch("app.providers.gemini_provider.ChatGoogleGenerativeAI") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("upstream boom"))

            with pytest.raises(ProviderCallError) as exc_info:
                await provider.generate([ChatMessage(role="user", content="hi")], "gemini-1.5-flash")

        assert exc_info.value.provider == "gemini"


class TestGeminiProviderListModels:
    async def test_list_models_returns_static_list_without_network_call(self):
        provider = GeminiProvider(api_key="test-key")

        models = await provider.list_models()

        assert any(m.id == "gemini-1.5-flash" for m in models)
        assert all(m.provider == "gemini" for m in models)


class TestGeminiProviderHealthCheck:
    async def test_health_check_healthy_when_endpoint_responds_ok(self):
        provider = GeminiProvider(api_key="test-key")
        mock_response = AsyncMock()
        mock_response.raise_for_status = lambda: None

        with patch("app.providers.gemini_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(return_value=mock_response)

            status = await provider.health_check()

        assert status.healthy is True
        assert status.provider == "gemini"

    async def test_health_check_unhealthy_on_request_failure(self):
        provider = GeminiProvider(api_key="test-key")

        with patch("app.providers.gemini_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=RuntimeError("network down"))

            status = await provider.health_check()

        assert status.healthy is False
        assert status.detail is not None
