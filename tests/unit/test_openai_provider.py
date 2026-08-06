from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.providers.base import ProviderCallError
from app.providers.openai_provider import OpenAIProvider
from app.schemas.chat import ChatMessage


class TestOpenAIProviderGenerate:
    async def test_generate_maps_usage_metadata_and_content(self):
        provider = OpenAIProvider(api_key="test-key")
        fake_response = AIMessage(
            content="hello!",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        )

        with patch("app.providers.openai_provider.ChatOpenAI") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(return_value=fake_response)

            result = await provider.generate(
                [ChatMessage(role="user", content="hi")], "gpt-4o-mini"
            )

        assert result.content == "hello!"
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 4
        assert result.usage.total_tokens == 14

    async def test_generate_wraps_sdk_exception_as_provider_call_error(self):
        provider = OpenAIProvider(api_key="test-key")

        with patch("app.providers.openai_provider.ChatOpenAI") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("upstream boom"))

            with pytest.raises(ProviderCallError) as exc_info:
                await provider.generate([ChatMessage(role="user", content="hi")], "gpt-4o-mini")

        assert exc_info.value.provider == "openai"

    async def test_generate_with_no_usage_metadata_returns_none_usage(self):
        provider = OpenAIProvider(api_key="test-key")
        fake_response = AIMessage(content="hi", response_metadata={}, usage_metadata=None)

        with patch("app.providers.openai_provider.ChatOpenAI") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(return_value=fake_response)

            result = await provider.generate([ChatMessage(role="user", content="hi")], "gpt-4o-mini")

        assert result.usage is None


class TestOpenAIProviderBaseUrl:
    async def test_custom_base_url_is_passed_to_chat_model(self):
        provider = OpenAIProvider(api_key="sk-or-v1-x", base_url="https://openrouter.ai/api/v1")
        fake_response = AIMessage(content="hi", response_metadata={}, usage_metadata=None)

        with patch("app.providers.openai_provider.ChatOpenAI") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(return_value=fake_response)
            await provider.generate([ChatMessage(role="user", content="hi")], "openai/gpt-4o-mini")

        _, kwargs = mock_chat_model_cls.call_args
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"

    async def test_default_base_url_is_none(self):
        provider = OpenAIProvider(api_key="sk-x")
        fake_response = AIMessage(content="hi", response_metadata={}, usage_metadata=None)

        with patch("app.providers.openai_provider.ChatOpenAI") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(return_value=fake_response)
            await provider.generate([ChatMessage(role="user", content="hi")], "gpt-4o-mini")

        _, kwargs = mock_chat_model_cls.call_args
        assert kwargs["base_url"] is None


class TestOpenAIProviderListModels:
    async def test_list_models_returns_static_list_without_network_call(self):
        provider = OpenAIProvider(api_key="test-key")

        models = await provider.list_models()

        assert any(m.id == "gpt-4o-mini" for m in models)
        assert all(m.provider == "openai" for m in models)
