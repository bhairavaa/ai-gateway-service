from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.providers.base import ProviderCallError
from app.providers.claude_provider import ClaudeProvider
from app.schemas.chat import ChatMessage


class TestClaudeProviderGenerate:
    async def test_generate_maps_usage_metadata_and_stop_reason(self):
        provider = ClaudeProvider(api_key="test-key")
        fake_response = AIMessage(
            content="hello from claude!",
            response_metadata={"stop_reason": "end_turn"},
            usage_metadata={"input_tokens": 12, "output_tokens": 6, "total_tokens": 18},
        )

        with patch("app.providers.claude_provider.ChatAnthropic") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(return_value=fake_response)

            result = await provider.generate([ChatMessage(role="user", content="hi")], "claude-haiku-4-5")

        assert result.content == "hello from claude!"
        assert result.finish_reason == "end_turn"
        assert result.usage.prompt_tokens == 12
        assert result.usage.completion_tokens == 6
        assert result.usage.total_tokens == 18

    async def test_generate_defaults_max_tokens_when_not_provided(self):
        provider = ClaudeProvider(api_key="test-key")
        fake_response = AIMessage(content="hi", response_metadata={}, usage_metadata=None)

        with patch("app.providers.claude_provider.ChatAnthropic") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(return_value=fake_response)

            await provider.generate([ChatMessage(role="user", content="hi")], "claude-haiku-4-5")

            # Anthropic requires max_tokens on every request; the provider must supply a default
            # when the caller doesn't pass one, rather than sending max_tokens=None upstream.
            _, kwargs = mock_chat_model_cls.call_args
            assert kwargs["max_tokens"] == 1024

    async def test_generate_wraps_sdk_exception_as_provider_call_error(self):
        provider = ClaudeProvider(api_key="test-key")

        with patch("app.providers.claude_provider.ChatAnthropic") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("upstream boom"))

            with pytest.raises(ProviderCallError) as exc_info:
                await provider.generate([ChatMessage(role="user", content="hi")], "claude-haiku-4-5")

        assert exc_info.value.provider == "anthropic"

    async def test_generate_handles_list_shaped_content_blocks(self):
        # Regression: Anthropic doesn't always return AIMessage.content as a plain string —
        # it can be a list of content blocks, e.g. when a response mixes text with citations
        # or other block types. This crashed ChatResponse's pydantic validation in production
        # (content expected `str`, got `list`) before app.providers.common.normalize_content.
        provider = ClaudeProvider(api_key="test-key")
        fake_response = AIMessage(
            content=[
                {"type": "text", "text": "The answer is "},
                {"type": "text", "text": "42."},
            ],
            response_metadata={"stop_reason": "end_turn"},
            usage_metadata=None,
        )

        with patch("app.providers.claude_provider.ChatAnthropic") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(return_value=fake_response)

            result = await provider.generate([ChatMessage(role="user", content="hi")], "claude-haiku-4-5")

        assert result.content == "The answer is 42."
        assert isinstance(result.content, str)


class TestClaudeProviderListModels:
    async def test_list_models_returns_static_list_without_network_call(self):
        provider = ClaudeProvider(api_key="test-key")

        models = await provider.list_models()

        assert any(m.id == "claude-haiku-4-5" for m in models)
        assert all(m.provider == "anthropic" for m in models)
