from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.providers.base import ProviderCallError
from app.providers.ollama_provider import OllamaProvider
from app.schemas.chat import ChatMessage


class TestOllamaProviderGenerate:
    async def test_generate_maps_usage_metadata_and_done_reason(self):
        provider = OllamaProvider(base_url="http://localhost:11434")
        fake_response = AIMessage(
            content="hello from ollama!",
            response_metadata={"done_reason": "stop"},
            usage_metadata={"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
        )

        with patch("app.providers.ollama_provider.ChatOllama") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(return_value=fake_response)

            result = await provider.generate([ChatMessage(role="user", content="hi")], "llama3.1")

        assert result.content == "hello from ollama!"
        assert result.finish_reason == "stop"
        assert result.usage.total_tokens == 10

    async def test_generate_wraps_sdk_exception_as_provider_call_error(self):
        provider = OllamaProvider(base_url="http://localhost:11434")

        with patch("app.providers.ollama_provider.ChatOllama") as mock_chat_model_cls:
            mock_chat_model_cls.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("connection refused"))

            with pytest.raises(ProviderCallError) as exc_info:
                await provider.generate([ChatMessage(role="user", content="hi")], "llama3.1")

        assert exc_info.value.provider == "ollama"


class TestOllamaProviderListModels:
    async def test_list_models_parses_local_tags_endpoint(self):
        provider = OllamaProvider(base_url="http://localhost:11434")
        mock_response = AsyncMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json = lambda: {"models": [{"name": "llama3.1"}, {"name": "mistral"}]}

        with patch("app.providers.ollama_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(return_value=mock_response)

            models = await provider.list_models()

        assert {m.id for m in models} == {"llama3.1", "mistral"}
        assert all(m.provider == "ollama" for m in models)

    async def test_list_models_returns_empty_list_when_daemon_unreachable(self):
        provider = OllamaProvider(base_url="http://localhost:11434")

        with patch("app.providers.ollama_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=RuntimeError("connection refused"))

            models = await provider.list_models()

        assert models == []


class TestOllamaProviderHealthCheck:
    async def test_health_check_unhealthy_when_daemon_unreachable(self):
        provider = OllamaProvider(base_url="http://localhost:11434")

        with patch("app.providers.ollama_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=RuntimeError("connection refused"))

            status = await provider.health_check()

        assert status.healthy is False
        assert status.provider == "ollama"
