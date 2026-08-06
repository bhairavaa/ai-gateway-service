"""Ollama adapter — thin wrapper around langchain_ollama.ChatOllama.

Unlike the cloud providers, Ollama has no API key and no fixed model catalog —
`list_models()` deliberately breaks from BaseProvider's "static list, no network
call" norm and queries the local daemon's /api/tags instead, since a hardcoded
list would almost certainly be wrong for whatever models the user has actually
pulled locally.
"""

from collections.abc import AsyncIterator

import httpx
from langchain_ollama import ChatOllama

from app.providers.base import (
    BaseProvider,
    HealthStatus,
    ModelInfo,
    ProviderCallError,
    ProviderResult,
    StreamChunk,
    UsageInfo,
)
from app.providers.common import normalize_content, to_langchain_messages, usage_from_metadata
from app.schemas.chat import ChatMessage


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url: str, *, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _build_chat_model(
        self, model: str, *, temperature: float | None = None, max_tokens: int | None = None
    ) -> ChatOllama:
        return ChatOllama(
            model=model,
            base_url=self._base_url,
            temperature=temperature if temperature is not None else 0.7,
            num_predict=max_tokens,  # Ollama's name for the output-token-limit parameter
        )

    async def generate(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResult:
        chat_model = self._build_chat_model(model, temperature=temperature, max_tokens=max_tokens)
        try:
            ai_message = await chat_model.ainvoke(to_langchain_messages(messages))
        except Exception as exc:  # noqa: BLE001 - normalize any ollama client exception into ProviderCallError
            raise ProviderCallError(self.name, str(exc), original=exc) from exc

        return ProviderResult(
            content=normalize_content(ai_message.content),
            model=model,
            finish_reason=(ai_message.response_metadata or {}).get("done_reason"),
            usage=usage_from_metadata(ai_message.usage_metadata),
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        chat_model = self._build_chat_model(model, temperature=temperature, max_tokens=max_tokens)
        lc_messages = to_langchain_messages(messages)
        try:
            final_usage: UsageInfo | None = None
            async for chunk in chat_model.astream(lc_messages):
                final_usage = usage_from_metadata(chunk.usage_metadata) or final_usage
                delta = normalize_content(chunk.content)
                if delta:
                    yield StreamChunk(delta=delta, is_final=False)
            yield StreamChunk(delta="", is_final=True, usage=final_usage)
        except Exception as exc:  # noqa: BLE001
            raise ProviderCallError(self.name, str(exc), original=exc) from exc

    async def list_models(self) -> list[ModelInfo]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
            return [ModelInfo(id=m["name"], provider=self.name) for m in data.get("models", [])]
        except Exception:  # noqa: BLE001
            # Local daemon unreachable — this is a discovery endpoint, not a critical path
            # (health_check is what surfaces reachability problems), so fail soft to an empty list.
            return []

    async def health_check(self) -> HealthStatus:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
            return HealthStatus(provider=self.name, healthy=True)
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(provider=self.name, healthy=False, detail=str(exc))
