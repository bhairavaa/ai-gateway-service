"""Google Gemini adapter — thin wrapper around langchain_google_genai.ChatGoogleGenerativeAI."""

from collections.abc import AsyncIterator

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI

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

_SUPPORTED_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]

# Google's public, free models-list REST endpoint. Hit directly with httpx rather than through
# the google-genai/langchain SDK internals, whose async models.list() surface has varied across
# package versions — this endpoint is stable and doesn't consume generation quota.
_MODELS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, api_key: str, *, timeout: float = 30.0):
        self._api_key = api_key
        self._timeout = timeout

    def _build_chat_model(
        self, model: str, *, temperature: float | None = None, max_tokens: int | None = None
    ) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=self._api_key,
            temperature=temperature if temperature is not None else 0.7,
            max_output_tokens=max_tokens,
            timeout=self._timeout,
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
        except Exception as exc:  # noqa: BLE001 - normalize any google.* SDK exception into ProviderCallError
            raise ProviderCallError(self.name, str(exc), original=exc) from exc

        return ProviderResult(
            content=normalize_content(ai_message.content),
            model=model,
            finish_reason=(ai_message.response_metadata or {}).get("finish_reason"),
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
                # Gemini's streaming usage reporting has historically lagged OpenAI/Anthropic — treated
                # as best-effort: if the final chunk carries no usage_metadata, the caller (chat_service /
                # cost tracking) simply sees usage=None rather than a guessed value.
                final_usage = usage_from_metadata(chunk.usage_metadata) or final_usage
                delta = normalize_content(chunk.content)
                if delta:
                    yield StreamChunk(delta=delta, is_final=False)
            yield StreamChunk(delta="", is_final=True, usage=final_usage)
        except Exception as exc:  # noqa: BLE001
            raise ProviderCallError(self.name, str(exc), original=exc) from exc

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=m, provider=self.name) for m in _SUPPORTED_MODELS]

    async def health_check(self) -> HealthStatus:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(_MODELS_ENDPOINT, params={"key": self._api_key})
                resp.raise_for_status()
            return HealthStatus(provider=self.name, healthy=True)
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(provider=self.name, healthy=False, detail=str(exc))
