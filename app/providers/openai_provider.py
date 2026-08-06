"""OpenAI adapter — thin wrapper around langchain_openai.ChatOpenAI.

Translates the gateway's ChatMessage list to LangChain message objects, calls
the stable LangChain interface, and reads usage from AIMessage.usage_metadata
(the standardized shape LangChain's stable provider packages populate) instead
of parsing OpenAI's raw response ourselves.
"""

from collections.abc import AsyncIterator

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

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

# Curated static list — list_models() must not make a network call (see BaseProvider docstring),
# so this is hand-maintained rather than fetched from OpenAI's /models endpoint.
_SUPPORTED_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, api_key: str, *, timeout: float = 30.0, base_url: str | None = None):
        self._api_key = api_key
        self._timeout = timeout
        # None -> the SDK's own default (api.openai.com). Set for OpenAI-compatible backends
        # (e.g. OpenRouter) — see Settings.openai_base_url.
        self._base_url = base_url

    def _build_chat_model(
        self,
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream_usage: bool = False,
    ) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            api_key=self._api_key,
            base_url=self._base_url,
            temperature=temperature if temperature is not None else 0.7,
            max_tokens=max_tokens,
            timeout=self._timeout,
            # OpenAI only reports usage_metadata on the final streamed chunk when this is set;
            # non-streaming responses always include it, so this flag is irrelevant to generate().
            stream_usage=stream_usage,
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
        except Exception as exc:  # noqa: BLE001 - normalize any openai.* SDK exception into ProviderCallError
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
        chat_model = self._build_chat_model(model, temperature=temperature, max_tokens=max_tokens, stream_usage=True)
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
        return [ModelInfo(id=m, provider=self.name) for m in _SUPPORTED_MODELS]

    async def health_check(self) -> HealthStatus:
        # models.list() is a free/non-billed endpoint — deliberately not a chat completion,
        # per BaseProvider.health_check's "never a paid call" contract.
        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._timeout)
        try:
            await client.models.list()
            return HealthStatus(provider=self.name, healthy=True)
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(provider=self.name, healthy=False, detail=str(exc))
