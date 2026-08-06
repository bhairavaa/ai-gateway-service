"""Anthropic Claude adapter — thin wrapper around langchain_anthropic.ChatAnthropic."""

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic
from langchain_anthropic import ChatAnthropic

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

_SUPPORTED_MODELS = ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"]

# Anthropic's API requires max_tokens on every request (unlike OpenAI, where it's optional) —
# this is the fallback used when the caller doesn't specify one.
_DEFAULT_MAX_TOKENS = 1024


class ClaudeProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str, *, timeout: float = 30.0):
        self._api_key = api_key
        self._timeout = timeout

    def _build_chat_model(
        self, model: str, *, temperature: float | None = None, max_tokens: int | None = None
    ) -> ChatAnthropic:
        return ChatAnthropic(
            model=model,
            api_key=self._api_key,
            temperature=temperature if temperature is not None else 0.7,
            max_tokens=max_tokens or _DEFAULT_MAX_TOKENS,
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
        except Exception as exc:  # noqa: BLE001 - normalize any anthropic.* SDK exception into ProviderCallError
            raise ProviderCallError(self.name, str(exc), original=exc) from exc

        return ProviderResult(
            content=normalize_content(ai_message.content),
            model=model,
            finish_reason=(ai_message.response_metadata or {}).get("stop_reason"),
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
                # Unlike OpenAI, Anthropic always includes usage on the final streamed chunk with
                # no special flag needed — input_tokens arrives on message_start, output_tokens on
                # message_delta/message_stop, and LangChain aggregates both onto usage_metadata.
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
        # models.list() is a free/non-billed endpoint — deliberately not a chat completion.
        client = AsyncAnthropic(api_key=self._api_key, timeout=self._timeout)
        try:
            await client.models.list()
            return HealthStatus(provider=self.name, healthy=True)
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(provider=self.name, healthy=False, detail=str(exc))
