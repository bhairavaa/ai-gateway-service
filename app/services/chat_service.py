"""Business logic layer for chat completions."""

import uuid
from dataclasses import dataclass

from starlette.requests import Request

from app.config import Settings
from app.providers.base import UsageInfo
from app.providers.registry import ProviderRegistry
from app.schemas.chat import ChatRequest, ChatResponse, UsageResponse
from app.services.fallback import AllProvidersFailedError, FallbackExecutor
from app.services.pricing import calculate_cost


@dataclass
class ChatOutcome:
    """Stashed on request.state so CostTrackingMiddleware can persist request/usage
    metadata after the response is sent, without that middleware needing to know
    anything about how a chat completion is actually produced.
    """

    provider: str
    model: str
    status: str  # "success" | "fallback" | "error"
    usage: UsageInfo | None
    cost_usd: float | None
    error_message: str | None = None


class ProviderRequestError(Exception):
    """Raised when the whole fallback chain is exhausted. Carries an HTTP status
    code so the route layer can translate it without knowing why the call failed.
    """

    def __init__(self, message: str, *, status_code: int):
        super().__init__(message)
        self.status_code = status_code


_DEFAULT_MODEL_BY_PROVIDER_ATTR = {
    "openai": "openai_default_model",
    "anthropic": "anthropic_default_model",
    "gemini": "gemini_default_model",
    "ollama": "ollama_default_model",
}


def build_fallback_chain(chat_request: ChatRequest, settings: Settings) -> list[tuple[str, str]]:
    primary = (chat_request.provider, chat_request.model)
    if chat_request.fallback:
        return [primary] + [(f.provider, f.model) for f in chat_request.fallback]

    # No explicit per-request fallback — fall back to the configured default chain
    # (Settings.default_fallback_chain), skipping the primary provider itself and using
    # each fallback provider's configured default model.
    extras = [
        (name, getattr(settings, _DEFAULT_MODEL_BY_PROVIDER_ATTR[name]))
        for name in settings.default_fallback_chain_list
        if name != chat_request.provider and name in _DEFAULT_MODEL_BY_PROVIDER_ATTR
    ]
    return [primary] + extras


class ChatService:
    def __init__(self, provider_registry: ProviderRegistry, settings: Settings):
        self._registry = provider_registry
        self._settings = settings
        self._fallback = FallbackExecutor(provider_registry)

    async def complete(self, request: Request, chat_request: ChatRequest) -> ChatResponse:
        chain = build_fallback_chain(chat_request, self._settings)

        try:
            served_by, result = await self._fallback.run(
                chain,
                chat_request.messages,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
            )
        except AllProvidersFailedError as exc:
            request.state.chat_outcome = ChatOutcome(
                provider=chat_request.provider,
                model=chat_request.model,
                status="error",
                usage=None,
                cost_usd=None,
                error_message=str(exc),
            )
            raise ProviderRequestError(str(exc), status_code=502) from exc

        fallback_used = served_by != chat_request.provider
        cost_usd = calculate_cost(served_by, result.model, result.usage)
        request.state.chat_outcome = ChatOutcome(
            provider=served_by,
            model=result.model,
            status="fallback" if fallback_used else "success",
            usage=result.usage,
            cost_usd=cost_usd,
        )

        usage = None
        if result.usage:
            usage = UsageResponse(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
                cost_usd=cost_usd,
            )

        # Reuse the gateway-assigned request id when available so ChatResponse.id lines up
        # with the X-Request-ID header the client already sees.
        response_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex

        return ChatResponse(
            id=response_id,
            provider=served_by,
            model=result.model,
            content=result.content,
            finish_reason=result.finish_reason,
            usage=usage,
            fallback_used=fallback_used,
        )
