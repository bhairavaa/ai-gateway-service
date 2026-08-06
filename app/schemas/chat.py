"""Chat request/response DTOs.

`ChatMessage` doubles as the shared vocabulary between the API layer and the
provider layer (see app/providers/base.py) — one shape for a role+content pair
is enough here, so it isn't duplicated as a separate provider-side dataclass.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class FallbackTarget(BaseModel):
    provider: Literal["openai", "anthropic", "gemini", "ollama"]
    model: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                    "fallback": [{"provider": "anthropic", "model": "claude-haiku-4-5"}],
                    "stream": False,
                }
            ]
        }
    )

    provider: Literal["openai", "anthropic", "gemini", "ollama"]
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    # Explicit per-request override of the fallback chain, tried in order after the primary
    # (provider, model) fails. Left empty, the gateway falls back to Settings.default_fallback_chain
    # (see app/services/chat_service.py::build_fallback_chain) rather than trying nothing else.
    fallback: list[FallbackTarget] = Field(default_factory=list)
    stream: bool = False


class UsageResponse(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # None means "not computed" (unmapped model pricing), never a silent free ride the way
    # a default of 0 would imply. See app/services/pricing.py.
    cost_usd: float | None = None


class ChatResponse(BaseModel):
    id: str
    provider: str  # the provider that actually served the request — may differ from the request's
    model: str
    content: str
    finish_reason: str | None
    usage: UsageResponse | None
    fallback_used: bool = False
