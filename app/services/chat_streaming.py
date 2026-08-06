"""Streaming (SSE) chat completions.

Shares the same fallback-chain construction and provider registry as the
non-streaming path (app/services/chat_service.py) — only the response
transport and the fallback-commitment semantics differ (see
app/services/fallback.py::run_streaming). The pre-first-chunk fallback attempt
happens as a plain awaited call here, *before* the route returns a
StreamingResponse — so a chain that's exhausted before any provider commits
still produces a normal HTTP error response, not a broken/empty stream.
"""

import time
import uuid
from collections.abc import AsyncIterator

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from app.config import Settings
from app.providers.base import ProviderCallError, StreamChunk, UsageInfo
from app.providers.registry import ProviderRegistry
from app.schemas.chat import ChatRequest
from app.schemas.errors import ErrorDetail, GatewayErrorResponse
from app.services.chat_service import ChatOutcome, build_fallback_chain
from app.services.fallback import AllProvidersFailedError, FallbackExecutor
from app.services.outcome_persistence import persist_chat_outcome
from app.services.pricing import calculate_cost
from app.utils.sse import SSE_DONE, format_sse_event


async def stream_chat_completion(
    http_request: Request, chat_request: ChatRequest, registry: ProviderRegistry, settings: Settings
) -> StreamingResponse:
    chain = build_fallback_chain(chat_request, settings)
    executor = FallbackExecutor(registry)
    stream_iter = executor.run_streaming(
        chain, chat_request.messages, temperature=chat_request.temperature, max_tokens=chat_request.max_tokens
    ).__aiter__()

    request_id = getattr(http_request.state, "request_id", None) or uuid.uuid4().hex

    try:
        first_provider, first_chunk = await stream_iter.__anext__()
    except AllProvidersFailedError as exc:
        await persist_chat_outcome(
            request_id=request_id,
            auth_context=getattr(http_request.state, "auth_context", None),
            outcome=ChatOutcome(
                provider=chat_request.provider,
                model=chat_request.model,
                status="error",
                usage=None,
                cost_usd=None,
                error_message=str(exc),
            ),
            duration_ms=0,
        )
        raise HTTPException(
            status_code=502,
            detail=GatewayErrorResponse(error=ErrorDetail(type="provider_error", message=str(exc))).model_dump(),
        ) from exc

    generator = _sse_generator(
        http_request, chat_request, stream_iter, request_id, first_provider, first_chunk
    )
    return StreamingResponse(generator, media_type="text/event-stream")


async def _sse_generator(
    http_request: Request,
    chat_request: ChatRequest,
    stream_iter: AsyncIterator,
    request_id: str,
    first_provider: str,
    first_chunk: StreamChunk,
) -> AsyncIterator[str]:
    start = time.perf_counter()
    served_by = first_provider
    final_usage: UsageInfo | None = None
    error_message: str | None = None
    chunk = first_chunk

    try:
        while True:
            if chunk.delta:
                yield format_sse_event(
                    {
                        "id": request_id,
                        "provider": served_by,
                        "model": chat_request.model,
                        "delta": chunk.delta,
                        "finish_reason": chunk.finish_reason,
                    }
                )
            if chunk.is_final:
                final_usage = chunk.usage
                break
            served_by, chunk = await stream_iter.__anext__()
    except StopAsyncIteration:
        pass
    except ProviderCallError as exc:
        # Committed to `served_by` already (its first chunk was sent) — no silent fallback
        # mid-stream, just a terminal error event (see fallback.py's run_streaming docstring).
        error_message = str(exc)
        yield format_sse_event({"error": {"type": "provider_error", "message": error_message}})
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        cost_usd = calculate_cost(served_by, chat_request.model, final_usage) if final_usage else None

        if final_usage is not None:
            yield format_sse_event(
                {
                    "usage": {
                        "prompt_tokens": final_usage.prompt_tokens,
                        "completion_tokens": final_usage.completion_tokens,
                        "total_tokens": final_usage.total_tokens,
                        "cost_usd": cost_usd,
                    }
                }
            )
        yield SSE_DONE

        status = "error" if error_message else ("fallback" if served_by != chat_request.provider else "success")
        await persist_chat_outcome(
            request_id=request_id,
            auth_context=getattr(http_request.state, "auth_context", None),
            outcome=ChatOutcome(
                provider=served_by,
                model=chat_request.model,
                status=status,
                usage=final_usage,
                cost_usd=cost_usd,
                error_message=error_message,
            ),
            duration_ms=duration_ms,
        )
