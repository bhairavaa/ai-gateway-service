"""Shared persistence for a completed chat request's outcome.

Used by both CostTrackingMiddleware (non-streaming path, where the outcome is
known by the time the route returns) and the SSE streaming generator (where the
outcome — final usage, whether a mid-stream error occurred — is only known once
the generator itself finishes, well after CostTrackingMiddleware's call_next()
has already returned with just the StreamingResponse object).
"""

import logging

from app.auth.base import AuthContext
from app.db.repositories.request_log_repository import SqliteRequestLogRepository
from app.db.repositories.usage_repository import SqliteUsageRepository
from app.db.session import AsyncSessionLocal
from app.services.chat_service import ChatOutcome

logger = logging.getLogger("app.cost_tracking")


async def persist_chat_outcome(
    *, request_id: str, auth_context: AuthContext | None, outcome: ChatOutcome, duration_ms: int
) -> None:
    try:
        async with AsyncSessionLocal() as session:
            request_log_repo = SqliteRequestLogRepository(session)
            usage_repo = SqliteUsageRepository(session)

            log_entry = await request_log_repo.create(
                request_id=request_id,
                api_key_id=auth_context.api_key_id if auth_context else None,
                provider=outcome.provider,
                model=outcome.model,
                status=outcome.status,
                duration_ms=duration_ms,
                error_message=outcome.error_message,
            )
            if outcome.usage is not None:
                await usage_repo.record(
                    request_log_id=log_entry.id,
                    provider=outcome.provider,
                    model=outcome.model,
                    prompt_tokens=outcome.usage.prompt_tokens,
                    completion_tokens=outcome.usage.completion_tokens,
                    total_tokens=outcome.usage.total_tokens,
                    cost_usd=outcome.cost_usd,
                )
    except Exception:
        # A persistence failure must never fail the API response the client already got.
        logger.exception("Failed to persist request/usage metadata [request_id=%s]", request_id)
