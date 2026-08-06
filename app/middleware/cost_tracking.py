"""Persists request/usage metadata after the response is sent (non-streaming path).

Reads request.state.chat_outcome, stashed by ChatService (see
app/services/chat_service.py). The streaming path persists its own outcome
directly at the end of the SSE generator instead (see
app/services/chat_streaming.py) — by the time this middleware's call_next()
returns for a StreamingResponse, the route has only just constructed the
response object; the generator producing chat_outcome hasn't run yet.
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.services.outcome_persistence import persist_chat_outcome


class CostTrackingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        outcome = getattr(request.state, "chat_outcome", None)
        if outcome is None:
            return response

        auth_context = getattr(request.state, "auth_context", None)
        request_id = getattr(request.state, "request_id", None) or "unknown"
        await persist_chat_outcome(
            request_id=request_id, auth_context=auth_context, outcome=outcome, duration_ms=int(duration_ms)
        )
        return response
