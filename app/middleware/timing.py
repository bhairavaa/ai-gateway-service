"""Measures wall-clock request duration.

Uses try/finally so `request.state.duration_ms` is always populated — including
when the handler raises — since other consumers (cost-tracking persistence,
later) may want it regardless of outcome.
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

RESPONSE_TIME_HEADER = "X-Response-Time-ms"


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request.state.duration_ms = (time.perf_counter() - start) * 1000
        response.headers[RESPONSE_TIME_HEADER] = f"{request.state.duration_ms:.2f}"
        return response
