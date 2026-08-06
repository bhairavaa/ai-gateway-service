"""Redis-backed fixed-window rate limiter, keyed by authenticated API key.

Fixed-window (INCR + EXPIRE) rather than a sliding-window log: two Redis calls,
easy to reason about/demo, at the cost of allowing up to ~2x the configured
limit in a burst straddling a window boundary — an accepted, documented
tradeoff (a sliding-window log is more accurate but meaningfully more code for
marginal benefit at this scale).

Runs after AuthenticationMiddleware (needs request.state.auth_context to key
by client identity, not by IP) and skips the same public paths auth skips,
since there's no identity to key by there.
"""

import logging
import time

from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.config import Settings
from app.middleware.authentication import PUBLIC_PATHS
from app.middleware.error_response import build_error_response

logger = logging.getLogger("app.rate_limit")


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings, redis_client: Redis | None = None):
        super().__init__(app)
        self._limit = settings.rate_limit_requests_per_window
        self._window_seconds = settings.rate_limit_window_seconds
        # Constructed once at startup and reused for the process lifetime — redis.asyncio.Redis
        # manages its own connection pool internally, so this is safe to share across requests.
        self._redis = redis_client or Redis.from_url(settings.redis_url, decode_responses=True)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        auth_context = getattr(request.state, "auth_context", None)
        if auth_context is None:
            # AuthenticationMiddleware runs before this (see registration order in main.py) and
            # already rejects unauthenticated requests — this is a defensive fallback only.
            return await call_next(request)

        request_id = getattr(request.state, "request_id", None)
        bucket = int(time.time() // self._window_seconds)
        redis_key = f"ratelimit:{auth_context.api_key_id}:{bucket}"

        try:
            count = await self._redis.incr(redis_key)
            if count == 1:
                await self._redis.expire(redis_key, self._window_seconds)
        except Exception:
            # Redis unreachable — fail open rather than taking the whole gateway down over a
            # non-critical dependency; logged so the degradation is visible, not silent.
            logger.warning("Rate limiter unavailable, allowing request through", exc_info=True)
            return await call_next(request)

        if count > self._limit:
            retry_after = self._window_seconds - (int(time.time()) % self._window_seconds)
            return build_error_response(
                status_code=429,
                error_type="rate_limited",
                message=f"Rate limit exceeded: {self._limit} requests per {self._window_seconds}s",
                request_id=request_id,
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
