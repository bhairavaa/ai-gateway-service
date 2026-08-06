import fakeredis.aioredis
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.config import Settings
from app.middleware.rate_limit import RateLimitMiddleware


class _FakeAuthMiddleware(BaseHTTPMiddleware):
    """Stands in for AuthenticationMiddleware so these tests don't need a real DB —
    just sets the request.state.auth_context RateLimitMiddleware reads."""

    async def dispatch(self, request, call_next):
        class _Ctx:
            api_key_id = "test-key-id"

        request.state.auth_context = _Ctx()
        return await call_next(request)


async def _ok(request):
    return JSONResponse({"ok": True})


def _build_test_app(*, limit: int, window_seconds: int = 60) -> Starlette:
    settings = Settings(rate_limit_requests_per_window=limit, rate_limit_window_seconds=window_seconds)
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)

    app = Starlette(routes=[Route("/ok", _ok)])
    app.add_middleware(RateLimitMiddleware, settings=settings, redis_client=redis_client)
    app.add_middleware(_FakeAuthMiddleware)
    return app


class TestRateLimitMiddleware:
    def test_requests_under_limit_are_allowed(self):
        client = TestClient(_build_test_app(limit=3))
        for _ in range(3):
            resp = client.get("/ok")
            assert resp.status_code == 200

    def test_request_over_limit_returns_429_with_retry_after(self):
        client = TestClient(_build_test_app(limit=2))
        client.get("/ok")
        client.get("/ok")

        resp = client.get("/ok")

        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        assert resp.json()["error"]["type"] == "rate_limited"

    def test_unauthenticated_requests_pass_through(self):
        # No _FakeAuthMiddleware here — simulates the defensive fallback when
        # request.state.auth_context was never set.
        settings = Settings(rate_limit_requests_per_window=1)
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        app = Starlette(routes=[Route("/ok", _ok)])
        app.add_middleware(RateLimitMiddleware, settings=settings, redis_client=redis_client)
        client = TestClient(app)

        for _ in range(5):
            resp = client.get("/ok")
            assert resp.status_code == 200

    def test_redis_failure_fails_open(self):
        class _BrokenRedis:
            async def incr(self, *_args, **_kwargs):
                raise ConnectionError("redis down")

        settings = Settings(rate_limit_requests_per_window=1)
        app = Starlette(routes=[Route("/ok", _ok)])
        app.add_middleware(RateLimitMiddleware, settings=settings, redis_client=_BrokenRedis())
        app.add_middleware(_FakeAuthMiddleware)
        client = TestClient(app)

        resp = client.get("/ok")

        assert resp.status_code == 200
