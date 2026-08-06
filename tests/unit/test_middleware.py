"""Middleware tests.

Request ID / timing / logging / exception-handling are exercised through a
small standalone Starlette app carrying the exact same middleware stack as
app/main.py, plus one deliberately-broken route — keeps the real app's routes
free of test-only "raise on purpose" endpoints.
"""

import logging

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.middleware.exception_handling import ExceptionHandlingMiddleware
from app.middleware.logging import LoggingMiddleware
from app.middleware.request_id import REQUEST_ID_HEADER, RequestIDMiddleware
from app.middleware.timing import RESPONSE_TIME_HEADER, TimingMiddleware


async def _ok(request):
    return JSONResponse({"ok": True})


async def _boom(request):
    raise RuntimeError("deliberate failure for exception-handling tests")


def _build_test_app() -> Starlette:
    app = Starlette(routes=[Route("/ok", _ok), Route("/boom", _boom)])
    # Same registration order/rationale as app/main.py.
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(ExceptionHandlingMiddleware)
    return app


class TestRequestIDMiddleware:
    def test_mints_a_request_id_when_none_supplied(self):
        client = TestClient(_build_test_app())
        resp = client.get("/ok")
        assert REQUEST_ID_HEADER in resp.headers
        assert len(resp.headers[REQUEST_ID_HEADER]) > 0

    def test_honors_inbound_request_id(self):
        client = TestClient(_build_test_app())
        resp = client.get("/ok", headers={REQUEST_ID_HEADER: "caller-supplied-id"})
        assert resp.headers[REQUEST_ID_HEADER] == "caller-supplied-id"


class TestTimingMiddleware:
    def test_sets_response_time_header(self):
        client = TestClient(_build_test_app())
        resp = client.get("/ok")
        assert RESPONSE_TIME_HEADER in resp.headers
        assert float(resp.headers[RESPONSE_TIME_HEADER]) >= 0


class TestLoggingMiddleware:
    def test_logs_successful_request(self, caplog):
        client = TestClient(_build_test_app())
        with caplog.at_level(logging.INFO, logger="app.access"):
            client.get("/ok")

        assert any("GET /ok -> 200" in record.message for record in caplog.records)

    def test_logs_and_reraises_on_unhandled_exception(self, caplog):
        client = TestClient(_build_test_app(), raise_server_exceptions=False)
        with caplog.at_level(logging.INFO, logger="app.access"):
            client.get("/boom")

        assert any("unhandled exception" in record.message for record in caplog.records)


class TestExceptionHandlingMiddleware:
    def test_unhandled_exception_returns_standardized_envelope(self):
        client = TestClient(_build_test_app(), raise_server_exceptions=False)
        resp = client.get("/boom")

        assert resp.status_code == 500
        body = resp.json()
        assert body["error"]["type"] == "internal_error"
        assert body["error"]["request_id"] is not None
        assert resp.headers[REQUEST_ID_HEADER] == body["error"]["request_id"]

    def test_normal_requests_are_unaffected(self):
        client = TestClient(_build_test_app())
        resp = client.get("/ok")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
