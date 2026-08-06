from fastapi.testclient import TestClient

from app.main import app


class TestCORS:
    def test_preflight_request_is_answered_without_auth(self):
        # Browsers send OPTIONS before a real cross-origin request, with no X-API-Key —
        # CORSMiddleware must answer this directly, never reaching AuthenticationMiddleware.
        with TestClient(app) as client:
            resp = client.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "https://example.com",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "X-API-Key,Content-Type",
                },
            )

        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "*"

    def test_actual_request_gets_cors_header_even_on_error(self, auth_headers):
        # Cross-origin error responses (e.g. this 400/502) still need the CORS header, or the
        # browser can't let JS read the response body at all.
        with TestClient(app) as client:
            resp = client.get(
                "/v1/does-not-exist", headers={**auth_headers, "Origin": "https://example.com"}
            )

        assert "access-control-allow-origin" in resp.headers

    def test_successful_response_includes_cors_header(self, auth_headers):
        with TestClient(app) as client:
            resp = client.get("/health", headers={"Origin": "https://example.com"})

        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "*"
