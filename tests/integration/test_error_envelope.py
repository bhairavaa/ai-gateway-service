from fastapi.testclient import TestClient

from app.api.deps import get_provider_registry
from app.main import app
from app.providers.registry import ProviderRegistry


class TestErrorEnvelope:
    def test_provider_error_body_is_unwrapped_not_nested_under_detail(self, auth_headers):
        app.dependency_overrides[get_provider_registry] = lambda: ProviderRegistry({})
        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={"provider": "openai", "model": "x", "messages": [{"role": "user", "content": "hi"}]},
                )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 502
        body = resp.json()
        assert "detail" not in body
        assert body["error"]["type"] == "provider_error"
        assert body["error"]["request_id"] is not None
        assert resp.headers["x-request-id"] == body["error"]["request_id"]

    def test_validation_error_uses_standard_envelope(self, auth_headers):
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", headers=auth_headers, json={"provider": "openai"})

        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["type"] == "validation_error"
        assert "model" in body["error"]["message"]
        assert body["error"]["request_id"] is not None

    def test_unknown_route_still_returns_gateway_shaped_404(self, auth_headers):
        with TestClient(app) as client:
            resp = client.get("/v1/does-not-exist", headers=auth_headers)

        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["type"] == "http_error"

    def test_unauthenticated_request_gets_401_before_routing(self):
        # Unknown /v1 path with no API key — auth is enforced before the router even
        # gets a chance to 404, so this is 401, not 404.
        with TestClient(app) as client:
            resp = client.get("/v1/does-not-exist")

        assert resp.status_code == 401
