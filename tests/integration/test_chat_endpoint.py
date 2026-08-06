from fastapi.testclient import TestClient

from app.api.deps import get_provider_registry
from app.main import app
from app.providers.base import ProviderCallError, ProviderResult, UsageInfo
from app.providers.registry import ProviderRegistry
from tests.fakes import FakeProvider


def _override_registry(registry: ProviderRegistry) -> None:
    app.dependency_overrides[get_provider_registry] = lambda: registry


def _clear_overrides() -> None:
    app.dependency_overrides.clear()


class TestChatCompletionsEndpoint:
    def test_successful_completion(self, auth_headers):
        result = ProviderResult(
            content="hi from fake provider",
            model="fake-model",
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=4, completion_tokens=6, total_tokens=10),
        )
        _override_registry(ProviderRegistry({"openai": FakeProvider(result=result)}))
        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "provider": "openai",
                        "model": "fake-model",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        finally:
            _clear_overrides()

        assert resp.status_code == 200
        body = resp.json()
        assert body["content"] == "hi from fake provider"
        assert body["provider"] == "openai"
        assert body["usage"]["total_tokens"] == 10

    def test_no_provider_registered_returns_502(self, auth_headers):
        # Empty registry -> primary attempt fails, and so does every entry in the default
        # fallback chain (nothing is registered) -> the whole chain is exhausted -> 502.
        _override_registry(ProviderRegistry({}))
        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "provider": "openai",
                        "model": "fake-model",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        finally:
            _clear_overrides()

        assert resp.status_code == 502

    def test_invalid_request_body_returns_422(self, auth_headers):
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", headers=auth_headers, json={"provider": "openai"})

        assert resp.status_code == 422

    def test_missing_api_key_returns_401(self):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"provider": "openai", "model": "x", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"

    def test_explicit_fallback_serves_from_second_provider(self, auth_headers):
        fallback_result = ProviderResult(
            content="served by anthropic", model="claude-haiku-4-5", finish_reason="stop", usage=None
        )
        _override_registry(
            ProviderRegistry(
                {
                    "openai": FakeProvider(name="openai", error=ProviderCallError("openai", "down")),
                    "anthropic": FakeProvider(name="anthropic", result=fallback_result),
                }
            )
        )
        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hi"}],
                        "fallback": [{"provider": "anthropic", "model": "claude-haiku-4-5"}],
                    },
                )
        finally:
            _clear_overrides()

        assert resp.status_code == 200
        body = resp.json()
        assert body["provider"] == "anthropic"
        assert body["fallback_used"] is True

    def test_invalid_api_key_returns_401(self):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers={"X-API-Key": "not-a-real-key"},
                json={"provider": "openai", "model": "x", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 401
