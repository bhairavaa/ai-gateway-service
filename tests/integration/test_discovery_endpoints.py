from fastapi.testclient import TestClient

from app.api.deps import get_provider_registry
from app.main import app
from app.providers.base import HealthStatus
from app.providers.registry import ProviderRegistry
from tests.fakes import FakeProvider


def _override_registry(registry: ProviderRegistry) -> None:
    app.dependency_overrides[get_provider_registry] = lambda: registry


def _clear_overrides() -> None:
    app.dependency_overrides.clear()


class TestModelsEndpoint:
    def test_aggregates_models_across_registered_providers(self, auth_headers):
        openai_provider = FakeProvider(name="openai")
        anthropic_provider = FakeProvider(name="anthropic")
        _override_registry(ProviderRegistry({"openai": openai_provider, "anthropic": anthropic_provider}))
        try:
            with TestClient(app) as client:
                resp = client.get("/v1/models", headers=auth_headers)
        finally:
            _clear_overrides()

        assert resp.status_code == 200
        providers_seen = {entry["provider"] for entry in resp.json()["data"]}
        assert providers_seen == {"openai", "anthropic"}

    def test_empty_registry_returns_empty_list(self, auth_headers):
        _override_registry(ProviderRegistry({}))
        try:
            with TestClient(app) as client:
                resp = client.get("/v1/models", headers=auth_headers)
        finally:
            _clear_overrides()

        assert resp.status_code == 200
        assert resp.json() == {"data": []}

    def test_missing_api_key_returns_401(self):
        with TestClient(app) as client:
            resp = client.get("/v1/models")

        assert resp.status_code == 401


class TestProvidersHealthEndpoint:
    def test_reports_health_per_provider(self, auth_headers):
        class HealthyFake(FakeProvider):
            async def health_check(self):
                return HealthStatus(provider=self.name, healthy=True)

        class UnhealthyFake(FakeProvider):
            async def health_check(self):
                return HealthStatus(provider=self.name, healthy=False, detail="boom")

        _override_registry(
            ProviderRegistry({"openai": HealthyFake(name="openai"), "ollama": UnhealthyFake(name="ollama")})
        )
        try:
            with TestClient(app) as client:
                resp = client.get("/v1/health/providers", headers=auth_headers)
        finally:
            _clear_overrides()

        assert resp.status_code == 200
        by_provider = {entry["provider"]: entry for entry in resp.json()["providers"]}
        assert by_provider["openai"]["healthy"] is True
        assert by_provider["ollama"]["healthy"] is False
        assert by_provider["ollama"]["detail"] == "boom"
