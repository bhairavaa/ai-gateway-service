from fastapi.testclient import TestClient

from app.api.deps import get_provider_registry
from app.db.repositories.request_log_repository import SqliteRequestLogRepository
from app.db.repositories.usage_repository import SqliteUsageRepository
from app.db.session import AsyncSessionLocal
from app.main import app
from app.providers.base import ProviderResult, UsageInfo
from app.providers.registry import ProviderRegistry
from tests.fakes import FakeProvider


class TestCostTrackingPersistence:
    async def test_successful_completion_persists_request_and_usage_logs(self, auth_headers):
        result = ProviderResult(
            content="hi",
            model="gpt-4o-mini",
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        app.dependency_overrides[get_provider_registry] = lambda: ProviderRegistry(
            {"openai": FakeProvider(result=result)}
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
                    },
                )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        request_id = resp.headers["x-request-id"]

        async with AsyncSessionLocal() as session:
            log_repo = SqliteRequestLogRepository(session)
            recent = await log_repo.list_recent(limit=50)
            matching = [entry for entry in recent if entry.request_id == request_id]
            assert len(matching) == 1
            assert matching[0].status == "success"
            assert matching[0].provider == "openai"

            usage_repo = SqliteUsageRepository(session)
            summary = await usage_repo.aggregate_cost()
            assert summary.total_requests >= 1

    async def test_provider_error_still_persists_a_request_log(self, auth_headers):
        app.dependency_overrides[get_provider_registry] = lambda: ProviderRegistry({})
        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 502
        request_id = resp.headers["x-request-id"]

        async with AsyncSessionLocal() as session:
            log_repo = SqliteRequestLogRepository(session)
            recent = await log_repo.list_recent(limit=50)
            matching = [entry for entry in recent if entry.request_id == request_id]
            assert len(matching) == 1
            assert matching[0].status == "error"
            assert matching[0].error_message is not None
