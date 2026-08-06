import json

from fastapi.testclient import TestClient

from app.api.deps import get_provider_registry
from app.db.repositories.request_log_repository import SqliteRequestLogRepository
from app.db.session import AsyncSessionLocal
from app.main import app
from app.providers.base import ProviderCallError, StreamChunk, UsageInfo
from app.providers.registry import ProviderRegistry
from tests.fakes import FakeProvider


def _override_registry(registry: ProviderRegistry) -> None:
    app.dependency_overrides[get_provider_registry] = lambda: registry


def _clear_overrides() -> None:
    app.dependency_overrides.clear()


class TestStreamingChatCompletions:
    def test_streams_deltas_then_usage_then_done(self, auth_headers):
        chunks = [
            StreamChunk(delta="Hel", is_final=False),
            StreamChunk(delta="lo!", is_final=False),
            StreamChunk(delta="", is_final=True, usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5)),
        ]
        _override_registry(ProviderRegistry({"openai": FakeProvider(stream_chunks=chunks)}))
        try:
            with TestClient(app) as client:
                with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                ) as resp:
                    assert resp.status_code == 200
                    assert resp.headers["content-type"].startswith("text/event-stream")
                    lines = [line for line in resp.iter_lines() if line]
        finally:
            _clear_overrides()

        events = [json.loads(line[len("data: "):]) for line in lines if line != "data: [DONE]"]
        assert lines[-1] == "data: [DONE]"
        deltas = [e["delta"] for e in events if "delta" in e]
        assert "".join(deltas) == "Hello!"
        usage_events = [e for e in events if "usage" in e]
        assert usage_events[0]["usage"]["total_tokens"] == 5

    def test_pre_stream_all_providers_failed_returns_normal_502_not_a_stream(self, auth_headers):
        _override_registry(
            ProviderRegistry({"openai": FakeProvider(error=ProviderCallError("openai", "down"))})
        )
        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "stream": True,
                        "fallback": [],
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        finally:
            _clear_overrides()

        assert resp.status_code == 502
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["error"]["type"] == "provider_error"

    async def test_streaming_completion_persists_request_log(self, auth_headers):
        chunks = [StreamChunk(delta="hi", is_final=False), StreamChunk(delta="", is_final=True, usage=None)]
        _override_registry(ProviderRegistry({"openai": FakeProvider(stream_chunks=chunks)}))
        try:
            with TestClient(app) as client:
                with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers=auth_headers,
                    json={
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                ) as resp:
                    request_id = resp.headers["x-request-id"]
                    for _ in resp.iter_lines():
                        pass  # drain the stream so the generator's finally block runs
        finally:
            _clear_overrides()

        async with AsyncSessionLocal() as session:
            log_repo = SqliteRequestLogRepository(session)
            recent = await log_repo.list_recent(limit=50)
            matching = [entry for entry in recent if entry.request_id == request_id]
            assert len(matching) == 1
            assert matching[0].status == "success"
