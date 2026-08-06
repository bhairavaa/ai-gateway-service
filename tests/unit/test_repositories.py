import hashlib

from app.db.repositories.api_key_repository import SqliteApiKeyRepository
from app.db.repositories.audit_log_repository import SqliteAuditLogRepository
from app.db.repositories.request_log_repository import SqliteRequestLogRepository
from app.db.repositories.usage_repository import SqliteUsageRepository


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


class TestApiKeyRepository:
    async def test_create_and_get_by_hash(self, db_session):
        repo = SqliteApiKeyRepository(db_session)
        created = await repo.create(key_hash=_hash("secret-1"), name="ci-bot")

        fetched = await repo.get_by_hash(_hash("secret-1"))

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "ci-bot"
        assert fetched.is_active is True
        assert fetched.scopes == []
        assert fetched.last_used_at is None

    async def test_get_by_hash_unknown_returns_none(self, db_session):
        repo = SqliteApiKeyRepository(db_session)
        assert await repo.get_by_hash(_hash("never-created")) is None

    async def test_touch_last_used_sets_timestamp(self, db_session):
        repo = SqliteApiKeyRepository(db_session)
        created = await repo.create(key_hash=_hash("secret-2"), name="dashboard")
        assert created.last_used_at is None

        await repo.touch_last_used(created.id)

        refreshed = await repo.get_by_id(created.id)
        assert refreshed.last_used_at is not None

    async def test_set_active_deactivates_key(self, db_session):
        repo = SqliteApiKeyRepository(db_session)
        created = await repo.create(key_hash=_hash("secret-3"), name="legacy-service")

        await repo.set_active(created.id, False)

        refreshed = await repo.get_by_id(created.id)
        assert refreshed.is_active is False

    async def test_list_all_active_only_filters_inactive(self, db_session):
        repo = SqliteApiKeyRepository(db_session)
        active = await repo.create(key_hash=_hash("secret-4"), name="active-service")
        inactive = await repo.create(key_hash=_hash("secret-5"), name="inactive-service")
        await repo.set_active(inactive.id, False)

        all_keys = await repo.list_all()
        active_keys = await repo.list_all(active_only=True)

        assert {k.id for k in all_keys} == {active.id, inactive.id}
        assert {k.id for k in active_keys} == {active.id}


class TestRequestLogRepository:
    async def test_create_and_list_recent(self, db_session):
        repo = SqliteRequestLogRepository(db_session)
        entry = await repo.create(
            request_id="req-1",
            api_key_id=None,
            provider="openai",
            model="gpt-4o-mini",
            status="success",
            duration_ms=245,
        )

        recent = await repo.list_recent()

        assert entry.id is not None
        assert entry.created_at is not None
        assert len(recent) == 1
        assert recent[0].request_id == "req-1"
        assert recent[0].error_message is None

    async def test_list_recent_filters_by_api_key(self, db_session):
        repo = SqliteRequestLogRepository(db_session)
        await repo.create(
            request_id="req-a", api_key_id="key-1", provider="openai", model="gpt-4o-mini",
            status="success", duration_ms=100,
        )
        await repo.create(
            request_id="req-b", api_key_id="key-2", provider="anthropic", model="claude-haiku-4-5",
            status="success", duration_ms=150,
        )

        only_key_1 = await repo.list_recent(api_key_id="key-1")

        assert len(only_key_1) == 1
        assert only_key_1[0].request_id == "req-a"


class TestUsageRepository:
    async def test_record_and_aggregate_cost(self, db_session):
        request_log_repo = SqliteRequestLogRepository(db_session)
        usage_repo = SqliteUsageRepository(db_session)

        log = await request_log_repo.create(
            request_id="req-usage-1", api_key_id="key-1", provider="openai", model="gpt-4o-mini",
            status="success", duration_ms=300,
        )
        await usage_repo.record(
            request_log_id=log.id, provider="openai", model="gpt-4o-mini",
            prompt_tokens=100, completion_tokens=50, total_tokens=150, cost_usd=0.002,
        )

        summary = await usage_repo.aggregate_cost(api_key_id="key-1")

        assert summary.total_requests == 1
        assert summary.total_tokens == 150
        assert summary.total_cost_usd == 0.002

    async def test_aggregate_cost_treats_unmapped_pricing_as_zero_in_rollup(self, db_session):
        request_log_repo = SqliteRequestLogRepository(db_session)
        usage_repo = SqliteUsageRepository(db_session)

        log = await request_log_repo.create(
            request_id="req-usage-2", api_key_id=None, provider="ollama", model="some-new-model",
            status="success", duration_ms=400,
        )
        # cost_usd=None represents "no pricing table entry" (see UsageLogORM.cost_usd comment) —
        # it must not blow up the aggregate, just contribute 0 to the sum.
        await usage_repo.record(
            request_log_id=log.id, provider="ollama", model="some-new-model",
            prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=None,
        )

        summary = await usage_repo.aggregate_cost()

        assert summary.total_requests == 1
        assert summary.total_tokens == 15
        assert summary.total_cost_usd == 0.0


class TestAuditLogRepository:
    async def test_record_does_not_raise(self, db_session):
        repo = SqliteAuditLogRepository(db_session)
        await repo.record(action="key_created", api_key_id="key-1", detail={"name": "ci-bot"})
