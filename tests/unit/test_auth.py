from datetime import datetime, timedelta, timezone

import pytest

from app.auth.api_key_backend import ApiKeyAuthBackend, hash_api_key
from app.auth.base import AuthenticationError
from app.db.repositories.api_key_repository import SqliteApiKeyRepository


class TestApiKeyAuthBackend:
    async def test_valid_key_returns_auth_context(self, db_session):
        repo = SqliteApiKeyRepository(db_session)
        await repo.create(key_hash=hash_api_key("raw-key-1"), name="ci-bot", scopes=["chat:write"])
        backend = ApiKeyAuthBackend(repo)

        context = await backend.authenticate("raw-key-1")

        assert context.name == "ci-bot"
        assert context.scopes == ["chat:write"]

    async def test_unknown_key_raises(self, db_session):
        backend = ApiKeyAuthBackend(SqliteApiKeyRepository(db_session))

        with pytest.raises(AuthenticationError):
            await backend.authenticate("never-created")

    async def test_inactive_key_raises(self, db_session):
        repo = SqliteApiKeyRepository(db_session)
        record = await repo.create(key_hash=hash_api_key("raw-key-2"), name="disabled-service")
        await repo.set_active(record.id, False)
        backend = ApiKeyAuthBackend(repo)

        with pytest.raises(AuthenticationError, match="inactive"):
            await backend.authenticate("raw-key-2")

    async def test_expired_key_raises(self, db_session):
        repo = SqliteApiKeyRepository(db_session)
        await repo.create(
            key_hash=hash_api_key("raw-key-3"),
            name="temp-service",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        backend = ApiKeyAuthBackend(repo)

        with pytest.raises(AuthenticationError, match="expired"):
            await backend.authenticate("raw-key-3")

    async def test_valid_key_touches_last_used(self, db_session):
        repo = SqliteApiKeyRepository(db_session)
        record = await repo.create(key_hash=hash_api_key("raw-key-4"), name="dashboard")
        backend = ApiKeyAuthBackend(repo)

        await backend.authenticate("raw-key-4")

        refreshed = await repo.get_by_id(record.id)
        assert refreshed.last_used_at is not None
