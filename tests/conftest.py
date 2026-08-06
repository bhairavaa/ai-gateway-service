"""Shared pytest fixtures.

Repository/unit tests use a fresh in-memory SQLite DB per test (see db_session).
Integration tests that go through the real app + AuthenticationMiddleware need
a DB the middleware itself can see — middleware builds its own AsyncSession
from app.db.session's module-level engine, outside FastAPI's Depends/override
machinery, so DATABASE_URL is pointed at a dedicated on-disk test file *before*
app.db.session is ever imported (its engine is built at import time).
"""

import asyncio
import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/test_gateway.db")

from collections.abc import AsyncGenerator  # noqa: E402

import fakeredis.aioredis  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from redis.asyncio import Redis as RedisAsync  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.auth.api_key_backend import hash_api_key  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db.repositories.api_key_repository import SqliteApiKeyRepository  # noqa: E402
from app.db.session import AsyncSessionLocal, build_engine, build_sessionmaker, init_models  # noqa: E402

TEST_API_KEY_RAW = "agw_test_fixed_key_for_pytest"

# app/middleware/rate_limit.py builds its Redis client via `Redis.from_url(...)` at middleware
# construction time (app startup) if no client is injected. Patched here, at import time, so the
# real app (built once at `app.main` import) gets a fake in-memory Redis instead of trying (and
# slowly failing) to reach a real one — every test would otherwise pay a connection-timeout tax.
_fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
RedisAsync.from_url = classmethod(lambda cls, *args, **kwargs: _fake_redis)


@pytest_asyncio.fixture(autouse=True)
async def _reset_rate_limiter():
    # Function-scoped + autouse: without this, the shared test API key (see auth_headers)
    # would accumulate a rate-limit count across the whole test session and start 429ing
    # unrelated tests once enough requests had been made.
    await _fake_redis.flushdb()
    yield


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """A fresh in-memory SQLite DB (schema created straight from ORM metadata,
    bypassing Alembic — real deployments still migrate via `alembic upgrade head`)."""
    engine = build_engine("sqlite+aiosqlite:///:memory:")
    await init_models(engine)
    session_factory = build_sessionmaker(engine)
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def _seed_test_database():
    """Ensures the shared on-disk test DB has tables and one active API key, so
    integration tests can authenticate against the real AuthenticationMiddleware
    instead of mocking auth away. Sync fixture wrapping asyncio.run() so it
    doesn't need to participate in pytest-asyncio's per-test event loop scoping.
    """

    async def _seed() -> None:
        await init_models()
        async with AsyncSessionLocal() as session:
            repo = SqliteApiKeyRepository(session)
            existing = await repo.get_by_hash(hash_api_key(TEST_API_KEY_RAW))
            if existing is None:
                await repo.create(key_hash=hash_api_key(TEST_API_KEY_RAW), name="pytest")

    asyncio.run(_seed())


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {get_settings().api_key_header_name: TEST_API_KEY_RAW}
