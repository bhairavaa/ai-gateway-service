"""Async SQLAlchemy engine/session factory.

Only place in the app that constructs an `AsyncEngine`. Repositories receive an
`AsyncSession` via dependency injection (see app/api/deps.py) — they never touch
the engine directly, which keeps the Postgres-swap story to "change DATABASE_URL
+ install asyncpg", nothing else.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.models import Base


def build_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, future=True)


def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


_settings = get_settings()
engine: AsyncEngine = build_engine(_settings.database_url)
AsyncSessionLocal: async_sessionmaker[AsyncSession] = build_sessionmaker(engine)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def init_models(target_engine: AsyncEngine | None = None) -> None:
    """Create all tables directly from ORM metadata.

    Used by tests (in-memory SQLite, needs no migration history) and as a local
    dev convenience. Real deployments use Alembic (`alembic upgrade head`), which
    is the only path that runs against the persistent data volume.
    """
    target_engine = target_engine or engine
    async with target_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
