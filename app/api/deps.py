"""Shared FastAPI dependency providers.

The one place that wires concrete implementations (SQLite repositories, the
async session) behind the interfaces the rest of the app depends on. Swapping
a backing store later means changing constructions here, not call sites.
"""

from collections.abc import AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.repositories.api_key_repository import SqliteApiKeyRepository
from app.db.repositories.audit_log_repository import SqliteAuditLogRepository
from app.db.repositories.interfaces import ApiKeyRepository, AuditLogRepository, RequestLogRepository, UsageRepository
from app.db.repositories.request_log_repository import SqliteRequestLogRepository
from app.db.repositories.usage_repository import SqliteUsageRepository
from app.db.session import get_db_session as _get_db_session
from app.providers.registry import ProviderRegistry


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in _get_db_session():
        yield session


def get_settings_dep() -> Settings:
    return get_settings()


def get_api_key_repository(session: AsyncSession = Depends(get_db_session)) -> ApiKeyRepository:
    return SqliteApiKeyRepository(session)


def get_request_log_repository(session: AsyncSession = Depends(get_db_session)) -> RequestLogRepository:
    return SqliteRequestLogRepository(session)


def get_usage_repository(session: AsyncSession = Depends(get_db_session)) -> UsageRepository:
    return SqliteUsageRepository(session)


def get_audit_log_repository(session: AsyncSession = Depends(get_db_session)) -> AuditLogRepository:
    return SqliteAuditLogRepository(session)


def get_provider_registry(request: Request) -> ProviderRegistry:
    # Built once at startup (see app/main.py's lifespan) and stashed on app.state —
    # constructing per-request would rebuild every provider's LangChain client on every call.
    return request.app.state.provider_registry
