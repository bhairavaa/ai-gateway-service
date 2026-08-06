"""Repository interfaces (abstract base classes).

Services and API routes depend on these ABCs, injected via FastAPI `Depends`
(see app/api/deps.py) — never on a concrete SQLite/Postgres implementation
directly. Swapping the storage backend means adding a new implementation of
these same interfaces and changing what `deps.py` constructs; no caller changes.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from app.db.repositories.dto import ApiKeyRecord, CostSummary, RequestLogEntry, UsageEntry


class ApiKeyRepository(ABC):
    @abstractmethod
    async def create(
        self,
        *,
        key_hash: str,
        name: str,
        scopes: list[str] | None = None,
        expires_at: datetime | None = None,
    ) -> ApiKeyRecord: ...

    @abstractmethod
    async def get_by_hash(self, key_hash: str) -> ApiKeyRecord | None: ...

    @abstractmethod
    async def get_by_id(self, api_key_id: str) -> ApiKeyRecord | None: ...

    @abstractmethod
    async def touch_last_used(self, api_key_id: str) -> None: ...

    @abstractmethod
    async def set_active(self, api_key_id: str, is_active: bool) -> None: ...

    @abstractmethod
    async def list_all(self, *, active_only: bool = False) -> list[ApiKeyRecord]: ...


class RequestLogRepository(ABC):
    @abstractmethod
    async def create(
        self,
        *,
        request_id: str,
        api_key_id: str | None,
        provider: str,
        model: str,
        status: str,
        duration_ms: int,
        error_message: str | None = None,
    ) -> RequestLogEntry: ...

    @abstractmethod
    async def list_recent(self, *, api_key_id: str | None = None, limit: int = 50) -> list[RequestLogEntry]: ...


class UsageRepository(ABC):
    @abstractmethod
    async def record(
        self,
        *,
        request_log_id: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float | None,
    ) -> UsageEntry: ...

    @abstractmethod
    async def aggregate_cost(
        self, *, api_key_id: str | None = None, since: datetime | None = None
    ) -> CostSummary: ...


class AuditLogRepository(ABC):
    @abstractmethod
    async def record(self, *, action: str, api_key_id: str | None = None, detail: dict | None = None) -> None: ...
