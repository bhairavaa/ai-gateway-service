"""Plain data-transfer objects returned by repositories.

Services and routes depend on these, never on the ORM classes in app/db/models.py.
That boundary is what makes the Postgres swap a repositories/-only change: nothing
above this layer imports SQLAlchemy.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ApiKeyRecord:
    id: str
    key_hash: str
    name: str
    is_active: bool
    scopes: list[str]
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None


@dataclass(frozen=True)
class RequestLogEntry:
    id: str
    request_id: str
    api_key_id: str | None
    provider: str
    model: str
    status: str  # "success" | "error" | "fallback"
    error_message: str | None
    duration_ms: int
    created_at: datetime


@dataclass(frozen=True)
class UsageEntry:
    id: str
    request_log_id: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None
    created_at: datetime


@dataclass(frozen=True)
class CostSummary:
    total_requests: int
    total_tokens: int
    total_cost_usd: float
