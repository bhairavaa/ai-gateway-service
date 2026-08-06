"""API-key auth backend.

Hashes the incoming key and looks it up via ApiKeyRepository — the raw key is
never persisted, only its hash.
"""

import hashlib
from datetime import datetime, timezone

from app.auth.base import AuthBackend, AuthContext, AuthenticationError
from app.db.repositories.interfaces import ApiKeyRepository


def hash_api_key(raw_key: str) -> str:
    # SHA-256 (not bcrypt/argon2) is sufficient here: this hashes a high-entropy,
    # machine-generated credential (see scripts/create_api_key.py), not a human
    # password subject to dictionary/brute-force attack against a leaked hash.
    return hashlib.sha256(raw_key.encode()).hexdigest()


class ApiKeyAuthBackend(AuthBackend):
    def __init__(self, repo: ApiKeyRepository):
        self._repo = repo

    async def authenticate(self, credential: str) -> AuthContext:
        record = await self._repo.get_by_hash(hash_api_key(credential))
        if record is None:
            raise AuthenticationError("Invalid API key")
        if not record.is_active:
            raise AuthenticationError("API key is inactive")
        if record.expires_at is not None:
            # SQLite has no native tz-aware datetime type — SQLAlchemy round-trips it as naive,
            # even though it was stored from a tz-aware value (see ApiKeyORM, always UTC). Treat
            # a naive value as UTC rather than comparing naive-vs-aware, which raises TypeError.
            expires_at = record.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at < datetime.now(timezone.utc):
                raise AuthenticationError("API key has expired")

        await self._repo.touch_last_used(record.id)
        return AuthContext(api_key_id=record.id, name=record.name, scopes=record.scopes)
