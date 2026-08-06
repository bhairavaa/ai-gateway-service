from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ApiKeyORM
from app.db.repositories.dto import ApiKeyRecord
from app.db.repositories.interfaces import ApiKeyRepository


def _to_record(row: ApiKeyORM) -> ApiKeyRecord:
    return ApiKeyRecord(
        id=row.id,
        key_hash=row.key_hash,
        name=row.name,
        is_active=row.is_active,
        scopes=list(row.scopes or []),
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        expires_at=row.expires_at,
    )


class SqliteApiKeyRepository(ApiKeyRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(
        self,
        *,
        key_hash: str,
        name: str,
        scopes: list[str] | None = None,
        expires_at: datetime | None = None,
    ) -> ApiKeyRecord:
        row = ApiKeyORM(key_hash=key_hash, name=name, scopes=scopes or [], expires_at=expires_at)
        self._session.add(row)
        await self._session.commit()
        await self._session.refresh(row)
        return _to_record(row)

    async def get_by_hash(self, key_hash: str) -> ApiKeyRecord | None:
        result = await self._session.execute(select(ApiKeyORM).where(ApiKeyORM.key_hash == key_hash))
        row = result.scalar_one_or_none()
        return _to_record(row) if row else None

    async def get_by_id(self, api_key_id: str) -> ApiKeyRecord | None:
        row = await self._session.get(ApiKeyORM, api_key_id)
        return _to_record(row) if row else None

    async def touch_last_used(self, api_key_id: str) -> None:
        row = await self._session.get(ApiKeyORM, api_key_id)
        if row is None:
            return
        row.last_used_at = datetime.now(timezone.utc)
        await self._session.commit()

    async def set_active(self, api_key_id: str, is_active: bool) -> None:
        row = await self._session.get(ApiKeyORM, api_key_id)
        if row is None:
            return
        row.is_active = is_active
        await self._session.commit()

    async def list_all(self, *, active_only: bool = False) -> list[ApiKeyRecord]:
        stmt = select(ApiKeyORM)
        if active_only:
            stmt = stmt.where(ApiKeyORM.is_active.is_(True))
        result = await self._session.execute(stmt.order_by(ApiKeyORM.created_at.desc()))
        return [_to_record(row) for row in result.scalars().all()]
