from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RequestLogORM
from app.db.repositories.dto import RequestLogEntry
from app.db.repositories.interfaces import RequestLogRepository


def _to_entry(row: RequestLogORM) -> RequestLogEntry:
    return RequestLogEntry(
        id=row.id,
        request_id=row.request_id,
        api_key_id=row.api_key_id,
        provider=row.provider,
        model=row.model,
        status=row.status,
        error_message=row.error_message,
        duration_ms=row.duration_ms,
        created_at=row.created_at,
    )


class SqliteRequestLogRepository(RequestLogRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

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
    ) -> RequestLogEntry:
        row = RequestLogORM(
            request_id=request_id,
            api_key_id=api_key_id,
            provider=provider,
            model=model,
            status=status,
            duration_ms=duration_ms,
            error_message=error_message,
        )
        self._session.add(row)
        await self._session.commit()
        await self._session.refresh(row)
        return _to_entry(row)

    async def list_recent(self, *, api_key_id: str | None = None, limit: int = 50) -> list[RequestLogEntry]:
        stmt = select(RequestLogORM)
        if api_key_id is not None:
            stmt = stmt.where(RequestLogORM.api_key_id == api_key_id)
        stmt = stmt.order_by(RequestLogORM.created_at.desc()).limit(limit)
        result = await self._session.execute(stmt)
        return [_to_entry(row) for row in result.scalars().all()]
