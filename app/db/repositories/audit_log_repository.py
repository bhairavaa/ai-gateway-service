from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditLogORM
from app.db.repositories.interfaces import AuditLogRepository


class SqliteAuditLogRepository(AuditLogRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def record(self, *, action: str, api_key_id: str | None = None, detail: dict | None = None) -> None:
        row = AuditLogORM(action=action, api_key_id=api_key_id, detail=detail or {})
        self._session.add(row)
        await self._session.commit()
