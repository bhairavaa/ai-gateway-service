from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RequestLogORM, UsageLogORM
from app.db.repositories.dto import CostSummary, UsageEntry
from app.db.repositories.interfaces import UsageRepository


def _to_entry(row: UsageLogORM) -> UsageEntry:
    return UsageEntry(
        id=row.id,
        request_log_id=row.request_log_id,
        provider=row.provider,
        model=row.model,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        total_tokens=row.total_tokens,
        cost_usd=row.cost_usd,
        created_at=row.created_at,
    )


class SqliteUsageRepository(UsageRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

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
    ) -> UsageEntry:
        row = UsageLogORM(
            request_log_id=request_log_id,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )
        self._session.add(row)
        await self._session.commit()
        await self._session.refresh(row)
        return _to_entry(row)

    async def aggregate_cost(self, *, api_key_id: str | None = None, since: datetime | None = None) -> CostSummary:
        stmt = select(
            func.count(UsageLogORM.id),
            func.coalesce(func.sum(UsageLogORM.total_tokens), 0),
            # Individual rows can have cost_usd = None (unmapped model pricing); the rollup
            # coalesces those to 0 for the sum so one unpriced row doesn't null out the whole total.
            func.coalesce(func.sum(UsageLogORM.cost_usd), 0.0),
        )
        if api_key_id is not None:
            stmt = stmt.join(RequestLogORM, RequestLogORM.id == UsageLogORM.request_log_id).where(
                RequestLogORM.api_key_id == api_key_id
            )
        if since is not None:
            stmt = stmt.where(UsageLogORM.created_at >= since)

        result = await self._session.execute(stmt)
        total_requests, total_tokens, total_cost = result.one()
        return CostSummary(
            total_requests=total_requests,
            total_tokens=int(total_tokens),
            total_cost_usd=float(total_cost),
        )
