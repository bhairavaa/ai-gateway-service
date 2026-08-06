"""GET /v1/health/providers — per-provider deep health check.

Distinct from the root /health liveness probe (app/api/health.py), which never
touches upstream providers and always returns instantly. This endpoint fans out
a real (but non-billed) reachability check to every registered provider.
"""

import asyncio

from fastapi import APIRouter, Depends

from app.api.deps import get_provider_registry
from app.providers.registry import ProviderRegistry
from app.schemas.providers import ProviderHealthEntry, ProvidersHealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/providers", response_model=ProvidersHealthResponse)
async def providers_health(registry: ProviderRegistry = Depends(get_provider_registry)) -> ProvidersHealthResponse:
    statuses = await asyncio.gather(*(provider.health_check() for provider in registry.all().values()))
    entries = [ProviderHealthEntry(provider=s.provider, healthy=s.healthy, detail=s.detail) for s in statuses]
    return ProvidersHealthResponse(providers=entries)
