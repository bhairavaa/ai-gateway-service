"""GET /v1/models — aggregates list_models() across every registered provider."""

import asyncio

from fastapi import APIRouter, Depends

from app.api.deps import get_provider_registry
from app.providers.registry import ProviderRegistry
from app.schemas.providers import ModelEntry, ModelsResponse

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
async def list_models(registry: ProviderRegistry = Depends(get_provider_registry)) -> ModelsResponse:
    results = await asyncio.gather(*(provider.list_models() for provider in registry.all().values()))
    entries = [ModelEntry(id=m.id, provider=m.provider) for models in results for m in models]
    return ModelsResponse(data=entries)
