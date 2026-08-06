"""Response DTOs for the provider-discovery/health endpoints."""

from pydantic import BaseModel


class ModelEntry(BaseModel):
    id: str
    provider: str


class ModelsResponse(BaseModel):
    data: list[ModelEntry]


class ProviderHealthEntry(BaseModel):
    provider: str
    healthy: bool
    detail: str | None = None


class ProvidersHealthResponse(BaseModel):
    providers: list[ProviderHealthEntry]
