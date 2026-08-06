"""Root-level liveness endpoint.

Deliberately outside the /v1 API surface and outside auth/rate-limit middleware
(see the public-path exemption in app/middleware/authentication.py once that lands)
so container orchestrators can probe it without credentials.
"""

from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def liveness() -> dict:
    settings = get_settings()
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}
