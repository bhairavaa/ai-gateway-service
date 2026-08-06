"""Aggregates all /v1 sub-routers into one — main.py mounts only this."""

from fastapi import APIRouter, Depends

from app.api.security import api_key_scheme
from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as providers_health_router
from app.api.v1.models import router as models_router

# `dependencies=` here is OpenAPI documentation only (see app/api/security.py) — the actual
# 401 enforcement is AuthenticationMiddleware, already applied to every non-public path.
router = APIRouter(prefix="/v1", dependencies=[Depends(api_key_scheme)])
router.include_router(chat_router)
router.include_router(models_router)
router.include_router(providers_health_router)
