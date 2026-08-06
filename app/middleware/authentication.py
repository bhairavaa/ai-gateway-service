"""Validates the X-API-Key header and attaches an AuthContext to request.state
for downstream code (rate limiting, cost tracking, routes).

Builds its own AsyncSession per request via app.db.session's module-level
session factory rather than FastAPI's Depends machinery — middleware runs
outside FastAPI's dependency-injection system, so this goes through the same
factory Depends-based code uses, just constructed directly.
"""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.auth.api_key_backend import ApiKeyAuthBackend
from app.auth.base import AuthenticationError
from app.config import Settings
from app.db.repositories.api_key_repository import SqliteApiKeyRepository
from app.db.session import AsyncSessionLocal
from app.middleware.error_response import build_error_response

# Reachable without a key: liveness probe and API docs. Everything else — including
# discovery endpoints like /v1/models — requires authentication for a consistent story.
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class AuthenticationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        request_id = getattr(request.state, "request_id", None)
        header_name = self._settings.api_key_header_name
        api_key = request.headers.get(header_name)
        if not api_key:
            return build_error_response(
                status_code=401,
                error_type="authentication_error",
                message=f"Missing {header_name} header",
                request_id=request_id,
            )

        async with AsyncSessionLocal() as session:
            backend = ApiKeyAuthBackend(SqliteApiKeyRepository(session))
            try:
                auth_context = await backend.authenticate(api_key)
            except AuthenticationError as exc:
                return build_error_response(
                    status_code=401, error_type="authentication_error", message=str(exc), request_id=request_id
                )

        request.state.auth_context = auth_context
        return await call_next(request)
