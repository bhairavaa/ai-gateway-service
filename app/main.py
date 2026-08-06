"""FastAPI application factory.

Wires together config, routers, middleware, and (as later milestones land) the
DB engine via startup/shutdown hooks. Kept as a factory (`create_app`) rather
than a bare module-level `app` so tests can build isolated app instances with
different settings/overrides.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.exception_handlers import register_exception_handlers
from app.api.health import router as health_router
from app.api.v1.router import router as v1_router
from app.config import get_settings
from app.logging_config import configure_logging
from app.middleware.authentication import AuthenticationMiddleware
from app.middleware.cost_tracking import CostTrackingMiddleware
from app.middleware.exception_handling import ExceptionHandlingMiddleware
from app.middleware.logging import LoggingMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.timing import TimingMiddleware
from app.providers.registry import build_provider_registry

OPENAPI_TAGS = [
    {"name": "health", "description": "Liveness and per-provider health checks."},
    {"name": "chat", "description": "Unified chat completions — one API for OpenAI, Anthropic, Gemini, and Ollama."},
    {"name": "models", "description": "Discover which models are available across registered providers."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    # Built once per process and stashed on app.state — see app/api/deps.py::get_provider_registry.
    app.state.provider_registry = build_provider_registry(settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Unified API gateway for multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama), "
            "with automatic fallback, per-key rate limiting, and cost/usage tracking. "
            "Authenticate with an `X-API-Key` header — see `scripts/create_api_key.py` to mint one."
        ),
        version="0.1.0",
        debug=settings.debug,
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
    )

    register_exception_handlers(app)

    # Registration order is reversed at request-time (Starlette: last-added = outermost), so this
    # order produces, outer -> inner: CORS -> ExceptionHandling -> RequestID -> Timing ->
    # Authentication -> RateLimit -> Logging -> CostTracking -> route. ExceptionHandling must be
    # outermost of the app's own layers to catch anything escaping every inner layer; RequestID
    # must run before everything else that wants request.state.request_id. Authentication runs
    # before RateLimit (which keys by the authenticated client, not IP); CostTracking is
    # innermost, right next to the route, since it reads request.state.chat_outcome — only ever
    # set by the route/service that produced it.
    #
    # CORS is the true outermost layer, ahead of even ExceptionHandling: browsers send an
    # unauthenticated OPTIONS preflight before a real cross-origin request, and CORSMiddleware
    # answers that directly rather than letting it reach AuthenticationMiddleware and 401. It
    # also needs to add CORS headers to *every* response, including error responses from the
    # layers inside it, or the browser can't read those responses' bodies at all.
    app.add_middleware(CostTrackingMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, settings=settings)
    app.add_middleware(AuthenticationMiddleware, settings=settings)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(ExceptionHandlingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins_list,
        # Auth here is a header (X-API-Key), never cookies, so credentialed CORS requests
        # aren't needed — keeps allow_origins=["*"] valid (browsers reject "*" combined with
        # allow_credentials=True).
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(v1_router)

    return app


app = create_app()
