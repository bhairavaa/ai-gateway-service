"""OpenAPI-visible API key security scheme.

Real enforcement happens in app/middleware/authentication.py (ASGI middleware,
outside FastAPI's dependency-injection system) — that's the only thing that
actually rejects a request. This dependency exists purely so the header shows
up in the OpenAPI schema, giving Swagger UI a working "Authorize" button and
letting "Try it out" attach the key. `auto_error=False` so it never itself
blocks anything — a route would still execute even with this "unauthenticated"
if the middleware weren't there, which is exactly why the middleware is the
one source of truth for enforcement, not this.
"""

from fastapi.security import APIKeyHeader

from app.config import get_settings

api_key_scheme = APIKeyHeader(name=get_settings().api_key_header_name, auto_error=False)
