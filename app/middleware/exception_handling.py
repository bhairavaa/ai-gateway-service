"""Outermost middleware — the last line of defense.

Typed errors (HTTPException, RequestValidationError) never reach this — they're
handled by app/api/exception_handlers.py, registered via FastAPI's own
exception-handler mechanism, which runs *inside* this middleware (further in,
closer to the route). This middleware exists purely to catch anything that
escapes every inner layer — a genuine bug, an exception type nobody registered
a handler for — so a client never sees a raw traceback or an unstructured 500.
"""

import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.middleware.error_response import build_error_response

logger = logging.getLogger("app.errors")


class ExceptionHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)
        except Exception:
            request_id = getattr(request.state, "request_id", None)
            logger.exception(
                "Unhandled exception on %s %s [request_id=%s]", request.method, request.url.path, request_id
            )
            return build_error_response(
                status_code=500,
                error_type="internal_error",
                message="An unexpected error occurred",
                request_id=request_id,
            )
