"""Shared helper for middleware that needs to short-circuit a request with a
standardized error response.

FastAPI's route-level exception handlers (app/api/exception_handlers.py) only
catch exceptions raised during routing/route execution — an HTTPException
raised from inside a middleware's own dispatch() propagates straight past
them, since user-added middleware sits outside that layer. So middleware that
needs to reject a request (auth, rate limiting) builds and returns the
JSONResponse directly instead of raising.
"""

from starlette.responses import JSONResponse

from app.middleware.request_id import REQUEST_ID_HEADER
from app.schemas.errors import ErrorDetail, GatewayErrorResponse


def build_error_response(
    *, status_code: int, error_type: str, message: str, request_id: str | None, headers: dict | None = None
) -> JSONResponse:
    body = GatewayErrorResponse(error=ErrorDetail(type=error_type, message=message, request_id=request_id)).model_dump()
    response = JSONResponse(status_code=status_code, content=body, headers=headers)
    if request_id:
        response.headers[REQUEST_ID_HEADER] = request_id
    return response
