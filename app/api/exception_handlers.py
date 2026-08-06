"""Typed FastAPI exception handlers.

The normal path for turning application errors (HTTPException, request
validation failures) into the standardized GatewayErrorResponse envelope.
app/middleware/exception_handling.py is the separate, outer safety net for
anything that isn't one of these.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.middleware.request_id import REQUEST_ID_HEADER
from app.schemas.errors import ErrorDetail, GatewayErrorResponse


def _format_validation_errors(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
        message = err.get("msg", "invalid request")
        parts.append(f"{loc}: {message}" if loc else message)
    return "; ".join(parts) or "invalid request"


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        # Route already built a GatewayErrorResponse-shaped detail (see app/api/v1/chat.py) —
        # reuse it directly rather than double-wrapping it under "detail", just backfill
        # request_id if the route didn't have one to attach. Pydantic's .model_dump() always
        # includes the key (as None when unset), so a plain `.setdefault()` would never fire —
        # this checks the *value*, not just key presence.
        body = exc.detail
        if not body["error"].get("request_id"):
            body["error"]["request_id"] = request_id
    else:
        body = GatewayErrorResponse(
            error=ErrorDetail(type="http_error", message=str(exc.detail), request_id=request_id)
        ).model_dump()

    response = JSONResponse(status_code=exc.status_code, content=body)
    if request_id:
        response.headers[REQUEST_ID_HEADER] = request_id
    return response


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    body = GatewayErrorResponse(
        error=ErrorDetail(type="validation_error", message=_format_validation_errors(exc), request_id=request_id)
    ).model_dump()

    response = JSONResponse(status_code=422, content=body)
    if request_id:
        response.headers[REQUEST_ID_HEADER] = request_id
    return response


def register_exception_handlers(app: FastAPI) -> None:
    # Registered against Starlette's base HTTPException, not fastapi.HTTPException: routing
    # failures (404/405) are raised by Starlette's router as the base class, and a handler
    # registered only for the FastAPI subclass would miss those, leaving them in FastAPI's
    # default (unwrapped) shape. The base-class handler catches both, since fastapi.HTTPException
    # is itself a subclass and exception dispatch walks the raised instance's MRO.
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
