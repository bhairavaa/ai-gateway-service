"""Assigns a unique ID to every request.

Exposed both to downstream code (`request.state.request_id` — read by logging,
error responses, and later the request-log/cost-tracking persistence) and to
the client (`X-Request-ID` response header, for support/debugging correlation).
"""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Honor an inbound request id (e.g. from an upstream load balancer) so traces correlate
        # end-to-end; otherwise mint a new one.
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
