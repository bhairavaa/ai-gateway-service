"""Standardized error envelope every error response in the gateway uses."""

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    type: str
    message: str
    request_id: str | None = None


class GatewayErrorResponse(BaseModel):
    error: ErrorDetail
