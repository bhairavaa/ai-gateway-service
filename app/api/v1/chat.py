"""POST /v1/chat/completions — the gateway's single unified endpoint.

Handles both plain JSON (stream=false) and SSE streaming (stream=true)
responses, sharing the same auth/rate-limit/provider-registry wiring — only
the response transport and fallback-commitment semantics differ, in
app/services/chat_streaming.py.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.deps import get_provider_registry, get_settings_dep
from app.config import Settings
from app.providers.registry import ProviderRegistry
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.errors import ErrorDetail, GatewayErrorResponse
from app.services.chat_service import ChatService, ProviderRequestError
from app.services.chat_streaming import stream_chat_completion

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/completions", response_model=ChatResponse)
async def create_chat_completion(
    http_request: Request,
    chat_request: ChatRequest,
    registry: ProviderRegistry = Depends(get_provider_registry),
    settings: Settings = Depends(get_settings_dep),
) -> ChatResponse | StreamingResponse:
    if chat_request.stream:
        return await stream_chat_completion(http_request, chat_request, registry, settings)

    service = ChatService(registry, settings)
    try:
        return await service.complete(http_request, chat_request)
    except ProviderRequestError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=GatewayErrorResponse(
                error=ErrorDetail(type="provider_error", message=str(exc))
            ).model_dump(),
        ) from exc
