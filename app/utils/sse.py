"""Server-Sent Events framing."""

import json
from typing import Any

SSE_DONE = "data: [DONE]\n\n"


def format_sse_event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"
