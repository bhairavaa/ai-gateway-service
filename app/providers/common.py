"""Helpers shared by every provider adapter.

Kept here instead of duplicated across openai_provider.py/claude_provider.py/
gemini_provider.py/ollama_provider.py because LangChain's stable provider
packages standardize both message construction and the `usage_metadata` shape
— every adapter needs the exact same two conversions.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.providers.base import UsageInfo
from app.schemas.chat import ChatMessage

_ROLE_TO_LC = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}


def to_langchain_messages(messages: list[ChatMessage]) -> list:
    return [_ROLE_TO_LC[m.role](content=m.content) for m in messages]


def usage_from_metadata(usage_metadata: dict | None) -> UsageInfo | None:
    if not usage_metadata:
        return None
    return UsageInfo(
        prompt_tokens=usage_metadata.get("input_tokens", 0),
        completion_tokens=usage_metadata.get("output_tokens", 0),
        total_tokens=usage_metadata.get("total_tokens", 0),
    )


def normalize_content(content: str | list) -> str:
    """LangChain's AIMessage(Chunk).content is typed `str | list[str | dict]` — Anthropic in
    particular returns a list of content blocks (not a plain string) whenever the response
    includes more than a single text block, e.g. `[{"type": "text", "text": "..."}]`, or mixes
    in non-text blocks. The gateway's response schema is plain text only, so this extracts and
    concatenates just the text parts, silently dropping block types it can't represent
    (tool_use, thinking, images, ...) rather than crashing on them.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)
