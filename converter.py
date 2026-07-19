"""Format conversion: OpenAI API shapes <-> Meoo internal shapes."""

from __future__ import annotations

import time
import uuid
from typing import Any, Final

# Default model ids used when callers omit model.
DEFAULT_TEXT_MODEL: Final[str] = "qwen3.7-max"
_DEFAULT_CHUNK_MODEL: Final[str] = "meoo-bolt-claude"


def _content_to_text(content: Any) -> str:
    """Normalize OpenAI message content (string or multimodal parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if isinstance(text, str) and text:
                    texts.append(text)
            elif isinstance(part, str) and part:
                texts.append(part)
        return "\n".join(texts)
    return str(content)


def openai_messages_to_text(messages: list[dict]) -> str:
    """
    Serialize a full OpenAI messages array into one Meoo prompt.

    Includes system / user / assistant / tool turns so multi-turn clients
    do not silently drop history.
    """
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        text = _content_to_text(msg.get("content"))
        if not text.strip():
            continue

        match role:
            case "system":
                label = "System"
            case "assistant":
                label = "Assistant"
            case "user":
                label = "User"
            case "tool":
                label = "Tool"
            case "function":
                label = "Function"
            case _:
                label = role

        parts.append(f"[{label}]\n{text}")

    return "\n\n".join(parts)


def meoo_message_to_openai_chat(
    assistant_msg: dict,
    model: str = _DEFAULT_CHUNK_MODEL,
) -> dict:
    """Convert a Meoo assistant message into an OpenAI chat.completion object."""
    content = assistant_msg.get("content", "")
    if not isinstance(content, str):
        content = str(content) if content is not None else ""

    metadata = assistant_msg.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    usage = metadata.get("usageInfo")
    if not isinstance(usage, dict):
        usage = {}

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(usage.get("inputTokens") or 0),
            "completion_tokens": int(usage.get("outputTokens") or 0),
            "total_tokens": int(usage.get("totalTokens") or 0),
        },
    }


def meoo_message_to_openai_chunk(
    delta_text: str,
    model: str = _DEFAULT_CHUNK_MODEL,
    index: int = 0,
    finish_reason: str | None = None,
    *,
    chunk_id: str | None = None,
    created: int | None = None,
) -> dict:
    """
    Convert a streaming text delta into an OpenAI chat.completion.chunk.

    `chunk_id` / `created` should stay stable for the whole stream.
    """
    if chunk_id is None:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    if created is None:
        created = int(time.time())

    delta: dict[str, str]
    if finish_reason is None:
        delta = {"content": delta_text}
    else:
        delta = {}

    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


# -- Model list (from Meoo capture) ----------------
#
# Text models: POST /api/agent/start model / llmMode fields
#   chatType fixed to "BOLT_CLAUDE", mode fixed to "swarms"
#
# Source: 2026-07-05 Reqable capture analysis

OPENAI_MODELS: Final[list[dict[str, object]]] = [
    {
        "id": "auto",
        "object": "model",
        "created": 1720000000,
        "owned_by": "meoo",
        "type": "text",
    },
    {
        "id": "qwen3.7-max",
        "object": "model",
        "created": 1720000000,
        "owned_by": "meoo",
        "type": "text",
    },
    {
        "id": "qwen3.7-plus",
        "object": "model",
        "created": 1720000000,
        "owned_by": "meoo",
        "type": "text",
    },
    {
        "id": "qwen3.6-plus",
        "object": "model",
        "created": 1720000000,
        "owned_by": "meoo",
        "type": "text",
    },
    {
        "id": "kimi-k2.5",
        "object": "model",
        "created": 1720000000,
        "owned_by": "meoo",
        "type": "text",
    },
    {
        "id": "glm-5.2",
        "object": "model",
        "created": 1720000000,
        "owned_by": "meoo",
        "type": "text",
    },
    {
        "id": "glm-5.1",
        "object": "model",
        "created": 1720000000,
        "owned_by": "meoo",
        "type": "text",
    },
    {
        "id": "glm-5",
        "object": "model",
        "created": 1720000000,
        "owned_by": "meoo",
        "type": "text",
    },
    {
        "id": "MiniMax-M2.5",
        "object": "model",
        "created": 1720000000,
        "owned_by": "meoo",
        "type": "text",
    },
]


def get_model_list() -> dict:
    """Return OpenAI-compatible /v1/models payload."""
    return {
        "object": "list",
        "data": list(OPENAI_MODELS),
    }
