"""格式转换模块 - OpenAI API 格式 ↔ Meoo 内部格式"""

import time
import uuid
from typing import Any


# ── OpenAI → Meoo ────────────────────────────────

def openai_messages_to_text(messages: list[dict]) -> str:
    """提取最后一条用户消息的文本内容"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # 支持多模态 content 数组
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
            return content if isinstance(content, str) else str(content)
    # fallback: 合并所有 user 消息
    return "\n".join(
        m.get("content", "") for m in messages
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    )


# ── Meoo → OpenAI ────────────────────────────────

def meoo_message_to_openai_chat(
    assistant_msg: dict,
    model: str = "meoo-bolt-claude",
) -> dict:
    """将 Meoo assistant 消息转为 OpenAI chat completion 响应"""
    content = assistant_msg.get("content", "")
    metadata = assistant_msg.get("metadata", {})
    usage = metadata.get("usageInfo", {})

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
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
        },
    }


def meoo_message_to_openai_chunk(
    delta_text: str,
    model: str = "meoo-bolt-claude",
    index: int = 0,
    finish_reason: str | None = None,
) -> dict:
    """将流式文本片段转为 OpenAI SSE chunk"""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": {"content": delta_text} if not finish_reason else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return chunk


# ── 模型列表（从 Meoo 抓包提取）───────────────────
#
# 文本模型: POST /api/agent/start 的 model/llmMode 字段
#   chatType 固定为 "BOLT_CLAUDE"，mode 固定为 "swarms"
# 图片模型: 通过 agent/start 发送图片生成 prompt 时触发
#
# 来源: 2026-07-05 Reqable 抓包分析

OPENAI_MODELS = [
    # ── 文本模型 ──
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


# 默认模型
DEFAULT_TEXT_MODEL = "qwen3.7-max"


def get_model_list() -> dict:
    return {
        "object": "list",
        "data": OPENAI_MODELS,
    }
