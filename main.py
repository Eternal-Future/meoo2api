"""
Meoo2API - OpenAI compatible API proxy.

Converts Meoo internal APIs into OpenAI-compatible chat endpoints.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import config
from converter import (
    DEFAULT_TEXT_MODEL,
    get_model_list,
    meoo_message_to_openai_chat,
    meoo_message_to_openai_chunk,
    openai_messages_to_text,
)
from meoo_client import MeooAPIError, MeooTimeoutError, client

# -- Logging ---------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("meoo2api")


def _public_detail(exc: Exception, generic: str) -> str:
    """Expose controlled Meoo errors; hide unexpected internals."""
    if isinstance(exc, MeooAPIError):
        return str(exc)
    return generic


# -- Lifecycle -------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    config.validate()
    logger.info("Meoo2API starting, listen: %s:%s", config.HOST, config.PORT)
    logger.info("Upstream: %s", config.MEOO_BASE_URL)
    logger.info("API auth: %s", "enabled" if config.API_KEY else "disabled (open)")
    logger.info("Skip content security: %s", config.MEOO_SKIP_SECURITY)
    yield
    await client.close()
    logger.info("Meoo2API stopped")


app = FastAPI(
    title="Meoo2API",
    description="OpenAI-compatible API proxy for Meoo",
    version="1.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """API key middleware. Returns JSONResponse (not raise) for reliable 401/403."""
    if request.url.path == "/health":
        return await call_next(request)

    if config.API_KEY:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Missing Authorization: Bearer <api_key>",
                        "type": "auth_error",
                    }
                },
            )
        token = auth[7:]
        if token != config.API_KEY:
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "message": "Invalid API key",
                        "type": "auth_error",
                    }
                },
            )

    return await call_next(request)


# -- OpenAI request models -------------------------

class Message(BaseModel):
    role: str
    content: str | list


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_TEXT_MODEL
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    # Accepted for OpenAI SDK compatibility; Meoo has no direct mapping yet.
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# -- Routes ----------------------------------------

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return get_model_list()


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions.

    Serializes full message history into a Meoo agent/start call, then polls
    for the assistant reply (SSE when stream=true).
    """
    prompt = openai_messages_to_text([m.model_dump() for m in req.messages])
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="No non-empty messages found")

    logger.info("Chat request model=%s chars=%s", req.model, len(prompt))

    try:
        project_id = await client.get_or_create_project()
    except MeooAPIError as exc:
        logger.error("Project init failed: %s", exc)
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BROAD_EXCEPT_OK - route boundary
        logger.exception("Project init unexpected error")
        raise HTTPException(
            status_code=500,
            detail=_public_detail(exc, "Project initialization failed"),
        ) from exc

    try:
        send_result = await client.send_message(
            message=prompt,
            project_id=project_id,
            task_id="",
            model=req.model,
        )
        task_id = str(send_result.get("taskId") or "")
        if not task_id:
            raise MeooAPIError("Meoo did not return taskId")
        logger.info("Message sent taskId=%s projectId=%s", task_id, project_id)
    except MeooAPIError as exc:
        logger.error("Send failed: %s", exc)
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BROAD_EXCEPT_OK - route boundary
        logger.exception("Send unexpected error")
        raise HTTPException(
            status_code=500,
            detail=_public_detail(exc, "Failed to send message"),
        ) from exc

    if req.stream:
        return StreamingResponse(
            _stream_chat(task_id, req.model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        assistant_msg = await client.poll_assistant_message(task_id)
    except MeooTimeoutError as exc:
        logger.error("Poll timeout: %s", exc)
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except MeooAPIError as exc:
        logger.error("Poll failed: %s", exc)
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BROAD_EXCEPT_OK - route boundary
        logger.exception("Poll unexpected error")
        raise HTTPException(
            status_code=500,
            detail=_public_detail(exc, "Failed to fetch reply"),
        ) from exc

    result = meoo_message_to_openai_chat(assistant_msg, model=req.model)
    logger.info(
        "Chat done tokens=%s",
        result.get("usage", {}).get("total_tokens", "?"),
    )
    return result


async def _stream_chat(task_id: str, model: str):
    """SSE stream chat content with a stable chunk id for the whole response."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    finished_cleanly = False

    try:
        async for delta_text in client.poll_stream(task_id):
            chunk = meoo_message_to_openai_chunk(
                delta_text,
                model=model,
                finish_reason=None,
                chunk_id=chunk_id,
                created=created,
            )
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        final_chunk = meoo_message_to_openai_chunk(
            "",
            model=model,
            finish_reason="stop",
            chunk_id=chunk_id,
            created=created,
        )
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        finished_cleanly = True

    except MeooTimeoutError as exc:
        logger.error("Stream timeout: %s", exc)
        error_chunk = {
            "error": {"message": str(exc), "type": "timeout_error"},
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except MeooAPIError as exc:
        logger.error("Stream Meoo error: %s", exc)
        error_chunk = {
            "error": {"message": str(exc), "type": "upstream_error"},
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:  # noqa: BROAD_EXCEPT_OK - stream boundary
        logger.exception("Stream unexpected error")
        error_chunk = {
            "error": {
                "message": _public_detail(exc, "Stream failed"),
                "type": "stream_error",
            }
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        if not finished_cleanly:
            logger.debug("Stream closed without clean finish taskId=%s", task_id)


@app.get("/health")
async def health():
    return {"status": "ok", "meoo_base_url": config.MEOO_BASE_URL}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
    )
