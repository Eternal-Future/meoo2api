"""
Meoo2API - OpenAI 兼容 API 代理

将 Meoo(秒悟) 内部 API 转换为 OpenAI 兼容格式，
支持 /v1/chat/completions 和 /v1/images/generations

用法:
    pip install -r requirements.txt
    cp .env.example .env   # 编辑填入 MEOO_COOKIE
    python main.py
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import config
from converter import (
    DEFAULT_TEXT_MODEL,
    get_model_list,
    meoo_message_to_openai_chat,
    meoo_message_to_openai_chunk,
    openai_messages_to_text,
)
from meoo_client import client

# ── 日志 ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("meoo2api")


# ── 生命周期 ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    config.validate()
    logger.info(f"Meoo2API 启动, 监听: {config.HOST}:{config.PORT}")
    logger.info(f"代理目标: {config.MEOO_BASE_URL}")
    logger.info(f"API 鉴权: {'已启用' if config.API_KEY else '未启用（开放访问）'}")
    logger.info(f"跳过内容安全检测: {config.MEOO_SKIP_SECURITY}")
    yield
    await client.close()
    logger.info("Meoo2API 关闭")


# ── 鉴权中间件 ────────────────────────────────────


app = FastAPI(
    title="Meoo2API",
    description="OpenAI-compatible API proxy for Meoo(秒悟)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """API Key 鉴权中间件"""
    # 健康检查不鉴权
    if request.url.path == "/health":
        return await call_next(request)

    if config.API_KEY:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="缺少 Authorization: Bearer <api_key>")
        token = auth[7:]  # 去掉 "Bearer " 前缀
        if token != config.API_KEY:
            raise HTTPException(status_code=403, detail="API Key 无效")

    return await call_next(request)


# ── OpenAI 请求模型 ──────────────────────────────

class Message(BaseModel):
    role: str
    content: str | list


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_TEXT_MODEL  # qwen3.7-max
    messages: list[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ── 路由 ──────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return get_model_list()


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI 兼容的聊天补全端点
    
    将请求转换为 Meoo agent/start 调用，轮询获取结果后返回
    """
    # 1. 提取用户消息
    user_text = openai_messages_to_text(
        [m.model_dump() for m in req.messages]
    )
    if not user_text:
        raise HTTPException(status_code=400, detail="未找到用户消息")

    logger.info(f"收到对话请求: {user_text[:100]}...")

    # 2. 获取或创建项目
    try:
        project_id = await client.get_or_create_project()
    except Exception as e:
        logger.error(f"获取项目失败: {e}")
        raise HTTPException(status_code=500, detail=f"项目初始化失败: {e}")

    # 3. 发送消息
    try:
        send_result = await client.send_message(
            message=user_text,
            project_id=project_id,
            task_id="",  # 每次新会话
            model=req.model,
        )
        task_id = send_result.get("taskId", "")
        logger.info(f"消息已发送, taskId={task_id}")
    except Exception as e:
        logger.error(f"发送消息失败: {e}")
        raise HTTPException(status_code=500, detail=f"发送失败: {e}")

    # 4. 流式模式
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

    # 5. 非流式模式 - 轮询等待
    try:
        assistant_msg = await client.poll_assistant_message(task_id)
    except Exception as e:
        logger.error(f"轮询消息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取回复失败: {e}")

    if assistant_msg is None:
        raise HTTPException(
            status_code=504,
            detail=f"等待回复超时 ({config.MEOO_POLL_TIMEOUT}s)",
        )

    # 6. 转换为 OpenAI 格式
    result = meoo_message_to_openai_chat(assistant_msg, model=req.model)
    logger.info(f"对话完成, tokens={result.get('usage', {}).get('total_tokens', '?')}")
    return result


# ── 流式生成器 ────────────────────────────────────

async def _stream_chat(task_id: str, model: str):
    """SSE 流式输出聊天内容"""
    try:
        async for delta_text in client.poll_stream(task_id):
            chunk = meoo_message_to_openai_chunk(
                delta_text, model=model, finish_reason=None
            )
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # 发送结束标记
        final_chunk = meoo_message_to_openai_chunk(
            "", model=model, finish_reason="stop"
        )
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"流式输出异常: {e}")
        error_chunk = {
            "error": {"message": str(e), "type": "stream_error"}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


# ── 健康检查 ──────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "meoo_base_url": config.MEOO_BASE_URL}


# ── 启动入口 ──────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
    )
