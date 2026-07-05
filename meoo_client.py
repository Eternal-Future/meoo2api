"""Meoo API 客户端 - 封装与 Meoo 后端的通信"""

import asyncio
import json
import re
import uuid
from typing import Optional

import httpx

from config import config


class MeooClient:
    """Meoo API 异步客户端"""

    def __init__(self):
        self._http = httpx.AsyncClient(
            base_url=config.MEOO_BASE_URL,
            cookies=config.parse_cookies(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/149.0.0.0 Safari/537.36"
                ),
                "Accept": "*/*",
                "Origin": config.MEOO_BASE_URL,
            },
            timeout=httpx.Timeout(120),
        )
        self._project_task_map: dict[str, str] = {}  # projectId -> taskId

    async def close(self):
        await self._http.aclose()

    # ── 内容安全检测 ──────────────────────────────

    async def check_text(self, text: str, app_id: str = "") -> bool:
        """内容安全检测，返回是否通过"""
        if config.MEOO_SKIP_SECURITY:
            return True
        headers = {}
        if app_id:
            headers["oneday-app-id"] = app_id
        resp = await self._http.post(
            "/api/v1/content-security/check-text",
            json={"text": text},
            headers=headers,
        )
        data = resp.json()
        return data.get("data", {}).get("passed", False)

    # ── 项目管理 ──────────────────────────────────

    async def create_project(self, project_type: str = "create-desktop") -> dict:
        """创建新项目，返回 {id, url_id, ...}"""
        resp = await self._http.post(
            "/api/v1/project",
            json={"type": project_type},
        )
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"创建项目失败: {data}")
        return data["data"]

    async def get_or_create_project(self) -> str:
        """获取或创建项目，返回 url_id"""
        if config.MEOO_PROJECT_ID:
            return config.MEOO_PROJECT_ID

        resp = await self._http.get("/api/v1/user/env")
        # 创建新项目
        project = await self.create_project()
        return project["url_id"]

    # ── 核心：发送消息 ────────────────────────────

    async def send_message(
        self,
        message: str,
        project_id: str,
        task_id: str = "",
        mode: str = "swarms",
        chat_type: str = "BOLT_CLAUDE",
        port: int = 3000,
        model: str = "auto",
    ) -> dict:
        """
        发送消息到 Meoo Agent，返回 {taskId, projectId, message}
        
        这是 Meoo 的核心 API：POST /api/agent/start
        对话和图片生成都走这个接口
        """
        if not task_id:
            task_id = str(uuid.uuid4())

        # 根据模型决定 llmInfo
        is_auto = model == "auto"
        llm_mode = model if not is_auto else "auto"
        model_source = "INIT_FROM_CONFIG" if is_auto else "PATCH_STATE"

        body = {
            "question": message,
            "chatType": chat_type,
            "args": {
                "inputs": {"create_mode": "create-desktop"},
                "skillsSnapshot": [],
            },
            "role": "user",
            "taskId": task_id,
            "message": message,
            "model": model,
            "mode": mode,
            "port": port,
            "projectId": project_id,
            "skills": [],
            "mcpServers": {"baseUrl": "", "mcpServers": {}},
            "message_owner": {
                "user_id": "",
                "username": "API",
                "avatar": "",
            },
            "createMode": "create-desktop",
            "extraParams": {
                "args": {
                    "inputs": {"create_mode": "create-desktop"},
                    "skillsSnapshot": [],
                },
                "llmInfo": {
                    "createMode": "create-desktop",
                    "llmMode": llm_mode,
                    "modelSource": model_source,
                },
            },
        }

        headers = {"oneday-app-id": project_id}
        resp = await self._http.post(
            "/api/agent/start",
            json=body,
            headers=headers,
        )
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"发送消息失败: {data}")

        result = data["data"]
        actual_task_id = result.get("taskId", task_id)
        self._project_task_map[project_id] = actual_task_id
        return result

    # ── 轮询获取消息 ──────────────────────────────

    async def fetch_messages(
        self,
        task_id: str,
        change_id: int = 0,
        page: int = 1,
        page_size: int = 1000,
    ) -> dict:
        """获取会话消息列表"""
        params = {
            "change_id": change_id,
            "task_id": task_id,
            "page": page,
            "page_size": page_size,
        }
        resp = await self._http.get(
            "/api/v1/agent/chat/messages",
            params=params,
        )
        return resp.json()

    async def poll_assistant_message(
        self,
        task_id: str,
        timeout: int = None,
    ) -> dict | None:
        """
        轮询等待 assistant 回复（文本对话用）
        
        取第一个 complete 状态的 assistant 消息即返回。
        返回 assistant 消息 dict，超时返回 None
        """
        if timeout is None:
            timeout = config.MEOO_POLL_TIMEOUT

        delays = [1, 1, 2, 2, 3, 3, 4, 4, 5]  # 指数退避, max 5s
        elapsed = 0
        delay_idx = 0

        while elapsed < timeout:
            delay = delays[min(delay_idx, len(delays) - 1)]
            await asyncio.sleep(delay)
            elapsed += delay
            delay_idx += 1

            data = await self.fetch_messages(task_id)
            messages = data.get("data", {}).get("messages", [])

            # 找第一条 status=complete 的 assistant 消息
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("status") == "complete":
                    # 如果有 tool_calls，跳过（等下一个 assistant）
                    if msg.get("tool_calls"):
                        continue
                    return msg

        return None

    async def poll_stream(
        self,
        task_id: str,
        timeout: int = None,
    ):
        """
        轮询并流式返回 assistant 回复内容（async generator）
        
        Yields: str - 每次 yield 新增的文本
        """
        if timeout is None:
            timeout = config.MEOO_POLL_TIMEOUT

        delays = [1, 1, 2, 2, 3, 3, 4, 4, 5]
        elapsed = 0
        delay_idx = 0
        last_content = ""

        while elapsed < timeout:
            delay = delays[min(delay_idx, len(delays) - 1)]
            await asyncio.sleep(delay)
            elapsed += delay
            delay_idx += 1

            data = await self.fetch_messages(task_id)
            messages = data.get("data", {}).get("messages", [])

            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and content != last_content:
                        # 返回新增内容
                        new_text = content[len(last_content):]
                        if new_text:
                            yield new_text
                        last_content = content
                        if msg.get("status") == "complete":
                            return
                    elif msg.get("status") == "complete" and last_content:
                        return

    # ── 图片 URL 提取 ─────────────────────────────

    @staticmethod
    def extract_image_urls(content: str) -> list[str]:
        """从 Meoo 回复中提取图片 URL"""
        urls = []
        # Markdown 图片语法: ![alt](url)
        md_pattern = r'!\[.*?\]\((https?://[^\s\)]+)\)'
        urls.extend(re.findall(md_pattern, content))
        # 直接的图片 URL
        direct_pattern = r'(https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp))'
        urls.extend(re.findall(direct_pattern, content, re.IGNORECASE))
        # 去重
        return list(dict.fromkeys(urls))

    @staticmethod
    def is_image_message(content: str) -> bool:
        """判断 content 是否包含图片"""
        return bool(
            re.search(r'!\[.*?\]\(https?://', content) or
            re.search(r'https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp)', content, re.IGNORECASE)
        )


# 全局客户端实例
client = MeooClient()
