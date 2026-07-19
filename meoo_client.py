"""Meoo API client - async communication with Meoo backend."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import AsyncIterator, Final

import httpx

from config import config

_POLL_DELAYS: Final[list[int]] = [1, 1, 2, 2, 3, 3, 4, 4, 5]
_IMAGE_MD_RE: Final[re.Pattern[str]] = re.compile(r"!\[.*?\]\((https?://[^\s\)]+)\)")
_IMAGE_URL_RE: Final[re.Pattern[str]] = re.compile(
    r"(https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp))",
    re.IGNORECASE,
)


class MeooAPIError(Exception):
    """Controlled Meoo backend / client error safe to surface publicly."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class MeooTimeoutError(MeooAPIError):
    """Raised when polling exceeds the configured timeout."""


class MeooClient:
    """Meoo API async client with project caching and status-checked HTTP."""

    def __init__(self) -> None:
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
            timeout=httpx.Timeout(120.0),
        )
        self._project_task_map: dict[str, str] = {}
        self._cached_project_id: str | None = config.MEOO_PROJECT_ID or None
        self._project_lock = asyncio.Lock()

    async def close(self) -> None:
        await self._http.aclose()

    async def _read_json(self, resp: httpx.Response) -> dict:
        """Parse JSON after enforcing HTTP success."""
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = (resp.text or "")[:300]
            raise MeooAPIError(
                f"Meoo HTTP {resp.status_code}: {body or resp.reason_phrase}",
                status_code=resp.status_code,
            ) from exc

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise MeooAPIError(
                "Meoo returned non-JSON response",
                status_code=resp.status_code,
            ) from exc

        if not isinstance(data, dict):
            raise MeooAPIError("Meoo response root must be an object")
        return data

    # -- Content security --------------------------

    async def check_text(self, text: str, app_id: str = "") -> bool:
        """Content security check; returns whether text passed."""
        if config.MEOO_SKIP_SECURITY:
            return True
        headers: dict[str, str] = {}
        if app_id:
            headers["oneday-app-id"] = app_id
        resp = await self._http.post(
            "/api/v1/content-security/check-text",
            json={"text": text},
            headers=headers,
        )
        data = await self._read_json(resp)
        payload = data.get("data")
        if not isinstance(payload, dict):
            return False
        return bool(payload.get("passed", False))

    # -- Project management ------------------------

    async def create_project(self, project_type: str = "create-desktop") -> dict:
        """Create a new project and return its data payload."""
        resp = await self._http.post(
            "/api/v1/project",
            json={"type": project_type},
        )
        data = await self._read_json(resp)
        if not data.get("success"):
            raise MeooAPIError(f"Create project failed: {data.get('message', data)}")
        project = data.get("data")
        if not isinstance(project, dict) or "url_id" not in project:
            raise MeooAPIError("Create project response missing url_id")
        return project

    async def get_or_create_project(self) -> str:
        """Return a stable project url_id, creating at most once per process."""
        if self._cached_project_id:
            return self._cached_project_id

        async with self._project_lock:
            if self._cached_project_id:
                return self._cached_project_id

            project = await self.create_project()
            url_id = str(project["url_id"])
            self._cached_project_id = url_id
            return url_id

    # -- Core: send message ------------------------

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
        Send a message to Meoo Agent via POST /api/agent/start.

        Returns the `data` object containing taskId / projectId / message.
        """
        if not task_id:
            task_id = str(uuid.uuid4())

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
        data = await self._read_json(resp)
        if not data.get("success"):
            raise MeooAPIError(f"Send message failed: {data.get('message', data)}")

        result = data.get("data")
        if not isinstance(result, dict):
            raise MeooAPIError("Send message response missing data object")

        actual_task_id = str(result.get("taskId", task_id))
        self._project_task_map[project_id] = actual_task_id
        return result

    # -- Poll messages -----------------------------

    async def fetch_messages(
        self,
        task_id: str,
        change_id: int = 0,
        page: int = 1,
        page_size: int = 1000,
    ) -> dict:
        """Fetch chat message list for a task."""
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
        return await self._read_json(resp)

    @staticmethod
    def _assistant_messages(messages: list) -> list[dict]:
        result: list[dict] = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                result.append(msg)
        return result

    @staticmethod
    def _is_final_assistant(msg: dict) -> bool:
        """True when message is complete text (not a tool-call intermediate)."""
        return msg.get("status") == "complete" and not msg.get("tool_calls")

    async def poll_assistant_message(
        self,
        task_id: str,
        timeout: int | None = None,
    ) -> dict:
        """
        Poll until a final assistant message is available.

        Raises MeooTimeoutError on timeout (never returns None).
        """
        if timeout is None:
            timeout = config.MEOO_POLL_TIMEOUT

        elapsed = 0
        delay_idx = 0

        while elapsed < timeout:
            delay = _POLL_DELAYS[min(delay_idx, len(_POLL_DELAYS) - 1)]
            await asyncio.sleep(delay)
            elapsed += delay
            delay_idx += 1

            data = await self.fetch_messages(task_id)
            payload = data.get("data")
            messages = payload.get("messages", []) if isinstance(payload, dict) else []
            if not isinstance(messages, list):
                continue

            for msg in self._assistant_messages(messages):
                if self._is_final_assistant(msg):
                    return msg

        raise MeooTimeoutError(f"Waiting for reply timed out ({timeout}s)")

    async def poll_stream(
        self,
        task_id: str,
        timeout: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Poll and yield incremental assistant text deltas.

        Skips tool-call intermediate assistants (aligned with non-stream path).
        Raises MeooTimeoutError if no complete final reply arrives in time.
        """
        if timeout is None:
            timeout = config.MEOO_POLL_TIMEOUT

        elapsed = 0
        delay_idx = 0
        last_content = ""
        saw_complete = False

        while elapsed < timeout:
            delay = _POLL_DELAYS[min(delay_idx, len(_POLL_DELAYS) - 1)]
            await asyncio.sleep(delay)
            elapsed += delay
            delay_idx += 1

            data = await self.fetch_messages(task_id)
            payload = data.get("data")
            messages = payload.get("messages", []) if isinstance(payload, dict) else []
            if not isinstance(messages, list):
                continue

            # Newest first: prefer the latest non-tool assistant bubble.
            for msg in reversed(self._assistant_messages(messages)):
                if msg.get("tool_calls"):
                    continue

                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = str(content) if content is not None else ""

                if content and content != last_content:
                    if content.startswith(last_content):
                        new_text = content[len(last_content) :]
                    else:
                        # Upstream rewrote the bubble; emit full text once.
                        new_text = content
                    if new_text:
                        yield new_text
                    last_content = content

                if self._is_final_assistant(msg):
                    saw_complete = True
                    return

        if saw_complete or last_content:
            return
        raise MeooTimeoutError(f"Waiting for reply timed out ({timeout}s)")

    # -- Image helpers (unused by routes; kept for callers) --

    @staticmethod
    def extract_image_urls(content: str) -> list[str]:
        """Extract image URLs from Meoo reply content."""
        urls: list[str] = []
        urls.extend(_IMAGE_MD_RE.findall(content))
        urls.extend(_IMAGE_URL_RE.findall(content))
        return list(dict.fromkeys(urls))

    @staticmethod
    def is_image_message(content: str) -> bool:
        """Return True when content appears to contain an image URL."""
        return bool(_IMAGE_MD_RE.search(content) or _IMAGE_URL_RE.search(content))


# Process-wide client (cookies fixed at import / process start)
client = MeooClient()
