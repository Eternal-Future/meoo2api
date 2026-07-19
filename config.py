"""Configuration loaded from environment variables / .env."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


class Config:
    """Process-wide configuration snapshot (read at import time)."""

    # -- Server --
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = _env_int("PORT", "8000")

    # -- Auth (empty => disabled) --
    API_KEY: str = os.getenv("API_KEY", "")

    # -- Meoo backend --
    MEOO_BASE_URL: str = os.getenv("MEOO_BASE_URL", "https://meoo.com")
    MEOO_COOKIE: str = os.getenv("MEOO_COOKIE", "")
    MEOO_PROJECT_ID: str = os.getenv("MEOO_PROJECT_ID", "")
    MEOO_SKIP_SECURITY: bool = os.getenv("MEOO_SKIP_SECURITY", "true").lower() == "true"
    MEOO_POLL_TIMEOUT: int = _env_int("MEOO_POLL_TIMEOUT", "120")

    @classmethod
    def parse_cookies(cls) -> dict[str, str]:
        """Parse Cookie header string into a dict."""
        cookies: dict[str, str] = {}
        if not cls.MEOO_COOKIE:
            return cookies
        for item in cls.MEOO_COOKIE.split(";"):
            item = item.strip()
            if "=" not in item:
                continue
            key, _, val = item.partition("=")
            cookies[key.strip()] = val.strip()
        return cookies

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration before serving traffic."""
        if not cls.MEOO_COOKIE:
            raise ValueError(
                "MEOO_COOKIE is not set; configure the environment or .env file"
            )
        cookies = cls.parse_cookies()
        required = ["oneday_sid", "login_oneday_ticket"]
        missing = [k for k in required if k not in cookies]
        if missing:
            raise ValueError(f"Cookie missing required fields: {missing}")
        if cls.MEOO_POLL_TIMEOUT <= 0:
            raise ValueError("MEOO_POLL_TIMEOUT must be > 0")


config = Config()
