"""配置管理模块 - 从环境变量读取 Meoo API 代理配置"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """全局配置单例"""

    # ── 服务配置 ──
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # ── 鉴权配置 ──
    # API Key，为空则不启用鉴权
    API_KEY: str = os.getenv("API_KEY", "")

    # ── Meoo 后端配置 ──
    MEOO_BASE_URL: str = os.getenv("MEOO_BASE_URL", "https://meoo.com")

    # 认证 Cookie（必须包含 oneday_sid, login_oneday_ticket）
    MEOO_COOKIE: str = os.getenv("MEOO_COOKIE", "")

    # 默认项目 ID（可选）
    MEOO_PROJECT_ID: str = os.getenv("MEOO_PROJECT_ID", "")

    # 是否跳过内容安全检测
    MEOO_SKIP_SECURITY: bool = os.getenv("MEOO_SKIP_SECURITY", "true").lower() == "true"

    # 轮询超时（秒）
    MEOO_POLL_TIMEOUT: int = int(os.getenv("MEOO_POLL_TIMEOUT", "120"))

    @classmethod
    def parse_cookies(cls) -> dict:
        """将 Cookie 字符串解析为字典"""
        cookies = {}
        if cls.MEOO_COOKIE:
            for item in cls.MEOO_COOKIE.split(";"):
                item = item.strip()
                if "=" in item:
                    key, _, val = item.partition("=")
                    cookies[key.strip()] = val.strip()
        return cookies

    @classmethod
    def validate(cls):
        """验证必要配置"""
        if not cls.MEOO_COOKIE:
            raise ValueError("MEOO_COOKIE 未设置，请配置环境变量或在 .env 文件中设置")
        cookies = cls.parse_cookies()
        required = ["oneday_sid", "login_oneday_ticket"]
        missing = [k for k in required if k not in cookies]
        if missing:
            raise ValueError(f"Cookie 缺少必要字段: {missing}")


config = Config()
