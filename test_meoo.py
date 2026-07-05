#!/usr/bin/env python3
"""
Meoo2API 测试脚本 — 支持直连模式和代理模式

直连模式 (默认):
    直接调用 MeooClient 与 Meoo API 通信，无需启动 API 服务器。
    需在 .env 中配置 MEOO_COOKIE。

代理模式 (指定 --base-url):
    通过已启动的 API 代理服务器进行测试。

用法:
    # 直连模式 — 对话测试
    python test_meoo.py --text --prompt "你好，介绍一下你自己"
    python test_meoo.py --text --prompt "写一首诗" --model qwen3.7-max
    python test_meoo.py --text --prompt "讲个笑话" --stream

    # 代理模式
    python test_meoo.py --text --prompt "你好" --base-url http://127.0.0.1:8000 --api-key sk-xxx

    # 列出模型
    python test_meoo.py --list-models
"""

import argparse
import asyncio
import json
import sys
import time

import httpx

# ── 参数解析 ──────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Meoo2API 测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", action="store_true", help="对话补全测试")
    group.add_argument("--list-models", action="store_true", help="列出可用模型")

    parser.add_argument("--model", type=str, default=None, help="模型名称")
    parser.add_argument("--prompt", type=str, default="", help="提示词")
    parser.add_argument("--project-id", type=str, default="", help="指定项目ID (可选)")

    # 代理模式参数
    parser.add_argument("--base-url", type=str, default="", help="API 代理地址，留空则直连 Meoo")
    parser.add_argument("--api-key", type=str, default="", help="API Key（代理模式鉴权）")

    # 对话参数
    parser.add_argument("--stream", action="store_true", help="流式输出")
    parser.add_argument("--temperature", type=float, default=None, help="温度参数")
    parser.add_argument("--max-tokens", type=int, default=None, help="最大 token 数")

    return parser.parse_args()


# ── 代理模式 HTTP 客户端逻辑 ──────────────────────


def _build_headers(api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _proxy_list_models(base_url: str, api_key: str):
    print("=" * 60)
    print("获取模型列表 (代理模式)...")
    print("=" * 60)
    try:
        resp = httpx.get(f"{base_url}/v1/models", headers=_build_headers(api_key), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"\n状态码: {resp.status_code}")
        print(f"模型数量: {len(data.get('data', []))}")
        for model in data.get("data", []):
            tag = f" [{model.get('type', '')}]" if model.get("type") else ""
            print(f"  - {model['id']}{tag}")
        print()
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        sys.exit(1)


def _proxy_chat(args):
    model = args.model or "qwen3.7-max"
    prompt = args.prompt or "你好，请介绍一下你自己"
    base_url = args.base_url.rstrip("/")

    print("=" * 60)
    print(f"对话测试 (代理模式 → {base_url})")
    print(f"  模型: {model}")
    print(f"  提示: {prompt}")
    print(f"  流式: {'是' if args.stream else '否'}")
    print("=" * 60)

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": args.stream,
    }
    if args.temperature is not None:
        body["temperature"] = args.temperature
    if args.max_tokens is not None:
        body["max_tokens"] = args.max_tokens

    try:
        if args.stream:
            _proxy_chat_stream(base_url, args.api_key, body)
        else:
            _proxy_chat_sync(base_url, args.api_key, body)
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        sys.exit(1)


def _proxy_chat_sync(base_url: str, api_key: str, body: dict):
    start = time.time()
    resp = httpx.post(
        f"{base_url}/v1/chat/completions",
        json=body,
        headers=_build_headers(api_key),
        timeout=300,
    )
    resp.raise_for_status()
    elapsed = time.time() - start
    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})

    print(f"\n✅ 回复 (耗时 {elapsed:.1f}s):")
    print("-" * 40)
    print(choice["message"]["content"])
    print("-" * 40)
    print(f"finish_reason: {choice['finish_reason']}")
    print(f"tokens: prompt={usage.get('prompt_tokens','?')}, "
          f"completion={usage.get('completion_tokens','?')}, "
          f"total={usage.get('total_tokens','?')}")
    print()


def _proxy_chat_stream(base_url: str, api_key: str, body: dict):
    print("\n📡 流式输出:")
    print("-" * 40)
    full_content = ""
    start = time.time()
    with httpx.stream("POST", f"{base_url}/v1/chat/completions", json=body, headers=_build_headers(api_key), timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("data: "):
                ds = line[6:]
                if ds == "[DONE]":
                    break
                try:
                    chunk = json.loads(ds)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        print(content, end="", flush=True)
                        full_content += content
                except json.JSONDecodeError:
                    pass
    elapsed = time.time() - start
    print(f"\n{'-' * 40}")
    print(f"✅ 流式完成 (耗时 {elapsed:.1f}s, 共 {len(full_content)} 字符)")
    print()


# ── 直连模式（直接调 MeooClient）───────────────────

# 延迟导入，避免在代理模式下触发配置校验
_meoo_client = None


def _get_meoo_client():
    global _meoo_client
    if _meoo_client is None:
        from config import config as cfg

        cfg.validate()
        from meoo_client import client as c

        _meoo_client = c
    return _meoo_client


def _direct_list_models():
    from converter import get_model_list

    print("=" * 60)
    print("模型列表 (直连模式)")
    print("=" * 60)
    data = get_model_list()
    print(f"模型数量: {len(data['data'])}")
    for model in data["data"]:
        tag = f" [{model.get('type', '')}]" if model.get("type") else ""
        print(f"  - {model['id']}{tag}")
    print()


async def _direct_chat(args):
    from converter import DEFAULT_TEXT_MODEL

    model = args.model or DEFAULT_TEXT_MODEL
    prompt = args.prompt or "你好，请介绍一下你自己"

    print("=" * 60)
    print(f"对话测试 (直连 Meoo)")
    print(f"  模型: {model}")
    print(f"  提示: {prompt}")
    print(f"  流式: {'是' if args.stream else '否'}")
    print("=" * 60)

    c = _get_meoo_client()
    project_id = args.project_id or await c.get_or_create_project()
    print(f"  项目: {project_id}")

    # 发送消息
    send_result = await c.send_message(
        message=prompt,
        project_id=project_id,
        task_id="",
        model=model,
    )
    task_id = send_result.get("taskId", "")
    print(f"  taskId: {task_id}")

    if args.stream:
        print("\n📡 流式输出:")
        print("-" * 40)
        full_content = ""
        start = time.time()
        try:
            async for delta_text in c.poll_stream(task_id):
                print(delta_text, end="", flush=True)
                full_content += delta_text
        except Exception as e:
            print(f"\n⚠️ 流式中断: {e}")
        elapsed = time.time() - start
        print(f"\n{'-' * 40}")
        print(f"✅ 流式完成 (耗时 {elapsed:.1f}s, 共 {len(full_content)} 字符)")
    else:
        print("\n⏳ 等待回复...")
        start = time.time()
        assistant_msg = await c.poll_assistant_message(task_id)
        elapsed = time.time() - start

        if assistant_msg is None:
            print("❌ 超时未收到回复")
            return

        content = assistant_msg.get("content", "")
        metadata = assistant_msg.get("metadata", {})
        usage = metadata.get("usageInfo", {})

        print(f"\n✅ 回复 (耗时 {elapsed:.1f}s):")
        print("-" * 40)
        print(content)
        print("-" * 40)
        print(f"tokens: input={usage.get('inputTokens','?')}, "
              f"output={usage.get('outputTokens','?')}, "
              f"total={usage.get('totalTokens','?')}")
        print(f"credit: {metadata.get('creditConsumed', '?')}")
    print()


# ── 主入口 ────────────────────────────────────────


def main():
    args = parse_args()

    # 判断模式
    use_proxy = bool(args.base_url)

    if args.list_models:
        if use_proxy:
            _proxy_list_models(args.base_url.rstrip("/"), args.api_key)
        else:
            _direct_list_models()
    else:
        # 默认 text
        if use_proxy:
            _proxy_chat(args)
        else:
            asyncio.run(_direct_chat(args))


if __name__ == "__main__":
    main()
