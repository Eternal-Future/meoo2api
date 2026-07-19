# Meoo2API

将 [秒悟(Meoo)](https://meoo.com) 内部 API 转换为 OpenAI 兼容格式的代理服务。

支持直接调用（无需启动服务器）和 HTTP 代理两种模式。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 Cookie
cp .env.example .env
# 编辑 .env，填入 MEOO_COOKIE

# 3. 直连测试（不需要启动服务器）
python test_meoo.py --text --prompt "你好"
python test_meoo.py --list-models

# 4. 或启动 HTTP 代理服务
python main.py
```

## 项目结构

```
meoo2api/
├── main.py           # FastAPI 入口，OpenAI 兼容路由
├── meoo_client.py    # Meoo API 异步客户端
├── converter.py      # OpenAI ↔ Meoo 格式转换 + 模型列表
├── config.py         # 环境变量配置
├── test_meoo.py      # 测试脚本（直连 + 代理双模式）
├── requirements.txt
├── .env.example
└── README.md
```

## 使用方式

### 直连模式（推荐测试用）

不启动服务器，直接调用 Meoo API。

```bash
# 对话
python test_meoo.py --text --prompt "你好"
python test_meoo.py --text --prompt "写一首诗" --model qwen3.7-max
python test_meoo.py --text --prompt "讲个笑话" --stream

# 列出模型
python test_meoo.py --list-models
```

### 代理模式（HTTP 服务）

```bash
# 终端 1：启动服务
python main.py
# 默认监听 127.0.0.1:8000

# 终端 2：通过代理测试
python test_meoo.py --text --prompt "你好" --base-url http://127.0.0.1:8000

# 或直接用 curl / OpenAI SDK
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"model":"qwen3.7-max"}'
```

### OpenAI SDK 兼容

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-xxx"  # 如果配置了 API_KEY 则需要
)

response = client.chat.completions.create(
    model="qwen3.7-max",
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 对话补全（支持 stream） |
| `/v1/models` | GET | 模型列表 |
| `/health` | GET | 健康检查 |

## 可用模型

### 文本模型（9个）

| 模型 ID | 说明 |
|---------|------|
| `auto` | 自动选择（默认） |
| `qwen3.7-max` | 通义千问 3.7 Max |
| `qwen3.7-plus` | 通义千问 3.7 Plus |
| `qwen3.6-plus` | 通义千问 3.6 Plus |
| `kimi-k2.5` | Kimi K2.5 |
| `glm-5.2` | 智谱 GLM 5.2 |
| `glm-5.1` | 智谱 GLM 5.1 |
| `glm-5` | 智谱 GLM 5 |
| `MiniMax-M2.5` | MiniMax M2.5 |

## 配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MEOO_COOKIE` | (必填) | Meoo 登录 Cookie |
| `API_KEY` | (空) | API 鉴权 Key，为空则不鉴权 |
| `HOST` | `127.0.0.1` | 监听地址 |
| `PORT` | `8000` | 监听端口 |
| `MEOO_BASE_URL` | `https://meoo.com` | Meoo API 地址 |
| `MEOO_PROJECT_ID` | (空) | 指定项目 ID，空则自动创建 |
| `MEOO_SKIP_SECURITY` | `true` | 跳过内容安全检测 |
| `MEOO_POLL_TIMEOUT` | `120` | 轮询超时（秒） |

### 获取 Cookie

1. 浏览器打开 https://meoo.com 并登录
2. F12 → Application → Cookies → `meoo.com`
3. 复制 `oneday_sid` 和 `login_oneday_ticket` 的值
4. 填入 `.env`：`MEOO_COOKIE=oneday_sid=xxx; login_oneday_ticket=xxx; lang=zh`

## 工作原理

```
┌──────────────┐     OpenAI Format     ┌──────────────┐     Meoo API      ┌───────────┐
│  OpenAI SDK   │ ──────────────────→  │   Proxy       │ ───────────────→  │  Meoo     │
│  / curl       │ ←──────────────────  │  (FastAPI)    │ ←───────────────  │  Server   │
└──────────────┘                       └──────────────┘                   └───────────┘
```

**对话流程：**
1. 将完整 `messages`（system / user / assistant / tool）序列化为单条 prompt → 构建 `POST /api/agent/start` 请求
2. 复用进程内缓存的 `projectId`（或 `MEOO_PROJECT_ID`），避免每次新建项目
3. 发送到 Meoo，获得 `taskId`
4. 轮询 `GET /api/v1/agent/chat/messages` 等待最终 assistant 回复（跳过 tool_calls 中间态）
5. 转为 OpenAI `choices[].message` 格式返回

**Stream 模式：**
- 轮询期间通过 SSE delta 推送新增内容；同一次响应的 `id` 保持稳定
- 超时返回 `timeout_error`，不再伪装成正常 `finish_reason=stop`

## 已知限制

- `temperature` / `max_tokens` 会被接受（兼容 OpenAI SDK），但暂未映射到 Meoo 上游参数
- 多轮历史通过拼进 prompt 传递，不是 Meoo 原生会话续聊（每次请求仍是新 `taskId`）
- 未实现 `/v1/images/generations`

## 许可

MIT
