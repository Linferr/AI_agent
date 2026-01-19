## Quickstart

### 0) 前置条件

- Python 3.10+（建议 3.11）
- 已开通阿里云 DashScope，并获取 API Key

### 1) 安装依赖

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

依赖安装检验：

```bash
python -c "import fastapi, uvicorn, dashscope, dotenv; print('deps ok')"
```

### 2) 配置环境变量

- 复制 `.env.example` 为 `.env`
- 填写 `DASHSCOPE_API_KEY`
- 可选：设置 `QWEN_MODEL`（默认 `qwen-turbo`）

配置检验（可选）：

```bash
python -c "import os; print('DASHSCOPE_API_KEY set' if os.getenv('DASHSCOPE_API_KEY') else 'missing DASHSCOPE_API_KEY')"
```

### 3) 启动

```bash
uvicorn app:app --reload --port 8000
```

打开 `http://127.0.0.1:8000/` 即可对话。

RAG/Tool/Agent 对话（需要先生成语料与 embeddings，见 `rag/README.md` 与 `rag/upgrade-embedding-vector-eval.md`）：
- 前端选择 `rag` / `tool` / `agent` 模式，或直接调用 `POST /api/rag_chat`、`POST /api/tool_chat`、`POST /api/agent_chat`
