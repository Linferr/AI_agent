# AI_agent

一个最小可运行的 LLM 应用骨架：后端用 FastAPI，对接阿里云 DashScope 通义千问；前端用一个静态页面做对话 UI。

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

依赖安装检验（可选）：

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

启动检验（可选）：

- 浏览器打开 `http://127.0.0.1:8000/healthz`，应返回 `{"status":"ok"}`
- PowerShell 测试接口：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/chat `
  -ContentType 'application/json' `
  -Body '{"message":"你好","history":[]}'
```

## 常见问题（依赖/配置）

- PowerShell 无法激活虚拟环境：先执行 `Set-ExecutionPolicy -Scope Process Bypass`，再运行 `.\.venv\Scripts\Activate.ps1`
- `ModuleNotFoundError: dashscope`：确认已在虚拟环境中执行 `pip install -r requirements.txt`
- `pip` 安装失败/版本过旧：先运行 `python -m pip install -U pip` 再重试
- 接口返回 `Missing DASHSCOPE_API_KEY`：在 `.env` 或系统环境变量中设置 `DASHSCOPE_API_KEY` 后重启服务

## 1. 目录与模块（建议的工程拆分）

当前仓库可以先以“文档驱动 + 骨架先行”的方式演进。建议结构：

```
AI_agent/
├── README.md
├── docs/
│   └── high-level.md
├── frontend/     # 最小对话前端（静态页面）
├── llm/          # LLM 调用与适配（provider、重试、限流、缓存）
├── rag/          # ingest / retrieve / build_context
├── tools/        # 工具定义、注册、执行与权限边界
├── agent/        # loop / planner / state
├── trace/        # 日志、trace、指标
├── eval/         # 测试用例、离线评估脚本
└── app.py        # 最小可运行入口（端到端）
```

说明：这里的目录是“规划目标”，不要求一开始全部实现；按里程碑逐步补齐即可。

## 2. 路线图（按阶段可验收）

每个 Phase 都有一个可运行的最小系统与清晰的“完成标准”。

### Phase 0（0–1 天）：仓库骨架与最小入口

- 产出：目录结构 + `app.py` 能运行（哪怕只返回固定字符串）
- 标准：能一条命令跑起来；README 说清楚目标与如何运行

### Phase 1（1–3 天）：LLM + RAG 最小闭环（先不做 Agent）

- 产出：文档 ingest + 检索 + 基于证据回答
- 标准：给定一份小文档集，问答能引用检索到的片段（至少可人工核对）

### Phase 2（4–7 天）：加入 1–2 个工具（Tool Calling）

- 产出：工具 schema + 执行器 + LLM 能选择是否调用工具
- 标准：工具调用失败可控（超时/重试/降级），并把结果回填后继续回答

### Phase 3（第 2 周）：最小 Agent Loop（受控 ReAct）

- 产出：多步任务的循环执行（限制步数、终止条件、失败兜底）
- 标准：在“最多 N 步”约束下完成一个可复现的多步任务

### Phase 4（可选但强烈建议）：可观测 + 可评估

- 产出：trace + 离线 eval 脚本 + 一组固定用例
- 标准：能量化比较两版策略/Prompt/RAG 设置的效果差异

## 3. 常见误区

- 不要把 Agent 当成“更强模型”；它首先是工程控制问题。
- 不要一上来追求“全家桶”；没有 Phase 1 的端到端闭环，上层只会更难调。
- 不要把 RAG 只当检索；关键是“回答是否被证据约束、是否可追溯”。
