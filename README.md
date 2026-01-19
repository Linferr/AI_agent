# AI_agent

一个最小可运行的 LLM 应用骨架：后端用 FastAPI，对接阿里云 DashScope 通义千问；前端用一个静态页面做对话 UI。

## 1. 目录与模块（建议的工程拆分）

当前仓库可以先以“文档驱动 + 骨架先行”的方式演进。结构：

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

## 2. 路线图（按阶段可验收）

每个 Phase 都有一个可运行的最小系统与清晰的“完成标准”。

### Phase 0：仓库骨架与最小入口

- 产出：目录结构 + `app.py` 能运行（哪怕只返回固定字符串）
- 标准：能一条命令跑起来；README 说清楚目标与如何运行
- 目前进度（已完成）：
  - 后端：`app.py`（FastAPI）+ `GET /` + `POST /api/chat` + `GET /healthz`
  - 前端：`frontend/index.html`（对话 UI，支持切换 `RAG`）
  - 配置：`.env.example` + `.gitignore` + `requirements.txt` + Quickstart 文档

### Phase 1：LLM + RAG 最小闭环（先不做 Agent）

- 产出：文档 ingest + 检索 + 基于证据回答
- 标准：给定一份小文档集，问答能引用检索到的片段（至少可人工核对）
- 目前进度（已完成，AWS Troubleshooting 主题）：
  - 语料蒸馏：`rag/script.py`（seed/crawl 抓取并蒸馏为 JSONL，保留 `source_url`）
  - 检索：
    - BM25：`rag/retrieve.py`
    - 向量检索：`rag/embed.py`（DashScope embedding）+ `rag/vector_retrieve.py`
  - RAG 对话接口：`POST /api/rag_chat`（top-k 检索 → 证据注入 prompt → 回答输出链接）
  - 评估：
    - 离线检索评估：`eval/run_eval.py`（对比 BM25 vs 向量检索 hit@k）
    - output 报告：`data/eval/report.json`，可查看每条用例的 top urls 以及是否命中
  - 过程文档：`rag/README.md`、`rag/upgrade-embedding-vector-eval.md`、`rag/qa.md`

### Phase 2：加入 1–2 个工具（Tool Calling）

- 产出：工具 schema + 执行器 + LLM 能选择是否调用工具
- 标准：工具调用失败可控（超时/重试/降级），并把结果回填后继续回答
- 目前进度（未开始）：建议优先做 `open_url`（打开命中链接抓正文）和 `search`（检索语料）两类工具

### Phase 3：最小 Agent Loop（受控 ReAct）

- 产出：多步任务的循环执行（限制步数、终止条件、失败兜底）
- 标准：在“最多 N 步”约束下完成一个可复现的多步任务
- 目前进度（未开始）：建议在 Tool Calling 稳定后再做

### Phase 4：可观测 + 可评估

- 产出：trace + 离线 eval 脚本 + 一组固定用例
- 标准：能量化比较两版策略/Prompt/RAG 设置的效果差异
- 目前进度（部分完成）：
  - 检索离线评估：`eval/aws_troubleshooting_cases.json` + `eval/run_eval.py`（对比 BM25 vs 向量检索 hit@k，输出 `data/eval/report.json`）
  - 下一步：增加“回答质量”评估（是否引用证据、是否包含链接、是否胡编）
