# RAG 升级：Embedding + 向量检索 + `/api/rag_chat` + 离线评估

本仓库已经有 RAG v1 的“蒸馏语料”（JSONL）与 BM25 本地检索。本文档把下一步升级串起来：

1) 对语料做 embedding，生成向量索引（本地 JSONL）
2) 后端新增 `/api/rag_chat`：先检索 top-k，再把 `summary + source_url` 注入 prompt，让 LLM 基于证据回答并输出链接
3) 建立离线评估集：对比 BM25 与向量检索的 hit@k

## 0. 前置条件

- 已生成语料（推荐 seed 模式）：`data/rag/aws_troubleshooting_seed.jsonl`
- 已配置 `DASHSCOPE_API_KEY`（用于 embedding + LLM）

## 1) 生成向量索引（embedding）

对每条语料（一个 section）生成 embedding，并输出到：
- `data/index/aws_troubleshooting_seed.embeddings.jsonl`

命令：

```bash
python -m rag.embed --corpus "data/rag/aws_troubleshooting_seed.jsonl" --out "data/index/aws_troubleshooting_seed.embeddings.jsonl"
```

可选参数：
- `DASHSCOPE_EMBEDDING_MODEL`：embedding 模型名（默认 `text-embedding-v1`）
- `--batch-size`：批大小（默认 16）
- `--no-resume`：不基于已有输出断点续跑

验证方法（看输出是否有向量字段）：

```powershell
Get-Content -TotalCount 1 -Encoding UTF8 data/index/aws_troubleshooting_seed.embeddings.jsonl
```

## 2) 后端：`/api/rag_chat` 如何工作

已实现：`app.py` 新增 `/api/rag_chat`。

请求流程：
1) 读取语料 `RAG_CORPUS_PATH`（默认 `data/rag/aws_troubleshooting_seed.jsonl`）
2) 如果存在 `RAG_EMBEDDINGS_PATH`（默认 `data/index/aws_troubleshooting_seed.embeddings.jsonl`），优先走“向量检索”
   - 用 `TextEmbedding` 为 query 生成向量
   - 对比语料向量余弦相似度，取 top-k
3) 如果 embeddings 不存在，则 fallback 到 BM25（`rag/retrieve.py` 的实现）
4) 将 top-k 命中的 `title + summary + source_url` 拼成 evidence block，注入 system prompt
5) 调用通义千问（`QWEN_MODEL`）生成回答

环境变量：
- `RAG_CORPUS_PATH`：语料 JSONL 路径
- `RAG_EMBEDDINGS_PATH`：向量索引 JSONL 路径
- `RAG_TOP_K`：检索条数（默认 5）
- `QWEN_MODEL`：LLM 模型（默认 `qwen-turbo`）
- `DASHSCOPE_EMBEDDING_MODEL`：embedding 模型（默认 `text-embedding-v1`）

验证方法（不依赖前端）：

```powershell
uvicorn app:app --reload --port 8000

Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/rag_chat `
  -ContentType 'application/json' `
  -Body '{"message":"SigV4 签名不匹配怎么排查？","history":[]}'
```

预期现象：
- 返回的 `reply` 中会有排障步骤
- 回答末尾会包含若干参考链接（来自 evidence）

## 3) 离线评估：对比检索策略与 prompt 版本

本仓库提供了一个“检索离线评估”脚手架（不评判答案质量，先评判检索是否命中）：

- 用例集：`eval/aws_troubleshooting_cases.json`
  - 每条包含：`question` + `expected_url_contains`
- 评估脚本：`eval/run_eval.py`
  - 计算 `hit@k`：top-k 结果的 URL 是否包含期望字符串
  - 输出报告：默认 `data/eval/report.json`

运行：

```bash
python -m eval.run_eval --cases "eval/aws_troubleshooting_cases.json" --corpus "data/rag/aws_troubleshooting_seed.jsonl" --embeddings "data/index/aws_troubleshooting_seed.embeddings.jsonl" --top-k 5 --out "data/eval/report.json"
```

验证方法：
- 控制台会打印 `bm25_hit@k` 与 `vector_hit@k`
- 打开 `data/eval/report.json` 查看每条用例的 top urls 以及是否命中
- 报告示例：
```bash
  "cases": 20,
  "top_k": 5,
  "bm25_hit_rate": 0.9,
  "vector_hit_rate": 1.0
  ```

## 4) 迭代建议

1) **先把检索命中率做稳**：完善 `eval/aws_troubleshooting_cases.json`（补到 20–50 条），并不断改进蒸馏字段（`summary/keywords`）。
2) **加 chunking**：目前按 `<h2>/<h3>` 切；可以把正文再切成不同tokens（300-800）的 chunk，以提升引用精度。
3) **答案评估**：在检索稳定后，想办法提高答案质量。
   - 先用更多用例跑离线评估，观察回答质量。
   - 考虑引入 LLM judge 或人工评分，对“回答质量/引用准确性/是否胡编”做评估。
   - 基于评估结果，调整 prompt 设计与蒸馏字段。
   - 
4）**多轮 RAG**：目前是单轮问答，可以考虑多轮对话场景下的 RAG（需要设计上下文管理与检索策略）。


