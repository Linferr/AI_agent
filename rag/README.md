# RAG v1（Link-first）：AWS Troubleshooting 语料蒸馏

目标：把 AWS troubleshooting 文档“蒸馏”成可检索的 JSONL 语料；命中后优先返回 `source_url`（尽量带 `#anchor`），让回答可追溯到原文链接。

> 说明：本仓库已将 `data/` 加入 `.gitignore`，缓存与语料默认不提交到 GitHub。

## 最推荐的最小流程

1) 在 `rag/data.md` 放入你挑选的 AWS 链接（支持一行一个 URL，也支持 markdown 里混着链接）。
2) 运行蒸馏脚本，生成 JSONL 语料库。
3) 用本地检索（BM25）先验证“能搜到 + 能给出处链接”。

## 依赖

使用仓库依赖即可：

```bash
pip install -r requirements.txt
```

## 生成语料（蒸馏）

### A. Seed 模式（推荐：只处理你挑选的链接）

```bash
python -m rag.script --seed-file rag/data.md --out "data/rag/aws_troubleshooting_seed.jsonl"
```

### B. Crawl 模式（从入口页递归抓取）

```bash
python -m rag.script --start-url "https://docs.aws.amazon.com/zh_cn/awssupport/latest/user/troubleshooting.html" --allowed-prefix "/zh_cn/awssupport/latest/user/" --max-pages 200 --sleep 0.5 --out "data/rag/aws_troubleshooting.jsonl"
```
## 脚本参数说明

常用参数：
- `--sleep`：每次请求前等待秒数（建议保留，避免对站点造成压力）
- `--max-pages`：最多抓取页数（crawl 模式有效）
- `--no-cache`：禁用缓存（调试时可用）
- `--debug`：打印 traceback（定位解析失败原因）

## 产物在哪里？分别是什么？

脚本会自动创建目录`data/`：

- `data/cache_html/`：抓取到的原始 HTML 缓存（用于加速重复运行、排查解析问题）
- `data/rag/aws_troubleshooting_seed.jsonl`：seed 模式的蒸馏语料（推荐作为 v1 语料库）
- `data/rag/aws_troubleshooting.jsonl`：crawl 模式的蒸馏语料（覆盖更广，但噪声更高）

## 这些产物有什么用？

- JSONL 的每一行是一个“可检索条目”（通常对应一个页面小节）。
- 条目包含 `summary/keywords`（用于检索）和 `source_url`（用于回答时提供出处链接）。
- 你可以先用 `rag/retrieve.py` 做本地检索验证（使用 BM25 -- 通过关键词匹配，利用 summary 与 keywords 字段返回相关条目，不需要 embedding/向量库）。

```bash
python -m rag.retrieve --corpus "data/rag/aws_troubleshooting_seed.jsonl" --query "SigV4 AccessDenied" --top-k 5
```

- 返回结果示例：
```bash
1. score=3.897 title=Credential errors
   url=https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv-troubleshooting.html#signature-v4-troubleshooting-credential
2. score=3.343 title=Permissions
   url=https://docs.aws.amazon.com/athena/latest/ug/troubleshooting-athena.html#troubleshooting-athena-permissions
```

## JSONL 字段（每行一个 JSON 对象）

- `id`：稳定条目 ID（`page_url + section_title` hash）
- `page_title`：页面标题
- `section_title`：小节标题（优先 `<h2>`，否则 `<h3>`）
- `source_url`：原文链接（尽量带 `#anchor`）
- `summary`：启发式摘要（不调用外部 LLM）
- `key_errors`：错误短语/错误码（启发式抽取）
- `keywords`：检索关键词（启发式抽取）
- `section_path`：层级路径（当前用于组织/展示）
- `anchor_missing`：是否未可靠获取锚点

## 下一步建议（从 v1 到“真正 RAG”）

1) 在后端新增 `/api/rag_chat`：先检索 top-k，再把命中条目的 `summary + source_url` 拼进 prompt，让 LLM 基于证据回答并输出链接。
2) 升级 embedding + 向量检索：对条目（或更细的 chunk）做 embedding，再用向量库检索（本地/云均可）。
3) 做 chunking：把小节正文再切成更细粒度片段，提高召回与引用精度。
4) 建立评估集：固定 20–50 条 AWS 排障问题做离线评估，对比检索策略与 prompt 版本。
