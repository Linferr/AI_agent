from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from llm.dashscope_client import DashScopeChatClient
from rag.retrieve import BM25Index, load_jsonl
from rag.vector_retrieve import VectorIndex

_dotenv_path = Path(__file__).parent / ".env"
if _dotenv_path.exists():
    load_dotenv(dotenv_path=_dotenv_path, override=True)
else:
    # Fallback to CWD-based discovery (e.g., when running from repo root)
    load_dotenv(override=True)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    system: str | None = None


class ChatResponse(BaseModel):
    reply: str
    model: str


class RAGChatResponse(BaseModel):
    reply: str
    model: str
    sources: list[dict[str, str]] = Field(default_factory=list)


def _frontend_dir() -> Path:
    return Path(__file__).parent / "frontend"


app = FastAPI(title="AI_agent", version="0.1.0")

frontend_dir = _frontend_dir()
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
def index() -> Any:
    index_path = _frontend_dir() / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index_path)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Missing DASHSCOPE_API_KEY (set it in env or .env).",
        )

    model = os.getenv("QWEN_MODEL", "qwen-turbo").strip() or "qwen-turbo"

    messages: list[dict[str, str]] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.extend([m.model_dump() for m in req.history])
    messages.append({"role": "user", "content": req.message})

    client = DashScopeChatClient(api_key=api_key)
    reply = client.chat(messages=messages, model=model)
    return ChatResponse(reply=reply, model=model)


_bm25_cache: dict[str, tuple[float, BM25Index]] = {}
_vec_cache: dict[str, tuple[float, VectorIndex]] = {}


def _get_mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _load_bm25(corpus_path: str) -> BM25Index:
    mtime = _get_mtime(corpus_path)
    cached = _bm25_cache.get(corpus_path)
    if cached and cached[0] == mtime:
        return cached[1]
    entries = load_jsonl(corpus_path)
    index = BM25Index(entries)
    _bm25_cache[corpus_path] = (mtime, index)
    return index


def _load_vec(embeddings_path: str) -> VectorIndex:
    mtime = _get_mtime(embeddings_path)
    cached = _vec_cache.get(embeddings_path)
    if cached and cached[0] == mtime:
        return cached[1]
    index = VectorIndex.load(embeddings_path)
    _vec_cache[embeddings_path] = (mtime, index)
    return index


def _embed_query(*, api_key: str, model: str, query: str) -> list[float]:
    from dashscope import TextEmbedding

    resp = TextEmbedding.call(
        model=model,
        input=query,
        api_key=api_key,
        text_type="query",
    )
    if not getattr(resp, "output", None) or "embeddings" not in resp.output:
        raise RuntimeError(f"Unexpected embedding response: {resp}")
    item = resp.output["embeddings"][0]
    vec = item.get("embedding")
    if not isinstance(vec, list):
        raise RuntimeError(f"Missing query embedding: {resp}")
    return [float(x) for x in vec]


def _build_evidence_block(sources: list[dict[str, str]]) -> str:
    lines = ["以下是检索到的证据（请优先基于这些证据作答，并在回答末尾给出链接）："]
    for i, s in enumerate(sources, start=1):
        title = s.get("title", "").strip() or "Untitled"
        url = s.get("url", "").strip()
        summary = s.get("summary", "").strip()
        lines.append(f"{i}. {title}")
        if summary:
            lines.append(f"   摘要：{summary}")
        if url:
            lines.append(f"   链接：{url}")
    return "\n".join(lines)


@app.post("/api/rag_chat", response_model=RAGChatResponse)
def rag_chat(req: ChatRequest) -> RAGChatResponse:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Missing DASHSCOPE_API_KEY (set it in env or .env).",
        )

    llm_model = os.getenv("QWEN_MODEL", "qwen-turbo").strip() or "qwen-turbo"
    embedding_model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v1").strip() or "text-embedding-v1"

    corpus_path = os.getenv("RAG_CORPUS_PATH", "data/rag/aws_troubleshooting_seed.jsonl").strip()
    embeddings_path = os.getenv("RAG_EMBEDDINGS_PATH", "data/index/aws_troubleshooting_seed.embeddings.jsonl").strip()
    top_k = int(os.getenv("RAG_TOP_K", "5"))

    if not Path(corpus_path).exists():
        raise HTTPException(status_code=400, detail=f"Corpus not found: {corpus_path}")

    sources: list[dict[str, str]] = []

    # Prefer vector retrieval if embeddings exist, fallback to BM25.
    if Path(embeddings_path).exists():
        try:
            qvec = _embed_query(api_key=api_key, model=embedding_model, query=req.message)
            vindex = _load_vec(embeddings_path)
            hits = vindex.search(qvec, top_k=top_k)
            for _score, d in hits:
                title = d.section_title or d.page_title
                sources.append({"title": title, "summary": d.summary, "url": d.source_url})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vector retrieval failed: {e}")
    else:
        bm25 = _load_bm25(corpus_path)
        hits = bm25.search(req.message, top_k=top_k)
        for score, e in hits:
            title = e.section_title or e.page_title
            sources.append({"title": title, "summary": e.summary, "url": e.source_url, "score": f"{score:.3f}"})

    evidence = _build_evidence_block(sources)
    system = (
        (req.system.strip() + "\n\n") if req.system else ""
    ) + (
        "你是 AWS 排障助手。你必须优先基于给定证据回答；如果证据不足，请明确说明不确定，并给出需要补充的信息。"
        "回答时请给出可执行的排查步骤，并在末尾输出参考链接列表（用证据里的链接）。\n\n"
        + evidence
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend([m.model_dump() for m in req.history])
    messages.append({"role": "user", "content": req.message})

    client = DashScopeChatClient(api_key=api_key)
    reply = client.chat(messages=messages, model=llm_model)
    return RAGChatResponse(reply=reply, model=llm_model, sources=[{"title": s["title"], "url": s.get("url", "")} for s in sources])
