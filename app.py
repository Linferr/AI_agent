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
from agent.loop import run_agent_chat, run_tool_chat
from rag.retrieval_service import get_env_paths, search
from tools.types import ToolContext

_dotenv_path = Path(__file__).parent / ".env"
if _dotenv_path.exists():
    load_dotenv(dotenv_path=_dotenv_path, override=True)
else:
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


class ToolChatResponse(BaseModel):
    reply: str
    model: str
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)


class AgentChatResponse(BaseModel):
    reply: str
    model: str
    steps: list[dict[str, Any]] = Field(default_factory=list)


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


def _require_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Missing DASHSCOPE_API_KEY (set it in env or .env).",
        )
    return api_key


def _build_messages(system: str | None, req: ChatRequest) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend([m.model_dump() for m in req.history])
    messages.append({"role": "user", "content": req.message})
    return messages


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


def _tool_context(api_key: str) -> ToolContext:
    corpus_path, embeddings_path = get_env_paths()
    embedding_model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v1").strip() or "text-embedding-v1"
    default_top_k = int(os.getenv("RAG_TOP_K", "5"))
    open_url_timeout_s = float(os.getenv("OPEN_URL_TIMEOUT_S", "15"))
    open_url_max_chars = int(os.getenv("OPEN_URL_MAX_CHARS", "1200"))
    return ToolContext(
        api_key=api_key,
        embedding_model=embedding_model,
        corpus_path=corpus_path,
        embeddings_path=embeddings_path,
        default_top_k=default_top_k,
        open_url_timeout_s=open_url_timeout_s,
        open_url_max_chars=open_url_max_chars,
    )


def _call_llm(api_key: str, model: str, messages: list[dict[str, str]]) -> str:
    client = DashScopeChatClient(api_key=api_key)
    return client.chat(messages=messages, model=model)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    api_key = _require_api_key()
    model = os.getenv("QWEN_MODEL", "qwen-turbo").strip() or "qwen-turbo"
    messages = _build_messages(req.system, req)
    reply = _call_llm(api_key, model, messages)
    return ChatResponse(reply=reply, model=model)


@app.post("/api/rag_chat", response_model=RAGChatResponse)
def rag_chat(req: ChatRequest) -> RAGChatResponse:
    api_key = _require_api_key()
    llm_model = os.getenv("QWEN_MODEL", "qwen-turbo").strip() or "qwen-turbo"
    embedding_model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v1").strip() or "text-embedding-v1"
    corpus_path, embeddings_path = get_env_paths()
    top_k = int(os.getenv("RAG_TOP_K", "5"))

    if not Path(corpus_path).exists():
        raise HTTPException(status_code=400, detail=f"Corpus not found: {corpus_path}")

    result = search(
        query=req.message,
        top_k=top_k,
        corpus_path=corpus_path,
        embeddings_path=embeddings_path,
        api_key=api_key,
        embedding_model=embedding_model,
        mode="auto",
    )
    sources: list[dict[str, str]] = []
    for item in result["results"]:
        sources.append(
            {
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
            }
        )

    evidence = _build_evidence_block(sources)
    system = (
        (req.system.strip() + "\n\n") if req.system else ""
    ) + (
        "你是 AWS 排障助手。你必须优先基于给定证据回答；如果证据不足，请明确说明不确定，并给出需要补充的信息。"
        "回答时请给出可执行的排查步骤，并在末尾输出参考链接列表（用证据里的链接）。\n\n"
        + evidence
    )
    messages = _build_messages(system, req)
    reply = _call_llm(api_key, llm_model, messages)
    return RAGChatResponse(
        reply=reply,
        model=llm_model,
        sources=[{"title": s["title"], "url": s.get("url", "")} for s in sources],
    )


@app.post("/api/tool_chat", response_model=ToolChatResponse)
def tool_chat(req: ChatRequest) -> ToolChatResponse:
    api_key = _require_api_key()
    model = os.getenv("QWEN_MODEL", "qwen-turbo").strip() or "qwen-turbo"
    ctx = _tool_context(api_key)

    history = [m.model_dump() for m in req.history]
    reply, tool_trace = run_tool_chat(
        api_key=api_key,
        model=model,
        message=req.message,
        history=history,
        system=req.system,
        ctx=ctx,
    )
    return ToolChatResponse(reply=reply, model=model, tool_trace=tool_trace)


@app.post("/api/agent_chat", response_model=AgentChatResponse)
def agent_chat(req: ChatRequest) -> AgentChatResponse:
    api_key = _require_api_key()
    model = os.getenv("QWEN_MODEL", "qwen-turbo").strip() or "qwen-turbo"
    ctx = _tool_context(api_key)
    max_steps = int(os.getenv("AGENT_MAX_STEPS", "3"))

    history = [m.model_dump() for m in req.history]
    reply, steps = run_agent_chat(
        api_key=api_key,
        model=model,
        message=req.message,
        history=history,
        system=req.system,
        ctx=ctx,
        max_steps=max_steps,
    )
    return AgentChatResponse(reply=reply, model=model, steps=steps)
