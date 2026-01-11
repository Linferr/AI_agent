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

load_dotenv()


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

