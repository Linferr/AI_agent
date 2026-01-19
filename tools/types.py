from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: dict[str, str]


@dataclass(frozen=True)
class ToolContext:
    api_key: str
    embedding_model: str
    corpus_path: str
    embeddings_path: str
    default_top_k: int
    open_url_timeout_s: float
    open_url_max_chars: int
