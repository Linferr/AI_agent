from __future__ import annotations

from typing import Any

from rag.retrieval_service import search
from tools.types import ToolContext, ToolSpec


SPEC = ToolSpec(
    name="search_corpus",
    description="Search the local RAG corpus and return top-k evidence entries with title/summary/url.",
    args={
        "query": "string, required",
        "top_k": "int, optional (default from env)",
        "mode": "string, optional: auto|bm25|vector",
        "min_score": "float, optional (override env threshold)",
        "allowed_prefixes": "string, optional (comma-separated URL prefixes)",
    },
)


def run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "query is required"}
    top_k = int(args.get("top_k", ctx.default_top_k))
    mode = str(args.get("mode", "auto")).strip().lower() or "auto"
    min_score = args.get("min_score", None)
    allowed_prefixes_raw = str(args.get("allowed_prefixes", "")).strip()
    allowed_prefixes = [p.strip() for p in allowed_prefixes_raw.split(",") if p.strip()] or None

    result = search(
        query=query,
        top_k=top_k,
        corpus_path=ctx.corpus_path,
        embeddings_path=ctx.embeddings_path,
        api_key=ctx.api_key,
        embedding_model=ctx.embedding_model,
        mode=mode,
        min_score_vector=float(min_score) if min_score is not None else None,
        min_score_bm25=float(min_score) if min_score is not None else None,
        allowed_url_prefixes=allowed_prefixes,
    )
    return result
