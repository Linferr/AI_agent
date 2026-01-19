from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from rag.retrieve import BM25Index, load_jsonl
from rag.vector_retrieve import VectorIndex

_bm25_cache: dict[str, tuple[float, BM25Index]] = {}
_vec_cache: dict[str, tuple[float, VectorIndex]] = {}


def _get_mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except FileNotFoundError:
        return 0.0


def load_bm25(corpus_path: str) -> BM25Index:
    mtime = _get_mtime(corpus_path)
    cached = _bm25_cache.get(corpus_path)
    if cached and cached[0] == mtime:
        return cached[1]
    entries = load_jsonl(corpus_path)
    index = BM25Index(entries)
    _bm25_cache[corpus_path] = (mtime, index)
    return index


def load_vec(embeddings_path: str) -> VectorIndex:
    mtime = _get_mtime(embeddings_path)
    cached = _vec_cache.get(embeddings_path)
    if cached and cached[0] == mtime:
        return cached[1]
    index = VectorIndex.load(embeddings_path)
    _vec_cache[embeddings_path] = (mtime, index)
    return index


def embed_query(*, api_key: str, model: str, query: str) -> list[float]:
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


def search(
    *,
    query: str,
    top_k: int,
    corpus_path: str,
    embeddings_path: str,
    api_key: str,
    embedding_model: str,
    mode: str = "auto",
    allowed_url_prefixes: list[str] | None = None,
    min_score_bm25: float | None = None,
    min_score_vector: float | None = None,
) -> dict[str, Any]:
    top_k = max(1, int(top_k))
    mode = (mode or "auto").lower()

    use_vec = mode == "vector"
    if mode == "auto" and Path(embeddings_path).exists():
        use_vec = True

    if allowed_url_prefixes is None:
        prefixes = os.getenv("RAG_ALLOWED_URL_PREFIXES", "").strip()
        allowed_url_prefixes = [p.strip() for p in prefixes.split(",") if p.strip()]
    if min_score_bm25 is None:
        try:
            min_score_bm25 = float(os.getenv("RAG_MIN_SCORE_BM25", "0"))
        except ValueError:
            min_score_bm25 = 0.0
    if min_score_vector is None:
        try:
            min_score_vector = float(os.getenv("RAG_MIN_SCORE_VECTOR", "0"))
        except ValueError:
            min_score_vector = 0.0

    def _filter_results(
        rows: Iterable[dict[str, Any]], *, min_score: float, prefixes: list[str]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in rows:
            score = float(r.get("score", 0.0))
            url = str(r.get("url", "")).strip()
            if score < min_score:
                continue
            if prefixes and url and not any(url.startswith(p) for p in prefixes):
                continue
            out.append(r)
        return out

    if use_vec:
        qvec = embed_query(api_key=api_key, model=embedding_model, query=query)
        vindex = load_vec(embeddings_path)
        hits = vindex.search(qvec, top_k=top_k)
        raw = []
        for score, d in hits:
            title = d.section_title or d.page_title
            raw.append(
                {
                    "title": title,
                    "summary": d.summary,
                    "url": d.source_url,
                    "score": score,
                }
            )
        filtered = _filter_results(raw, min_score=min_score_vector, prefixes=allowed_url_prefixes)
        results = [
            {**r, "score": f"{float(r['score']):.4f}"}
            for r in filtered
        ]
        if results or mode == "vector":
            return {"mode": "vector", "results": results}

    bm25 = load_bm25(corpus_path)
    hits = bm25.search(query, top_k=top_k)
    raw = []
    for score, e in hits:
        title = e.section_title or e.page_title
        raw.append(
            {
                "title": title,
                "summary": e.summary,
                "url": e.source_url,
                "score": score,
            }
        )
    filtered = _filter_results(raw, min_score=min_score_bm25, prefixes=allowed_url_prefixes)
    results = [
        {**r, "score": f"{float(r['score']):.4f}"}
        for r in filtered
    ]
    return {"mode": "bm25", "results": results}


def get_env_paths() -> tuple[str, str]:
    corpus_path = os.getenv("RAG_CORPUS_PATH", "data/rag/aws_troubleshooting_seed.jsonl").strip()
    embeddings_path = os.getenv(
        "RAG_EMBEDDINGS_PATH", "data/index/aws_troubleshooting_seed.embeddings.jsonl"
    ).strip()
    return corpus_path, embeddings_path
