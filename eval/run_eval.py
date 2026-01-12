from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from rag.retrieve import BM25Index, load_jsonl
from rag.vector_retrieve import VectorIndex


_dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_dotenv_path):
    load_dotenv(dotenv_path=_dotenv_path, override=True)
else:
    load_dotenv(override=True)


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    expected_url_contains: str

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Case":
        return Case(
            id=str(d.get("id", "")).strip(),
            question=str(d.get("question", "")).strip(),
            expected_url_contains=str(d.get("expected_url_contains", "")).strip(),
        )


def load_cases(path: str) -> list[Case]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    out: list[Case] = []
    for item in data:
        c = Case.from_dict(item)
        if c.id and c.question and c.expected_url_contains:
            out.append(c)
    return out


def hit_at_k(urls: list[str], expected_contains: str) -> bool:
    expected_contains = expected_contains.lower()
    return any(expected_contains in (u or "").lower() for u in urls)


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline evaluation: compare BM25 vs vector retrieval.")
    p.add_argument("--cases", default="eval/aws_troubleshooting_cases.json", help="Cases JSON path.")
    p.add_argument("--corpus", default="data/rag/aws_troubleshooting_seed.jsonl", help="Corpus JSONL path.")
    p.add_argument("--embeddings", default="data/index/aws_troubleshooting_seed.embeddings.jsonl", help="Embeddings JSONL path.")
    p.add_argument("--top-k", type=int, default=5, help="Top K for hit@k.")
    p.add_argument("--out", default="data/eval/report.json", help="Output report JSON path.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases_path: str = args.cases
    corpus_path: str = args.corpus
    embeddings_path: str = args.embeddings
    top_k = max(1, int(args.top_k))
    out_path: str = args.out

    if not os.path.exists(cases_path):
        print(f"[ERR] cases not found: {cases_path}", file=sys.stderr)
        return 1
    if not os.path.exists(corpus_path):
        print(f"[ERR] corpus not found: {corpus_path}", file=sys.stderr)
        return 1

    cases = load_cases(cases_path)
    entries = load_jsonl(corpus_path)
    bm25 = BM25Index(entries)

    have_vec = os.path.exists(embeddings_path)
    embedding_model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v1").strip() or "text-embedding-v1"
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    vec_index = VectorIndex.load(embeddings_path) if have_vec else None

    if have_vec and not api_key:
        print("[ERR] embeddings exist but DASHSCOPE_API_KEY is missing (needed to embed queries).", file=sys.stderr)
        return 1

    details: list[dict[str, Any]] = []
    bm25_hits = 0
    vec_hits = 0

    for c in cases:
        bm25_results = bm25.search(c.question, top_k=top_k)
        bm25_urls = [e.source_url for _s, e in bm25_results]
        bm25_ok = hit_at_k(bm25_urls, c.expected_url_contains)
        if bm25_ok:
            bm25_hits += 1

        vec_urls: list[str] = []
        vec_ok = False
        if have_vec and vec_index is not None:
            qvec = embed_query(api_key=api_key, model=embedding_model, query=c.question)
            vec_results = vec_index.search(qvec, top_k=top_k)
            vec_urls = [d.source_url for _s, d in vec_results]
            vec_ok = hit_at_k(vec_urls, c.expected_url_contains)
            if vec_ok:
                vec_hits += 1

        details.append(
            {
                "id": c.id,
                "question": c.question,
                "expected_url_contains": c.expected_url_contains,
                "bm25": {"hit": bm25_ok, "top_urls": bm25_urls},
                "vector": {"hit": vec_ok, "top_urls": vec_urls} if have_vec else {"enabled": False},
            }
        )

    report: dict[str, Any] = {
        "cases": len(cases),
        "top_k": top_k,
        "bm25_hit_rate": bm25_hits / max(1, len(cases)),
        "vector_hit_rate": (vec_hits / max(1, len(cases))) if have_vec else None,
        "details": details,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[DONE] cases={len(cases)} bm25_hit@{top_k}={bm25_hits} vector_hit@{top_k}={(vec_hits if have_vec else 'N/A')} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
