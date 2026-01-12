from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable

from dotenv import load_dotenv


_dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_dotenv_path):
    load_dotenv(dotenv_path=_dotenv_path, override=True)
else:
    load_dotenv(override=True)


def iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_existing_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    ids: set[str] = set()
    for obj in iter_jsonl(path):
        sid = str(obj.get("id", "")).strip()
        if sid:
            ids.add(sid)
    return ids


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


@dataclass(frozen=True)
class CorpusRow:
    id: str
    page_title: str
    section_title: str
    summary: str
    source_url: str
    keywords: list[str]

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CorpusRow":
        return CorpusRow(
            id=str(d.get("id", "")).strip(),
            page_title=str(d.get("page_title", "")).strip(),
            section_title=str(d.get("section_title", "")).strip(),
            summary=str(d.get("summary", "")).strip(),
            source_url=str(d.get("source_url", "")).strip(),
            keywords=list(d.get("keywords", []) or []),
        )

    def to_embedding_text(self) -> str:
        parts = [
            f"page: {self.page_title}",
            f"section: {self.section_title}",
            f"summary: {self.summary}",
        ]
        if self.keywords:
            parts.append("keywords: " + ", ".join(map(str, self.keywords)))
        return "\n".join([p for p in parts if p.strip()])


class DashScopeEmbedder:
    def __init__(self, *, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            from dashscope import TextEmbedding
        except Exception as e:  # pragma: no cover
            raise RuntimeError("dashscope not installed. Run: pip install -r requirements.txt") from e

        resp = TextEmbedding.call(
            model=self.model,
            input=texts,
            api_key=self.api_key,
            text_type="document",
        )

        if not getattr(resp, "output", None) or "embeddings" not in resp.output:
            raise RuntimeError(f"Unexpected embedding response: {resp}")

        embeddings = resp.output["embeddings"]
        # Expect list like: [{"embedding": [...], "text_index": 0}, ...]
        vectors_by_index: dict[int, list[float]] = {}
        for item in embeddings:
            idx = int(item.get("text_index", -1))
            vec = item.get("embedding")
            if idx < 0 or not isinstance(vec, list):
                continue
            vectors_by_index[idx] = [float(x) for x in vec]

        out: list[list[float]] = []
        for i in range(len(texts)):
            vec = vectors_by_index.get(i)
            if not vec:
                raise RuntimeError(f"Missing embedding for batch index {i}")
            out.append(vec)
        return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DashScope embeddings JSONL for a distilled corpus.")
    p.add_argument("--corpus", default="data/rag/aws_troubleshooting_seed.jsonl", help="Input corpus JSONL.")
    p.add_argument("--out", default="data/index/aws_troubleshooting_seed.embeddings.jsonl", help="Output embeddings JSONL.")
    p.add_argument("--model", default=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v1"), help="Embedding model.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    p.add_argument("--no-resume", action="store_true", help="Do not resume from existing output.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # Load environment variables from .env file
    args = parse_args(argv)
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        print("[ERR] Missing DASHSCOPE_API_KEY (set it in env or .env).", file=sys.stderr)
        return 1

    corpus_path: str = args.corpus
    out_path: str = args.out
    model: str = args.model
    batch_size = max(1, int(args.batch_size))
    resume = not bool(args.no_resume)

    if not os.path.exists(corpus_path):
        print(f"[ERR] corpus not found: {corpus_path}", file=sys.stderr)
        return 1

    existing_ids = load_existing_ids(out_path) if resume else set()
    embedder = DashScopeEmbedder(api_key=api_key, model=model)

    rows: list[CorpusRow] = []
    for d in iter_jsonl(corpus_path):
        row = CorpusRow.from_dict(d)
        if not row.id:
            continue
        if row.id in existing_ids:
            continue
        rows.append(row)

    if not rows:
        print("[OK] Nothing to embed (already up-to-date).")
        return 0

    # We append by reading existing lines first, then writing full file once.
    existing: list[dict[str, Any]] = list(iter_jsonl(out_path)) if (resume and os.path.exists(out_path)) else []
    new_rows: list[dict[str, Any]] = []

    texts = [r.to_embedding_text() for r in rows]
    for batch_idx, batch in enumerate(chunked(texts, batch_size), start=1):
        vecs = embedder.embed_documents(batch)
        base = (batch_idx - 1) * batch_size
        for i, vec in enumerate(vecs):
            r = rows[base + i]
            new_rows.append(
                {
                    "id": r.id,
                    "vector": vec,
                    "page_title": r.page_title,
                    "section_title": r.section_title,
                    "summary": r.summary,
                    "source_url": r.source_url,
                    "keywords": r.keywords,
                    "embedding_model": model,
                }
            )
        print(f"[OK] embedded batch {batch_idx} ({len(batch)} texts)")

    write_jsonl(out_path, existing + new_rows)
    print(f"[DONE] embeddings_out={out_path} added={len(new_rows)} total={(len(existing) + len(new_rows))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
