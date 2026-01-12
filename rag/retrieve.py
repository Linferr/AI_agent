from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from typing import Iterable


STOP_WORDS = {
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "with",
    "a",
    "an",
    "is",
    "are",
    "be",
    "this",
    "that",
    "you",
    "your",
    "will",
    "can",
    "from",
    "on",
    "as",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    tokens.extend(re.findall(r"[a-z0-9][a-z0-9\-_]{1,30}", text))
    tokens.extend([t for t in re.findall(r"[\u4e00-\u9fff]{2,12}", text)])
    return [t for t in tokens if t not in STOP_WORDS]


@dataclass(frozen=True)
class Entry:
    id: str
    page_title: str
    section_title: str
    summary: str
    source_url: str
    keywords: list[str]

    @staticmethod
    def from_dict(d: dict) -> "Entry":
        return Entry(
            id=str(d.get("id", "")),
            page_title=str(d.get("page_title", "")),
            section_title=str(d.get("section_title", "")),
            summary=str(d.get("summary", "")),
            source_url=str(d.get("source_url", "")),
            keywords=list(d.get("keywords", []) or []),
        )

    def to_search_text(self) -> str:
        return " ".join(
            [
                self.page_title,
                self.section_title,
                self.summary,
                " ".join(self.keywords),
            ]
        )


def load_jsonl(path: str) -> list[Entry]:
    out: list[Entry] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Entry.from_dict(json.loads(line)))
    return out


class BM25Index:
    def __init__(self, entries: list[Entry], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.entries = entries
        self.k1 = k1
        self.b = b

        self.docs: list[list[str]] = [tokenize(e.to_search_text()) for e in entries]
        self.doc_lens = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0

        df: dict[str, int] = {}
        for doc in self.docs:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1
        self.df = df

    def idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score_doc(self, query_terms: Iterable[str], doc_idx: int) -> float:
        doc = self.docs[doc_idx]
        if not doc:
            return 0.0
        freq: dict[str, int] = {}
        for t in doc:
            freq[t] = freq.get(t, 0) + 1

        dl = self.doc_lens[doc_idx]
        denom_norm = self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))

        score = 0.0
        for term in query_terms:
            tf = freq.get(term, 0)
            if tf <= 0:
                continue
            score += self.idf(term) * (tf * (self.k1 + 1)) / (tf + denom_norm)
        return score

    def search(self, query: str, *, top_k: int = 5) -> list[tuple[float, Entry]]:
        terms = tokenize(query)
        if not terms:
            return []
        scored: list[tuple[float, Entry]] = []
        for i, e in enumerate(self.entries):
            s = self.score_doc(terms, i)
            if s > 0:
                scored.append((s, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[: max(1, top_k)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local retrieval demo (BM25) over distilled JSONL corpus.")
    p.add_argument("--corpus", default="data/rag/aws_troubleshooting.jsonl", help="JSONL corpus path.")
    p.add_argument("--query", required=True, help="Search query.")
    p.add_argument("--top-k", type=int, default=5, help="Top K results.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    entries = load_jsonl(args.corpus)
    index = BM25Index(entries)
    hits = index.search(args.query, top_k=int(args.top_k))
    if not hits:
        print("No hits.")
        return 0

    for rank, (score, e) in enumerate(hits, start=1):
        title = e.section_title or e.page_title
        print(f"{rank}. score={score:.3f} title={title}")
        print(f"   url={e.source_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

