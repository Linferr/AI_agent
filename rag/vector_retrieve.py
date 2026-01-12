from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable


def iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def l2_normalize(vec: list[float]) -> list[float]:
    s = math.sqrt(sum((x * x) for x in vec)) or 1.0
    return [x / s for x in vec]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class VectorDoc:
    id: str
    vector: list[float]
    page_title: str
    section_title: str
    summary: str
    source_url: str

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "VectorDoc":
        return VectorDoc(
            id=str(d.get("id", "")),
            vector=[float(x) for x in (d.get("vector") or [])],
            page_title=str(d.get("page_title", "")),
            section_title=str(d.get("section_title", "")),
            summary=str(d.get("summary", "")),
            source_url=str(d.get("source_url", "")),
        )


class VectorIndex:
    def __init__(self, docs: list[VectorDoc]) -> None:
        self.docs = docs
        self.norm_vectors = [l2_normalize(d.vector) for d in docs]

    @staticmethod
    def load(path: str) -> "VectorIndex":
        docs: list[VectorDoc] = []
        for obj in iter_jsonl(path):
            d = VectorDoc.from_dict(obj)
            if d.id and d.vector:
                docs.append(d)
        return VectorIndex(docs)

    def search(self, query_vector: list[float], *, top_k: int = 5) -> list[tuple[float, VectorDoc]]:
        q = l2_normalize(query_vector)
        scored: list[tuple[float, VectorDoc]] = []
        for v, d in zip(self.norm_vectors, self.docs):
            scored.append((dot(q, v), d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[: max(1, top_k)]

