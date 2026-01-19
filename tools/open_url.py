from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from tools.types import ToolContext, ToolSpec


SPEC = ToolSpec(
    name="open_url",
    description="Fetch an AWS docs URL and extract concise evidence snippets for the given query.",
    args={
        "url": "string, required (must be docs.aws.amazon.com)",
        "query": "string, optional (used to extract relevant snippets)",
        "max_chars": "int, optional (max characters to return)",
    },
)


def _is_allowed(url: str) -> bool:
    p = urlparse(url)
    return p.scheme in ("http", "https") and p.netloc.endswith("docs.aws.amazon.com")


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?\.])\s+", text)
    out = [p.strip() for p in parts if p.strip()]
    return out


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9\-_]{1,30}", text)
    tokens.extend(re.findall(r"[\u4e00-\u9fff]{2,12}", text))
    return tokens


def _extract_relevant(text: str, query: str, max_snippets: int = 5) -> list[str]:
    if not query:
        return _split_sentences(text)[:max_snippets]
    q_tokens = set(_tokenize(query))
    scored: list[tuple[int, str]] = []
    for sent in _split_sentences(text):
        if not sent:
            continue
        s_tokens = set(_tokenize(sent))
        score = len(q_tokens & s_tokens)
        if score > 0:
            scored.append((score, sent))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _score, s in scored[:max_snippets]] or _split_sentences(text)[:max_snippets]


def run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    url = str(args.get("url", "")).strip()
    if not url:
        return {"error": "url is required"}
    if not _is_allowed(url):
        return {"error": "url must be under docs.aws.amazon.com"}

    query = str(args.get("query", "")).strip()
    max_chars = int(args.get("max_chars", ctx.open_url_max_chars))

    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
    resp = requests.get(url, headers=headers, timeout=ctx.open_url_timeout_s)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type.lower():
        return {"error": f"non-html content-type: {content_type}"}

    html = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    main = soup.find("main") or soup.body
    if not main:
        return {"error": "main content not found"}

    for t in main.find_all(["script", "style", "noscript"]):
        t.decompose()

    text = _clean_text(main.get_text(" ", strip=True))
    if not text:
        return {"error": "empty content"}

    snippets = _extract_relevant(text, query=query, max_snippets=5)
    merged = " ".join(snippets)
    truncated = False
    if len(merged) > max_chars:
        merged = merged[: max_chars - 1] + "…"
        truncated = True

    return {
        "url": url,
        "title": title or "Untitled",
        "snippets": snippets,
        "excerpt": merged,
        "truncated": truncated,
    }
