#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWS Docs -> Link-first RAG v1 distiller

What it does:
- Crawls an AWS Docs entry page and follows nested links (same domain + allowed path prefix)
- Caches raw HTML to data/cache_html/ to avoid re-fetching
- Distills each page into section-level JSONL entries:
  - section title, heuristic summary, key error phrases, keywords, and source_url (with anchor if possible)

Only dependencies: requests + beautifulsoup4 (already in requirements.txt)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from typing import Iterable, Optional
from urllib.parse import urldefrag, urljoin, urlparse
import traceback


try:
    import requests
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: requests. Run: pip install -r requirements.txt") from e

try:
    from bs4 import BeautifulSoup
    from bs4.element import Tag
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: beautifulsoup4. Run: pip install -r requirements.txt"
    ) from e


DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stable_sha1_12(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def normalize_url(url: str) -> str:
    base, _frag = urldefrag(url)
    return base.strip()


def url_to_cache_path(cache_dir: str, url: str) -> str:
    ensure_dir(cache_dir)
    return os.path.join(cache_dir, f"{stable_sha1_12(normalize_url(url))}.html")


def read_cached(cache_dir: str, url: str) -> Optional[str]:
    path = url_to_cache_path(cache_dir, url)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write_cached(cache_dir: str, url: str, html: str) -> None:
    path = url_to_cache_path(cache_dir, url)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def fetch_with_retries(
    url: str,
    *,
    timeout_s: float,
    max_retries: int,
    backoff_base_s: float,
    user_agent: str,
) -> str:
    headers = {"User-Agent": user_agent, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type.lower():
                raise RuntimeError(f"Non-HTML content-type: {content_type}")

            charset = None
            m = re.search(r"charset=([^\s;]+)", content_type, re.I)
            if m:
                charset = m.group(1).strip().strip('"').strip("'")
            if not charset:
                # AWS docs pages are UTF-8; requests may mis-detect on Windows.
                charset = "utf-8"
            return resp.content.decode(charset, errors="replace")
        except Exception as e:  # noqa: PERF203
            last_err = e
            time.sleep(backoff_base_s * (2**attempt))
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} retries: {last_err}")


def load_html(
    url: str,
    *,
    cache_dir: str,
    use_cache: bool,
    timeout_s: float,
    max_retries: int,
    backoff_base_s: float,
    user_agent: str,
    sleep_s: float,
) -> str:
    if use_cache:
        cached = read_cached(cache_dir, url)
        if cached and cached.strip():
            return cached

    if sleep_s > 0:
        time.sleep(sleep_s)

    html = fetch_with_retries(
        url,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
        user_agent=user_agent,
    )
    if use_cache:
        write_cached(cache_dir, url, html)
    return html


def get_main_container(soup: BeautifulSoup) -> Tag:
    main = soup.find("main")
    if isinstance(main, Tag):
        return main
    body = soup.body
    if isinstance(body, Tag):
        return body
    return soup  # type: ignore[return-value]


def prune_noise(container: Tag) -> None:
    for t in container.find_all(["script", "style", "noscript"]):
        t.decompose()

    # Best-effort: remove common chrome by class/id name patterns.
    patterns = [
        re.compile(r"(breadcrumb|breadcrumbs|toc|table[-_ ]of[-_ ]contents)", re.I),
        re.compile(r"(feedback|rating|was[-_ ]this[-_ ]page|share|social)", re.I),
        re.compile(r"(side(nav|bar)|nav-?pane|menu|sidenav)", re.I),
        re.compile(r"(header|footer|masthead)", re.I),
    ]
    for t in list(container.find_all(True)):
        if not isinstance(t, Tag):
            continue
        # A tag could have been decomposed already (then attrs becomes None)
        if getattr(t, "attrs", None) is None:
            continue
        if t is container:
            continue
        cid = " ".join(t.get("class", [])) if t.get("class") else ""
        tid = t.get("id", "") or ""
        blob = f"{t.name} {cid} {tid}"
        if any(p.search(blob) for p in patterns):
            t.decompose()


def clean_text(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_page_title(soup: BeautifulSoup, container: Tag) -> str:
    t = soup.title.get_text(strip=True) if soup.title else ""
    t = clean_text(t)
    if t:
        return t
    h1 = container.find(["h1", "h2"])
    if isinstance(h1, Tag):
        return clean_text(h1.get_text(" ", strip=True))
    return "Untitled"


def iter_headings(container: Tag) -> tuple[int, list[Tag]]:
    h2s = [t for t in container.find_all("h2") if isinstance(t, Tag)]
    if h2s:
        return 2, h2s
    h3s = [t for t in container.find_all("h3") if isinstance(t, Tag)]
    if h3s:
        return 3, h3s
    return 0, []


def get_heading_text(h: Tag) -> str:
    return clean_text(h.get_text(" ", strip=True))


def collect_section_content(heading: Tag, *, boundary_level: int) -> list[Tag]:
    out: list[Tag] = []
    node = heading.next_sibling
    while node is not None:
        if isinstance(node, Tag):
            if node.name in ("h2", "h3", "h4", "h5", "h6"):
                level = int(node.name[1])
                if boundary_level and level <= boundary_level:
                    break
            out.append(node)
        node = node.next_sibling
    return out


def tags_to_blocks(tags: Iterable[Tag]) -> list[str]:
    blocks: list[str] = []
    for t in tags:
        if t.name in ("h2", "h3", "h4", "h5", "h6"):
            continue
        if t.name in ("ul", "ol"):
            items = [clean_text(li.get_text(" ", strip=True)) for li in t.find_all("li")]
            items = [x for x in items if x]
            if items:
                blocks.append("\n".join(f"- {x}" for x in items))
            continue

        txt = clean_text(t.get_text(" ", strip=True))
        if txt and len(txt) >= 20:
            blocks.append(txt)

    # De-dup consecutive blocks
    deduped: list[str] = []
    for b in blocks:
        if not deduped or deduped[-1] != b:
            deduped.append(b)
    return deduped


def infer_anchor(heading: Tag) -> tuple[Optional[str], bool]:
    hid = heading.get("id")
    if isinstance(hid, str) and hid.strip():
        return hid.strip(), False

    # Sometimes anchors are on a nested <a id="...">
    a = heading.find("a")
    if isinstance(a, Tag):
        aid = a.get("id") or a.get("name")
        if isinstance(aid, str) and aid.strip():
            return aid.strip(), False

    return None, True


def build_source_url(page_url: str, anchor: Optional[str]) -> str:
    if anchor:
        return f"{normalize_url(page_url)}#{anchor}"
    return normalize_url(page_url)


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?\.])\s+", text)
    out = [p.strip() for p in parts if p.strip()]
    return out


def summarize_heuristic(title: str, blocks: list[str], *, max_chars: int = 520) -> str:
    joined = clean_text(" ".join(blocks))
    sents = split_sentences(joined)
    if not sents:
        sents = [joined] if joined else []
    picked = []
    for s in sents:
        picked.append(s)
        if len(picked) >= 4:
            break
    summary = clean_text(" ".join(picked))
    if not summary:
        summary = title
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1] + "…"
    return summary


ERROR_PATTERNS = [
    re.compile(r"`([^`]{6,160})`"),
    re.compile(r"\b([A-Za-z]+(?:Error|Exception))\b"),
    re.compile(r"\b(AccessDenied|Unauthorized|Forbidden|InvalidSignatureException)\b", re.I),
    re.compile(r"\b(4\d\d|5\d\d)\b"),
]


def extract_key_errors(text: str, *, max_items: int = 12) -> list[str]:
    text = text.strip()
    found: list[str] = []
    for pat in ERROR_PATTERNS:
        for m in pat.finditer(text):
            s = m.group(1) if m.groups() else m.group(0)
            s = clean_text(s)
            if not s or len(s) < 4:
                continue
            if s not in found:
                found.append(s)
            if len(found) >= max_items:
                return found
    return found


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


def tokenize_for_keywords(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    tokens.extend(re.findall(r"[a-z0-9][a-z0-9\-_]{1,30}", text))
    # CJK: keep short phrases (len>=2) to reduce noise
    tokens.extend([t for t in re.findall(r"[\u4e00-\u9fff]{2,12}", text)])
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def extract_keywords(title: str, blocks: list[str], key_errors: list[str]) -> list[str]:
    raw = " ".join([title] + blocks + key_errors)
    tokens = tokenize_for_keywords(raw)
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    keywords = [k for k, _v in ranked[:20]]
    # keep at least some title tokens
    for t in tokenize_for_keywords(title):
        if t not in keywords:
            keywords.insert(0, t)
        if len(keywords) >= 20:
            break
    # de-dup preserve order
    out: list[str] = []
    for k in keywords:
        if k not in out:
            out.append(k)
    return out[:15]


@dataclass(frozen=True)
class DistilledEntry:
    id: str
    page_title: str
    section_title: str
    source_url: str
    summary: str
    key_errors: list[str]
    keywords: list[str]
    section_path: list[str]
    anchor_missing: bool

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "page_title": self.page_title,
            "section_title": self.section_title,
            "source_url": self.source_url,
            "summary": self.summary,
            "key_errors": self.key_errors,
            "keywords": self.keywords,
            "section_path": self.section_path,
            "anchor_missing": self.anchor_missing,
        }


def distill_page(page_url: str, html: str) -> list[DistilledEntry]:
    soup = BeautifulSoup(html, "html.parser")
    container = get_main_container(soup)
    prune_noise(container)

    page_title = extract_page_title(soup, container)
    boundary_level, headings = iter_headings(container)
    if not headings:
        text = clean_text(container.get_text(" ", strip=True))
        blocks = split_sentences(text)[:6]
        key_errors = extract_key_errors(text)
        keywords = extract_keywords(page_title, blocks, key_errors)
        sid = stable_sha1_12(f"{normalize_url(page_url)}|Full Page")
        return [
            DistilledEntry(
                id=sid,
                page_title=page_title,
                section_title="Full Page",
                source_url=normalize_url(page_url),
                summary=summarize_heuristic(page_title, blocks),
                key_errors=key_errors,
                keywords=keywords,
                section_path=["AWS", "Support Troubleshooting", page_title, "Full Page"],
                anchor_missing=True,
            )
        ]

    out: list[DistilledEntry] = []
    for h in headings:
        section_title = get_heading_text(h)
        if not section_title:
            continue
        tags = collect_section_content(h, boundary_level=boundary_level)
        blocks = tags_to_blocks(tags)
        if not blocks:
            continue

        raw_text = clean_text(" ".join(blocks))
        anchor, missing = infer_anchor(h)
        source_url = build_source_url(page_url, anchor)
        key_errors = extract_key_errors(raw_text)
        keywords = extract_keywords(section_title, blocks, key_errors)
        sid = stable_sha1_12(f"{normalize_url(page_url)}|{section_title}")

        out.append(
            DistilledEntry(
                id=sid,
                page_title=page_title,
                section_title=section_title,
                source_url=source_url,
                summary=summarize_heuristic(section_title, blocks),
                key_errors=key_errors,
                keywords=keywords,
                section_path=["AWS", "Support Troubleshooting", page_title, section_title],
                anchor_missing=missing,
            )
        )
    return out


def extract_links(
    page_url: str,
    html: str,
    *,
    allowed_netloc: str,
    allowed_prefix: str,
) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    container = get_main_container(soup)
    prune_noise(container)

    links: list[str] = []
    for a in container.find_all("a"):
        if not isinstance(a, Tag):
            continue
        href = a.get("href")
        if not isinstance(href, str) or not href.strip():
            continue
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue

        abs_url = urljoin(page_url, href)
        abs_url = normalize_url(abs_url)
        p = urlparse(abs_url)
        if p.scheme not in ("http", "https"):
            continue
        if p.netloc != allowed_netloc:
            continue
        if not p.path.startswith(allowed_prefix):
            continue
        lower_path = p.path.lower()
        if any(lower_path.endswith(ext) for ext in (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".zip")):
            continue

        if abs_url not in links:
            links.append(abs_url)
    return links


def write_jsonl(path: str, rows: list[dict]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crawl AWS docs and distill into Link-first RAG JSONL (v1).")
    p.add_argument(
        "--seed-file",
        default="",
        help="Optional seed file containing URLs (one per line or embedded in markdown).",
    )
    p.add_argument(
        "--start-url",
        default="https://docs.aws.amazon.com/zh_cn/awssupport/latest/user/troubleshooting.html",
        help="Start page URL (entry point).",
    )
    p.add_argument(
        "--allowed-prefix",
        default="/zh_cn/awssupport/latest/user/",
        help="Only crawl pages whose path starts with this prefix.",
    )
    p.add_argument("--max-pages", type=int, default=200, help="Maximum pages to crawl.")
    p.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between network fetches.")
    p.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds.")
    p.add_argument("--retries", type=int, default=3, help="Max retries per request.")
    p.add_argument("--backoff", type=float, default=1.0, help="Exponential backoff base seconds.")
    p.add_argument("--user-agent", default=DEFAULT_UA, help="HTTP User-Agent.")
    p.add_argument("--cache-dir", default="data/cache_html", help="Cache directory for raw HTML.")
    p.add_argument("--no-cache", action="store_true", help="Disable cache read/write.")
    p.add_argument("--out", default="data/rag/aws_troubleshooting.jsonl", help="Output JSONL path.")
    p.add_argument(
        "--no-follow",
        action="store_true",
        help="Do not follow nested links (only distill the given page(s)).",
    )
    p.add_argument("--debug", action="store_true", help="Print tracebacks for debugging.")
    return p.parse_args(argv)


def read_seed_urls(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    urls = re.findall(r"https?://[^\s)>\"]+", text)
    normed: list[str] = []
    seen: set[str] = set()
    for u in urls:
        u = normalize_url(u)
        if u and u not in seen:
            seen.add(u)
            normed.append(u)
    return normed


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    start_url = args.start_url
    allowed_prefix = args.allowed_prefix
    max_pages = max(1, int(args.max_pages))

    allowed_netloc = urlparse(start_url).netloc
    if not allowed_netloc:
        print(f"[ERR] Invalid start URL: {start_url}", file=sys.stderr)
        return 1

    cache_dir: str = args.cache_dir
    use_cache = not bool(args.no_cache)
    out_path: str = args.out
    debug: bool = bool(args.debug)
    no_follow: bool = bool(args.no_follow)

    seed_file = str(args.seed_file or "").strip()
    if seed_file:
        if not os.path.exists(seed_file):
            print(f"[ERR] seed file not found: {seed_file}", file=sys.stderr)
            return 1
        seeds = read_seed_urls(seed_file)
        if not seeds:
            print(f"[ERR] no URLs found in seed file: {seed_file}", file=sys.stderr)
            return 1
        queue: list[str] = list(seeds)
        if not no_follow:
            no_follow = True
    else:
        queue = [normalize_url(start_url)]
    visited: set[str] = set()
    distilled_rows: list[dict] = []
    failed: list[tuple[str, str]] = []

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            html = load_html(
                url,
                cache_dir=cache_dir,
                use_cache=use_cache,
                timeout_s=float(args.timeout),
                max_retries=int(args.retries),
                backoff_base_s=float(args.backoff),
                user_agent=str(args.user_agent),
                sleep_s=float(args.sleep),
            )
        except Exception as e:
            failed.append((url, f"fetch: {e}"))
            print(f"[WARN] Fetch failed: {url} ({e})", file=sys.stderr)
            if debug:
                traceback.print_exc()
            continue

        # enqueue links
        if not no_follow:
            try:
                links = extract_links(
                    url,
                    html,
                    allowed_netloc=allowed_netloc,
                    allowed_prefix=allowed_prefix,
                )
                for link in links:
                    if link not in visited and link not in queue:
                        queue.append(link)
            except Exception as e:
                failed.append((url, f"links: {e}"))
                print(f"[WARN] Link extraction failed: {url} ({e})", file=sys.stderr)
                if debug:
                    traceback.print_exc()

        # distill content
        try:
            entries = distill_page(url, html)
            distilled_rows.extend([e.to_dict() for e in entries])
            print(f"[OK] {url} -> {len(entries)} entries (queue={len(queue)})")
        except Exception as e:
            failed.append((url, f"distill: {e}"))
            print(f"[WARN] Distill failed: {url} ({e})", file=sys.stderr)
            if debug:
                traceback.print_exc()

    write_jsonl(out_path, distilled_rows)
    if failed:
        print(f"[WARN] failures={len(failed)} (use --debug for tracebacks)", file=sys.stderr)
    print(f"[DONE] pages={len(visited)} entries={len(distilled_rows)} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
