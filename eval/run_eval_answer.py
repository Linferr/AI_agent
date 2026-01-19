from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv


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
    must_mention: list[str]

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Case":
        return Case(
            id=str(d.get("id", "")).strip(),
            question=str(d.get("question", "")).strip(),
            expected_url_contains=str(d.get("expected_url_contains", "")).strip(),
            must_mention=list(d.get("must_mention", []) or []),
        )


def load_cases(path: str) -> list[Case]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    out: list[Case] = []
    for item in data:
        c = Case.from_dict(item)
        if c.id and c.question:
            out.append(c)
    return out


def extract_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)>\"]+", text or "")
    # de-dup preserve order
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def extract_source_urls(resp: dict[str, Any]) -> list[str]:
    urls: list[str] = []

    # rag_chat style
    sources = resp.get("sources") or []
    for s in sources:
        url = str(s.get("url", "")).strip()
        if url:
            urls.append(url)

    # tool_chat style
    tool_trace = resp.get("tool_trace") or []
    for t in tool_trace:
        result = t.get("result", {})
        urls.extend(_urls_from_result(result))

    # agent_chat style
    steps = resp.get("steps") or []
    for s in steps:
        result = s.get("result", {})
        urls.extend(_urls_from_result(result))

    # de-dup preserve order
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _urls_from_result(result: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    if not isinstance(result, dict):
        return urls
    if "url" in result:
        u = str(result.get("url", "")).strip()
        if u:
            urls.append(u)
    if "results" in result and isinstance(result["results"], list):
        for r in result["results"]:
            u = str(r.get("url", "")).strip()
            if u:
                urls.append(u)
    return urls


def normalize(text: str) -> str:
    return (text or "").lower()


def coverage_ratio(answer: str, must_mention: list[str]) -> tuple[float, list[str]]:
    if not must_mention:
        return 1.0, []
    answer_norm = normalize(answer)
    missing: list[str] = []
    hit = 0
    for term in must_mention:
        t = normalize(term)
        if t and t in answer_norm:
            hit += 1
        else:
            missing.append(term)
    return hit / max(1, len(must_mention)), missing


def is_uncertain(answer: str) -> bool:
    markers = ["不确定", "无法", "需要更多", "缺少信息", "不足以判断", "无法确定"]
    ans = answer or ""
    return any(m in ans for m in markers)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Answer-level evaluation (citation + coverage).")
    p.add_argument("--cases", default="eval/aws_troubleshooting_cases.json", help="Cases JSON path.")
    p.add_argument("--mode", default="rag", help="chat|rag|tool|agent")
    p.add_argument("--endpoint", default="", help="Override endpoint URL.")
    p.add_argument("--out", default="data/eval/report_answer.json", help="Output report JSON path.")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    p.add_argument("--max-cases", type=int, default=0, help="Max cases to run (0 means all).")
    return p.parse_args(argv)


def endpoint_for_mode(mode: str) -> str:
    mode = (mode or "rag").lower()
    if mode == "chat":
        return "http://127.0.0.1:8000/api/chat"
    if mode == "tool":
        return "http://127.0.0.1:8000/api/tool_chat"
    if mode == "agent":
        return "http://127.0.0.1:8000/api/agent_chat"
    return "http://127.0.0.1:8000/api/rag_chat"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_cases(args.cases)
    if args.max_cases and args.max_cases > 0:
        cases = cases[: int(args.max_cases)]

    endpoint = args.endpoint.strip() or endpoint_for_mode(args.mode)

    details: list[dict[str, Any]] = []
    cases_with_sources = 0
    citation_ok = 0
    coverage_sum = 0.0
    answer_url_rate = 0
    uncertain_count = 0

    for idx, c in enumerate(cases, start=1):
        payload = {"message": c.question, "history": []}
        resp = requests.post(endpoint, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer = str(data.get("reply", "")).strip()

        source_urls = extract_source_urls(data)
        answer_urls = extract_urls(answer)

        if answer_urls:
            answer_url_rate += 1

        citation_in_answer = any(u in answer for u in source_urls) if source_urls else False
        answer_urls_only_sources = all(u in source_urls for u in answer_urls) if answer_urls else True
        citation_consistent = citation_in_answer and answer_urls_only_sources

        if source_urls:
            cases_with_sources += 1
            if citation_consistent:
                citation_ok += 1

        cov_ratio, missing = coverage_ratio(answer, c.must_mention)
        coverage_sum += cov_ratio

        if is_uncertain(answer):
            uncertain_count += 1

        details.append(
            {
                "id": c.id,
                "question": c.question,
                "expected_url_contains": c.expected_url_contains,
                "must_mention": c.must_mention,
                "missing_terms": missing,
                "coverage_ratio": round(cov_ratio, 3),
                "source_urls": source_urls,
                "answer_urls": answer_urls,
                "citation_consistent": citation_consistent,
                "uncertain": is_uncertain(answer),
                "answer_preview": answer[:400],
            }
        )

        if args.sleep:
            time.sleep(float(args.sleep))

        print(f"[OK] {idx}/{len(cases)} {c.id} cov={cov_ratio:.2f} cite={citation_consistent}")

    report = {
        "mode": args.mode,
        "endpoint": endpoint,
        "cases": len(cases),
        "cases_with_sources": cases_with_sources,
        "citation_consistency_rate": (citation_ok / max(1, cases_with_sources)),
        "coverage_avg": (coverage_sum / max(1, len(cases))),
        "answer_url_rate": (answer_url_rate / max(1, len(cases))),
        "uncertainty_rate": (uncertain_count / max(1, len(cases))),
        "details": details,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[DONE] out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
