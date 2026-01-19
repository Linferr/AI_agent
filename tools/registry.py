from __future__ import annotations

from typing import Any, Callable

from tools import open_url, search_corpus
from tools.types import ToolContext, ToolSpec


ToolRunner = Callable[[dict[str, Any], ToolContext], dict[str, Any]]


def get_tool_specs() -> list[ToolSpec]:
    return [search_corpus.SPEC, open_url.SPEC]


def get_tool_runner(name: str) -> ToolRunner | None:
    if name == search_corpus.SPEC.name:
        return search_corpus.run
    if name == open_url.SPEC.name:
        return open_url.run
    return None


def run_tool(name: str, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    runner = get_tool_runner(name)
    if runner is None:
        return {"error": f"Unknown tool: {name}"}
    return runner(args, ctx)
