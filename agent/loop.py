from __future__ import annotations

import json
import re
from typing import Any

from llm.dashscope_client import DashScopeChatClient
from tools.registry import ToolContext, get_tool_specs, run_tool


def build_messages(system: str | None, history: list[dict[str, str]], message: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(history)
    messages.append({"role": "user", "content": message})
    return messages


def _format_tool_specs(allowed_tools: set[str] | None = None) -> str:
    lines = []
    for spec in get_tool_specs():
        if allowed_tools and spec.name not in allowed_tools:
            continue
        args = ", ".join(f"{k}: {v}" for k, v in spec.args.items())
        lines.append(f"- {spec.name}: {spec.description}\n  args: {args}")
    return "\n".join(lines)


def _format_tool_history(tool_trace: list[dict[str, Any]]) -> str:
    if not tool_trace:
        return "None"
    return json.dumps(tool_trace, ensure_ascii=False, indent=2)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def _call_llm(api_key: str, model: str, messages: list[dict[str, str]]) -> str:
    client = DashScopeChatClient(api_key=api_key)
    return client.chat(messages=messages, model=model)


def _tool_decide(
    *,
    api_key: str,
    model: str,
    message: str,
    history: list[dict[str, str]],
    system: str | None,
    tool_trace: list[dict[str, Any]],
    allowed_tools: set[str] | None = None,
    allow_final: bool = True,
    extra_instructions: str = "",
) -> dict[str, Any]:
    schema_lines = ['- To call a tool: {"action":"tool","tool_name":"...","arguments":{...}}']
    if allow_final:
        schema_lines.append('- To answer: {"action":"final","answer":"..."}')
    schema = "\n".join(schema_lines)

    tool_prompt = (
        "You are an AWS troubleshooting assistant. Decide whether to call a tool or answer directly.\n"
        "You MUST reply with a single JSON object and nothing else.\n"
        f"JSON schema:\n{schema}\n"
        "Never include extra keys. Never include markdown.\n\n"
        "Available tools:\n"
        f"{_format_tool_specs(allowed_tools)}\n\n"
        "Tool results so far:\n"
        f"{_format_tool_history(tool_trace)}"
    )
    if extra_instructions:
        tool_prompt = extra_instructions.strip() + "\n\n" + tool_prompt
    if system:
        tool_prompt = system.strip() + "\n\n" + tool_prompt

    messages = build_messages(tool_prompt, history, message)
    raw = _call_llm(api_key, model, messages)
    parsed = _parse_json_object(raw)
    if not parsed:
        return {"action": "final", "answer": raw}
    return parsed


def _tool_finalize(
    *,
    api_key: str,
    model: str,
    message: str,
    history: list[dict[str, str]],
    system: str | None,
    tool_trace: list[dict[str, Any]],
) -> str:
    prompt = (
        "You are an AWS troubleshooting assistant. Use the tool results to answer.\n"
        "Do NOT call tools now. Provide a clear checklist and include source URLs at the end.\n\n"
        "Tool results:\n"
        f"{_format_tool_history(tool_trace)}"
    )
    if system:
        prompt = system.strip() + "\n\n" + prompt
    messages = build_messages(prompt, history, message)
    return _call_llm(api_key, model, messages)


def _forced_search(message: str, ctx: ToolContext) -> dict[str, Any]:
    return run_tool(
        "search_corpus",
        {"query": message, "top_k": ctx.default_top_k, "mode": "auto"},
        ctx,
    )


def run_tool_chat(
    *,
    api_key: str,
    model: str,
    message: str,
    history: list[dict[str, str]],
    system: str | None,
    ctx: ToolContext,
) -> tuple[str, list[dict[str, Any]]]:
    tool_trace: list[dict[str, Any]] = []

    # Step 1: force search
    search_result = _forced_search(message, ctx)
    tool_trace.append({"tool_name": "search_corpus", "arguments": {"query": message}, "result": search_result})

    # Step 2: allow optional open_url
    decision = _tool_decide(
        api_key=api_key,
        model=model,
        message=message,
        history=history,
        system=system,
        tool_trace=tool_trace,
        allowed_tools={"open_url"},
        allow_final=True,
        extra_instructions="search_corpus is already executed. Only call open_url if evidence is insufficient.",
    )

    action = str(decision.get("action", "")).lower()
    if action == "tool":
        tool_name = str(decision.get("tool_name", "")).strip()
        if tool_name == "open_url":
            arguments = decision.get("arguments", {}) if isinstance(decision.get("arguments", {}), dict) else {}
            if "query" not in arguments:
                arguments["query"] = message
            try:
                result = run_tool(tool_name, arguments, ctx)
            except Exception as e:
                result = {"error": str(e)}
            tool_trace.append({"tool_name": tool_name, "arguments": arguments, "result": result})

    reply = _tool_finalize(
        api_key=api_key, model=model, message=message, history=history, system=system, tool_trace=tool_trace
    )
    return reply, tool_trace


def run_agent_chat(
    *,
    api_key: str,
    model: str,
    message: str,
    history: list[dict[str, str]],
    system: str | None,
    ctx: ToolContext,
    max_steps: int,
) -> tuple[str, list[dict[str, Any]]]:
    max_steps = max(1, int(max_steps))
    steps: list[dict[str, Any]] = []
    tool_trace: list[dict[str, Any]] = []

    # Step 1: force search
    search_result = _forced_search(message, ctx)
    tool_trace.append({"tool_name": "search_corpus", "arguments": {"query": message}, "result": search_result})
    steps.append({"step": 1, "action": "tool", "tool_name": "search_corpus"})

    final_answer: str | None = None

    for step in range(2, max_steps + 1):
        decision = _tool_decide(
            api_key=api_key,
            model=model,
            message=message,
            history=history,
            system=system,
            tool_trace=tool_trace,
            allowed_tools={"open_url"},
            allow_final=True,
            extra_instructions="search_corpus already executed. You may call open_url to fetch more evidence.",
        )
        action = str(decision.get("action", "")).lower()
        if action != "tool":
            final_answer = str(decision.get("answer", "")).strip() or json.dumps(decision, ensure_ascii=False)
            steps.append({"step": step, "action": "final"})
            break

        tool_name = str(decision.get("tool_name", "")).strip()
        if tool_name != "open_url":
            steps.append({"step": step, "action": "final"})
            final_answer = str(decision.get("answer", "")).strip() or ""
            break

        arguments = decision.get("arguments", {}) if isinstance(decision.get("arguments", {}), dict) else {}
        if "query" not in arguments:
            arguments["query"] = message
        try:
            result = run_tool(tool_name, arguments, ctx)
        except Exception as e:
            result = {"error": str(e)}
        tool_trace.append({"tool_name": tool_name, "arguments": arguments, "result": result})
        steps.append(
            {
                "step": step,
                "action": "tool",
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
            }
        )

    if final_answer is None:
        final_answer = _tool_finalize(
            api_key=api_key, model=model, message=message, history=history, system=system, tool_trace=tool_trace
        )
    return final_answer, steps
