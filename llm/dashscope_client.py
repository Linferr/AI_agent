from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DashScopeChatClient:
    api_key: str

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
    ) -> str:
        try:
            import dashscope
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "dashscope is not installed. Run: pip install -r requirements.txt"
            ) from e

        dashscope.api_key = self.api_key

        resp: Any = dashscope.Generation.call(
            model=model,
            messages=messages,
            result_format="message",
            temperature=temperature,
        )

        if not getattr(resp, "output", None) or not getattr(resp.output, "choices", None):
            raise RuntimeError(f"Unexpected DashScope response: {resp}")

        choice = resp.output.choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Unexpected DashScope message content: {resp}")

        return content.strip()

