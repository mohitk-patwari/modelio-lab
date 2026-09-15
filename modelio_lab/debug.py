"""Debuggability helpers (pain point #2: deep stack traces).

TraceHandler records exactly what went over the wire. format_error turns a deep framework
exception into one readable line that names the root cause.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class TraceHandler(BaseCallbackHandler):
    """Collects prompts sent to the model and raw text returned by it."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def _elapsed(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 1)

    def on_chat_model_start(self, serialized: dict[str, Any] | None, messages: list, **kwargs: Any) -> None:
        self.events.append(
            {
                "event": "request",
                "t_ms": self._elapsed(),
                "messages": [
                    {"role": m.type, "content": m.content if isinstance(m.content, str) else str(m.content)}
                    for batch in messages
                    for m in batch
                ],
            }
        )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        outputs = []
        for generations in response.generations:
            for gen in generations:
                message = getattr(gen, "message", None)
                tool_calls = getattr(message, "tool_calls", None) if message is not None else None
                outputs.append({"text": gen.text, "tool_calls": tool_calls or None})
        self.events.append({"event": "response", "t_ms": self._elapsed(), "outputs": outputs})

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.events.append({"event": "error", "t_ms": self._elapsed(), "error": format_error(error)})


def root_cause(exc: BaseException) -> BaseException:
    """Follow __cause__ / __context__ to the original exception."""
    seen = set()
    while id(exc) not in seen:
        seen.add(id(exc))
        nxt = exc.__cause__ or exc.__context__
        if nxt is None:
            break
        exc = nxt
    return exc


def format_error(exc: BaseException, limit: int = 300) -> str:
    """One readable line: top-level error plus the root cause if different."""

    def first_line(e: BaseException) -> str:
        msg = (str(e).strip().splitlines() or [""])[0]
        return f"{type(e).__name__}: {msg}"

    text = first_line(exc)
    cause = root_cause(exc)
    if cause is not exc:
        text += f" (root cause: {first_line(cause)})"
    return text if len(text) <= limit else text[: limit - 3] + "..."
