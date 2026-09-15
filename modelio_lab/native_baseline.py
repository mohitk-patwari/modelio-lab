"""The same task without LangChain: f-string prompt, raw SDK call, json + pydantic."""

from __future__ import annotations

import json
from time import perf_counter

from pydantic import ValidationError

from .config import DEFAULT_MODELS
from .debug import format_error
from .models import fake_responses_for
from .parsers import clean_llm_json
from .schemas import ReviewAnalysis, RunResult

NATIVE_PROVIDERS = ("fake", "openai", "anthropic")

SCHEMA = json.dumps(ReviewAnalysis.model_json_schema())
SYSTEM = (
    "You are a precise product-review analyst. Respond with one JSON object matching this "
    "JSON schema and nothing else:\n" + SCHEMA
)


def _call(provider: str, review: str) -> str:
    user = f"Review:\n{review}"
    if provider == "fake":
        return fake_responses_for(review)[0]
    if provider == "openai":
        from openai import OpenAI

        resp = OpenAI().chat.completions.create(
            model=DEFAULT_MODELS["openai"],
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""
    if provider == "anthropic":
        import anthropic

        resp = anthropic.Anthropic().messages.create(
            model=DEFAULT_MODELS["anthropic"],
            max_tokens=512,
            temperature=0,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")
    raise ValueError(f"Native baseline supports {NATIVE_PROVIDERS}, not '{provider}'")


def analyze_native(review: str, provider: str = "fake") -> RunResult:
    """Analyze one review with the provider SDK directly. Never raises."""
    result = RunResult(pipeline="native", provider=provider, parser_mode="native", attempts=1)
    start = perf_counter()
    try:
        raw = _call(provider, review)
        result.raw_output = raw
        result.result = ReviewAnalysis.model_validate(json.loads(clean_llm_json(raw))).model_dump()
        result.ok = True
    except (json.JSONDecodeError, ValidationError) as exc:
        result.error = format_error(exc)
    except Exception as exc:  # noqa: BLE001
        result.error = format_error(exc)
    finally:
        result.latency_ms = (perf_counter() - start) * 1000
    return result
