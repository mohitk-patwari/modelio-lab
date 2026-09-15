"""The full Model I/O pipeline: prompt -> model -> parser, with tracing and parse-retry."""

from __future__ import annotations

from time import perf_counter

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from .config import PARSER_MODES
from .debug import TraceHandler, format_error
from .models import fake_responses_for, get_chat_model
from .parsers import get_parser
from .prompts import RETRY_PROMPT, build_prompt
from .schemas import ReviewAnalysis, RunResult


def _to_dict(value) -> dict:
    return value.model_dump() if hasattr(value, "model_dump") else dict(value)


def analyze_review(
    review: str,
    provider: str = "fake",
    parser_mode: str = "robust",
    max_retries: int = 1,
    model: BaseChatModel | None = None,
) -> RunResult:
    """Analyze one review with LangChain Model I/O. Never raises; errors land in RunResult.error.

    parser_mode:
      strict     - stock PydanticOutputParser, no retries
      robust     - cleanup parser, plus up to `max_retries` LLM repair calls
      structured - model.with_structured_output (needs a tool-calling provider)
    """
    if parser_mode not in PARSER_MODES:
        raise ValueError(f"parser_mode must be one of {PARSER_MODES}")

    tracer = TraceHandler()
    config = {"callbacks": [tracer], "run_name": f"modelio-lab:{provider}:{parser_mode}"}
    result = RunResult(pipeline="langchain", provider=provider, parser_mode=parser_mode)
    start = perf_counter()

    try:
        if model is None:
            fake = fake_responses_for(review) if provider == "fake" else None
            model = get_chat_model(provider, fake_responses=fake)

        format_instructions = PydanticOutputParser(pydantic_object=ReviewAnalysis).get_format_instructions()

        if parser_mode == "structured":
            prompt = build_prompt(provider, format_instructions, structured=True)
            result.attempts = 1
            chain = prompt | model.with_structured_output(ReviewAnalysis)
            result.result = _to_dict(chain.invoke({"review": review}, config=config))
            result.ok = True
        else:
            prompt = build_prompt(provider, format_instructions)
            parser = get_parser(parser_mode)
            raw = (prompt | model | StrOutputParser()).invoke({"review": review}, config=config)
            result.raw_output, result.attempts = raw, 1

            while True:
                try:
                    parsed = parser.parse(raw)
                    break
                except OutputParserException as exc:
                    if parser_mode == "strict" or result.attempts > max_retries:
                        raise
                    raw = (RETRY_PROMPT | model | StrOutputParser()).invoke(
                        {"bad_output": raw, "error": str(exc), "format_instructions": format_instructions},
                        config=config,
                    )
                    result.raw_output = raw
                    result.attempts += 1

            result.result = _to_dict(parsed)
            result.ok = True
    except NotImplementedError as exc:
        result.error = f"Structured output needs a tool-calling provider ({format_error(exc)})"
    except Exception as exc:  # noqa: BLE001 - surface every failure as a readable message
        result.error = format_error(exc)
    finally:
        result.latency_ms = (perf_counter() - start) * 1000
        result.trace = tracer.events

    return result
