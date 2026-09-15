"""Output-parsing layer of Model I/O (pain point #4: fragile parsing).

- strict: LangChain's stock PydanticOutputParser, no recovery.
- robust: a custom BaseOutputParser that cleans common LLM drift before validating.
"""

from __future__ import annotations

import json
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from pydantic import BaseModel, ValidationError

from .schemas import ReviewAnalysis

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def extract_json_block(text: str) -> str:
    """Pull the first balanced JSON object out of messy model text."""
    fenced = FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    if start == -1:
        return text.strip()

    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]  # unbalanced; let json.loads report it


def clean_llm_json(text: str) -> str:
    """Strip fences and prose, then remove trailing commas."""
    return TRAILING_COMMA_RE.sub(r"\1", extract_json_block(text)).strip()


class RobustPydanticParser(BaseOutputParser[BaseModel]):
    """Pydantic parser that tolerates fences, surrounding prose and trailing commas."""

    pydantic_object: type[BaseModel] = ReviewAnalysis

    def parse(self, text: str) -> BaseModel:
        cleaned = clean_llm_json(text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise OutputParserException(f"Invalid JSON after cleanup: {exc}", llm_output=text) from exc
        try:
            return self.pydantic_object.model_validate(data)
        except ValidationError as exc:
            errors = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise OutputParserException(f"Schema validation failed: {errors}", llm_output=text) from exc

    def get_format_instructions(self) -> str:
        return PydanticOutputParser(pydantic_object=self.pydantic_object).get_format_instructions()

    @property
    def _type(self) -> str:
        return "robust_pydantic"


def get_parser(mode: str) -> BaseOutputParser:
    """Return the parser for 'strict' or 'robust' mode."""
    if mode == "strict":
        return PydanticOutputParser(pydantic_object=ReviewAnalysis)
    if mode == "robust":
        return RobustPydanticParser()
    raise ValueError(f"No text parser for mode '{mode}' (structured mode uses tool calling).")
