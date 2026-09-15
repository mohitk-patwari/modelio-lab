"""Output schema for the review-analysis task and the shared run-result record."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


class ReviewAnalysis(BaseModel):
    """Structured analysis of a single product review."""

    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Overall sentiment of the review"
    )
    rating: int = Field(ge=1, le=5, description="Inferred star rating from 1 to 5")
    pros: list[str] = Field(default_factory=list, description="Things the reviewer liked")
    cons: list[str] = Field(default_factory=list, description="Things the reviewer disliked")
    summary: str = Field(description="One-sentence summary of the review")


@dataclass
class RunResult:
    """Outcome of one analysis run, shared by the LangChain and native pipelines."""

    pipeline: Literal["langchain", "native"]
    provider: str
    parser_mode: str
    ok: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None
    raw_output: str | None = None
    latency_ms: float = 0.0
    attempts: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        """Flat dict for tables (drops the bulky trace and raw output)."""
        row = asdict(self)
        row.pop("trace")
        row.pop("raw_output")
        row["latency_ms"] = round(self.latency_ms, 1)
        row["result"] = None if self.result is None else self.result.get("summary")
        return row
