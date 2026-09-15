"""Experiments that turn the community complaints into numbers."""

from __future__ import annotations

import json
import re
from importlib import metadata
from pathlib import Path

from langchain_core.exceptions import OutputParserException

from .chains import analyze_review
from .config import DATA_DIR
from .parsers import get_parser

PACKAGE_DIR = Path(__file__).resolve().parent


def load_json(name: str) -> list[dict]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def load_sample_reviews() -> list[dict]:
    return load_json("sample_reviews.json")


# Pain point #4 -------------------------------------------------------------
def parser_stress_test() -> list[dict]:
    """Run strict and robust parsers over a corpus of realistic malformed outputs."""
    strict, robust = get_parser("strict"), get_parser("robust")
    rows = []
    for case in load_json("malformed_outputs.json"):
        row = {"case": case["name"], "robust_expected": case["robust_expected"]}
        for label, parser in (("strict", strict), ("robust", robust)):
            try:
                parser.parse(case["text"])
                row[label] = True
            except (OutputParserException, ValueError):
                row[label] = False
        rows.append(row)
    return rows


# Pain point #3 -------------------------------------------------------------
def model_swap(review: str, providers: list[str], parser_mode: str = "robust", max_retries: int = 1) -> list[dict]:
    """Same review, same code path, different providers."""
    return [analyze_review(review, p, parser_mode, max_retries).to_row() for p in providers]


# Pain point #5 -------------------------------------------------------------
def _requirement_name(req: str) -> str | None:
    if ";" in req and "extra" in req.split(";", 1)[1]:
        return None
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
    return match.group(1).lower().replace("_", "-") if match else None


def transitive_dependencies(package: str) -> set[str]:
    seen: set[str] = set()
    stack = [package]
    while stack:
        try:
            requirements = metadata.requires(stack.pop()) or []
        except metadata.PackageNotFoundError:
            continue
        for req in requirements:
            name = _requirement_name(req)
            if name and name not in seen:
                seen.add(name)
                stack.append(name)
    return seen


def dependency_footprint(packages: tuple[str, ...] = ("langchain", "langchain-openai", "openai", "anthropic")) -> list[dict]:
    rows = []
    for pkg in packages:
        try:
            version = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            rows.append({"package": pkg, "version": "not installed", "transitive_deps": None})
            continue
        rows.append({"package": pkg, "version": version, "transitive_deps": len(transitive_dependencies(pkg))})
    return rows


# Pain point #1 -------------------------------------------------------------
def _count_loc(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#"))


def loc_comparison() -> list[dict]:
    """Rough 'ceremony' metric: lines of code on each side (docstrings included)."""
    langchain_files = ["prompts.py", "models.py", "parsers.py", "chains.py"]
    return [
        {"pipeline": "langchain", "files": ", ".join(langchain_files), "loc": sum(_count_loc(PACKAGE_DIR / f) for f in langchain_files)},
        {"pipeline": "native", "files": "native_baseline.py", "loc": _count_loc(PACKAGE_DIR / "native_baseline.py")},
    ]
