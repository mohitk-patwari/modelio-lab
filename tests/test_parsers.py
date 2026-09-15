import pytest
from langchain_core.exceptions import OutputParserException

from modelio_lab.benchmark import load_json, parser_stress_test
from modelio_lab.parsers import clean_llm_json, get_parser

CASES = load_json("malformed_outputs.json")


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_robust_parser_matches_expectations(case):
    parser = get_parser("robust")
    if case["robust_expected"]:
        assert parser.parse(case["text"]).rating >= 1
    else:
        with pytest.raises(OutputParserException):
            parser.parse(case["text"])


def test_strict_parser_accepts_clean_json():
    clean = next(c for c in CASES if c["name"] == "clean_json")
    assert get_parser("strict").parse(clean["text"]).sentiment == "positive"


def test_robust_beats_strict_on_corpus():
    rows = parser_stress_test()
    assert sum(r["robust"] for r in rows) > sum(r["strict"] for r in rows)


def test_clean_llm_json_handles_fence_and_trailing_comma():
    assert clean_llm_json('```json\n{"a": [1, 2,],}\n```') == '{"a": [1, 2]}'


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        get_parser("structured")
