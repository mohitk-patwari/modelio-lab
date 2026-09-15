import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from modelio_lab.chains import analyze_review
from modelio_lab.models import CANNED_ANALYSIS
from modelio_lab.native_baseline import analyze_native

CLEAN = json.dumps(CANNED_ANALYSIS)
CHATTY = f"Sure thing!\n{CLEAN}\nCheers."


def fake(*responses):
    return FakeListChatModel(responses=list(responses))


def test_robust_parses_chatty_output_first_try():
    res = analyze_review("r", parser_mode="robust", model=fake(CHATTY))
    assert res.ok and res.attempts == 1 and res.result["rating"] == 3


def test_robust_retries_then_succeeds():
    res = analyze_review("r", parser_mode="robust", max_retries=1, model=fake("not json at all", CLEAN))
    assert res.ok and res.attempts == 2


def test_robust_gives_up_after_max_retries():
    res = analyze_review("r", parser_mode="robust", max_retries=1, model=fake("nope", "still nope"))
    assert not res.ok and res.attempts == 2 and res.error


def test_strict_fails_without_retry_and_error_is_one_line():
    res = analyze_review("r", parser_mode="strict", model=fake("not json at all", CLEAN))
    assert not res.ok and res.attempts == 1
    assert "\n" not in res.error and len(res.error) <= 300


def test_trace_records_request_and_response():
    res = analyze_review("r", model=fake(CLEAN))
    events = [e["event"] for e in res.trace]
    assert "request" in events and "response" in events


def test_structured_mode_on_fake_reports_clear_error():
    res = analyze_review("r", parser_mode="structured", model=fake(CLEAN))
    assert not res.ok and res.error


def test_default_fake_provider_runs_end_to_end():
    assert analyze_review("Battery is great, case is flimsy.", provider="fake").ok


def test_native_fake_baseline():
    res = analyze_native("Battery is great, case is flimsy.", provider="fake")
    assert res.ok and res.pipeline == "native"
