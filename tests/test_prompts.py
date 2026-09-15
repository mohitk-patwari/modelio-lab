from modelio_lab.prompts import build_prompt

INSTRUCTIONS = "FORMAT_MARKER {\"json\": true}"


def test_only_review_variable_remains():
    for provider in ("openai", "fake"):
        assert build_prompt(provider, INSTRUCTIONS).input_variables == ["review"]


def test_format_instructions_are_injected_verbatim():
    messages = build_prompt("openai", INSTRUCTIONS).format_messages(review="x")
    assert INSTRUCTIONS in messages[0].content


def test_small_models_get_few_shot_examples():
    large = build_prompt("openai", INSTRUCTIONS).format_messages(review="x")
    small = build_prompt("ollama", INSTRUCTIONS).format_messages(review="x")
    assert len(small) > len(large)
    assert "Do not use markdown" in small[0].content
