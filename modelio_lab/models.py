"""Model layer of Model I/O: one factory for every provider, plus an offline fake model."""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from .config import DEFAULT_MODELS

CANNED_ANALYSIS = {
    "sentiment": "mixed",
    "rating": 3,
    "pros": ["solid build quality"],
    "cons": ["short battery life"],
    "summary": "Well built, but the battery life disappoints.",
}


def fake_responses_for(review: str) -> list[str]:
    """Return [messy_response, clean_response] for the fake model.

    The messy style is picked deterministically from the review text so that the offline
    demo shows real parser failures (fences, chatty prose, trailing commas). The clean
    response comes second so a retry can succeed.
    """
    clean = json.dumps(CANNED_ANALYSIS, indent=2)
    styles = [
        clean,
        f"```json\n{clean}\n```",
        f"Sure! Here is the analysis you asked for:\n\n{clean}\n\nLet me know if you need anything else.",
        clean[:-1].rstrip() + ",\n}",
    ]
    pick = sum(map(ord, review)) % len(styles)
    return [styles[pick], clean]


def get_chat_model(
    provider: str,
    temperature: float = 0.0,
    fake_responses: list[str] | None = None,
) -> BaseChatModel:
    """Create a chat model for `provider` using LangChain's unified interface."""
    if provider == "fake":
        return FakeListChatModel(responses=fake_responses or [json.dumps(CANNED_ANALYSIS)])

    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {', '.join(DEFAULT_MODELS)}")

    from langchain.chat_models import init_chat_model

    return init_chat_model(DEFAULT_MODELS[provider], model_provider=provider, temperature=temperature)
