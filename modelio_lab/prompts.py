"""Prompt layer of Model I/O: provider-aware ChatPromptTemplates.

Pain point #3 ("swap-in, swap-out is a myth") is handled here: small or local models
get a stricter system prompt plus few-shot examples instead of the same prompt as GPT-4-class models.
"""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from .config import SMALL_MODEL_PROVIDERS

BASE_SYSTEM = (
    "You are a precise product-review analyst. Read the review and extract its sentiment, "
    "an inferred 1-5 star rating, pros, cons and a one-sentence summary."
    "\n\n{format_instructions}"
)

STRICT_SUFFIX = (
    "\n\nRules: respond with exactly one JSON object and nothing else. "
    "Do not use markdown code fences. Do not add explanations before or after the JSON."
)

FEW_SHOT_EXAMPLES = [
    {
        "review": "Arrived fast and the sound is amazing, but the ear cushions started peeling after a month.",
        "output": json.dumps(
            {
                "sentiment": "mixed",
                "rating": 3,
                "pros": ["fast delivery", "great sound"],
                "cons": ["ear cushions peel quickly"],
                "summary": "Great-sounding headphones let down by poor cushion durability.",
            }
        ),
    },
    {
        "review": "Stopped charging after two days and support never replied.",
        "output": json.dumps(
            {
                "sentiment": "negative",
                "rating": 1,
                "pros": [],
                "cons": ["stopped charging", "unresponsive support"],
                "summary": "The device failed within days and support was unhelpful.",
            }
        ),
    },
]

RETRY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You repair malformed model outputs. Return only a valid JSON object that satisfies "
            "the format instructions. No markdown, no commentary.\n\n{format_instructions}",
        ),
        (
            "human",
            "This output failed to parse:\n\n{bad_output}\n\nParser error:\n{error}\n\n"
            "Return the corrected JSON object.",
        ),
    ]
)


def build_prompt(provider: str, format_instructions: str) -> ChatPromptTemplate:
    """Build the analysis prompt for a provider. The only remaining input variable is `review`."""
    small = provider in SMALL_MODEL_PROVIDERS
    system = BASE_SYSTEM + (STRICT_SUFFIX if small else "")
    messages: list = [("system", system)]

    if small:
        example_prompt = ChatPromptTemplate.from_messages([("human", "{review}"), ("ai", "{output}")])
        messages.append(
            FewShotChatMessagePromptTemplate(examples=FEW_SHOT_EXAMPLES, example_prompt=example_prompt)
        )

    messages.append(("human", "Review:\n{review}"))
    return ChatPromptTemplate.from_messages(messages).partial(format_instructions=format_instructions)
