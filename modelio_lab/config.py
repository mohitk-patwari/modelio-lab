"""Provider configuration and API-key detection."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Default model per provider. Override any of them with env vars, e.g. OPENAI_MODEL=gpt-4o.
DEFAULT_MODELS: dict[str, str | None] = {
    "fake": None,
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
    "groq": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.2"),
}

# Env var that must be set for the provider to be usable (None = no key needed).
KEY_ENV: dict[str, str | None] = {
    "fake": None,
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": None,
}

# Smaller / local models get a stricter prompt with few-shot examples (pain point #3).
SMALL_MODEL_PROVIDERS = {"fake", "groq", "ollama"}

PARSER_MODES = ("strict", "robust", "structured")


def available_providers() -> list[str]:
    """Providers that can run right now. Ollama is listed only if OLLAMA_ENABLED=1."""
    providers = []
    for name, env in KEY_ENV.items():
        if name == "ollama":
            if os.getenv("OLLAMA_ENABLED") == "1":
                providers.append(name)
            continue
        if env is None or os.getenv(env):
            providers.append(name)
    return providers
