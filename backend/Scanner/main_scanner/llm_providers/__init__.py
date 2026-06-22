"""Pluggable LLM providers — Claude and OpenAI for now.

Usage:
    from main_scanner.llm_providers import build_llm_client
    client = build_llm_client("claude")                  # default Haiku
    client = build_llm_client("openai", model="gpt-4o")  # specific model
    text = client.complete(system, messages, max_tokens=500)
"""

from __future__ import annotations

from .base import LLMClient
from .claude import DEFAULT_CLAUDE_MODEL, ClaudeClient
from .openai import DEFAULT_OPENAI_MODEL, OpenAIClient


PROVIDER_NAMES = ("claude", "openai")

DEFAULT_MODELS: dict = {
    "claude": DEFAULT_CLAUDE_MODEL,
    "openai": DEFAULT_OPENAI_MODEL,
}


def build_llm_client(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """Factory — returns a provider client implementing the LLMClient protocol."""
    p = provider.lower()
    if p == "claude":
        return ClaudeClient(model=model or DEFAULT_CLAUDE_MODEL, api_key=api_key)
    if p == "openai":
        return OpenAIClient(model=model or DEFAULT_OPENAI_MODEL, api_key=api_key)
    raise ValueError(
        f"Unknown provider: {provider!r}. Use one of: {PROVIDER_NAMES}"
    )


__all__ = [
    "ClaudeClient",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_MODELS",
    "DEFAULT_OPENAI_MODEL",
    "LLMClient",
    "OpenAIClient",
    "PROVIDER_NAMES",
    "build_llm_client",
]
