"""Claude (Anthropic) provider wrapper."""

from __future__ import annotations

import os


DEFAULT_CLAUDE_MODEL: str = "claude-haiku-4-5-20251001"


class ClaudeClient:
    name: str = "claude"

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, api_key: str | None = None):
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set — required for the Claude provider."
            )
        # max_retries: the SDK auto-retries 429 (rate limit) with exponential
        # backoff, honouring the Retry-After header. Default is 2; bump it so
        # concurrent scoring runs survive transient rate limits.
        self._client = anthropic.Anthropic(api_key=key, max_retries=5)
        self.model = model

    def complete(self, system: str, messages: list[dict], max_tokens: int) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return msg.content[0].text.strip()
