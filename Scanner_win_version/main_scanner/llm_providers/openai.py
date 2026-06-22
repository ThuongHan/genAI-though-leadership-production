"""OpenAI provider wrapper.

Translates the Anthropic-style (system, messages) call shape into OpenAI's
chat-completion API where the system prompt is just the first message.
"""

from __future__ import annotations

import os


# Default to a small/fast model in the same tier as claude-haiku.
DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"


class OpenAIClient:
    name: str = "openai"

    def __init__(self, model: str = DEFAULT_OPENAI_MODEL, api_key: str | None = None):
        from openai import OpenAI  # local import — package optional
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set — required for the OpenAI provider."
            )
        # max_retries: SDK auto-retries 429 with backoff. Default is 2; bump
        # it so concurrent scoring runs survive transient rate limits.
        self._client = OpenAI(api_key=key, max_retries=5)
        self.model = model

    def complete(self, system: str, messages: list[dict], max_tokens: int) -> str:
        oai_messages = [{"role": "system", "content": system}]
        # Anthropic uses 'user'/'assistant' roles which OpenAI also accepts.
        oai_messages.extend(messages)
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=oai_messages,
        )
        return (resp.choices[0].message.content or "").strip()
