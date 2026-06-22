"""Provider-agnostic LLM client interface.

Each concrete provider implements `complete(system, messages, max_tokens) -> str`.
The evaluators in main_scanner/evaluators/ call only this method, so they
work with any provider that implements the contract.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Minimal contract every provider wrapper must satisfy."""

    name: str   # e.g. "claude" / "openai"
    model: str  # full model id reported by the SDK

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int,
    ) -> str:
        """Run one completion turn and return the assistant text.

        - `system` is a free-form system prompt.
        - `messages` follows the Anthropic shape: list of
          `{"role": "user"|"assistant", "content": "..."}`.
          Provider wrappers translate to native shapes as needed.
        - `max_tokens` is the cap on the response length.
        """
        ...
