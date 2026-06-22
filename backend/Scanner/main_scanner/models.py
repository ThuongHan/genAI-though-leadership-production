"""Article dataclass — the canonical record produced by all scanner sources."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from .text import detect_language


@dataclass
class Article:
    id: str
    name: str
    url: str
    tag: str  # news | dutch_news | policy_regulation | research_reports | dutch_ecosystem
    source: str
    author: Optional[str]
    published_at: Optional[str]
    full_text: str
    paywalled: bool = False  # True => full_text is the free preview, not the full body
    full_text_character_count: int = field(init=False)
    language: str = field(init=False)

    def __post_init__(self) -> None:
        self.full_text_character_count = len(self.full_text)
        self.language = detect_language(self.name, self.full_text)

    @staticmethod
    def make_id(url: str) -> str:
        return hashlib.sha256(url.strip().lower().encode()).hexdigest()[:16]
