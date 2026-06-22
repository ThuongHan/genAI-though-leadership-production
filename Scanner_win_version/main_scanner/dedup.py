"""Deduplication of Article lists by id (URL hash)."""

from __future__ import annotations

from .models import Article


def deduplicate(articles: list[Article]) -> list[Article]:
    seen: set[str] = set()
    unique: list[Article] = []
    for art in articles:
        if art.id not in seen:
            seen.add(art.id)
            unique.append(art)
    return unique
