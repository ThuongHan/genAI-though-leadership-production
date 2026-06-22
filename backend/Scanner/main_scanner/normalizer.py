"""Convert raw API/RSS/scraper records into Article instances."""

from __future__ import annotations

from typing import Optional

import arxiv

from . import settings as config
from .extractor import _extractor
from .models import Article
from .text import (
    free_preview,
    is_ai_related,
    is_consent_wall,
    is_paywalled_source,
    looks_paywalled,
)


_MIN_BODY_CHARS = 100  # below this, body is considered too thin to be useful


def _resolve_body(url: str, extracted: str, snippet: str) -> tuple[str, bool]:
    """Decide the stored body for one article, respecting paywalls.

    Free/open sources keep the full extracted text. Subscription sources (known
    domains, or caught at runtime) keep ONLY the freely-visible preview (the lede,
    capped at PAYWALL_PREVIEW_CHARS); if that is too thin, the licensed news-API
    snippet is used instead. Returns (body, paywalled).
    """
    if is_paywalled_source(url) or looks_paywalled(extracted):
        preview = free_preview(extracted, config.PAYWALL_PREVIEW_CHARS)
        body = preview if len(preview) >= _MIN_BODY_CHARS else snippet
        return body, True
    return (extracted or snippet), False


class ArticleNormalizer:
    """Convert raw dicts from various sources into Article instances."""

    def from_newsapi(self, item: dict, tag: str) -> Optional[Article]:
        url = item.get("url", "")
        if not url or url == "https://removed.com":
            return None
        title = item.get("title") or ""
        description = item.get("description") or ""
        content_snippet = item.get("content") or ""
        source_name = (item.get("source") or {}).get("name", "NewsAPI")
        author = item.get("author") or None
        published_at = item.get("publishedAt") or None

        snippet = f"{title}\n\n{description}\n\n{content_snippet}".strip()
        extracted = _extractor.fetch(url)                 # fetch + clean the article body
        full_text, paywalled = _resolve_body(url, extracted, snippet)

        if not is_ai_related(f"{title} {full_text}"):
            return None

        return Article(
            id=Article.make_id(url),
            name=title,
            url=url,
            tag=tag,
            source=source_name,
            author=author,
            published_at=published_at,
            full_text=full_text,
            paywalled=paywalled,
        )

    def from_rss(self, entry: object, tag: str, source_name: str) -> Optional[Article]:
        url = getattr(entry, "link", "") or ""
        if not url:
            return None
        title = getattr(entry, "title", "") or ""
        rss_summary = getattr(entry, "summary", "") or ""

        published_at = None
        if hasattr(entry, "published"):
            published_at = entry.published
        elif hasattr(entry, "updated"):
            published_at = entry.updated

        author = None
        if hasattr(entry, "author"):
            author = entry.author or None

        snippet = f"{title}\n\n{rss_summary}".strip()
        extracted = _extractor.fetch(url)                 # fetch + clean the article body
        # Discard junk extractions (consent wall / too thin) so we fall back to the RSS summary.
        if extracted and (is_consent_wall(extracted) or len(extracted) < _MIN_BODY_CHARS):
            extracted = ""
        # Free sources keep full text; paywalled sources keep only the free preview.
        full_text, paywalled = _resolve_body(url, extracted, snippet)
        # If even the fallback is too thin, skip this article.
        if len(full_text) < _MIN_BODY_CHARS:
            return None

        if not is_ai_related(f"{title} {full_text}"):
            return None

        return Article(
            id=Article.make_id(url),
            name=title,
            url=url,
            tag=tag,
            source=source_name,
            author=author,
            published_at=published_at,
            full_text=full_text,
            paywalled=paywalled,
        )

    def from_arxiv(self, result: arxiv.Result) -> Optional[Article]:
        url = result.entry_id
        title = result.title or ""
        authors = ", ".join(str(a) for a in result.authors) if result.authors else None
        published_at = result.published.isoformat() if result.published else None
        full_text = f"{title}\n\nAuthors: {authors or 'N/A'}\n\nAbstract:\n{result.summary or ''}"

        if not is_ai_related(full_text):
            return None

        return Article(
            id=Article.make_id(url),
            name=title,
            url=url,
            tag="research_reports",
            source="arXiv",
            author=authors,
            published_at=published_at,
            full_text=full_text,
        )

    def from_scraped(
        self, url: str, title: str, full_text: str, tag: str, source_name: str
    ) -> Optional[Article]:
        if not url or not full_text:
            return None
        if not is_ai_related(f"{title} {full_text}"):
            return None
        return Article(
            id=Article.make_id(url),
            name=title,
            url=url,
            tag=tag,
            source=source_name,
            author=None,
            published_at=None,
            full_text=full_text,
        )


_normalizer = ArticleNormalizer()
