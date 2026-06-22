"""RSS / Atom feed scanner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import feedparser

from .. import settings as config
from ..logging_setup import logger
from ..models import Article
from ..normalizer import _normalizer


class RSSScanner:
    def fetch(self, url: str, tag: str, source_name: str) -> list[Article]:
        try:
            feed = feedparser.parse(url)
            entries = list(feed.entries)
            if not entries:
                logger.info("RSS '%s': 0 articles", source_name)
                return []

            def _safe(entry) -> Optional[Article]:
                try:
                    return _normalizer.from_rss(entry, tag, source_name)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("RSS entry failed (%s): %s", source_name, exc)
                    return None

            workers = max(1, getattr(config, "MAX_WORKERS", 8))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = ex.map(_safe, entries)
            articles = [a for a in results if a]
            logger.info("RSS '%s': %d articles", source_name, len(articles))
            return articles
        except Exception as exc:
            logger.warning("RSS feed failed (%s): %s", source_name, exc)
            return []
