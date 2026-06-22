"""NewsAPI source scanner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from .. import settings as config
from ..logging_setup import logger
from ..models import Article
from ..normalizer import _normalizer


def _normalize_items_parallel(items: list[dict], tag: str) -> list[Article]:
    """Fetch + normalise NewsAPI items concurrently (one worker per article).

    Each worker independently fetches full text and builds an Article; the
    extractor's per-fetch delay throttles each worker, so up to MAX_WORKERS
    fetches run at once. Per-item errors are swallowed so one bad article
    can't kill the batch.
    """
    if not items:
        return []

    def _safe(item: dict) -> Optional[Article]:
        try:
            return _normalizer.from_newsapi(item, tag)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("NewsAPI item failed: %s", exc)
            return None

    workers = max(1, getattr(config, "MAX_WORKERS", 8))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = ex.map(_safe, items)
    return [a for a in results if a]


class NewsAPIScanner:
    BASE = "https://newsapi.org/v2"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"X-Api-Key": api_key, **config.REQUEST_HEADERS})

    def _get(self, endpoint: str, params: dict) -> dict:
        try:
            resp = self.session.get(
                f"{self.BASE}/{endpoint}", params=params, timeout=config.REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("NewsAPI request failed (%s): %s", endpoint, exc)
            return {}

    def fetch_everything(
        self,
        query: str,
        domains: Optional[str] = None,
        language: str | None = None,
        page_size: int | None = None,
        days_back: int | None = None,
        tag: str = "news",
    ) -> list[Article]:
        if language is None:
            language = config.NEWSAPI_LANGUAGE
        if page_size is None:
            page_size = config.NEWSAPI_PAGE_SIZE
        if days_back is None:
            days_back = config.NEWSAPI_DAYS_BACK

        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime(
            "%Y-%m-%d"
        )
        params: dict = {
            "q": query,
            "language": language,
            "pageSize": page_size,
            "from": from_date,
            "sortBy": "publishedAt",
        }
        if domains:
            params["domains"] = domains

        data = self._get("everything", params)
        articles = _normalize_items_parallel(data.get("articles", []), tag)
        logger.info("NewsAPI '%s': %d articles", query[:60], len(articles))
        return articles

    def fetch_top_headlines(
        self,
        query: str,
        language: str | None = None,
        page_size: int | None = None,
        tag: str = "news",
    ) -> list[Article]:
        if language is None:
            language = config.NEWSAPI_LANGUAGE
        if page_size is None:
            page_size = config.NEWSAPI_PAGE_SIZE

        params = {"q": query, "language": language, "pageSize": page_size}
        data = self._get("top-headlines", params)
        articles = _normalize_items_parallel(data.get("articles", []), tag)
        logger.info("NewsAPI headlines '%s': %d articles", query, len(articles))
        return articles
