"""arXiv source scanner."""

from __future__ import annotations

import time

import arxiv

from .. import settings as config
from ..logging_setup import logger
from ..models import Article
from ..normalizer import _normalizer


class ArxivScanner:
    def fetch(
        self,
        query: str,
        max_results: int = 20,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
    ) -> list[Article]:
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
            )
            articles: list[Article] = []
            for result in arxiv.Client().results(search):
                art = _normalizer.from_arxiv(result)
                if art:
                    articles.append(art)
                time.sleep(config.REQUEST_DELAY)
            logger.info("arXiv '%s': %d papers", query[:60], len(articles))
            return articles
        except Exception as exc:
            logger.warning("arXiv query failed (%s): %s", query, exc)
            return []
