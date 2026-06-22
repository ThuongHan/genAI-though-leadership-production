"""Web scraper for listing pages and individual article pages."""

from __future__ import annotations

import re
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
import trafilatura

from .. import settings as config
from ..extractor import _extractor
from ..logging_setup import logger
from ..models import Article
from ..normalizer import _normalizer


class WebScraper:
    """
    Scrapes a listing/index page for article links, then extracts full text
    from each article page.
    """

    def scrape_listing(
        self,
        listing_url: str,
        tag: str,
        source_name: str,
        link_filter: Optional[str] = None,
        max_articles: int = 10,
    ) -> list[Article]:
        try:
            time.sleep(config.REQUEST_DELAY)
            resp = requests.get(
                listing_url, headers=config.REQUEST_HEADERS, timeout=config.REQUEST_TIMEOUT
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Scraper listing failed (%s): %s", listing_url, exc)
            return []

        raw_links = re.findall(r'href=["\']([^"\'#?][^"\']*)["\']', resp.text)
        base = f"{urlparse(listing_url).scheme}://{urlparse(listing_url).netloc}"
        seen: set[str] = set()
        article_urls: list[str] = []

        for href in raw_links:
            full = urljoin(base, href) if not href.startswith("http") else href
            if full in seen:
                continue
            if link_filter and link_filter not in full:
                continue
            if full.rstrip("/") == listing_url.rstrip("/"):
                continue
            seen.add(full)
            article_urls.append(full)
            if len(article_urls) >= max_articles:
                break

        articles: list[Article] = []
        for url in article_urls:
            try:
                time.sleep(config.REQUEST_DELAY)
                page_resp = requests.get(
                    url, headers=config.REQUEST_HEADERS, timeout=config.REQUEST_TIMEOUT
                )
                page_resp.raise_for_status()
                meta = trafilatura.extract_metadata(page_resp.text, default_url=url)
                title = (meta.title if meta and meta.title else "") or url
                full_text = _extractor._clean(
                    trafilatura.extract(
                        page_resp.text,
                        include_comments=False,
                        include_tables=False,
                        include_links=False,
                        favor_recall=True,
                    )
                    or ""
                )
                art = _normalizer.from_scraped(url, title, full_text, tag, source_name)
                if art:
                    articles.append(art)
            except Exception as exc:
                logger.debug("Scraper article failed (%s): %s", url, exc)

        logger.info("Scraper '%s': %d articles", source_name, len(articles))
        return articles

    def scrape_single(
        self, url: str, tag: str, source_name: str
    ) -> Optional[Article]:
        try:
            time.sleep(config.REQUEST_DELAY)
            resp = requests.get(url, headers=config.REQUEST_HEADERS, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            meta = trafilatura.extract_metadata(resp.text, default_url=url)
            title = (meta.title if meta and meta.title else "") or url
            full_text = _extractor.fetch(url)
            return _normalizer.from_scraped(url, title, full_text, tag, source_name)
        except Exception as exc:
            logger.warning("Single scrape failed (%s): %s", url, exc)
            return None
