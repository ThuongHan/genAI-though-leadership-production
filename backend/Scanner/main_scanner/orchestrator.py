"""Main scanner orchestrator — runs all sources and writes output."""

from __future__ import annotations

import time

from . import settings as config
from .dedup import deduplicate
from .logging_setup import logger
from .models import Article
from .output import write_output
from .paths import timestamped_path
from .sources import ArxivScanner, NewsAPIScanner, RSSScanner, WebScraper


def run_scanner() -> list[Article]:
    all_articles: list[Article] = []

    # 1. NewsAPI
    if config.NEWSAPI_KEY and config.NEWSAPI_KEY != "YOUR_NEWSAPI_KEY":
        newsapi = NewsAPIScanner(config.NEWSAPI_KEY)

        for query in config.NEWSAPI_SOURCES["queries_en"]:
            all_articles += newsapi.fetch_everything(query=query, language="en", tag="news")
            time.sleep(config.REQUEST_DELAY)

        for query in config.NEWSAPI_SOURCES["queries_nl"]:
            all_articles += newsapi.fetch_everything(query=query, language="nl", tag="dutch_news")
            time.sleep(config.REQUEST_DELAY)

        for entry in config.NEWSAPI_SOURCES["domain_queries"]:
            all_articles += newsapi.fetch_everything(
                query=entry["query"],
                domains=entry["domains"],
                language=entry.get("language", "en"),
                tag=entry.get("tag", "news"),
            )
            time.sleep(config.REQUEST_DELAY)
    else:
        logger.warning("NEWSAPI_KEY not set — skipping NewsAPI sources.")

    # 2. RSS feeds
    rss_scanner = RSSScanner()
    for feed_cfg in config.RSS_FEEDS:
        all_articles += rss_scanner.fetch(
            url=feed_cfg["url"],
            tag=feed_cfg["tag"],
            source_name=feed_cfg["source"],
        )

    # 3. arXiv
    arxiv_scanner = ArxivScanner()
    for query in [
        "artificial intelligence",
        "large language model",
        "generative AI",
        "AI regulation governance",
        "foundation model",
        "responsible AI trustworthy",
    ]:
        all_articles += arxiv_scanner.fetch(query=query, max_results=15)

    # 4. Web scraping
    scraper = WebScraper()
    for target in config.SCRAPE_TARGETS:
        if target.get("type") == "listing":
            all_articles += scraper.scrape_listing(
                listing_url=target["url"],
                tag=target["tag"],
                source_name=target["source"],
                link_filter=target.get("link_filter"),
                max_articles=target.get("max_articles", 10),
            )
        elif target.get("type") == "single":
            art = scraper.scrape_single(
                url=target["url"],
                tag=target["tag"],
                source_name=target["source"],
            )
            if art:
                all_articles.append(art)

    unique = deduplicate(all_articles)
    filtered = [a for a in unique if a.source not in config.EXCLUDED_SOURCES]
    logger.info(
        "Total: %d unique (from %d raw), %d kept after excluded-source filter",
        len(unique), len(all_articles), len(filtered),
    )
    # Date/time-stamped filename so each scan is kept (never overwrites a past run).
    out_path = timestamped_path(config.OUTPUT_FILE)
    write_output(filtered, out_path)
    return filtered
