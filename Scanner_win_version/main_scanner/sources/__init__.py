"""Source-specific scanner classes."""

from .arxiv_source import ArxivScanner
from .newsapi import NewsAPIScanner
from .rss import RSSScanner
from .scraper import WebScraper

__all__ = ["ArxivScanner", "NewsAPIScanner", "RSSScanner", "WebScraper"]
