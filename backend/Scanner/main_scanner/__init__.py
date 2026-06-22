"""
KickstartAI Industry Scanner package.

Public API re-exported for backward compatibility with existing imports
(scanner_annotation.py, build_annotation_db.py).
"""

from __future__ import annotations

# Re-exports
from .dedup import deduplicate
from .extractor import FullTextExtractor, _extractor
from .logging_setup import logger
from .models import Article
from .normalizer import ArticleNormalizer, _normalizer
from .orchestrator import run_scanner
from .output import write_output
from .sources import ArxivScanner, NewsAPIScanner, RSSScanner, WebScraper
from .text import (
    _ALL_KEYWORDS,
    _CONSENT_MARKERS,
    _NL_KEYWORDS,
    detect_language,
    is_ai_related,
    is_consent_wall,
)

__all__ = [
    "Article",
    "ArticleNormalizer",
    "ArxivScanner",
    "FullTextExtractor",
    "NewsAPIScanner",
    "RSSScanner",
    "WebScraper",
    "deduplicate",
    "detect_language",
    "is_ai_related",
    "is_consent_wall",
    "logger",
    "run_scanner",
    "write_output",
]
