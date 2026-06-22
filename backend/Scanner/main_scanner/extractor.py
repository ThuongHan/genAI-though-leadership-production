"""Full-text extraction from URLs via trafilatura."""

from __future__ import annotations

import re
import time

import requests
import trafilatura

from . import settings as config
from .logging_setup import logger


# Lines matching any of these patterns are stripped from extracted article text.
# Conservative: only well-known boilerplate / promo phrases, no length-based filtering.
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(subscribe|sign\s*up)\s+(to|for)\s+(our|the|this)\s+newsletter", re.IGNORECASE),
    re.compile(r"^\s*subscribe\s+to\s+our\s+", re.IGNORECASE),
    re.compile(r"^\s*sign\s*up\s+for\s+", re.IGNORECASE),
    re.compile(r"^\s*follow\s+us\s+on\b", re.IGNORECASE),
    re.compile(r"^\s*advertisement\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*sponsored(\s+content|\s+story|\s+post)?\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(read\s+more|related|see\s+also)\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^\s*share\s+(this|on)\b", re.IGNORECASE),
    re.compile(r"^\s*©\s*\d{4}", re.IGNORECASE),
    re.compile(r"^\s*copyright\s+©", re.IGNORECASE),
    re.compile(r"^\s*all\s+rights\s+reserved\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*click\s+here\s+to\b", re.IGNORECASE),
    re.compile(r"^\s*image\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*photo\s*(credit)?\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*(view|download)\s+(pdf|the\s+report)\b", re.IGNORECASE),
    re.compile(r"^\s*(cookies?|privacy)\s+(policy|notice|settings)\s*$", re.IGNORECASE),
    re.compile(r"^\s*back\s+to\s+top\s*$", re.IGNORECASE),
]


class FullTextExtractor:
    """Extract and clean article text from a URL using trafilatura."""

    def fetch(self, url: str, timeout: int | None = None) -> str:
        if timeout is None:
            timeout = config.REQUEST_TIMEOUT
        try:
            time.sleep(config.REQUEST_DELAY)
            resp = requests.get(url, headers=config.REQUEST_HEADERS, timeout=timeout)
            resp.raise_for_status()
            text = trafilatura.extract(
                resp.text,
                include_comments=False,
                include_tables=False,
                include_links=False,
                no_fallback=False,
                favor_recall=True,
            )
            return self._clean(text or "")
        except Exception as exc:
            logger.debug("Full-text extraction failed for %s: %s", url, exc)
            return ""

    @staticmethod
    def _strip_boilerplate(text: str) -> str:
        if not text:
            return text
        lines = text.split("\n")
        kept = [
            line for line in lines
            if not any(p.search(line) for p in _BOILERPLATE_PATTERNS)
        ]
        return "\n".join(kept)

    @classmethod
    def _clean(cls, text: str) -> str:
        text = cls._strip_boilerplate(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()


_extractor = FullTextExtractor()
