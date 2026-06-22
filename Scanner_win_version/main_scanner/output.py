"""JSON output writer for scanner runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from . import settings as config
from .logging_setup import logger
from .models import Article


def write_output(articles: list[Article], path: str | None = None) -> None:
    if path is None:
        path = config.OUTPUT_FILE
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "article_count": len(articles),
        "articles": [asdict(a) for a in articles],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote %d articles to %s", len(articles), path)
