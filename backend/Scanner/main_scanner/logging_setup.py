"""Shared logger for the scanner package."""

from __future__ import annotations

import logging

from .settings import LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

logger = logging.getLogger("scanner")
