from __future__ import annotations

import json
from pathlib import Path

from extractors.base_extractor import BaseExtractor


def deduplicate_beliefs(beliefs: list[dict]) -> list[dict]:
    """
    Exact deduplication on normalised belief text.
    Preserves first occurrence; ignores case and trailing punctuation.
    """
    seen: set[str] = set()
    deduped: list[dict] = []

    for b in beliefs:
        belief = str(b.get("belief", "")).strip()
        if not belief:
            continue
        key = belief.lower().rstrip(".").strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(b)

    return deduped


def load_seed_beliefs(seed_path: str | Path) -> list[dict]:
    """
    Load an optional seed belief file (e.g. beliefs_raw.json from a prior run).
    Returns an empty list if the file does not exist.
    """
    seed_path = Path(seed_path)

    if not seed_path.exists():
        return []

    with open(seed_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Seed file must contain a JSON list: {seed_path}")

    for b in data:
        b.setdefault("belief_type",      "primary")
        b.setdefault("source_id",        "seed")
        b.setdefault("source_document",  "seed_file")

    return data


def get_extractor(method: str) -> BaseExtractor:
    """
    Factory function — return the correct extractor instance by method name.

    Usage:
        extractor = get_extractor("llm")
        extractor = get_extractor("sensemaking")

    Args:
        method: one of 'llm' | 'sensemaking'

    Returns:
        An instance of the corresponding BaseExtractor subclass.
    """
    from extractors.extractor_llm import LLMExtractor
    from extractors.extractor_sensemaking import SensemakingExtractor
    from extractors.extractor_multimodel import MultiModelExtractor

    EXTRACTOR_MAP: dict[str, type[BaseExtractor]] = {
        "llm":          LLMExtractor,
        "sensemaking":  SensemakingExtractor,
        "multimodel":   MultiModelExtractor,
    }

    if method not in EXTRACTOR_MAP:
        raise ValueError(
            f"Unknown extractor method '{method}'. "
            f"Available: {list(EXTRACTOR_MAP.keys())}"
        )

    return EXTRACTOR_MAP[method]()