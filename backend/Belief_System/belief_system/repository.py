from __future__ import annotations

"""
repository.py — Belief Repository I/O Interface
================================================

Provides save_beliefs and load_beliefs for reading and writing the canonical
belief repository produced by Step 3 (structure.py). All file paths are
namespaced by extractor method so that each method's repository is kept
separate and results can be compared without risk of overwriting.

Method-namespaced file layout
------------------------------
    data/processed/belief_repository_llm.json
    data/processed/belief_repository_sensemaking.json
    data/processed/belief_repository_multimodel.json

Canonical schema (per belief)
------------------------------
    id                 str   — method-prefixed unique identifier (e.g. LLM-B001)
    belief             str   — canonical declarative belief statement
    category           str   — one of: mission | strategy | values | stance | domain_knowledge
    sources            list  — source_document labels from Step 2
    extractor_method   str   — which extractor produced this belief
    belief_type        str?  — primary | secondary  (LLM / Sensemaking / MultiModel)
    confidence         str?  — high | medium | low
    [method extras]    *     — agreement_count, inference_type, etc. where present

Usage (standalone)
------------------
    python -m belief_system.repository                        # default: llm
    python -m belief_system.repository --extractor sensemaking
    python -m belief_system.repository --extractor multimodel
"""

import argparse
import json
from pathlib import Path

# Required fields that every canonical belief must have.
_REQUIRED_FIELDS = ("id", "belief", "category")


# ── SAVE ──────────────────────────────────────────────────────────────────────

def save_beliefs(beliefs: list[dict], path: str | Path) -> None:
    """
    Write a list of canonical belief dicts to a JSON file.
    Creates parent directories if they do not exist.

    The output filename should already be namespaced by extractor method
    (e.g. belief_repository_llm.json). This function does not enforce naming;
    callers (main.py / structure.py) are responsible for correct path selection.

    Args:
        beliefs: list of canonical belief dicts
                 (each must have: id, belief, category, sources)
        path:    output file path (e.g. 'data/processed/belief_repository_llm.json')
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(beliefs, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"[save_beliefs] Written {len(beliefs)} beliefs → {out_path}")


# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_beliefs(path: str | Path) -> list[dict]:
    """
    Load and validate canonical beliefs from a belief_repository_<method>.json.

    Skips entries missing any of the required fields: id, belief, category.
    Normalises the 'sources' field to a clean list of strings.
    Preserves all method-specific extra fields (belief_type, confidence,
    agreement_count, inference_type, extractor_method, etc.) so downstream
    consumers can filter or inspect by method.

    Args:
        path: path to belief_repository_<method>.json

    Returns:
        List of cleaned canonical belief dicts.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if the file is empty or contains no valid beliefs.
    """
    in_path = Path(path)

    if not in_path.is_file():
        raise FileNotFoundError(f"Belief repository not found: {in_path}")

    data = json.loads(in_path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {in_path}, got {type(data).__name__}")

    cleaned = []
    for b in data:
        # ── Required fields ───────────────────────────────────────────────────
        belief_id   = str(b.get("id",       "")).strip()
        belief_text = str(b.get("belief",   "")).strip()
        category    = str(b.get("category", "")).strip()

        if not belief_id or not belief_text or not category:
            continue

        # ── Normalise sources ─────────────────────────────────────────────────
        sources = b.get("sources", [])
        if not isinstance(sources, list):
            sources = [str(sources)]
        sources = [str(s).strip() for s in sources if str(s).strip()]

        # ── Build entry, preserving all extra fields ──────────────────────────
        entry: dict = {
            "id":       belief_id,
            "belief":   belief_text,
            "category": category,
            "sources":  sources,
        }

        # Carry forward every field beyond the four canonical ones.
        # This preserves method-specific signals for downstream consumers.
        skip = {"id", "belief", "category", "sources"}
        for key, value in b.items():
            if key not in skip:
                entry[key] = value

        cleaned.append(entry)

    if not cleaned:
        raise ValueError(f"No usable beliefs found in {in_path}")

    print(f"[load_beliefs] Loaded {len(cleaned)} beliefs from {in_path}")
    return cleaned


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_repository_path(extractor_method: str, base_dir: str | Path = "data/processed") -> Path:
    """
    Return the canonical repository path for a given extractor method.

    Args:
        extractor_method: One of "llm", "sensemaking", "multimodel".
        base_dir:         Directory containing processed outputs.

    Returns:
        Path — e.g. data/processed/belief_repository_llm.json
    """
    return Path(base_dir) / f"belief_repository_{extractor_method}.json"


def summarise_repository(beliefs: list[dict]) -> None:
    """
    Print a formatted summary of a loaded belief repository to stdout.
    Groups counts by category and by extractor_method (where present).

    Args:
        beliefs: list of canonical belief dicts from load_beliefs.
    """
    print(f"\n{'─' * 70}")
    print(f"  Repository summary — {len(beliefs)} canonical beliefs")
    print(f"{'─' * 70}")

    # By category
    from collections import Counter
    by_cat = Counter(b.get("category", "unknown") for b in beliefs)
    print(f"\n  By category:")
    for cat, count in sorted(by_cat.items()):
        print(f"    {cat:<25} {count:>4}")

    # By extractor_method (if field is present)
    methods = [b.get("extractor_method") for b in beliefs if b.get("extractor_method")]
    if methods:
        by_method = Counter(methods)
        print(f"\n  By extractor method:")
        for method, count in sorted(by_method.items()):
            print(f"    {method:<25} {count:>4}")

    # By belief_type (if field is present)
    types = [b.get("belief_type") for b in beliefs if b.get("belief_type")]
    if types:
        by_type = Counter(types)
        print(f"\n  By belief type:")
        for btype, count in sorted(by_type.items()):
            print(f"    {btype:<25} {count:>4}")

    print(f"{'─' * 70}\n")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a method-namespaced belief repository."
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default="llm",
        choices=["llm", "sensemaking", "multimodel"],
        help="Extractor method whose repository to load (default: llm)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load and compare all three method repositories side-by-side"
    )
    args = parser.parse_args()

    PROCESSED_DIR = Path("data/processed")

    if args.all:
        # ── Compare all three repositories ────────────────────────────────────
        all_beliefs: list[dict] = []
        for method in ["llm", "sensemaking", "multimodel"]:
            repo_path = get_repository_path(method, PROCESSED_DIR)
            if repo_path.is_file():
                method_beliefs = load_beliefs(repo_path)
                all_beliefs.extend(method_beliefs)
                print(f"  [{method:<12}] {len(method_beliefs)} canonical beliefs loaded")
            else:
                print(f"  [{method:<12}] NOT FOUND — {repo_path}")

        summarise_repository(all_beliefs)

    else:
        # ── Single method ─────────────────────────────────────────────────────
        method    = args.extractor
        repo_path = get_repository_path(method, PROCESSED_DIR)
        beliefs   = load_beliefs(repo_path)

        summarise_repository(beliefs)

        print(f"{'ID':<12} {'Category':<22} {'Type':<12} Belief")
        print("─" * 90)
        for b in beliefs:
            btype = b.get("belief_type", "—")
            print(f"{b['id']:<12} {b['category']:<22} {btype:<12} {b['belief'][:50]}")