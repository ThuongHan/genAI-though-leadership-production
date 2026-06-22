from __future__ import annotations

"""
structure.py — Step 3: Semantic Deduplication and Canonical Structuring
========================================================================

Theoretical basis
-----------------
This step operationalises the consolidation phase described in knowledge
repository design literature (Schein, 2010; van Dijk, 1998). Raw beliefs
extracted by three distinct methods (LLM, Sensemaking, MultiModel) are
semantically deduplicated and assigned stable canonical identifiers. Each
method produces its own named repository, allowing cross-method comparison
while preventing overwriting.

Method-namespaced outputs
-------------------------
    belief_repository_llm.json
    belief_repository_sensemaking.json
    belief_repository_multimodel.json

Belief ID format: <METHOD_PREFIX>-B001, e.g. LLM-B001, SNS-B001, MMD-B001.
This makes every belief's provenance traceable from its ID alone.

Usage (standalone)
------------------
    python -m belief_system.structure                        # default: llm
    python -m belief_system.structure --extractor sensemaking
    python -m belief_system.structure --extractor multimodel
"""

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── OPENAI CLIENT ─────────────────────────────────────────────────────────────
_api_key  = os.getenv("OPENAI_API_KEY")
_base_url = os.getenv("OPENAI_BASE_URL")

client = OpenAI(api_key=_api_key, base_url=_base_url) if _base_url else OpenAI(api_key=_api_key)

MODEL       = os.getenv("OPENAI_MODEL", "gpt-5.1")
TEMPERATURE = 0.0

# ── METHOD CONFIGURATION ──────────────────────────────────────────────────────

# Short prefix used in belief IDs to encode provenance, e.g. LLM-B001.
METHOD_PREFIXES: dict[str, str] = {
    "llm":         "LLM",
    "sensemaking": "SNS",
    "multimodel":  "MMD",
}

# Per-method fields to carry forward into the canonical repository.
# Standard fields (belief, category, sources) are always included.
# These extras allow downstream consumers to inspect method-specific metadata.
METHOD_EXTRA_FIELDS: dict[str, list[str]] = {
    "llm": [
        "belief_type",       # primary | secondary  (Schein, 2010 layer distinction)
        "confidence",        # high | medium | low
        "source_quote",      # verbatim excerpt grounding the belief
        "inference_reasoning",  # present on secondary beliefs (van Dijk, 1998)
    ],
    "sensemaking": [
        "belief_type",       # primary | secondary
        "confidence",        # high | medium | low
        "inference_type",    # explicit | implicit  (Weick, 1995 framing)
        "source_quote",      # grounding excerpt
    ],
    "multimodel": [
        "belief_type",       # primary | secondary
        "confidence",        # high | medium | low
        "agreement_count",   # 1–3 model votes (Du et al., 2023)
        "agreement_score",   # Jaccard-based consensus score
        "source_excerpt",    # grounding excerpt
    ],
}


# ── DEDUP PROMPT ──────────────────────────────────────────────────────────────

DEDUP_PROMPT = """
You are given a list of belief statements extracted from KickstartAI's documents
using the {method} extraction method.

Some beliefs are semantically equivalent or redundant.

Your task:
1. Merge semantically equivalent beliefs into a single canonical belief.
2. Preserve all genuinely distinct beliefs.
3. Assign a unique ID to each canonical belief using the format {prefix}-B001,
   {prefix}-B002, {prefix}-B003, etc.
4. For each canonical belief, keep:
   - "id":       unique belief identifier (format: {prefix}-B###)
   - "belief":   canonical belief statement
   - "category": one of ["mission", "strategy", "domain_knowledge", "values", "stance"]
   - "sources":  list of unique source_document values where this belief appeared
5. Be conservative: only merge beliefs that clearly express the same underlying
   proposition. Do NOT merge beliefs that differ in belief_type (primary vs
   secondary), as these represent analytically distinct layers (Schein, 2010).
6. Return ONLY valid JSON.

Input beliefs:
{beliefs_json}
""".strip()

# ── STRUCTURED OUTPUT SCHEMA ──────────────────────────────────────────────────

CANONICAL_SCHEMA = {
    "name": "canonical_belief_repository",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "beliefs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":       {"type": "string"},
                        "belief":   {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": ["mission", "strategy", "domain_knowledge", "values", "stance"]
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["id", "belief", "category", "sources"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["beliefs"],
        "additionalProperties": False
    }
}


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def load_raw_beliefs(path: str | Path, extractor_method: str = "llm") -> list[dict]:
    """
    Load and clean raw beliefs from the Step 2 output JSON.

    Preserves method-specific extra fields (e.g. belief_type, confidence,
    agreement_count) alongside the three standard fields required by Step 3
    (belief, category, source_document). Skips entries missing 'belief' or
    'category'.

    Args:
        path:             Path to the Step 2 output JSON.
        extractor_method: One of "llm", "sensemaking", "multimodel".
                          Controls which extra fields are forwarded.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if the file contains no usable beliefs.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 2 output: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")

    extra_fields = METHOD_EXTRA_FIELDS.get(extractor_method, [])

    cleaned = []
    for b in data:
        belief          = str(b.get("belief", "")).strip()
        category        = str(b.get("category", "")).strip()
        source_document = str(b.get("source_document", "")).strip()

        if not belief or not category:
            continue

        entry: dict = {
            "belief":          belief,
            "category":        category,
            "source_document": source_document or "unknown",
        }

        # Forward method-specific fields so the canonical repository retains
        # extractor provenance and quality signals.
        for field in extra_fields:
            value = b.get(field)
            if value is not None:
                entry[field] = value

        cleaned.append(entry)

    if not cleaned:
        raise ValueError(f"No usable beliefs found in {path}")

    print(f"[load_raw_beliefs] Loaded {len(cleaned)} beliefs from {path}")
    return cleaned


def deduplicate_and_structure(
    beliefs: list[dict],
    extractor_method: str = "llm",
) -> list[dict]:
    """
    Send raw beliefs to the LLM for semantic deduplication and canonical
    structuring. Returns a list of canonical belief dicts.

    Belief IDs are prefixed with the method code (e.g. LLM-B001, SNS-B001,
    MMD-B001) so provenance is traceable from the ID alone.

    The structured output contains the four canonical fields (id, belief,
    category, sources). Method-specific extra fields from the input are
    merged back onto each canonical belief after the LLM call, keyed by
    the canonical belief text for best-effort matching.

    Args:
        beliefs:          List of cleaned belief dicts from load_raw_beliefs.
        extractor_method: One of "llm", "sensemaking", "multimodel".
    """
    prefix       = METHOD_PREFIXES.get(extractor_method, extractor_method.upper()[:3])
    beliefs_json = json.dumps(beliefs, ensure_ascii=False, indent=2)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_completion_tokens=32000,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful analyst that consolidates extracted beliefs "
                    "into a canonical repository. Preserve analytical distinctions "
                    "between belief layers (Schein, 2010)."
                )
            },
            {
                "role": "user",
                "content": DEDUP_PROMPT.format(
                    method=extractor_method,
                    prefix=prefix,
                    beliefs_json=beliefs_json,
                )
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": CANONICAL_SCHEMA,
        },
    )

    content    = response.choices[0].message.content
    payload    = json.loads(content)
    structured = payload["beliefs"]

    # ── Merge method-specific fields back onto canonical beliefs ──────────────
    # Build a lookup from belief text → extra fields from the raw input.
    # This is a best-effort merge: canonical beliefs that resulted from merging
    # multiple raw beliefs will inherit the extra fields of the first match.
    extra_fields = METHOD_EXTRA_FIELDS.get(extractor_method, [])
    if extra_fields:
        raw_lookup: dict[str, dict] = {}
        for b in beliefs:
            key = b["belief"].strip().lower()
            if key not in raw_lookup:
                raw_lookup[key] = {f: b[f] for f in extra_fields if f in b}

        for canon in structured:
            canon_key = canon["belief"].strip().lower()
            extras    = raw_lookup.get(canon_key, {})
            # Also try prefix match for merged beliefs
            if not extras:
                for raw_key, raw_extras in raw_lookup.items():
                    if raw_key[:60] in canon_key or canon_key[:60] in raw_key:
                        extras = raw_extras
                        break
            canon.update(extras)

    # ── Tag with extractor method for audit ───────────────────────────────────
    for canon in structured:
        canon["extractor_method"] = extractor_method

    print(
        f"[deduplicate_and_structure] "
        f"{len(beliefs)} raw → {len(structured)} canonical beliefs "
        f"[method: {extractor_method}, prefix: {prefix}]"
    )
    return structured


def validate_ids(structured: list[dict], extractor_method: str = "llm") -> None:
    """
    Ensure every canonical belief has a unique ID with the correct method prefix.

    Args:
        structured:       List of canonical belief dicts.
        extractor_method: Expected method prefix (e.g. "llm" → "LLM-").

    Raises:
        ValueError: if any duplicate IDs or incorrectly prefixed IDs are found.
    """
    prefix   = METHOD_PREFIXES.get(extractor_method, extractor_method.upper()[:3])
    expected = f"{prefix}-"
    seen: set[str] = set()

    for b in structured:
        bid = b["id"]

        if not bid.startswith(expected):
            raise ValueError(
                f"Belief ID '{bid}' does not match expected prefix '{expected}' "
                f"for extractor method '{extractor_method}'."
            )

        if bid in seen:
            raise ValueError(f"Duplicate belief ID found: {bid}")

        seen.add(bid)

    print(f"[validate_ids] All {len(structured)} IDs are unique and correctly prefixed ({expected}*).")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3 — Deduplicate and canonicalise beliefs into a method-namespaced repository."
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default="llm",
        choices=["llm", "sensemaking", "multimodel"],
        help="Extractor method whose Step 2 output to process (default: llm)"
    )
    args = parser.parse_args()

    method   = args.extractor
    RAW_PATH = Path(f"data/processed/beliefs_extracted_{method}.json")
    OUT_PATH = Path(f"data/processed/belief_repository_{method}.json")

    raw_beliefs = load_raw_beliefs(RAW_PATH, extractor_method=method)
    structured  = deduplicate_and_structure(raw_beliefs, extractor_method=method)
    validate_ids(structured, extractor_method=method)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(structured, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\n[Step 3] Canonical repository written : {OUT_PATH}")
    print(f"[Step 3] Extractor method             : {method}")
    print(f"[Step 3] Total canonical beliefs      : {len(structured)}")
    print()

    for b in structured:
        print(f"  {b['id']} [{b['category']}]: {b['belief']}")