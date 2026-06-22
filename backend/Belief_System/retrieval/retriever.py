from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from embeddings.embedder import get_embedding
from embeddings.index    import cosine_similarity

# ── DEFAULT PATHS ─────────────────────────────────────────────────────────────
_REPO_PATH  = Path("data/processed/belief_repository.json")
_STORE_PATH = Path("data/belief_store/beliefs_with_embeddings.json")

# ── VALID CATEGORIES ──────────────────────────────────────────────────────────
_VALID_CATEGORIES = {"mission", "strategy", "domain_knowledge", "values", "stance"}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _ensure_file_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


# ── PUBLIC INTERFACE ──────────────────────────────────────────────────────────

def get_all_beliefs(
    repo_path: str | Path = _REPO_PATH,
) -> list[dict]:
    """
    Return the full canonical belief repository.
    Used by Member 3 (Interpreter) to access all beliefs at once.

    Args:
        repo_path: path to belief_repository.json (Step 3 output)

    Returns:
        List of all canonical belief dicts.
    """
    repo_path = Path(repo_path)
    _ensure_file_exists(repo_path)

    beliefs = json.loads(repo_path.read_text(encoding="utf-8"))

    if not isinstance(beliefs, list):
        raise ValueError(f"Expected a list in {repo_path}")

    return beliefs


def get_beliefs_by_category(
    category: str,
    repo_path: str | Path = _REPO_PATH,
) -> list[dict]:
    """
    Filter and return beliefs matching a specific category.

    Args:
        category:  one of: mission | strategy | domain_knowledge | values | stance
        repo_path: path to belief_repository.json

    Returns:
        Filtered list of belief dicts.

    Raises:
        ValueError: if category is not one of the valid options.
    """
    if category not in _VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. "
            f"Must be one of: {sorted(_VALID_CATEGORIES)}"
        )

    return [
        b for b in get_all_beliefs(repo_path)
        if b.get("category") == category
    ]


def retrieve_relevant_beliefs(
    query: str,
    k: int = 3,
    store_path: str | Path = _STORE_PATH,
) -> list[dict]:
    """
    Retrieve the k most relevant beliefs for a given query string.
    Uses cosine similarity on placeholder embeddings (Step 4).
    Used by the Interpreter module during RAG-based contextualisation.

    Args:
        query:      natural language query from the Interpreter
        k:          number of top beliefs to return
        store_path: path to beliefs_with_embeddings.json (Step 4 output)

    Returns:
        List of up to k belief dicts, each with a 'relevance_score' field,
        sorted by descending relevance.
    """
    store_path = Path(store_path)
    _ensure_file_exists(store_path)

    records   = json.loads(store_path.read_text(encoding="utf-8"))
    query_vec = np.array(get_embedding(query), dtype=np.float32)

    scored = []
    for r in records:
        belief_vec = np.array(r["embedding"], dtype=np.float32)
        sim = cosine_similarity(query_vec, belief_vec)

        scored.append({
            "id":              r["id"],
            "belief":          r["belief"],
            "category":        r["category"],
            "sources":         r.get("sources", []),
            "relevance_score": round(sim, 6),
        })

    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:k]


def format_beliefs_for_prompt(beliefs: list[dict]) -> str:
    """
    Format retrieved beliefs into a prompt-injectable string block.
    Standardised format consumed by the Interpreter's system prompt.

    Args:
        beliefs: list of belief dicts (from get_all_beliefs or retrieve_relevant_beliefs)

    Returns:
        A formatted string block ready to inject into a prompt.

    Example output:
        [KickstartAI Belief Repository — Retrieved Context]
        - [B001] (values): AI should serve society rather than ...
        - [B002] (strategy): Organisations must embed AI literacy ...
    """
    lines = ["[KickstartAI Belief Repository — Retrieved Context]"]

    for b in beliefs:
        lines.append(f"- [{b['id']}] ({b['category']}): {b['belief']}")

    return "\n".join(lines)


# ── ENTRY POINT (smoke test) ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Demonstrate the full public interface

    print("=== get_all_beliefs ===")
    all_beliefs = get_all_beliefs()
    print(f"Total: {len(all_beliefs)} beliefs\n")

    print("=== get_beliefs_by_category('values') ===")
    values_beliefs = get_beliefs_by_category("values")
    for b in values_beliefs:
        print(f"  [{b['id']}] {b['belief'][:70]}")

    print("\n=== retrieve_relevant_beliefs ===")
    query   = "AI regulation and societal responsibility"
    results = retrieve_relevant_beliefs(query, k=3)
    print(format_beliefs_for_prompt(results))