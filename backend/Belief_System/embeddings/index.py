from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from embeddings.embedder import get_embedding

# ── DEFAULT PATHS ─────────────────────────────────────────────────────────────
_STORE_DIR  = Path("data/belief_store")
_STORE_FILE = _STORE_DIR / "beliefs_with_embeddings.json"


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Small epsilon (1e-8) prevents division by zero.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def build_vectorstore(
    beliefs: list[dict],
    store_file: str | Path = _STORE_FILE,
) -> list[dict]:
    """
    Embed every belief and write the resulting records to a JSON file.
    Called once after Step 3 produces belief_repository.json.

    Args:
        beliefs:    list of canonical belief dicts
                    (each must have: id, belief, category, sources)
        store_file: path to write the vectorstore JSON

    Returns:
        List of belief records, each with an added 'embedding' field.
    """
    store_file = Path(store_file)
    store_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for belief in beliefs:
        embedding = get_embedding(belief["belief"])
        records.append({
            "id":        belief["id"],
            "belief":    belief["belief"],
            "category":  belief["category"],
            "sources":   belief.get("sources", []),
            "embedding": embedding,
        })

    store_file.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[build_vectorstore] {len(records)} beliefs embedded → {store_file}")
    return records


def retrieve_top_k(
    query: str,
    k: int = 3,
    store_file: str | Path = _STORE_FILE,
) -> list[dict]:
    """
    Return the k most similar beliefs to a query string.
    Used by retrieval/retriever.py (Step 5) to serve Member 3's Interpreter.

    Args:
        query:      natural language query
        k:          number of top results to return
        store_file: path to the vectorstore JSON

    Returns:
        List of up to k belief dicts, each with an added 'score' field,
        sorted by descending cosine similarity.

    Raises:
        FileNotFoundError: if the vectorstore has not been built yet.
    """
    store_file = Path(store_file)

    if not store_file.is_file():
        raise FileNotFoundError(
            f"Vector store not found: {store_file}\n"
            f"Run build_vectorstore() first (Step 4)."
        )

    records   = json.loads(store_file.read_text(encoding="utf-8"))
    query_vec = np.array(get_embedding(query), dtype=np.float32)

    scored = []
    for r in records:
        belief_vec = np.array(r["embedding"], dtype=np.float32)
        score = cosine_similarity(query_vec, belief_vec)
        scored.append({**r, "score": round(score, 6)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


# ── ENTRY POINT (smoke test) ──────────────────────────────────────────────────

if __name__ == "__main__":
    from belief_system.repository import load_beliefs

    REPO_PATH  = Path("data/processed/belief_repository.json")
    STORE_PATH = Path("data/belief_store/beliefs_with_embeddings.json")

    # Step 4: build
    beliefs = load_beliefs(REPO_PATH)
    build_vectorstore(beliefs, store_file=STORE_PATH)

    # Smoke test: retrieve
    test_query = "How should AI contribute to Dutch society?"
    results = retrieve_top_k(test_query, k=3, store_file=STORE_PATH)

    print(f"\n[index] Top-3 beliefs for: '{test_query}'")
    print("-" * 60)
    for r in results:
        print(f"  [{r['id']}] score={r['score']:.4f}  {r['belief'][:70]}")