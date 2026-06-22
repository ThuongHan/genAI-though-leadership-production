from __future__ import annotations

import hashlib

import numpy as np


def get_embedding(text: str, dim: int = 64) -> list[float]:
    """
    Deterministic local placeholder embedding.

    This is NOT a semantic embedding API.
    It creates a stable numeric vector from the text using SHA-256 hashing,
    so the pipeline runs end-to-end without an external embedding service.

    Replace this with a real embedding model (e.g. OpenAI text-embedding-3-small,
    sentence-transformers) once you are ready to move beyond the prototype stage.

    Args:
        text: input text to embed
        dim:  vector dimensionality (default 64)

    Returns:
        A normalised list of floats of length `dim`.
    """
    text = (text or "").strip()

    if not text:
        return [0.0] * dim

    values = []
    counter = 0

    while len(values) < dim:
        digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
        values.extend(digest)   # each digest adds 32 integers in [0, 255]
        counter += 1

    vec = np.array(values[:dim], dtype=np.float32)

    # centre and normalise to unit sphere
    vec = (vec - 127.5) / 127.5
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()


# ── ENTRY POINT (smoke test) ──────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "AI should serve society rather than function as a technology race."
    vec = get_embedding(sample)
    print(f"[embedder] Input  : {sample}")
    print(f"[embedder] Dim    : {len(vec)}")
    print(f"[embedder] First 8: {[round(v, 4) for v in vec[:8]]}")
    print(f"[embedder] Norm   : {round(sum(v**2 for v in vec) ** 0.5, 6)}")