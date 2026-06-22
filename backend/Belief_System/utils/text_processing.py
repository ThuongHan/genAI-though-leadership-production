import re
import pandas as pd
from pathlib import Path


def chunk_text(text: str, max_chars: int = 8000) -> list[str]:
    """
    Split long blog text into paragraph-preserving chunks.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []

    for para in paragraphs:
        tentative = "\n\n".join(current + [para])
        if len(tentative) <= max_chars:
            current.append(para)
        else:
            if current:
                chunks.append("\n\n".join(current))
            if len(para) <= max_chars:
                current = [para]
            else:
                # hard split oversized paragraph
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i:i + max_chars])
                current = []

    if current:
        chunks.append("\n\n".join(current))

    return chunks

def build_corpus(blog_path: Path, posts_path: Path) -> list[dict]:
    """
    Build corpus from Step 1 outputs.
    - blog.txt becomes multiple chunked documents
    - each LinkedIn post title becomes one document
    """
    blog_text = blog_path.read_text(encoding="utf-8").strip()
    if not blog_text:
        raise ValueError(f"Blog file is empty: {blog_path}")

    df_posts = pd.read_csv(posts_path)
    required_cols = {"Post title", "Created date", "Engagement rate"}
    missing = required_cols - set(df_posts.columns)
    if missing:
        raise KeyError(f"Missing columns in {posts_path.name}: {missing}")

    corpus = []

    # Blog chunks
    blog_chunks = chunk_text(blog_text, max_chars=4000)
    for i, chunk in enumerate(blog_chunks, start=1):
        corpus.append(
            {
                "id": f"blog_chunk_{i:03d}",
                "source": "blog",
                "text": chunk,
                "meta": {"chunk_index": i}
            }
        )

    # LinkedIn post titles
    df_posts = df_posts.dropna(subset=["Post title"]).reset_index(drop=True)
    for i, row in df_posts.iterrows():
        title = str(row["Post title"]).strip()
        if not title:
            continue

        corpus.append(
            {
                "id": f"linkedin_post_{i+1:04d}",
                "source": "linkedin_posts",
                "text": title,
                "meta": {
                    "created_date": None if pd.isna(row["Created date"]) else str(row["Created date"]),
                    "engagement_rate": None if pd.isna(row["Engagement rate"]) else float(row["Engagement rate"]),
                }
            }
        )

    return corpus

def strip_code_fences(text: str) -> str:
    """Remove accidental markdown code fences from model output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()