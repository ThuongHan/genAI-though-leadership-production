"""
main.py — KickstartAI Belief System Pipeline
=============================================
Orchestrates all five steps end-to-end:

    Step 1 — Ingestion      : load raw data (blog + LinkedIn posts)
    Step 2 — Extraction     : extract beliefs from text
    Step 3 — Structuring    : deduplicate and canonicalise beliefs
    Step 4 — Vectorstore    : embed beliefs and build vector index
    Step 5 — Interface      : expose retrieval interface for Member 3

Usage:
    python main.py                          # run full pipeline (default: llm extractor)
    python main.py --extractor sensemaking  # run with sensemaking extractor
    python main.py --extractor multimodel   # run with multi-model agreement extractor
    python main.py --steps 1 2              # run specific steps only
    python main.py --steps 4 5              # rebuild vectorstore + interface only

Output files are namespaced by extractor method so results never overwrite each other:
    data/processed/beliefs_extracted_llm.json
    data/processed/beliefs_extracted_sensemaking.json
    data/processed/beliefs_extracted_multimodel.json
    data/processed/belief_repository_llm.json
    ...
    data/belief_store/beliefs_with_embeddings_llm.json
    ...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# ── PATHS (static) ────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
STORE_DIR     = Path("data/belief_store")

BLOG_PATH      = RAW_DIR / "blog.txt"
POSTS_CSV_PATH = RAW_DIR / "linkedin_posts.csv"
LINKEDIN_XLSX  = RAW_DIR / "KickstartAI_LinkedIn_post_year_to_date.xlsx"


# ── DYNAMIC PATHS (per extractor) ─────────────────────────────────────────────

def get_paths(extractor_method: str) -> tuple[Path, Path, Path]:
    """
    Return output paths namespaced by extractor method.
    This ensures each extractor writes to its own files so results
    can be compared side-by-side without any overwrites.

    Returns:
        (beliefs_raw_path, repo_path, store_path)
    """
    beliefs_raw = PROCESSED_DIR / f"beliefs_extracted_{extractor_method}.json"
    repo = STORE_DIR / f"belief_repository_{extractor_method}.json"
    store       = STORE_DIR     / f"beliefs_with_embeddings_{extractor_method}.json"
    return beliefs_raw, repo, store


# ── STEP RUNNERS ──────────────────────────────────────────────────────────────

def run_step1() -> None:
    """
    Step 1 — Ingestion
    Load blog.txt and LinkedIn xlsx, save cleaned outputs to data/raw/.
    """
    print("\n" + "=" * 60)
    print("STEP 1 — Ingestion")
    print("=" * 60)

    from utils.file_io import load_txt, load_linkedin_xlsx, save_txt, save_csv

    # Load blog text
    blog_text  = load_txt(BLOG_PATH)

    # Load and filter LinkedIn posts
    df_organic = load_linkedin_xlsx(LINKEDIN_XLSX)

    # Save cleaned outputs for Step 2
    save_txt(blog_text, BLOG_PATH)
    save_csv(df_organic, POSTS_CSV_PATH)

    print(f"\n[Step 1] Done. Outputs in: {RAW_DIR}")


def run_step2(extractor_method: str, beliefs_raw_path: Path) -> None:
    """
    Step 2 — Belief Extraction
    Extract beliefs from blog + LinkedIn posts using selected extractor method.
    Output is written to beliefs_raw_path (namespaced by extractor).

    Note: SensemakingExtractor uses output_dir + its own fixed filename, so we
    call it differently and then rename the output to match our naming convention.
    """
    print("\n" + "=" * 60)
    print(f"STEP 2 — Extraction  [method: {extractor_method}]")
    print("=" * 60)

    from extractors import get_extractor

    extractor = get_extractor(extractor_method)

    if extractor_method == "sensemaking":
        # SensemakingExtractor accepts both output_dir and output_path.
        # Passing output_path ensures files are named after the extractor method.
        extractor.run_pipeline(
            blog_path   = BLOG_PATH,
            posts_path  = POSTS_CSV_PATH,
            output_dir  = PROCESSED_DIR,
            output_path = beliefs_raw_path,
        )

        # Normalise sensemaking schema → standard schema expected by Step 3
        # Sensemaking fields : belief_statement | inference_type | domain | confidence | source_label
        # Standard fields    : belief           | belief_type    | category             | source_document
        with open(beliefs_raw_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        normalised = []
        for b in raw:
            normalised.append({
                "belief":          b.get("belief_statement", b.get("belief", "")),
                "category":        b.get("domain",           "domain_knowledge"),
                "source_quote":    b.get("belief_statement", "")[:120],
                "belief_type":     "secondary" if b.get("inference_type", "") == "implicit" else "primary",
                "source_document": b.get("source_label",    b.get("source_document", "sensemaking")),
                "inference_type":  b.get("inference_type",  ""),
                "confidence":      b.get("confidence",      ""),
            })

        with open(beliefs_raw_path, "w", encoding="utf-8") as f:
            json.dump(normalised, f, indent=2, ensure_ascii=False)

        print(f"  [Normalised] {len(normalised)} beliefs → {beliefs_raw_path.name}")

    elif extractor_method == "multimodel":
        # MultiModelExtractor outputs extra fields (agreement_count, agreement_score,
        # confidence) that load_raw_beliefs does not expect. Run extraction first,
        # then normalise to the standard schema while preserving the extra fields.
        extractor.run_pipeline(
            blog_path   = BLOG_PATH,
            posts_path  = POSTS_CSV_PATH,
            output_path = beliefs_raw_path,
        )

        with open(beliefs_raw_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        normalised = []
        for b in raw:
            normalised.append({
                # ── standard fields (required by Step 3) ──────────────────────
                # Both field name variants are preserved so downstream steps
                # can read whichever name they expect without KeyErrors.
                "belief":           b.get("belief_statement", b.get("belief", "")),
                "belief_statement": b.get("belief_statement", b.get("belief", "")),
                "category":         b.get("category",         "domain_knowledge"),
                "source_quote":     b.get("source_excerpt",   b.get("source_quote", "")),
                "source_excerpt":   b.get("source_excerpt",   b.get("source_quote", "")),
                "belief_type":      b.get("belief_type",      "primary"),
                "source_document":  b.get("source_document",  ""),
                "source_id":        b.get("source_id",        ""),
                "source_text":      b.get("source_text",      ""),
                "meta":             b.get("meta",             {}),
                # ── multimodel-specific fields (preserved for analysis) ───────
                "belief_id":        b.get("belief_id",        ""),
                "agreement_count":  b.get("agreement_count",  1),
                "agreement_score":  b.get("agreement_score",  0.0),
                "confidence":       b.get("confidence",       "medium"),
            })

        with open(beliefs_raw_path, "w", encoding="utf-8") as f:
            json.dump(normalised, f, indent=2, ensure_ascii=False)

        print(f"  [Normalised] {len(normalised)} beliefs -> {beliefs_raw_path.name}")

    else:
        # LLMExtractor — standard schema, no normalisation needed
        extractor.run_pipeline(
            blog_path   = BLOG_PATH,
            posts_path  = POSTS_CSV_PATH,
            output_path = beliefs_raw_path,
        )

    print(f"\n[Step 2] Done. Output: {beliefs_raw_path}")


def run_step3(beliefs_raw_path: Path, repo_path: Path, extractor_method: str) -> None:
    from belief_system.structure  import load_raw_beliefs, deduplicate_and_structure, validate_ids
    from belief_system.repository import save_beliefs

    raw_beliefs = load_raw_beliefs(beliefs_raw_path, extractor_method=extractor_method)
    structured  = deduplicate_and_structure(raw_beliefs, extractor_method=extractor_method)
    validate_ids(structured, extractor_method=extractor_method)
    save_beliefs(structured, repo_path)


def run_step4(repo_path: Path, store_path: Path) -> None:
    """
    Step 4 — Vectorstore
    Embed all canonical beliefs and build the vector index.
    """
    print("\n" + "=" * 60)
    print("STEP 4 — Vectorstore")
    print("=" * 60)

    from belief_system.repository import load_beliefs
    from embeddings.index         import build_vectorstore

    beliefs = load_beliefs(repo_path)
    build_vectorstore(beliefs, store_file=store_path)

    print(f"\n[Step 4] Done. Vectorstore → {store_path}")


def run_step5(repo_path: Path, store_path: Path) -> None:
    """
    Step 5 — Interface (smoke test)
    Verify the retrieval interface works end-to-end.
    This is the public interface consumed by Member 3 (Interpreter).
    """
    print("\n" + "=" * 60)
    print("STEP 5 — Interface (smoke test)")
    print("=" * 60)

    from retrieval.retriever import (
        get_all_beliefs,
        get_beliefs_by_category,
        retrieve_relevant_beliefs,
        format_beliefs_for_prompt,
    )

    # 5a. All beliefs
    all_beliefs = get_all_beliefs(repo_path)
    print(f"[Step 5] Total beliefs in repository : {len(all_beliefs)}")

    # 5b. By category
    for cat in ["values", "strategy", "mission", "stance", "domain_knowledge"]:
        filtered = get_beliefs_by_category(cat, repo_path)
        print(f"  [{cat:<20}] {len(filtered)} beliefs")

    # 5c. RAG retrieval
    test_query = "How should AI contribute to Dutch society?"
    results    = retrieve_relevant_beliefs(test_query, k=3, store_path=store_path)
    print(f"\n[Step 5] Top-3 beliefs for: '{test_query}'")
    print(format_beliefs_for_prompt(results))

    print(f"\n[Step 5] Done. Retrieval interface is ready for Member 3.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(steps: list[int], extractor_method: str) -> None:

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve namespaced output paths for this extractor
    BELIEFS_RAW_PATH, REPO_PATH, STORE_PATH = get_paths(extractor_method)

    print("\n" + "=" * 60)
    print(f"[Config] extractor  : {extractor_method}")
    print(f"[Config] beliefs    : {BELIEFS_RAW_PATH}")
    print(f"[Config] repository : {REPO_PATH}")
    print(f"[Config] store      : {STORE_PATH}")
    print("=" * 60)

    step_runners = {
        1: run_step1,
        2: lambda: run_step2(extractor_method, BELIEFS_RAW_PATH),
        3: lambda: run_step3(BELIEFS_RAW_PATH, REPO_PATH, extractor_method),
        4: lambda: run_step4(REPO_PATH, STORE_PATH),
        5: lambda: run_step5(REPO_PATH, STORE_PATH),
    }

    for step in steps:
        step_runners[step]()

    print("\n" + "=" * 60)
    print(f"Pipeline complete. Steps run: {steps}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KickstartAI Belief System Pipeline")

    parser.add_argument(
        "--extractor",
        type=str,
        default="llm",
        choices=["llm", "sensemaking", "multimodel"],
        help="Extractor method for Step 2 (default: llm)"
    )

    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
        help="Steps to run (default: 1 2 3 4 5)"
    )

    args = parser.parse_args()
    main(steps=sorted(args.steps), extractor_method=args.extractor)