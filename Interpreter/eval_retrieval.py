"""
eval_retrieval.py — RQ1: TF-IDF vs Embedding retrieval quality
================================================================
Evaluates belief retrieval against human-annotated gold relevance labels.

Metrics:
  - Hit@k   : fraction of news where ≥1 relevant belief appears in top-k
  - AvgRel@k: average number of relevant beliefs in top-k

Usage:
  python eval_retrieval.py
  python eval_retrieval.py --news my_news.json --gold my_labels.json --top-k 3
"""

import json
import os
import sys
import argparse
from typing import List, Dict, Set, Tuple

# ── Replace with your real module when ready ──
from pipeline import retrieve, load_beliefs

# ╔══════════════════════════════════════════════════════════════╗
# ║                      DATA LOADERS                            ║
# ╚══════════════════════════════════════════════════════════════╝

def load_news(path: str | None = None) -> List[dict]:
    """
    Load ~20 news articles for retrieval evaluation.

    Each news dict MUST contain at least:
      - "news_id":  str   — unique identifier
      - "title":    str   — article headline
      - "excerpt":  str   — article body or abstract (fed to retrieve())

    Args:
        path: Path to a JSON file. If None, looks for
              NEWS_PATH env var or defaults to "eval_news.json".

    Returns:
        List of news dicts, each with news_id, title, excerpt.

    Expected JSON format (one of):
      1. A dict with an "articles" key → list of news objects
      2. A top-level list of news objects

    TODO: Populate eval_news.json with ~20 articles selected from
          your existing kickstartai_blogs.json or scanner_output.json.
          Ensure each article has news_id, title, and excerpt fields.
    """
    if path is None:
        path = os.getenv("EVAL_NEWS_PATH", "eval_news.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"News file not found: {path}\n"
            f"  TODO: Create this file with ~20 articles from your blog/news corpus.\n"
            f"  Each article needs: news_id, title, excerpt (full_text)."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Accept both {"articles": [...]} and [...] formats
    if isinstance(raw, list):
        items = raw
    else:
        items = raw.get("articles", raw.get("results", []))

    # Normalise to the news_obj shape expected by retrieve()
    news_list: List[dict] = []
    for i, item in enumerate(items):
        news_obj = {
            "news_id": item.get("news_id", item.get("id", f"news-{i:03d}")),
            "title": item.get("title", item.get("name", "Unknown")),
            "excerpt": item.get("excerpt", item.get("full_text", item.get("description", ""))),
        }
        news_list.append(news_obj)

    return news_list


def load_gold_relevance(path: str | None = None) -> Dict[str, Set[str]]:
    """
    Load human-annotated gold relevance labels.

    Args:
        path: Path to a JSON or CSV file. If None, looks for
              GOLD_PATH env var or defaults to "eval_gold_labels.json".

    Returns:
        Dict mapping news_id → set of relevant belief_ids.
        Example: {"news-001": {"B003", "B015"}, "news-002": {"B042"}}

    Expected JSON format:
        {
            "news-001": ["B003", "B015"],
            "news-002": ["B042"],
            ...
        }

    TODO: For each of the ~20 news articles in eval_news.json,
          manually annotate 1–3 truly relevant beliefs from the
          belief repository (belief_repository.json).
          Save the result as eval_gold_labels.json.
    """
    if path is None:
        path = os.getenv("EVAL_GOLD_PATH", "eval_gold_labels.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gold labels file not found: {path}\n"
            f"  TODO: Create this file with manual relevance annotations.\n"
            f'  Format: {{"news_id": ["belief_id1", "belief_id2", ...]}}'
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    gold: Dict[str, Set[str]] = {}
    for news_id, belief_ids in raw.items():
        gold[str(news_id)] = set(str(bid) for bid in belief_ids)

    return gold


# ╔══════════════════════════════════════════════════════════════╗
# ║                     EVALUATION LOGIC                         ║
# ╚══════════════════════════════════════════════════════════════╝

def evaluate_retriever(
    retriever_type: str,
    news_list: List[dict],
    beliefs: List[dict],
    gold_relevance: Dict[str, Set[str]],
    top_k: int = 3,
) -> Tuple[float, float]:
    """
    Evaluate one retriever on all news articles.

    For each news article:
      1. Call retrieve(news_obj, beliefs, retriever_type, top_k)
      2. Extract the top_k belief_ids from the RAGResult
      3. Compare against gold_relevance[news_id] to compute:
         - Hit@k : 1 if ≥1 relevant belief in top_k, else 0
         - Rel@k : count of relevant beliefs in top_k

    Args:
        retriever_type: "tfidf" or "embedding"
        news_list:      list of news_obj dicts
        beliefs:        list of belief dicts (belief_id, belief_text)
        gold_relevance: {news_id: {relevant belief_ids}}
        top_k:          number of top beliefs to consider

    Returns:
        (hit_rate: float, avg_relevant: float)
          - hit_rate    = mean of Hit@k across all news
          - avg_relevant = mean of relevant beliefs in top_k
    """
    hits: List[int] = []        # 0 or 1 per news
    relevants: List[int] = []   # count of relevant beliefs per news
    skipped = 0

    for news_obj in news_list:
        news_id = news_obj["news_id"]

        # ── Skip news without gold labels ──
        if news_id not in gold_relevance:
            skipped += 1
            continue

        gold_ids = gold_relevance[news_id]

        # ── Call the unified retrieve() interface ──
        rag_result = retrieve(news_obj, beliefs, retriever_type, top_k)

        # ── Extract retrieved belief_ids ──
        retrieved_ids: List[str] = [b["belief_id"] for b in rag_result["beliefs"]]

        # ── Compute Hit@k: at least one relevant belief retrieved? ──
        hit = 1 if any(bid in gold_ids for bid in retrieved_ids) else 0
        hits.append(hit)

        # ── Compute Rel@k: how many relevant beliefs? ──
        rel_count = sum(1 for bid in retrieved_ids if bid in gold_ids)
        relevants.append(rel_count)

        # Per-article detail
        marker = "✅" if hit else "❌"
        print(f"  {marker} {news_id} | retrieved: {retrieved_ids} | gold: {gold_ids} | rel: {rel_count}/{len(gold_ids)}")

    if skipped > 0:
        print(f"  ⚠️  {skipped} news skipped (no gold labels)")

    n = len(hits)
    if n == 0:
        return 0.0, 0.0

    hit_rate = sum(hits) / n
    avg_relevant = sum(relevants) / n

    return hit_rate, avg_relevant


# ╔══════════════════════════════════════════════════════════════╗
# ║                         MAIN                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ1: Evaluate TF-IDF vs Embedding belief retrieval"
    )
    parser.add_argument(
        "--news", default=None,
        help="Path to eval news JSON (default: eval_news.json or $EVAL_NEWS_PATH)"
    )
    parser.add_argument(
        "--gold", default=None,
        help="Path to gold labels JSON (default: eval_gold_labels.json or $EVAL_GOLD_PATH)"
    )
    parser.add_argument(
        "--beliefs", default=None,
        help="Path to belief repository JSON (default: belief_repository.json)"
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of top beliefs to retrieve (default: 3)"
    )
    args = parser.parse_args()

    # ── Load data ──
    print("=" * 60)
    print("  RQ1: Retrieval Quality — TF-IDF vs Embedding")
    print("=" * 60)

    print("\n📂 Loading data...")
    news_list = load_news(args.news)
    print(f"   News articles: {len(news_list)}")

    beliefs = load_beliefs(args.beliefs) if args.beliefs else load_beliefs()
    print(f"   Beliefs: {len(beliefs)}")

    gold = load_gold_relevance(args.gold)
    print(f"   Gold-labelled news: {len(gold)}")

    # ── Filter to only news that have gold labels ──
    labelled_ids = set(gold.keys())
    labelled_news = [n for n in news_list if n["news_id"] in labelled_ids]
    print(f"   News with gold labels: {len(labelled_news)}")

    if len(labelled_news) == 0:
        print("\n⚠️  No news articles have gold labels. Exiting.")
        print("   TODO: Create eval_gold_labels.json with manual annotations.")
        return

    # ── Evaluate each retriever ──
    results: Dict[str, Tuple[float, float]] = {}

    for retriever_type in ["bm25", "embedding", "hybrid"]:
        print(f"\n{'─' * 40}")
        print(f"🔍 Retriever: {retriever_type.upper()} (top_k={args.top_k})")
        print(f"{'─' * 40}")

        hit_rate, avg_rel = evaluate_retriever(
            retriever_type=retriever_type,
            news_list=labelled_news,
            beliefs=beliefs,
            gold_relevance=gold,
            top_k=args.top_k,
        )
        results[retriever_type] = (hit_rate, avg_rel)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  📊 RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Retriever':<15} {'Hit@{0}'.format(args.top_k):<12} {'AvgRel@{0}'.format(args.top_k):<15}")
    print(f"  {'─' * 40}")
    for rt, (hit, avg) in results.items():
        print(f"  {rt.upper():<15} {hit:.3f}        {avg:.3f}")

    # ── Winner ──
    bm25_hit, bm25_avg = results.get("bm25", (0, 0))
    emb_hit, emb_avg = results.get("embedding", (0, 0))
    hyb_hit, hyb_avg = results.get("hybrid", (emb_hit, emb_avg))
    print(f"\n  Δ Hit@{args.top_k} (embedding − bm25):    {emb_hit - bm25_hit:+.3f}")
    print(f"  Δ Hit@{args.top_k} (hybrid − embedding):  {hyb_hit - emb_hit:+.3f}")
    print(f"  Δ Hit@{args.top_k} (hybrid − bm25):        {hyb_hit - bm25_hit:+.3f}")
    print(f"  Δ AvgRel@{args.top_k} (embedding − bm25):  {emb_avg - bm25_avg:+.3f}")
    print(f"  Δ AvgRel@{args.top_k} (hybrid − embedding): {hyb_avg - emb_avg:+.3f}")
    print(f"  Δ AvgRel@{args.top_k} (hybrid − bm25):      {hyb_avg - bm25_avg:+.3f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
