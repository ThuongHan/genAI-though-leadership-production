"""
scorer.py — Automatic Interpretation Quality Scorer (LLM-as-Judge)
====================================================================
Scores each interpreter output on 5 dimensions (0–10) using gpt-4o-mini.
No human annotation required — fully automatic.

Dimensions (aligned with diagnose.py framework):
  1. stance_accuracy   (20%) — stance vs news facts + belief
  2. argument_quality  (25%) — 3 arguments: specific, logical, defensible?
  3. belief_alignment  (25%) — "Why for KickstartAI" connects to belief?
  4. factual_precision (15%) — "What happened" accurate vs source?
  5. nl_perspective    (15%) — Dutch/NL context addressed?

Usage:
  # Score an existing batch output file
  python scorer.py --input output/test_new_input.json --output output/scores.json

  # Score a single interpretation directly
  python scorer.py --news "..." --belief "..." --interpretation output/interpretation.json
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone
from typing import Optional

from openai import OpenAI

# ╔══════════════════════════════════════════════════════════════╗
# ║                        CLIENT                                ║
# ╚══════════════════════════════════════════════════════════════╝

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


# ╔══════════════════════════════════════════════════════════════╗
# ║                   SCORING CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝

SCORING_WEIGHTS = {
    "stance_accuracy":   0.20,
    "argument_quality":  0.25,
    "belief_alignment":  0.25,
    "factual_precision": 0.15,
    "nl_perspective":    0.15,
}

DIMENSION_LABELS = {
    "stance_accuracy":   "Stance Accuracy",
    "argument_quality":  "Argument Quality",
    "belief_alignment":  "Belief Alignment",
    "factual_precision": "Factual Precision",
    "nl_perspective":    "NL Perspective",
}

# Anchor descriptions for the LLM
SCORING_ANCHORS = """
Score each dimension 0–10 using these anchors:
  0–2:  Completely wrong / missing / irrelevant
  3–4:  Major flaws, barely acceptable
  5–6:  Acceptable but generic, lacks depth or specificity
  7–8:  Good — specific, well-reasoned, mostly accurate
  9–10: Excellent — precise, insightful, fully aligned, NL context well integrated
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║              SCORING SYSTEM PROMPT                           ║
# ╚══════════════════════════════════════════════════════════════╝

_SCORING_SYSTEM_PROMPT = f"""You are an expert quality auditor for an AI strategy interpretation pipeline.
Your job: score the interpreter's output on 5 dimensions.

The pipeline flow:
  News article → RAG (belief retrieval) → Interpreter (LLM) → structured JSON

You receive: the news article, the retrieved beliefs, and the interpreter's JSON output.
You score the interpreter's output ONLY — not the beliefs, not the news.

DIMENSIONS:
1. **stance_accuracy** (0–10): Is the stance (Supportive/Critical/Neutral/Cautious) appropriate given the news facts AND the retrieved belief? A stance that contradicts the belief or misreads the news should score low.

2. **argument_quality** (0–10): Are the 3 Supporting arguments specific, logical, and defensible? Vague or generic arguments ("AI is important") score low. Arguments with concrete reasoning or data references score high.

3. **belief_alignment** (0–10): Does "Why does it matter for KickstartAI" meaningfully connect to the retrieved belief? A generic rewording of the belief scores medium. A specific, insightful connection scores high.

4. **factual_precision** (0–10): Does "What happened" accurately reflect the news article's key facts? Hallucinations or misrepresentations score low. Accurate, well-summarized facts score high.

5. **nl_perspective** (0–10): Does the output address Dutch/NL context where relevant? For Dutch-language news or NL-relevant topics, this should be specific. For purely global news, a brief NL mention is adequate.

{SCORING_ANCHORS}

Output ONLY valid JSON with this exact structure:
{{
    "stance_accuracy":   {{"score": <int 0-10>, "justification": "<1-2 sentences, English>"}},
    "argument_quality":  {{"score": <int 0-10>, "justification": "<1-2 sentences, English>"}},
    "belief_alignment":  {{"score": <int 0-10>, "justification": "<1-2 sentences, English>"}},
    "factual_precision": {{"score": <int 0-10>, "justification": "<1-2 sentences, English>"}},
    "nl_perspective":    {{"score": <int 0-10>, "justification": "<1-2 sentences, English>"}}
}}

No markdown, no extra text — ONLY the JSON object."""


# ╔══════════════════════════════════════════════════════════════╗
# ║                 SCORE SINGLE INTERPRETATION                  ║
# ╚══════════════════════════════════════════════════════════════╝

def score_interpretation(
    news_obj: dict,
    rag_result: dict,
    interpreter_result: dict,
) -> dict:
    """
    Score a single interpreter output on 5 quality dimensions.

    Args:
        news_obj:           {{"news_id", "title", "excerpt"}} — original news
        rag_result:         RAGResult with "beliefs" list
        interpreter_result: InterpreterResult with "parsed_json"

    Returns:
        {{
            "overall_score": float,       # weighted average 0–10
            "dimensions": {{...}},        # per-dimension score + justification
            "scored_at": str,             # ISO timestamp
        }}
        On failure: returns error dict with overall_score = -1
    """
    client = _get_client()

    news_title = news_obj.get("title", news_obj.get("name", "Unknown"))
    news_excerpt = news_obj.get("excerpt", news_obj.get("full_text", ""))[:2000]

    # ── Build belief summary ──
    beliefs = rag_result.get("beliefs", [])
    belief_lines = []
    for b in beliefs:
        bid = b.get("belief_id", "?")
        btext = b.get("belief_text", "")[:200]
        bscore = b.get("score", 0)
        belief_lines.append(f"  [{bid}] (score={bscore:.3f}) {btext}")
    belief_summary = "\n".join(belief_lines) if belief_lines else "(none)"

    # ── Build interpreter output summary ──
    parsed = interpreter_result.get("parsed_json") or {}
    interp_text = json.dumps(parsed, indent=2, ensure_ascii=False)

    # ── Build user prompt ──
    user_prompt = f"""
=== NEWS ARTICLE ===
Title: {news_title}
Content (first 2000 chars): {news_excerpt}

=== RETRIEVED BELIEFS ===
{belief_summary}

=== INTERPRETER OUTPUT (to be scored) ===
{interp_text}

Please score this interpreter output on all 5 dimensions.
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        dims = json.loads(response.choices[0].message.content or "{}")
    except Exception as e:
        print(f"  ⚠️  score_interpretation error: {e}")
        return {
            "overall_score": -1,
            "dimensions": {},
            "error": str(e),
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Compute weighted overall ──
    total = 0.0
    for dim_key, weight in SCORING_WEIGHTS.items():
        dim_data = dims.get(dim_key, {})
        score = dim_data.get("score", 0)
        if isinstance(score, (int, float)):
            total += float(score) * weight

    overall = round(total, 1)

    return {
        "overall_score": overall,
        "dimensions": dims,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║                 SCORE BATCH                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def score_batch(
    news_list: list[dict],
    rag_results: list[dict],
    interpreter_results: list[dict],
) -> list[dict]:
    """
    Score a batch of interpretations.

    Args:
        news_list:           list of news_obj dicts
        rag_results:         list of RAGResult dicts
        interpreter_results: list of InterpreterResult dicts

    Returns:
        list of score dicts (same order as input)
    """
    scores = []
    n = len(interpreter_results)
    for i in range(n):
        news = news_list[i] if i < len(news_list) else {}
        rag = rag_results[i] if i < len(rag_results) else {}
        interp = interpreter_results[i]

        title = news.get("title", "?")[:60]
        nid = news.get("news_id", f"item-{i}")
        print(f"  Scoring [{i+1}/{n}] {nid} — {title}...", end=" ")

        s = score_interpretation(news, rag, interp)
        overall = s.get("overall_score", -1)
        print(f"{overall:.1f}/10" if overall >= 0 else "❌")

        scores.append(s)

    return scores


# ╔══════════════════════════════════════════════════════════════╗
# ║                 SCORE FROM BATCH OUTPUT FILE                 ║
# ╚══════════════════════════════════════════════════════════════╝

def score_from_batch_file(input_path: str, output_path: str) -> None:
    """
    Read an existing batch output JSON, score all results, write augmented output.

    The input file should have a "results" list where each entry has:
      - news_id, title
      - parsed_json
      - rag_result

    Scores are written back as a new "scores" field per result.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        batch = json.load(f)

    results = batch.get("results", [])
    if not results:
        print("No results found in input file.")
        return

    print(f"Scoring {len(results)} interpretations from {input_path}...")
    print(f"{'=' * 60}")

    for i, entry in enumerate(results):
        news_obj = {
            "news_id": entry.get("news_id", f"item-{i}"),
            "title": entry.get("title", entry.get("news_title", "Unknown")),
            "excerpt": entry.get("excerpt", entry.get("full_text", "")),
        }
        rag_result = entry.get("rag_result", {})
        interpreter_result = {
            "parsed_json": entry.get("parsed_json"),
            "raw_llm_output": entry.get("raw_llm_output", ""),
            "schema_pass": entry.get("schema_pass", True),
        }

        title = news_obj["title"][:60]
        print(f"  [{i+1}/{len(results)}] {title}...", end=" ")

        s = score_interpretation(news_obj, rag_result, interpreter_result)
        overall = s.get("overall_score", -1)
        print(f"{overall:.1f}/10" if overall >= 0 else "❌")

        entry["scores"] = s

    # ── Write back ──
    batch["scoring_metadata"] = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "scorer_model": "gpt-4o-mini",
        "scoring_weights": SCORING_WEIGHTS,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    valid_scores = [r["scores"]["overall_score"] for r in results
                    if r.get("scores", {}).get("overall_score", -1) >= 0]
    if valid_scores:
        avg = sum(valid_scores) / len(valid_scores)
        print(f"\n{'=' * 60}")
        print(f"  ✅ Scored {len(valid_scores)}/{len(results)} interpretations")
        print(f"  Average overall score: {avg:.1f}/10")
        print(f"  Output: {output_path}")
        print(f"{'=' * 60}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                        CLI                                   ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatic Interpreter Quality Scorer (LLM-as-Judge)"
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to batch output JSON to score (e.g. output/test_new_input.json)"
    )
    parser.add_argument(
        "--output", default="output/scored_output.json",
        help="Output path for scored results (default: output/scored_output.json)"
    )
    args = parser.parse_args()

    if args.input:
        if not os.path.exists(args.input):
            print(f"Input file not found: {args.input}")
            sys.exit(1)
        score_from_batch_file(args.input, args.output)
    else:
        print("Usage: python scorer.py --input <batch_output.json> [--output <out.json>]")
        print("Example: python scorer.py --input output/test_new_input.json --output output/scored.json")
        sys.exit(0)
