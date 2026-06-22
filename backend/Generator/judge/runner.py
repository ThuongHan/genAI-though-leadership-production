"""
Shared logic for the LLM-as-a-judge evaluation panel.

Each judge script calls run_judge() with its LLM and a path to a JSON file
of generated posts.  For each post the runner:
  1. Embeds the post text.
  2. Retrieves the K most similar historical KickstartAI posts (cosine similarity).
  3. Fills the thesis-eval.md template with those references and the post.
  4. Calls the LLM and parses the returned JSON evaluation.
  5. Writes judge/results/<judge_name>.json.
"""

import json
import re
from pathlib import Path

from Generator.utils.embedder import Embedder
from Generator.utils.few_shot import FewShotPost

TEMPLATE_PATH = "Generator/config/eval-prompt.md"
RESULTS_DIR   = Path("judge/results")
DIMENSIONS    = [
    "tone_of_voice",
    "language_and_style",
    "coherence_readability",
    "discourse_structure",
    "specificity",
    "historical_similarity",
]


def load_posts(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["posts"]


def _build_prompt(post_text: str, historical_posts: list[dict]) -> str:
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()

    historical_block = "\n\n".join(
        f"REFERENCE {i + 1}:\n{p['text']}"
        for i, p in enumerate(historical_posts)
    )

    return (
        template
        .replace("{historical_posts}", historical_block)
        .replace("{generated_post}", post_text)
    )


def _extract_json(text: str) -> dict:
    """Parse the JSON object from an LLM response, tolerating markdown wrappers."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # Find the outermost {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response:\n{text[:300]}")
    return json.loads(match.group())


def _print_scores(post_idx: int, angle: str, evaluation: dict) -> None:
    print(f"\n  Post {post_idx} — {angle[:60]}")
    for dim in evaluation.get("dimensions", []):
        bar = "█" * dim["score"] + "░" * (5 - dim["score"])
        print(f"    {dim['name']:<25} {bar}  {dim['score']}/5")


def run_judge(
    judge_name: str,
    llm,
    posts_path: str,
    k_historical: int = 1,
) -> Path:
    """Evaluate all posts in posts_path and write results to judge/results/<judge_name>.json."""

    posts    = load_posts(posts_path)
    embedder = Embedder()
    few_shot = FewShotPost()

    print(f"\nJudge: {judge_name}  |  posts: {len(posts)}  |  k_historical: {k_historical}\n")

    all_results: list[dict] = []

    for post in posts:
        print(f"  [{post['post_idx']}/{len(posts)}] evaluating …", end=" ", flush=True)

        # 1. Embed the post text
        embedding = embedder.embed_text(post["content"])

        # 2. Retrieve closest historical posts as style reference
        references = few_shot.get_similar_posts(embedding, top_k=k_historical)

        # 3. Build prompt from template
        prompt = _build_prompt(post["content"], references)

        # 4. LLM call with up to 3 retries on parse failure
        evaluation: dict = {}
        for attempt in range(3):
            response = llm.invoke(prompt)
            assert isinstance(response.content, str)
            try:
                evaluation = _extract_json(response.content)
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                print(f"parse failed (attempt {attempt + 1}/3) — retrying…", end=" ", flush=True)

        print("done")
        _print_scores(post["post_idx"], post["angle"], evaluation)

        all_results.append({
            "post_idx":   post["post_idx"],
            "angle":      post["angle"],
            "content":    post["content"],
            "evaluation": evaluation,
        })

    # 5. Write output
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{judge_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 6. Print summary table
    print(f"\n{'─'*65}")
    print(f"  SUMMARY — {judge_name}")
    print(f"{'─'*65}")
    header = f"  {'Post':<6}" + "".join(f"{d[:10]:<12}" for d in DIMENSIONS) + "  avg"
    print(header)
    for r in all_results:
        dims = {d["name"]: d["score"] for d in r["evaluation"].get("dimensions", [])}
        scores = [dims.get(d, 0) for d in DIMENSIONS]
        avg = sum(scores) / len(scores) if scores else 0
        row = f"  {r['post_idx']:<6}" + "".join(f"{s:<12}" for s in scores) + f"  {avg:.1f}"
        print(row)
    print(f"{'─'*65}")
    print(f"\nSaved → {output_path}\n")

    return output_path
