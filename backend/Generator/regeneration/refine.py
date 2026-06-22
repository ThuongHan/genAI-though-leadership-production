"""
Dual-judge refinement loop.

Each iteration:
  1. Both judges (Sonnet + GPT-5) evaluate the current post.
  2. Print the post and the per-dimension score table.
  3. If every dimension scores ≥ 4 from every judge → stop.
  4. Otherwise collect justifications from all failing dimensions and
     regenerate the post using thesis-prompt-improve-v2.md + few-shot (k=1)
     + thesis-regeneration.md with the feedback filled in.

After at most MAX_ITER iterations, pick the version with the highest
average score across all judges and print it as the final result.

Run from repo root:
  python3 regeneration/refine.py
"""

import json

from Generator.utils.embedder import Embedder
from Generator.utils.few_shot import FewShotPost
from Generator.utils.llm.claude import ClaudeLLM
from Generator.utils.llm.gpt import GPTLLM
from Generator.judge.runner import _build_prompt as build_eval_prompt, _extract_json as extract_json

# --- Config ---
POSTS_PATH     = "Generator/example_generated/generated_posts.json"
GEN_CONFIG     = "Generator/config/post-reformulated-prompt.md"
REGEN_TEMPLATE = "Generator/regeneration/thesis-regeneration.md"
MAX_ITER       = 3
K_FEW_SHOT     = 1   # few-shot examples for regeneration
K_HISTORICAL   = 1   # historical refs for judge evaluation
PASS_THRESHOLD = 4   # every dimension must reach this to stop early

# gpt models: ["gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o"]
# cluade models: ["claude-haiku-4-6", "claude-sonnet-4-6", "claude-opus-4-8"]

GENERATOR_MODEL = "claude-sonnet-4-6"

DIMENSIONS = [
    "tone_of_voice",
    "language_and_style",
    "coherence_readability",
    "discourse_structure",
    "specificity",
    "historical_similarity",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_first_post(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["posts"][0]


def evaluate_post(
    post_text: str,
    judges: dict,
    embedder: Embedder,
    few_shot: FewShotPost,
) -> dict[str, dict]:
    """Return {judge_name: evaluation_dict} for every judge."""
    embedding  = embedder.embed_text(post_text)
    references = few_shot.get_similar_posts(embedding, top_k=K_HISTORICAL)
    prompt     = build_eval_prompt(post_text, references)

    results: dict[str, dict] = {}
    for name, llm in judges.items():
        print(f"    [{name}] evaluating …", end=" ", flush=True)
        for attempt in range(3):
            response = llm.invoke(prompt)
            assert isinstance(response.content, str)
            try:
                results[name] = extract_json(response.content)
                print("done")
                break
            except Exception:
                if attempt == 2:
                    raise
                print(f"parse error (attempt {attempt+1}/3) — retrying…", end=" ", flush=True)
    return results


def dim_scores(evaluation: dict) -> dict[str, int]:
    return {d["name"]: d["score"] for d in evaluation.get("dimensions", [])}


def all_pass(evaluations: dict[str, dict]) -> bool:
    for ev in evaluations.values():
        for d in ev.get("dimensions", []):
            if d["score"] < PASS_THRESHOLD:
                return False
    return True


def build_feedback(evaluations: dict[str, dict]) -> str:
    """Collect justifications from dimensions that failed for any judge."""
    failing: dict[str, list[str]] = {}
    for judge_name, ev in evaluations.items():
        for d in ev.get("dimensions", []):
            if d["score"] < PASS_THRESHOLD:
                failing.setdefault(d["name"], []).append(
                    f"- [{judge_name}, {d['score']}/5] {d['justification']}"
                )

    if not failing:
        return "(all dimensions pass)"

    parts = []
    for dim_name, lines in failing.items():
        parts.append(f"### {dim_name}\n" + "\n".join(lines))
    return "\n\n".join(parts)


def build_regen_prompt(
    current_post: str,
    feedback: str,
    few_shot: FewShotPost,
    embedder: Embedder,
) -> str:
    with open(GEN_CONFIG, encoding="utf-8") as f:
        system_prompt = f.read()
    with open(REGEN_TEMPLATE, encoding="utf-8") as f:
        regen_template = f.read()

    embedding  = embedder.embed_text(current_post)
    references = few_shot.get_similar_posts(embedding, top_k=K_FEW_SHOT)
    few_shot_block = "\n\n".join(
        f"REFERENCE {i+1}:\n{r['text']}" for i, r in enumerate(references)
    )

    regen_section = (
        regen_template
        .replace("{current_post}", current_post)
        .replace("{feedback}", feedback)
    )

    return (
        f"{system_prompt}\n\n---\n\n"
        f"## STYLE REFERENCES\n\n{few_shot_block}\n\n---\n\n"
        f"{regen_section}"
    )


# ── Display ───────────────────────────────────────────────────────────────────

def print_iteration(
    iteration: int,
    post_text: str,
    evaluations: dict[str, dict],
    overall_avg: float,
) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  ITERATION {iteration}  |  avg score: {overall_avg:.2f}")
    print(sep)
    print(f"\n{post_text}\n")

    col = 16
    # Header
    print(f"  {'Dimension':<28}", end="")
    for name in evaluations:
        print(f"{name[:col]:<{col}}", end="")
    print("  combined avg")
    print(f"  {'-'*28}", end="")
    for _ in evaluations:
        print(f"{'-'*col}", end="")
    print(f"  {'-'*12}")

    # Rows
    for dim in DIMENSIONS:
        print(f"  {dim:<28}", end="")
        row_scores = []
        for ev in evaluations.values():
            s = dim_scores(ev).get(dim, 0)
            row_scores.append(s)
            mark = "✓" if s >= PASS_THRESHOLD else "✗"
            cell = f"{mark} {s}/5"
            print(f"{cell:<{col}}", end="")
        avg = sum(row_scores) / len(row_scores) if row_scores else 0
        print(f"  {avg:.1f}")

    # Per-judge averages
    print(f"  {'AVERAGE':<28}", end="")
    judge_avgs = []
    for ev in evaluations.values():
        scores = list(dim_scores(ev).values())
        a = sum(scores) / len(scores) if scores else 0
        judge_avgs.append(a)
        print(f"{a:.2f}{' '*(col-4)}", end="")
    overall = sum(judge_avgs) / len(judge_avgs) if judge_avgs else 0
    print(f"  {overall:.2f}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    post_data    = load_first_post(POSTS_PATH)
    current_post = post_data["content"]

    print(f"\nStarting post — idx: {post_data['post_idx']}  angle: {post_data['angle'][:60]}")

    embedder  = Embedder()
    few_shot  = FewShotPost()
    generator = ClaudeLLM(GENERATOR_MODEL) # GENERATOR
    judges    = {
        "opus": ClaudeLLM("claude-opus-4-8"),
        "gpt5.5":   GPTLLM("gpt-5.5"),
    }

    # iteration history: (iteration_no, post_text, evaluations, avg_score)
    history: list[tuple[int, str, dict, float]] = []

    for iteration in range(1, MAX_ITER + 1):
        print(f"\n── Iteration {iteration}/{MAX_ITER} ── evaluating with both judges …")

        evaluations = evaluate_post(current_post, judges, embedder, few_shot)

        all_scores = [
            d["score"]
            for ev in evaluations.values()
            for d in ev.get("dimensions", [])
        ]
        avg = sum(all_scores) / len(all_scores) if all_scores else 0

        history.append((iteration, current_post, evaluations, avg))
        print_iteration(iteration, current_post, evaluations, avg)

        if all_pass(evaluations):
            print("\n✓ All dimensions ≥ 4 from all judges — stopping early.")
            break

        if iteration == MAX_ITER:
            print("\nMax iterations reached.")
            break

        # Regenerate
        print(f"\n── Iteration {iteration}/{MAX_ITER} ── regenerating …", end=" ", flush=True)
        feedback    = build_feedback(evaluations)
        regen_prompt = build_regen_prompt(current_post, feedback, few_shot, embedder)
        response    = generator.invoke(regen_prompt)
        assert isinstance(response.content, str)
        current_post = response.content.strip()
        print("done")

    # Pick best version
    best_iter, best_post, _, best_avg = max(history, key=lambda x: x[3])

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  FINAL BEST  |  iteration {best_iter}  |  avg score {best_avg:.2f}")
    print(sep)
    print(f"\n{best_post}\n")


if __name__ == "__main__":
    main()

# python3 -m Generator.regeneration.refine
