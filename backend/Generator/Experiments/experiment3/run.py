"""
Experiment 3 — FS-Post regeneration comparison.

Reads the 15 FS-Post posts from data/UvA Expert Voice - Output annotation.xlsx (rows 45-59) and runs each
through the dual-judge refinement loop from regeneration/refine.py:

  1. Evaluate current post with both judges (Opus + GPT-5.5).
  2. If every dimension scores >= 4 from every judge -> stop early.
  3. Otherwise collect feedback from failing dimensions and regenerate.
  4. Repeat up to MAX_ITER times; keep the version with highest avg score.

Output -> backend/Generator/Experiment/experiment3/results/regeneration_comparison.xlsx
  Columns: Topic | FS-Post (Original) | FS-Post (Regenerated)

After that the company will rate the post-reformulated post. The results 
of that are in backend/Generator/Experiments/experiment3/results/UvA Expert Voice _ Data Annotation (Mateusz) - Sheet1.csv
"""

from pathlib import Path

import pandas as pd

from backend.Generator.utils.embedder import Embedder
from backend.Generator.utils.few_shot import FewShotPost
from backend.Generator.utils.llm.claude import ClaudeLLM
from backend.Generator.utils.llm.gpt import GPTLLM
from backend.Generator.regeneration.refine import (
    MAX_ITER,
    GENERATOR_MODEL,
    evaluate_post,
    all_pass,
    build_feedback,
    build_regen_prompt,
)

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_FILE   = "backend/Generator/Experiments/data/UvA Expert Voice - Output annotation.xlsx"
RESULTS_DIR   = Path("backend/Generator/Experiments/experiment3/results")
FS_POST_START = 45
FS_POST_END   = 60  # exclusive


# ── Core refinement loop ──────────────────────────────────────────────────────

def refine_post(
    post_text: str,
    judges: dict,
    generator,
    embedder: Embedder,
    few_shot: FewShotPost,
) -> str:
    """Run the dual-judge refinement loop; return the best-scoring post."""
    current_post = post_text
    history: list[tuple[int, str, float]] = []

    for iteration in range(1, MAX_ITER + 1):
        print(f"    iter {iteration}/{MAX_ITER} — evaluating …", end=" ", flush=True)
        evaluations = evaluate_post(current_post, judges, embedder, few_shot)

        all_scores = [
            d["score"]
            for ev in evaluations.values()
            for d in ev.get("dimensions", [])
        ]
        avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        history.append((iteration, current_post, avg))
        print(f"avg {avg:.2f}")

        if all_pass(evaluations):
            print("    ✓ all dimensions ≥ 4 — stopping early")
            break

        if iteration == MAX_ITER:
            print("    max iterations reached")
            break

        print(f"    iter {iteration}/{MAX_ITER} — regenerating …", end=" ", flush=True)
        feedback     = build_feedback(evaluations)
        regen_prompt = build_regen_prompt(current_post, feedback, few_shot, embedder)
        response     = generator.invoke(regen_prompt)
        assert isinstance(response.content, str)
        current_post = response.content.strip()
        print("done")

    _, best_post, best_avg = max(history, key=lambda x: x[2])
    print(f"    → best avg score: {best_avg:.2f}")
    return best_post


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(SAMPLE_FILE, header=0)
    fs_post_df = df.iloc[FS_POST_START:FS_POST_END].reset_index(drop=True)
    print(f"Loaded {len(fs_post_df)} FS-Post rows from {SAMPLE_FILE}\n")

    embedder  = Embedder()
    few_shot  = FewShotPost()
    generator = ClaudeLLM(GENERATOR_MODEL)
    judges = {
        "opus": ClaudeLLM("claude-opus-4-8"),
        "gpt5.5": GPTLLM("gpt-5.5"),
    }

    rows = []
    for i, row in fs_post_df.iterrows():
        topic    = row["Topic"]
        original = row["Posts"]

        print(f"[{i+1}/15] {topic[:70]}")
        regenerated = refine_post(str(original), judges, generator, embedder, few_shot)
        print()

        rows.append({
            "Topic":                 topic,
            "FS-Post (Original)":    original,
            "FS-Post (Regenerated)": regenerated,
        })

    out_path = RESULTS_DIR / "regeneration_comparison.xlsx"
    pd.DataFrame(rows).to_excel(out_path, index=False)
    print(f"Saved {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()

# python3 -m backend.Generator.Experiments.experiment3.run