"""
Experiment 2: LLM-judge vs. human inter-rater agreement.

J1 = Claude Opus 4.8
J2 = GPT-5.5

The annotated posts are loaded from the annotation Excel file.
Both judges evaluate every post. Their scores are compared against
the human annotations to produce:

  Table 2 — Exact and adjacent (-+1) agreement proportions per dimension
             (rows = 6 dimensions, cols = exact J1, exact J2, adj J1, adj J2)

Output in backend/Generator/Experiments/experiment2/results/:
  raw_evaluations.json    — full judge JSON for all posts
  table2_agreement.xlsx
"""

import json
from pathlib import Path

import pandas as pd

from backend.Generator.utils.embedder import Embedder
from backend.Generator.utils.few_shot import FewShotPost
from backend.Generator.utils.llm.claude import ClaudeLLM
from backend.Generator.utils.llm.gpt import GPTLLM
from backend.Generator.judge.runner import _build_prompt as build_eval_prompt, _extract_json as extract_json

# ── Config ────────────────────────────────────────────────────────────────────

ANNOTATION_FILE = "backend/Generator/Experiments/data/UvA Expert Voice - Output annotation.xlsx"
RESULTS_DIR     = Path("backend/Generator/Experiment/experiment2/results")
K_EVAL_REFS     = 1

DIMENSION_MAP = {
    "tone_of_voice":             "tone_of_voice",
    "language_and_style":        "language_and_style",
    "coherence_and_readability": "coherence_readability",
    "discourse_structure":       "discourse_structure",
    "specificity":               "specificity",
    "historical_similarity":     "historical_similarity",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_annotation_data() -> pd.DataFrame:
    df = pd.read_excel(ANNOTATION_FILE, header=0)
    df.columns = [c.strip() for c in df.columns]
    return df


def human_scores(df: pd.DataFrame, excel_col: str) -> list[int]:
    return df[excel_col].fillna(0).astype(int).tolist()


# ── Evaluation ────────────────────────────────────────────────────────────────

def call_judge(text: str, llm, embedder: Embedder, few_shot: FewShotPost) -> dict:
    embedding  = embedder.embed_text(text)
    references = few_shot.get_similar_posts(embedding, top_k=K_EVAL_REFS)
    prompt     = build_eval_prompt(text, references)
    for attempt in range(3):
        resp = llm.invoke(prompt)
        try:
            return extract_json(resp.content)
        except Exception:
            if attempt == 2:
                raise
    return {}


def run_evaluations(df: pd.DataFrame) -> list[dict]:
    embedder = Embedder()
    few_shot = FewShotPost()
    judges = {
        "J1": ClaudeLLM("claude-opus-4-8"),
        "J2": GPTLLM("gpt-5.5"),
    }

    raw: list[dict] = []
    n = len(df)

    for i, row in df.iterrows():
        post_text = str(row["Posts"])
        print(f"  [{i+1:>2}/{n}] {str(row.get('Topic', ''))[:55]}")

        evaluations: dict[str, dict] = {}
        for judge_name, llm in judges.items():
            print(f"    [{judge_name}] …", end=" ", flush=True)
            evaluations[judge_name] = call_judge(post_text, llm, embedder, few_shot)
            print("done")

        raw.append({
            "post_idx":    i,
            "post_text":   post_text,
            "evaluations": evaluations,
        })

    return raw


# ── Score extraction ──────────────────────────────────────────────────────────

def llm_dim_score(evaluation: dict, dim_key: str) -> int | None:
    for d in evaluation.get("dimensions", []):
        if d["name"] == dim_key:
            return d.get("score")
    return None


# ── Statistics ────────────────────────────────────────────────────────────────

def exact_agreement(human: list[int], judge: list[int]) -> float:
    pairs = [(h, j) for h, j in zip(human, judge) if h is not None and j is not None]
    if not pairs:
        return float("nan")
    return sum(h == j for h, j in pairs) / len(pairs)


def adjacent_agreement(human: list[int], judge: list[int]) -> float:
    pairs = [(h, j) for h, j in zip(human, judge) if h is not None and j is not None]
    if not pairs:
        return float("nan")
    return sum(abs(h - j) <= 1 for h, j in pairs) / len(pairs)


# ── Table builder ─────────────────────────────────────────────────────────────

def build_table2(df: pd.DataFrame, raw: list[dict]) -> pd.DataFrame:
    """Exact and adjacent (±1) agreement per dimension for J1 and J2."""
    rows = []
    for excel_col, dim_key in DIMENSION_MAP.items():
        human     = human_scores(df, excel_col)
        j1_scores = [llm_dim_score(e["evaluations"].get("J1", {}), dim_key) for e in raw]
        j2_scores = [llm_dim_score(e["evaluations"].get("J2", {}), dim_key) for e in raw]
        rows.append({
            "Dimension":        dim_key,
            "Exact J1":         round(exact_agreement(human, j1_scores), 3),
            "Exact J2":         round(exact_agreement(human, j2_scores), 3),
            "Adjacent (±1) J1": round(adjacent_agreement(human, j1_scores), 3),
            "Adjacent (±1) J2": round(adjacent_agreement(human, j2_scores), 3),
        })
    return pd.DataFrame(rows).set_index("Dimension")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "raw_evaluations.json"

    df = load_annotation_data()
    print(f"Loaded {len(df)} annotated posts from {ANNOTATION_FILE}\n")

    if raw_path.exists():
        print("Found existing evaluations — skipping API calls.\n")
        with open(raw_path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        print("Running J1 (Claude Opus) and J2 (GPT-5.5) on all posts …\n")
        raw = run_evaluations(df)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
        print(f"\nRaw evaluations saved → {raw_path}\n")

    df.columns = [c.strip() for c in df.columns]

    table2 = build_table2(df, raw)

    print("── Table 2: Exact and Adjacent Agreement ───────────────────────")
    print(table2.to_string())

    t2 = RESULTS_DIR / "table2_agreement.xlsx"
    table2.to_excel(t2)
    print(f"\nTable 2 → {t2}")


if __name__ == "__main__":
    main()

# python3 -m backend.Generator.Experiments.experiment2.run