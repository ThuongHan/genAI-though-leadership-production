"""
2x2 experiment — post generation across four conditions.

Conditions:
  ZS-Pre:  zeroshot-prompt.md          + k=0 (no few-shot)
  FS-Pre:  zeroshot-prompt.md          + k=1 (1 few-shot example)
  ZS-Post: post-reformulated-prompt.md + k=0
  FS-Post: post-reformulated-prompt.md + k=1

For each condition 15 topics from 40_blog_posts.json are generated,
producing 60 posts total (15 x 4 conditions).
Generator model: claude-sonnet-4-6.

Output → backend/Generator/Experiments/data/sample_61.xlsx
  Columns: Topic | Posts | Condition
  Row order: ZS-Pre (0-14), FS-Pre (1-29), ZS-Post (30-44), FS-Post (45-59)
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add backend/ to sys.path so internal Generator imports (from Generator.xxx) resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.Generator.post_generator import PostGenerator

# ── Paths (absolute, independent of CWD) ─────────────────────────────────────

_EXP     = Path(__file__).resolve().parent        # experiment1/
_GEN     = _EXP.parents[1]                        # Generator/
_EXPDATA = _EXP.parent / "data"                   # Experiments/data/

DATA_FILE  = str(_GEN / "data" / "Interpreter_output" / "40_blog_posts.json")
OUTPUT     = _EXPDATA / "sample_61.xlsx"

# ── Config ────────────────────────────────────────────────────────────────────

N_TOPICS  = 15
GEN_MODEL = "claude-sonnet-4-6"

CONDITIONS = {
    "ZS-Pre":  {"config": str(_GEN / "config" / "zeroshot-prompt.md"),          "few_shot": False, "k": 0},
    "FS-Pre":  {"config": str(_GEN / "config" / "zeroshot-prompt.md"),          "few_shot": True,  "k": 1},
    "ZS-Post": {"config": str(_GEN / "config" / "post-reformulated-prompt.md"), "few_shot": False, "k": 0},
    "FS-Post": {"config": str(_GEN / "config" / "post-reformulated-prompt.md"), "few_shot": True,  "k": 1},
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_topics(n: int) -> list[dict]:
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data["results"][:n]


def map_fields(item: dict) -> dict:
    interp = item["interpretation"]
    meta   = item["metadata"]
    return {
        "what_happened":   interp["What happened"],
        "why_relevance":   interp["Why does it matter (globally and NL)"],
        "why_kickstartai": interp["Why does it matter for KickstartAI"],
        "stance":          interp["Key stance / opinion"],
        "arguments":       interp["Supporting arguments"],
        "source":          meta["source_article"]["url"],
    }


# ── Generation loop ───────────────────────────────────────────────────────────

def run_generation() -> list[dict]:
    topics = load_topics(N_TOPICS)
    rows: list[dict] = []

    for cond_name, cfg in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name}  |  few-shot: {cfg['few_shot']}  |  k={cfg['k']}")

        generator = PostGenerator(model=GEN_MODEL, config_path=cfg["config"])

        for i, topic in enumerate(topics):
            title      = topic["metadata"]["source_article"]["title"]
            event_input = map_fields(topic)

            print(f"  [{i+1:>2}/{N_TOPICS}] {title[:70]} … ", end="", flush=True)
            result = generator.generate(
                interpreter_output=event_input,
                k_posts=cfg["k"],
                use_few_shot=cfg["few_shot"],
                save=False,
            )
            post_text = result["posts"][0].content
            print("done")

            rows.append({
                "Topic":     title,
                "Posts":     post_text,
                "Condition": cond_name,
            })

    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _EXPDATA.mkdir(parents=True, exist_ok=True)

    rows = run_generation()

    df = pd.DataFrame(rows)
    df.to_excel(OUTPUT, index=False)

    print(f"\nGenerated {len(rows)} posts across {len(CONDITIONS)} conditions.")
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()

# python3 -m backend.Generator.Experiments.experiment1.generate_posts