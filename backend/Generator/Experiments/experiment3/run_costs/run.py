"""
Experiment 4 — Computational cost: generation vs. regeneration.

For each of the first 15 interpretations from 40_blog_posts.json:
The 40_blog_posts.json is the Interperatation produced by Interpreter
for 40 topics. The backend/Generator/Experiments/data/UvA Expert Voice - Output annotation.xlsx
was also generated from it (each config took the first 15 topics).

  Phase 1 — Generation (FS-Post condition):
    - 1 embedding call   (few-shot retrieval)
    - 1 LLM call         (claude-sonnet-4-6 -> 3 posts JSON; pick one)

  Phase 2 — Regeneration (dual-judge refinement loop, MAX_ITER=3):
    Per round:
      - 1 embedding call   (build judge eval prompt)
      - 2 LLM calls        (GPT-5.5 + Claude Opus 4-8 judges)
    If round not final and not converged:
      - 1 embedding call   (build regen few-shot block)
      - 1 LLM call         (claude-sonnet-4-6 generator)
    Loop ends when all dimensions >= 4 or MAX_ITER reached.
    Best-scoring version is kept.

Tracked metrics per phase per post:
  - Wall-clock time (s)
  - API calls (LLM + embedding separately)
  - Input / output tokens (per model role: generator / judge-gpt5_5 / judge-opus)
  - Estimated USD cost   (from PRICING table below — approximate public rates)
  - Refinement rounds    (regeneration phase only)
  - Early-stop flag      (regeneration phase only)

Outputs -> backend/Experiments/expriment3/run_costs/result
  raw_costs.json          — full per-post data
  cost_comparison.xlsx    — 15-row detail table + summary sheet
"""

import json
import time
from pathlib import Path

import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser

from backend.Generator.prompt_builder import PromptBuilder
from backend.Generator.schemas.generator_schema import GeneratedPosts
from backend.Generator.utils.embedder import Embedder
from backend.Generator.utils.few_shot import FewShotPost
from backend.Generator.utils.llm.claude import ClaudeLLM
from backend.Generator.utils.llm.gpt import GPTLLM
from backend.Generator.judge.runner import _build_prompt as build_eval_prompt, _extract_json as extract_json
from backend.Generator.regeneration.refine import (
    MAX_ITER,
    K_FEW_SHOT,
    K_HISTORICAL,
    PASS_THRESHOLD,
    GEN_CONFIG,
    REGEN_TEMPLATE,
    build_feedback,
)

# ── Config ────────────────────────────────────────────────────────────────────

BLOG_FILE    = "backend/Generator/data/Interpreter_output/40_blog_posts.json"
N_TOPICS     = 15
RESULTS_DIR  = Path("backend/Experiments/expriment3/run_costs/result")
GEN_MODEL   = "claude-sonnet-4-6"
JUDGE_GPT   = "gpt-5.5"
JUDGE_OPUS  = "claude-opus-4-8"

# Approximate public API pricing (USD per 1M tokens).
# Update these if proxy rates differ.
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6":        {"input": 3.00,  "output": 15.00},
    "gpt-5.5":                  {"input": 5.00, "output": 30.00},
    "claude-opus-4-8":          {"input": 5.00, "output": 25.00},
    "text-embedding-3-large":   {"input": 0.13,  "output": 0.00},
}


# ── Cost helpers ──────────────────────────────────────────────────────────────

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def extract_usage(response) -> tuple[int, int]:
    """Pull (input_tokens, output_tokens) from a LangChain AIMessage."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return 0, 0
    if isinstance(meta, dict):
        return meta.get("input_tokens", 0), meta.get("output_tokens", 0)
    return getattr(meta, "input_tokens", 0), getattr(meta, "output_tokens", 0)


# ── Instrumented call wrappers ────────────────────────────────────────────────

def timed_llm_call(llm, model_name: str, prompt: str) -> tuple[object, int, int, float, float]:
    """Return (response, in_tok, out_tok, cost_usd, elapsed_s)."""
    t0 = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - t0
    in_tok, out_tok = extract_usage(response)
    cost = estimate_cost(model_name, in_tok, out_tok)
    return response, in_tok, out_tok, cost, elapsed


def timed_embed(embedder: Embedder, text: str) -> tuple[list, float]:
    """Return (embedding, elapsed_s)."""
    t0 = time.perf_counter()
    embedding = embedder.embed_text(text)
    elapsed = time.perf_counter() - t0
    return embedding, elapsed


# ── Data loading ──────────────────────────────────────────────────────────────

def load_topics(n: int) -> list[dict]:
    with open(BLOG_FILE, encoding="utf-8") as f:
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


# ── Phase 1: Generation ───────────────────────────────────────────────────────

def run_generation(
    event_input: dict,
    topic_idx: int,
    generator_llm,
    embedder: Embedder,
    few_shot: FewShotPost,
    prompt_builder: PromptBuilder,
    parser: PydanticOutputParser,
) -> tuple[str, dict]:
    """Generate one post (FS-Post condition). Return (post_text, metrics)."""
    phase_start = time.perf_counter()

    # 1. Embed event text for few-shot retrieval
    event_text = (
        f"What happened: {event_input['what_happened']} "
        f"Why relevance: {event_input['why_relevance']} "
        f"Stance: {event_input['stance']} "
        f"Arguments: {', '.join(event_input['arguments'])}"
    )
    embedding, embed_time = timed_embed(embedder, event_text)
    few_shot_posts = few_shot.get_similar_posts(embedding, top_k=K_FEW_SHOT) or []

    # 2. Build prompt and call LLM
    prompt = prompt_builder.build(
        event=event_input,
        format_instructions=parser.get_format_instructions(),
        use_few_shot=True,
        few_shot_posts=few_shot_posts,
    )

    for attempt in range(5):
        response, in_tok, out_tok, cost, llm_time = timed_llm_call(
            generator_llm, GEN_MODEL, prompt
        )
        assert isinstance(response.content, str)
        try:
            parsed = parser.parse(response.content)
            break
        except Exception as e:
            if attempt == 4:
                raise
            print(f"      parse failed (attempt {attempt+1}/5): {e!s:.80} — retrying…")

    # Round-robin post selection (matches experiment_2 convention)
    post_text = parsed.posts[topic_idx % 3].content

    metrics = {
        "gen_time_s":         round(time.perf_counter() - phase_start, 2),
        "gen_embed_calls":    1,
        "gen_embed_time_s":   round(embed_time, 2),
        "gen_api_calls":      1,
        "gen_llm_time_s":     round(llm_time, 2),
        "gen_input_tokens":   in_tok,
        "gen_output_tokens":  out_tok,
        "gen_cost_usd":       round(cost, 6),
    }
    return post_text, metrics


# ── Phase 2: Regeneration ─────────────────────────────────────────────────────

def run_refinement(
    post_text: str,
    judges: dict,          # {judge_name: (llm, model_name)}
    generator_llm,
    embedder: Embedder,
    few_shot: FewShotPost,
) -> tuple[str, dict]:
    """Run dual-judge refinement loop. Return (best_post, metrics)."""
    phase_start = time.perf_counter()

    # Accumulated totals
    total_embed_calls  = 0
    total_embed_time_s = 0.0
    total_judge_calls  = 0
    total_judge_time_s = 0.0
    judge_tokens: dict[str, dict[str, int]] = {
        name: {"input": 0, "output": 0} for name in judges
    }
    total_gen_calls    = 0
    total_gen_time_s   = 0.0
    gen_in_tok_total   = 0
    gen_out_tok_total  = 0
    total_cost_usd     = 0.0
    rounds             = 0
    stopped_early      = False

    # History for picking best version
    history: list[tuple[int, str, float]] = []
    current_post = post_text

    with open(GEN_CONFIG, encoding="utf-8") as f:
        gen_config_text = f.read()
    with open(REGEN_TEMPLATE, encoding="utf-8") as f:
        regen_template = f.read()

    for iteration in range(1, MAX_ITER + 1):
        rounds = iteration
        print(f"    iter {iteration}/{MAX_ITER} — evaluating …", end=" ", flush=True)

        # Embed for judge eval
        embedding, embed_time = timed_embed(embedder, current_post)
        total_embed_calls  += 1
        total_embed_time_s += embed_time
        total_cost_usd += estimate_cost(
            "text-embedding-3-large", len(current_post.split()), 0
        )

        references  = few_shot.get_similar_posts(embedding, top_k=K_HISTORICAL)
        eval_prompt = build_eval_prompt(current_post, references)

        evaluations: dict[str, dict] = {}
        for judge_name, (judge_llm, judge_model) in judges.items():
            resp, in_tok, out_tok, cost, elapsed = timed_llm_call(
                judge_llm, judge_model, eval_prompt
            )
            assert isinstance(resp.content, str)
            for attempt in range(3):
                try:
                    evaluations[judge_name] = extract_json(resp.content)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    resp, in_tok, out_tok, cost, elapsed = timed_llm_call(
                        judge_llm, judge_model, eval_prompt
                    )
            judge_tokens[judge_name]["input"]  += in_tok
            judge_tokens[judge_name]["output"] += out_tok
            total_judge_calls  += 1
            total_judge_time_s += elapsed
            total_cost_usd     += cost

        # Compute avg score for this iteration
        all_scores = [
            d["score"]
            for ev in evaluations.values()
            for d in ev.get("dimensions", [])
        ]
        avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        history.append((iteration, current_post, avg))
        print(f"avg {avg:.2f}")

        # Check early stop
        def _all_pass(evals: dict) -> bool:
            return all(
                d["score"] >= PASS_THRESHOLD
                for ev in evals.values()
                for d in ev.get("dimensions", [])
            )

        if _all_pass(evaluations):
            print("    ✓ all dimensions ≥ 4 — stopping early")
            stopped_early = True
            break

        if iteration == MAX_ITER:
            print("    max iterations reached")
            break

        # Regenerate
        print(f"    iter {iteration}/{MAX_ITER} — regenerating …", end=" ", flush=True)
        feedback = build_feedback(evaluations)

        # Embed for regen few-shot
        regen_embedding, regen_embed_time = timed_embed(embedder, current_post)
        total_embed_calls  += 1
        total_embed_time_s += regen_embed_time
        regen_refs = few_shot.get_similar_posts(regen_embedding, top_k=K_FEW_SHOT)
        few_shot_block = "\n\n".join(
            f"REFERENCE {i+1}:\n{r['text']}" for i, r in enumerate(regen_refs)
        )
        regen_section = (
            regen_template
            .replace("{current_post}", current_post)
            .replace("{feedback}", feedback)
        )
        regen_prompt = (
            f"{gen_config_text}\n\n---\n\n"
            f"## STYLE REFERENCES\n\n{few_shot_block}\n\n---\n\n"
            f"{regen_section}"
        )

        resp, in_tok, out_tok, cost, elapsed = timed_llm_call(
            generator_llm, GEN_MODEL, regen_prompt
        )
        assert isinstance(resp.content, str)
        current_post     = resp.content.strip()
        gen_in_tok_total  += in_tok
        gen_out_tok_total += out_tok
        total_gen_calls   += 1
        total_gen_time_s  += elapsed
        total_cost_usd    += cost
        print("done")

    _, best_post, best_avg = max(history, key=lambda x: x[2])
    print(f"    → best avg: {best_avg:.2f}  rounds: {rounds}")

    metrics = {
        "regen_rounds":             rounds,
        "regen_early_stop":         stopped_early,
        "regen_time_s":             round(time.perf_counter() - phase_start, 2),
        "regen_embed_calls":        total_embed_calls,
        "regen_embed_time_s":       round(total_embed_time_s, 2),
        "regen_judge_calls":        total_judge_calls,
        "regen_judge_time_s":       round(total_judge_time_s, 2),
        "regen_gen_calls":          total_gen_calls,
        "regen_gen_time_s":         round(total_gen_time_s, 2),
        **{
            f"regen_{name}_input_tokens":  judge_tokens[name]["input"]
            for name in judges
        },
        **{
            f"regen_{name}_output_tokens": judge_tokens[name]["output"]
            for name in judges
        },
        "regen_gen_input_tokens":   gen_in_tok_total,
        "regen_gen_output_tokens":  gen_out_tok_total,
        "regen_cost_usd":           round(total_cost_usd, 6),
    }
    return best_post, metrics


# ── Excel builders ────────────────────────────────────────────────────────────

def build_detail_df(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        g = r["generation"]
        rg = r["regeneration"]
        total_cost = round(g["gen_cost_usd"] + rg["regen_cost_usd"], 6)
        total_time = round(g["gen_time_s"]   + rg["regen_time_s"],   2)
        total_api  = g["gen_api_calls"]       + rg["regen_judge_calls"] + rg["regen_gen_calls"]
        total_emb  = g["gen_embed_calls"]     + rg["regen_embed_calls"]

        rows.append({
            "Topic":                       r["topic"],
            # ── Generation ──────────────────────────────────────────────────
            "Gen Time (s)":                g["gen_time_s"],
            "Gen Embed Calls":             g["gen_embed_calls"],
            "Gen Embed Time (s)":          g["gen_embed_time_s"],
            "Gen API Calls":               g["gen_api_calls"],
            "Gen LLM Time (s)":            g["gen_llm_time_s"],
            "Gen Input Tokens":            g["gen_input_tokens"],
            "Gen Output Tokens":           g["gen_output_tokens"],
            "Gen Cost (USD)":              g["gen_cost_usd"],
            # ── Regeneration ─────────────────────────────────────────────────
            "Regen Rounds":                rg["regen_rounds"],
            "Regen Early Stop":            rg["regen_early_stop"],
            "Regen Time (s)":              rg["regen_time_s"],
            "Regen Embed Calls":           rg["regen_embed_calls"],
            "Regen Embed Time (s)":        rg["regen_embed_time_s"],
            "Regen Judge Calls":           rg["regen_judge_calls"],
            "Regen Judge Time (s)":        rg["regen_judge_time_s"],
            "Regen GPT5.5 Input Tokens":   rg.get("regen_gpt5_5_input_tokens", 0),
            "Regen GPT5.5 Output Tokens":  rg.get("regen_gpt5_5_output_tokens", 0),
            "Regen Opus Input Tokens":     rg.get("regen_opus_input_tokens", 0),
            "Regen Opus Output Tokens":    rg.get("regen_opus_output_tokens", 0),
            "Regen Gen Calls":             rg["regen_gen_calls"],
            "Regen Gen Time (s)":          rg["regen_gen_time_s"],
            "Regen Gen Input Tokens":      rg["regen_gen_input_tokens"],
            "Regen Gen Output Tokens":     rg["regen_gen_output_tokens"],
            "Regen Cost (USD)":            rg["regen_cost_usd"],
            # ── Totals ───────────────────────────────────────────────────────
            "Total Cost (USD)":            total_cost,
            "Total Time (s)":              total_time,
            "Total API Calls":             total_api,
            "Total Embed Calls":           total_emb,
        })
    return pd.DataFrame(rows)


def build_summary_df(detail_df: pd.DataFrame) -> pd.DataFrame:
    numeric = detail_df.select_dtypes(include="number").columns.tolist()
    totals  = detail_df[numeric].sum().rename("Total")
    means   = detail_df[numeric].mean().round(3).rename("Mean per post")
    return pd.DataFrame([totals, means])


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "raw_costs.json"

    topics = load_topics(N_TOPICS)
    print(f"Loaded {len(topics)} topics from {BLOG_FILE}\n")

    embedder      = Embedder()
    few_shot      = FewShotPost()
    generator_llm = ClaudeLLM(GEN_MODEL)
    prompt_builder = PromptBuilder(GEN_CONFIG)
    parser         = PydanticOutputParser(pydantic_object=GeneratedPosts)

    judges = {
        "gpt5_5": (GPTLLM(JUDGE_GPT),       JUDGE_GPT),
        "opus":   (ClaudeLLM(JUDGE_OPUS),    JUDGE_OPUS),
    }

    records: list[dict] = []

    for i, topic_entry in enumerate(topics):
        title       = topic_entry["metadata"]["source_article"]["title"]
        event_input = map_fields(topic_entry)

        print(f"\n{'='*70}")
        print(f"[{i+1}/{N_TOPICS}] {title[:68]}")
        print(f"{'='*70}")

        # Phase 1 — Generation
        print("  Phase 1: Generation")
        post_text, gen_metrics = run_generation(
            event_input, i, generator_llm,
            embedder, few_shot, prompt_builder, parser,
        )
        print(f"    done — cost ${gen_metrics['gen_cost_usd']:.5f}  time {gen_metrics['gen_time_s']}s")

        # Phase 2 — Regeneration
        print("  Phase 2: Regeneration")
        best_post, regen_metrics = run_refinement(
            post_text, judges, generator_llm, embedder, few_shot
        )
        print(f"    done — cost ${regen_metrics['regen_cost_usd']:.5f}  time {regen_metrics['regen_time_s']}s")

        records.append({
            "topic":        title,
            "generation":   gen_metrics,
            "regeneration": regen_metrics,
            "post_original":   post_text,
            "post_regenerated": best_post,
        })

    # Save raw JSON
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nRaw costs saved → {raw_path}")

    # Build Excel with two sheets
    detail_df  = build_detail_df(records)
    summary_df = build_summary_df(detail_df)

    out_path = RESULTS_DIR / "cost_comparison.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        detail_df.to_excel(writer, sheet_name="Per-Post Detail", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=True)

    print(f"Cost comparison saved → {out_path}")
    print(f"\n── Summary ────────────────────────────────────────────────────")
    print(summary_df.to_string())


if __name__ == "__main__":
    main()

# python3 -m backend.Generator.Experiments.experiment3.run_costs.run
