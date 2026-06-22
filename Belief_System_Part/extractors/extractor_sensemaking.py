from __future__ import annotations

"""
extractor_sensemaking.py — Method 2: Sensemaking-Informed Extraction
=====================================================================

Theoretical basis
-----------------
This extractor implements a sensemaking-informed approach to implicit
organisational belief extraction.

The method is grounded in three theoretical sources:

1. Weick (1995) — foundational architecture of organisational sensemaking:
   beliefs shape what organisations notice in their environment and give
   form to the actions they take. The most consequential organisational
   beliefs are those that operate beneath explicit articulation, shaping
   interpretive frames without ever being directly stated.

2. Maitlis & Christianson (2014) — sensemaking process moves:
   sensemaking is activated by violated expectations and manifests through
   three identifiable moves in organisational communication:
     (a) Noticing cues  → operationalised as: problem_framing
     (b) Creating interpretations → operationalised as: causal_attribution
     (c) Articulating calls to action → operationalised as: prescriptive_language

3. Malik et al. (2025) — organisational digital sensemaking:
   digitally enabled strategic agility emerges from a combination of
   meaning (digital orientation as discursive construct) and action
   (information governance and digital transformation as process
   facilitators). The beliefs extracted here constitute the substantive
   contents of the discursive construct — the shared interpretive premises
   through which KickstartAI makes sense of the AI landscape.

Pipeline
--------
  Step 1 — LLM extraction using three sensemaking lenses
           (problem_framing | causal_attribution | prescriptive_language)
  Step 2 — LLM semantic deduplication pass (group-scoped)

Changes from original
---------------------
  - DEDUP FIX: Replaced single-pass global deduplication (with [:12000]
    truncation that silently dropped beliefs) with group-scoped deduplication.
    Beliefs are grouped by (theoretical_construct, sensemaking_role, domain)
    and each group is deduplicated in a separate LLM call, eliminating
    truncation entirely and preventing cross-group over-merging.
  - AUDIT: _llm_deduplicate now returns (df_final, removed_records) so that
    a separate audit file can be written for thesis reporting of retention rates.
  - ISOLATION: Per-group dedup failures fall back to keeping the group intact,
    preventing a single failed LLM call from collapsing the entire pipeline.
  - raw_belief_id preserved throughout for traceability.

Output fields
-------------
Each extracted belief contains:
  - belief_id
  - raw_belief_id
  - belief_statement
  - inference_type        (problem_framing | prescriptive | causal_attribution)
  - sensemaking_move      (noticing | interpreting | action_articulation)
  - sensemaking_role      (meaning | action | meaning_action_link | equivocality_removal)
  - theoretical_construct (digital_orientation | information_governance |
                           digital_transformation | digitally_enabled_strategic_agility |
                           equivocality)
  - inference_logic
  - source_excerpt
  - confidence            (high | medium | low)
  - domain
  - source_label
  - source_document

References
----------
Weick, K. E. (1995). Sensemaking in Organizations. Sage.

Maitlis, S., & Christianson, M. (2014). Sensemaking in organizations:
Taking stock and moving forward. Academy of Management Annals, 8(1), 57-125.

Malik, M., Andargoli, A., Tallon, P., & Wickramasinghe, N. (2025).
An organizational sensemaking theorizing of how firms construct digitally
enabled strategic agility. Information & Management, 62, 104130.
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from extractors.base_extractor import BaseExtractor

load_dotenv()

# ── OPENAI CLIENT ─────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")


# ── PROMPTS ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert qualitative researcher specialising in organisational
sensemaking theory.

Your task is to extract implicit organisational beliefs from KickstartAI
communications — their public blog and LinkedIn posts.

─────────────────────────────────────────────────────────────────────────
THEORETICAL FRAMEWORK
─────────────────────────────────────────────────────────────────────────

You apply the sensemaking framework built from three sources:

1. Weick (1995): Sensemaking is belief-driven and retrospective. Beliefs
   shape what organisations notice in their environment and give form to
   the actions they take. The most consequential beliefs operate beneath
   explicit articulation — embedded in interpretive frames rather than
   stated directly.

2. Maitlis & Christianson (2014): Sensemaking is triggered by violated
   expectations and manifests through three process moves in communication:
     (a) NOTICING CUES — the organisation brackets a condition as a
         problem requiring attention. Signal: problem framing language.
     (b) CREATING INTERPRETATIONS — the organisation assigns meaning by
         linking outcomes to causes. Signal: causal attribution language.
     (c) ARTICULATING CALLS TO ACTION — the organisation asserts what
         must or should be done. Signal: prescriptive language.

3. Malik et al. (2025): In digital organisations, sensemaking operates
   through two linked devices:
     (a) MEANING / DISCOURSE — digital orientation as the shared cognitive
         framework that shapes how environmental signals are interpreted.
     (b) ACTIONS / PROCESS FACILITATORS — information governance and
         digital transformation as structures that translate meaning into
         organisational response.
   Strategic agility emerges when meaning and action are coordinated.

─────────────────────────────────────────────────────────────────────────
EXTRACTION INSTRUCTIONS
─────────────────────────────────────────────────────────────────────────

For each belief you identify, apply ALL of the following:

1. INFERENCE TYPE — which of the three Maitlis & Christianson process
   moves does this text signal?
     - problem_framing      → noticing move
     - causal_attribution   → interpreting move
     - prescriptive         → action-articulation move

2. SENSEMAKING MOVE — the corresponding Maitlis & Christianson label:
     - noticing
     - interpreting
     - action_articulation

3. SENSEMAKING ROLE — which Malik et al. sensemaking device is at work?
     - meaning              → belief is about shared cognition / orientation
     - action               → belief is about governance or transformation
     - meaning_action_link  → belief connects cognition to action
     - equivocality_removal → belief resolves ambiguity about AI/digital change

4. THEORETICAL CONSTRUCT — the most specific Malik et al. construct:
     - digital_orientation
     - information_governance
     - digital_transformation
     - digitally_enabled_strategic_agility
     - equivocality

5. INFERENCE LOGIC — one sentence explaining explicitly how the
   source_excerpt supports the belief_statement using the theory above.

─────────────────────────────────────────────────────────────────────────
OUTPUT FORMAT
─────────────────────────────────────────────────────────────────────────

Return a valid JSON array. Each element must have exactly these fields:

{
  "belief_id"            : "<sequential integer as string>",
  "belief_statement"     : "<concise declarative sentence stating the belief>",
  "inference_type"       : "<problem_framing | prescriptive | causal_attribution>",
  "sensemaking_move"     : "<noticing | interpreting | action_articulation>",
  "sensemaking_role"     : "<meaning | action | meaning_action_link | equivocality_removal>",
  "theoretical_construct": "<digital_orientation | information_governance | digital_transformation | digitally_enabled_strategic_agility | equivocality>",
  "inference_logic"      : "<one sentence connecting excerpt to theory>",
  "source_excerpt"       : "<verbatim quote of 25 words or fewer from the input>",
  "confidence"           : "<high | medium | low>",
  "domain"               : "<AI_adoption | societal_impact | organisational_capability | knowledge_sharing | collaboration | responsibility>"
}

─────────────────────────────────────────────────────────────────────────
RULES
─────────────────────────────────────────────────────────────────────────

- Extract only beliefs that are inferable from the text. Do not invent.
- Every belief must be grounded in a verbatim source_excerpt.
- Do not extract generic themes or factual observations without normative,
  causal, or sensemaking implication.
- Do not extract the same belief twice. Prefer the clearest formulation.
- Return ONLY valid JSON. No preamble, no markdown fences.
"""


DEDUP_SYSTEM = """
You are a qualitative research analyst using organisational sensemaking theory.

You will receive a JSON array of extracted implicit beliefs. All beliefs in
this array already share the same theoretical_construct, sensemaking_role,
and domain — they have been pre-grouped for you.

Your task: deduplicate semantically redundant beliefs within this group.

Merge two beliefs ONLY when they express the same underlying assumption AND
share the same values for ALL of these fields:
  - inference_type  (e.g. both must be problem_framing, or both prescriptive)
  - sensemaking_move (e.g. both must be noticing, or both action_articulation)

Do NOT merge beliefs that differ in:
  - inference_type  (e.g. problem_framing vs prescriptive → keep both)
  - sensemaking_move (e.g. noticing vs action_articulation → keep both)

When retaining one belief from a duplicate pair, keep the belief with:
  1. The clearest and most specific belief_statement
  2. The strongest and most verbatim source_excerpt
  3. The most explicit inference_logic
  4. The highest confidence

Return ONLY the deduplicated JSON array using the same schema. No markdown.
"""


# ── HELPERS ───────────────────────────────────────────────────────────────────

def strip_code_fences(text: str) -> str:
    """Remove accidental markdown code fences from model output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _llm_deduplicate(
    df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Group-scoped semantic deduplication via LLM.

    Beliefs are grouped by (theoretical_construct, sensemaking_role, domain).
    Each group is sent to a separate LLM call, eliminating the [:12000]
    truncation bug present in the original single-pass design and preventing
    cross-group over-merging.

    Within each group, the LLM is instructed to merge only beliefs that also
    share the same inference_type and sensemaking_move — preserving the
    theoretical distinction between Maitlis & Christianson's three process
    moves even when surface belief content appears similar.

    Per-group failures fall back to retaining the full group intact, so a
    single failed API call cannot collapse the entire deduplication pass.

    Args:
        df_raw: DataFrame of all raw extracted beliefs.

    Returns:
        Tuple of (df_final, removed_records) where removed_records is a list
        of belief dicts that were eliminated, retained for audit purposes.
    """
    df_raw = df_raw.copy()

    # Stamp raw_belief_id for traceability before any renumbering
    if "raw_belief_id" not in df_raw.columns:
        df_raw["raw_belief_id"] = df_raw["belief_id"].astype(str)

    available_cols = [
        c for c in [
            "belief_id", "raw_belief_id", "belief_statement", "inference_type",
            "sensemaking_move", "sensemaking_role", "theoretical_construct",
            "inference_logic", "source_excerpt", "confidence", "domain",
            "source_label", "source_document",
        ]
        if c in df_raw.columns
    ]

    group_keys = ["theoretical_construct", "sensemaking_role", "domain"]
    for k in group_keys:
        if k not in df_raw.columns:
            df_raw[k] = "unknown"

    retained_records: list[dict] = []
    removed_records:  list[dict] = []

    groups = list(df_raw.groupby(group_keys, dropna=False))
    print(f"\n[DEDUP] {len(groups)} groups identified for deduplication ...")

    for group_vals, group_df in groups:
        group_label = " | ".join(str(v) for v in group_vals)

        if len(group_df) == 1:
            # Single belief — nothing to deduplicate
            retained_records.append(group_df.iloc[0].to_dict())
            continue

        group_input = json.dumps(
            group_df[available_cols].to_dict(orient="records"),
            ensure_ascii=False,
        )

        print(f"  [{group_label}] {len(group_df)} beliefs ...")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": DEDUP_SYSTEM},
                    {"role": "user",   "content": f"Deduplicate this group:\n{group_input}"},
                ],
            )
            raw = strip_code_fences(response.choices[0].message.content or "")
            deduped = json.loads(raw)

            if not isinstance(deduped, list) or len(deduped) == 0:
                print(f"    [WARNING] Invalid output for '{group_label}'; keeping all.")
                retained_records.extend(group_df.to_dict(orient="records"))
                continue

            # Identify which raw beliefs were removed
            retained_raw_ids = {
                str(b.get("raw_belief_id", b.get("belief_id", "")))
                for b in deduped
            }
            removed_df = group_df[
                ~group_df["raw_belief_id"].astype(str).isin(retained_raw_ids)
            ]
            removed_records.extend(removed_df.to_dict(orient="records"))
            retained_records.extend(deduped)

            n_removed = len(group_df) - len(deduped)
            print(f"    -> {len(deduped)} retained, {n_removed} removed")

        except Exception as e:
            print(f"    [WARNING] Dedup failed for '{group_label}' ({e}); keeping all.")
            retained_records.extend(group_df.to_dict(orient="records"))

    df_final = pd.DataFrame(retained_records).reset_index(drop=True)

    # Re-number belief_id from 1; raw_belief_id preserved for audit trail
    df_final["belief_id"] = [str(i + 1) for i in df_final.index]

    print(
        f"\n[DEDUP] Raw: {len(df_raw)}  →  Final: {len(df_final)}  "
        f"(removed: {len(removed_records)})"
    )

    return df_final, removed_records


# ── EXTRACTOR CLASS ───────────────────────────────────────────────────────────

class SensemakingExtractor(BaseExtractor):
    """
    Method 2 — Sensemaking-Informed Extraction.

    Theoretical basis: Weick (1995), Maitlis & Christianson (2014),
    Malik et al. (2025).

    The extractor applies an LLM prompt structured around three sensemaking
    lenses — problem framing, causal attribution, and prescriptive language —
    to surface implicit beliefs embedded in KickstartAI's public communications.
    Each extracted belief is annotated with its sensemaking move, role, and
    theoretical construct. A second LLM pass performs group-scoped semantic
    deduplication to prevent over-merging across theoretically distinct belief
    categories.
    """

    def extract(self, text: str, source_label: str) -> list[dict]:
        """
        Extract implicit beliefs from a single text using sensemaking lenses.
        Implements the abstract method from BaseExtractor.

        Args:
            text:         Source text (blog passage or batch of LinkedIn posts).
            source_label: Identifier for the source document or batch.

        Returns:
            List of belief dicts, each annotated with sensemaking metadata.
        """
        user_message = (
            f"Source: {source_label}\n\n"
            f"--- BEGIN TEXT ---\n{text}\n--- END TEXT ---\n\n"
            "Extract all implicit organisational beliefs using the sensemaking "
            "framework (Weick 1995; Maitlis & Christianson 2014; Malik et al. 2025). "
            "For each belief, identify which of the three process moves it reflects "
            "(noticing, interpreting, action_articulation), which sensemaking device "
            "it represents (meaning or action), and which theoretical construct it maps to. "
            "Only extract beliefs supported by a verbatim source_excerpt. "
            "Return the full JSON array."
        )

        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            )
            raw = strip_code_fences(response.choices[0].message.content or "")
            beliefs = json.loads(raw)

            if not isinstance(beliefs, list):
                print(f"  [WARNING] Output for '{source_label}' is not a list.")
                return []

            for b in beliefs:
                if isinstance(b, dict):
                    b["source_label"]    = source_label
                    b["source_document"] = source_label

            return [b for b in beliefs if isinstance(b, dict)]

        except json.JSONDecodeError as e:
            print(f"  [WARNING] JSON parsing failed for '{source_label}': {e}")
            return []
        except Exception as e:
            print(f"  [WARNING] LLM call failed for '{source_label}': {e}")
            return []

    def extract_from_posts(self, posts: list[dict], batch_size: int = 20) -> list[dict]:
        """
        Process LinkedIn posts in batches.

        Batching reduces API calls and preserves inter-post context,
        which improves the LLM's ability to detect recurring belief patterns
        across a set of posts rather than treating each in isolation.

        Args:
            posts:      List of post dicts, each with a 'Post title' key.
            batch_size: Number of post titles per LLM call.

        Returns:
            List of belief dicts across all batches.
        """
        post_titles = [
            str(p.get("Post title", "")).strip()
            for p in posts
            if str(p.get("Post title", "")).strip()
        ]

        batches = [
            post_titles[i:i + batch_size]
            for i in range(0, len(post_titles), batch_size)
        ]
        all_beliefs: list[dict] = []

        for idx, batch in enumerate(batches, start=1):
            batch_text = "\n\n".join(
                [f"Post {i + 1}: {t}" for i, t in enumerate(batch)]
            )
            label = f"linkedin_batch_{idx:02d}"
            print(f"  Processing {label} ({len(batch)} posts) ...")
            beliefs = self.extract(text=batch_text, source_label=label)
            all_beliefs.extend(beliefs)
            time.sleep(1)

        return all_beliefs

    def run_pipeline(
        self,
        blog_path:          str | Path,
        posts_path:         str | Path,
        output_dir:         str | Path,
        output_path:        str | Path | None = None,
        batch_size:         int = 20,
        blog_char_limit:    int | None = None,
        linkedin_row_limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Full Method 2 pipeline.

        Loads blog and LinkedIn post data, runs LLM-based sensemaking
        extraction, applies group-scoped semantic deduplication, and saves
        outputs including a removed-beliefs audit file.

        Args:
            blog_path:          Path to blog.txt.
            posts_path:         Path to linkedin_posts.csv.
            output_dir:         Directory for all output files.
            output_path:        Optional explicit path for the final beliefs JSON.
            batch_size:         LinkedIn posts per LLM batch (default: 20).
            blog_char_limit:    Optional character limit on blog text.
            linkedin_row_limit: Optional row limit on LinkedIn posts.

        Returns:
            Deduplicated DataFrame of extracted implicit beliefs.
        """
        blog_path  = Path(blog_path)
        posts_path = Path(posts_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Load inputs ───────────────────────────────────────────────────────
        blog_text_full = blog_path.read_text(encoding="utf-8").strip()
        blog_text = blog_text_full[:blog_char_limit] if blog_char_limit else blog_text_full

        df_posts = pd.read_csv(posts_path)
        if linkedin_row_limit:
            df_posts = df_posts.head(linkedin_row_limit).copy()

        print(f"[SensemakingExtractor] Blog : {len(blog_text):,} chars")
        print(f"[SensemakingExtractor] Posts: {len(df_posts):,} rows")

        # ── LLM extraction ────────────────────────────────────────────────────
        print("\n[1] Blog extraction ...")
        blog_beliefs = self.extract(blog_text, source_label="blog")
        print(f"  -> {len(blog_beliefs)} beliefs extracted from blog")
        time.sleep(1)

        print("\n[2] LinkedIn posts extraction ...")
        posts_records    = df_posts.to_dict(orient="records")
        linkedin_beliefs = self.extract_from_posts(posts_records, batch_size=batch_size)
        print(f"  -> {len(linkedin_beliefs)} beliefs extracted from posts")

        # ── Consolidate ───────────────────────────────────────────────────────
        all_raw = blog_beliefs + linkedin_beliefs
        for i, b in enumerate(all_raw, start=1):
            b["belief_id"]     = str(i)
            b["raw_belief_id"] = str(i)

        df_raw = pd.DataFrame(all_raw)
        print(f"\n=== Raw beliefs before deduplication: {len(df_raw)} ===")

        if df_raw.empty:
            print("[WARNING] No beliefs extracted. Check source data and prompts.")
            return df_raw

        # ── Group-scoped semantic deduplication ───────────────────────────────
        df_final, removed = _llm_deduplicate(df_raw)

        # ── Resolve output paths ──────────────────────────────────────────────
        if output_path is not None:
            output_path  = Path(output_path)
            stem         = output_path.stem
            beliefs_out  = output_path
            raw_out      = output_dir / f"{stem}_raw.json"
            removed_out  = output_dir / f"{stem}_removed.json"
        else:
            beliefs_out  = output_dir / "beliefs_extracted_method2.json"
            raw_out      = output_dir / "beliefs_raw_method2.json"
            removed_out  = output_dir / "beliefs_removed_method2.json"

        # ── Save outputs ──────────────────────────────────────────────────────
        df_final.to_json(beliefs_out, orient="records", force_ascii=False, indent=2)
        df_raw.to_json(raw_out,       orient="records", force_ascii=False, indent=2)
        pd.DataFrame(removed).to_json(
            removed_out, orient="records", force_ascii=False, indent=2
        )

        print(f"\n[OUTPUT] Written:")
        print(f"  ├── {beliefs_out}  ({len(df_final)} beliefs, deduplicated)")
        print(f"  ├── {raw_out}  ({len(df_raw)} beliefs, raw)")
        print(f"  └── {removed_out}  ({len(removed)} beliefs removed, audit)")

        return df_final