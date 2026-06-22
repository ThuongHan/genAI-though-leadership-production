from __future__ import annotations

"""
extractor_llm.py — Method 1: LLM-Based Direct Extraction
=========================================================

Theoretical basis
-----------------
The dual-pass architecture operationalises a distinction between two layers
of organisational belief that is well established across three independent
scholarly traditions:

1. SCHEIN (2010) — Organisational Culture and Leadership (4th ed.)
   Schein's three-level model separates *espoused beliefs and values*
   (what an organisation explicitly claims to stand for) from *underlying
   basic assumptions* (taken-for-granted beliefs that are rarely articulated
   but manifest through language patterns and framing choices).

   Pass A targets Schein's espoused layer  → PRIMARY beliefs.
   Pass B targets Schein's assumption layer → SECONDARY beliefs.

2. VAN DIJK (1998) — Ideology: A Multidisciplinary Approach
   Van Dijk's socio-cognitive framework of ideological discourse analysis
   distinguishes *explicitly represented propositions* (conscious normative
   assertions) from *presupposed background beliefs* (naturalised assumptions
   communicated through framing, lexical choice, and rhetorical structure
   rather than direct statement).

   Pass A targets van Dijk's explicit propositions  → PRIMARY beliefs.
   Pass B targets van Dijk's presupposed background → SECONDARY beliefs.

3. SUN et al. (2024) — TrustLLM: Trustworthiness in Large Language Models
   In the machine-ethics dimension of TrustLLM, Sun et al. operationalise
   the same explicit/implicit split computationally: *explicit ethics* are
   the normatively prescribed behaviours a model displays when placed in a
   directly ethical scenario; *implicit ethics* are the internally encoded
   values the model reveals through judgment, without being asked to
   articulate them. This contemporary computational parallel confirms that
   the two-pass architecture reflects a robust conceptual distinction, not
   merely an engineering heuristic.

Deduplication strategy
----------------------
Three deduplication mechanisms are available, applied in priority order:

  Pass C-1 (Embedding):  Full pairwise cosine similarity using OpenAI
                          text-embedding-3-large. Every belief is compared
                          against every other belief within its type group.
                          Most thorough — no batch blindspots.

  Pass C-2 (LLM):        Semantic deduplication via GPT-5.1 in batches of 80.
                          Used automatically if the embedding API is unavailable.

  Pass C-3 (Jaccard):    Local bigram Jaccard similarity fallback. Runs when
                          both API-based methods fail (e.g. full connectivity
                          loss). Deterministic, no API calls required.

In all three paths, primary and secondary beliefs are deduplicated
STRUCTURALLY SEPARATELY — they are split at the Python level before any
dedup call and recombined only after, so no LLM or similarity function
can ever merge beliefs across the Schein (2010) layer boundary.

Design notes
------------
- TEMPERATURE = 0.0 ensures deterministic output across runs (reproducibility).
- Structured JSON schemas enforce downstream compatibility.
- The secondary pass adds an `inference_reasoning` field (grounded in van
  Dijk's requirement to make the inference chain explicit and traceable).
- `confidence` is present on both primary and secondary schemas for
  downstream uniformity (Sun et al., 2024).

References
----------
Schein, E. H. (2010). Organizational culture and leadership (4th ed.).
    Jossey-Bass.

Sun, L., Huang, Y., Wang, H., Wu, S., Zhang, Q., Gao, C., Huang, X.,
    Lyu, W., Zhang, Y., Li, X., Liu, Z., Liu, Y., Wang, Y., Tang, J.,
    Xiong, L., Tian, H., Qiu, X., He, X., Gui, T., & Zhang, X. (2024).
    TrustLLM: Trustworthiness in large language models.
    arXiv preprint arXiv:2401.05561.

van Dijk, T. A. (1998). Ideology: A multidisciplinary approach. Sage.
"""

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from extractors.base_extractor import BaseExtractor

load_dotenv()

# ── OPENAI CLIENT ────────────────────────────────────────────────────────────
# Single client — direct OpenAI endpoint, no proxy.
# Both chat (extraction + LLM dedup) and embeddings use the same client.

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
)

# Alias so both names resolve to the same object
chat_client  = client
embed_client = client

MODEL        = os.getenv("OPENAI_MODEL", "gpt-5.1")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# TEMPERATURE = 0.0 ensures deterministic output across runs (reproducibility).
# Sun et al. (2024) apply the same principle in TrustLLM's benchmarking.
TEMPERATURE  = 0.0

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert organisational analyst trained in belief system extraction.

Your analytical framework draws on two theoretical traditions:

1. Schein's (2010) model of organisational culture, which distinguishes
   espoused beliefs (explicitly stated values and positions) from underlying
   assumptions (taken-for-granted beliefs that surface through language
   patterns and framing rather than direct assertion).

2. Van Dijk's (1998) socio-cognitive framework, which separates explicitly
   represented ideological propositions from presupposed background beliefs
   that are naturalised within an organisation's discourse.

A BELIEF is a stable principle that defines what an organisation considers
essential, correct, or strategically important. Beliefs motivate decisions,
public positions, and behavioural patterns.

A belief is NOT:
- A factual report of an event
- A simple product announcement with no normative content
- A description of an activity without an implicit stance

Be precise, conservative, and faithful to the source text.
""".strip()

# ── PROMPT TEMPLATES ──────────────────────────────────────────────────────────

PROMPT_A_TEMPLATE = """
TASK — PRIMARY BELIEF EXTRACTION (Pass A)

Theoretical target: Schein's (2010) espoused beliefs and values /
van Dijk's (1998) explicitly represented ideological propositions.

You are given the following organisational text:

\"\"\"{text}\"\"\"

Your goal: identify all PRIMARY beliefs in this text.

A PRIMARY belief is one that is EXPLICITLY stated. Reliable indicators include:
- First-person declarations: "we believe", "we think", "our mission is", "we are convinced"
- Normative prescriptions directed at others: "organisations must", "the key is to", "AI should"
- Explicit value claims: "what matters is", "this is essential", "the priority is"
- Goal statements: "our aim is", "we strive to", "we are committed to"

Instructions:
- Extract only the MOST SALIENT primary beliefs.
- Maximum 3 beliefs for this text segment.
- Do not include factual observations unless they clearly express a belief.
- Do not split one idea into multiple near-duplicate beliefs.
- Write each belief as a full, standalone declarative sentence.
- Select the SHORTEST phrase from the text that triggered the belief as source_quote.
- Assign one category from: values | stance | strategy | mission | domain_knowledge
- Return an empty list if none are present.
""".strip()

PROMPT_B_TEMPLATE = """
TASK — SECONDARY BELIEF EXTRACTION (Pass B)

Theoretical target: Schein's (2010) underlying basic assumptions /
van Dijk's (1998) presupposed background beliefs.

You are given the following organisational text:

\"\"\"{text}\"\"\"

Your goal: identify all SECONDARY beliefs in this text.

A SECONDARY belief is NOT directly stated, but is clearly implied by how the
text is written. Following van Dijk (1998), presupposed beliefs surface through
the following rhetorical mechanisms — look specifically for these:
- Problem framing (the way a problem is defined implies a normative 'should-be')
- Causal attribution (attributing outcomes to specific causes reveals assumed
  mechanisms of change)
- Prescriptive urgency (urgency language implies an underlying normative premise)
- Contrasting language (in-group/out-group framing implies an ideological stance)
- Audience assumptions (what the text takes for granted that its reader knows
  or accepts)
- Metaphor and analogy choices (the conceptual frame chosen reveals assumptions)

Instructions:
- Only include beliefs that are strongly and clearly inferable from the text.
- Extract only the MOST SALIENT secondary beliefs.
- Maximum 2 beliefs for this text segment.
- Do not include weak, speculative, repetitive, or overlapping inferences.
- Do not restate a primary belief as a secondary belief.
- Write each belief as a full, standalone declarative sentence.
- Provide one sentence of reasoning explaining the inference (inference_reasoning).
- Select the SHORTEST phrase from the text that grounds the inference as source_quote.
- Assign one category from: values | stance | strategy | mission | domain_knowledge
- Assign a confidence level: high | medium | low
  (high = the inference is unambiguous; medium = clearly implied but requires
   interpretation; low = plausible but speculative — per Sun et al. (2024),
   implicit beliefs carry inherent inferential uncertainty that must be declared)
- Return an empty list if none are present.
""".strip()

DEDUP_SYSTEM_PROMPT = """
You are a qualitative research analyst performing semantic deduplication of
organisational belief statements.

All beliefs in this batch are of the SAME type (either all primary or all
secondary). Your task:
- Identify beliefs that express the same underlying proposition in different
  surface forms: paraphrase variants, rewording, or synonymous framing.
- Retain only the MOST CLEARLY FORMULATED version of each duplicated belief.
- Do NOT merge beliefs from different categories unless genuinely the same.
- Preserve ALL fields of the retained belief object exactly as received.

Return ONLY the deduplicated JSON array.
No preamble, no markdown fences, no explanation. Raw JSON array only.
""".strip()

# ── STRUCTURED OUTPUT SCHEMAS ─────────────────────────────────────────────────

PRIMARY_SCHEMA = {
    "name": "primary_beliefs",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "beliefs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "belief":       {"type": "string"},
                        "category":     {
                            "type": "string",
                            "enum": ["values", "stance", "strategy",
                                     "mission", "domain_knowledge"]
                        },
                        "source_quote": {"type": "string"},
                        "belief_type":  {"type": "string", "enum": ["primary"]},
                        "confidence":   {"type": "string",
                                         "enum": ["high", "medium", "low"]},
                    },
                    "required": ["belief", "category", "source_quote",
                                 "belief_type", "confidence"],
                    "additionalProperties": False,
                }
            }
        },
        "required": ["beliefs"],
        "additionalProperties": False,
    }
}

SECONDARY_SCHEMA = {
    "name": "secondary_beliefs",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "beliefs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "belief":              {"type": "string"},
                        "category":            {
                            "type": "string",
                            "enum": ["values", "stance", "strategy",
                                     "mission", "domain_knowledge"]
                        },
                        "source_quote":        {"type": "string"},
                        "inference_reasoning": {"type": "string"},
                        "belief_type":         {"type": "string",
                                                "enum": ["secondary"]},
                        "confidence":          {"type": "string",
                                                "enum": ["high", "medium", "low"]},
                    },
                    "required": ["belief", "category", "source_quote",
                                 "inference_reasoning", "belief_type", "confidence"],
                    "additionalProperties": False,
                }
            }
        },
        "required": ["beliefs"],
        "additionalProperties": False,
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# DEDUPLICATION — THREE-TIER STRATEGY
# ══════════════════════════════════════════════════════════════════════════════
#
# All three tiers enforce the same structural guarantee:
# primary and secondary beliefs are NEVER placed in the same comparison group.
# The split is done at the Python level before any dedup call is made.
#
# Priority order:
#   1. Embedding (Pass C-1) — full pairwise cosine similarity
#   2. LLM       (Pass C-2) — semantic dedup via GPT-5.1 in batches
#   3. Jaccard   (Pass C-3) — local bigram fallback, no API required
#
# ══════════════════════════════════════════════════════════════════════════════


# ── TIER 1: Embedding-based dedup (Pass C-1) ──────────────────────────────────

def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Fetch embeddings from OpenAI text-embedding-3-large in batches of 100.
    Retries up to 3 times per batch on transient failure.
    """
    BATCH = 100
    all_vecs: list[list[float]] = []
    n_batches = (len(texts) + BATCH - 1) // BATCH

    for i in range(0, len(texts), BATCH):
        batch     = texts[i : i + BATCH]
        batch_num = i // BATCH + 1
        print(f"    [EMBED] batch {batch_num}/{n_batches} ({len(batch)}) ...", end=" ")

        for attempt in range(3):
            try:
                resp = embed_client.embeddings.create(model=EMBED_MODEL, input=batch, timeout=30)
                all_vecs.extend([r.embedding for r in resp.data])
                print("ok")
                break
            except Exception as e:
                wait = (attempt + 1) * 3
                if attempt < 2:
                    print(f"\n      [WARN] attempt {attempt+1} failed: {e} — retry in {wait}s")
                    time.sleep(wait)
                else:
                    print(f"\n      [ERROR] all attempts failed: {e}")
                    raise
        time.sleep(0.3)

    return all_vecs


def _cosine_sim_matrix(vecs: list[list[float]]) -> np.ndarray:
    """Return the full (n×n) pairwise cosine similarity matrix."""
    mat   = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    mat   = mat / norms
    return mat @ mat.T


def _embedding_dedup_group(
    beliefs: list[dict],
    belief_type: str,
    threshold: float,
) -> list[dict]:
    """
    Deduplicate one same-type group using full pairwise cosine similarity.

    When two beliefs exceed the threshold the one with the longer (more
    informative) text is kept. Primary and secondary are never mixed —
    this function only ever receives one type at a time.

    Args:
        beliefs:     All beliefs of one belief_type.
        belief_type: "primary" or "secondary" — for logging.
        threshold:   Cosine similarity above which two beliefs are duplicates.
    """
    n   = len(beliefs)
    tag = f"[EMBED-{belief_type.upper()}]"

    if n == 0:
        return []

    print(f"\n  {tag} {n} beliefs — fetching embeddings ...")
    texts = [b.get("belief", b.get("belief_statement", "")) for b in beliefs]
    vecs  = _get_embeddings(texts)
    sim   = _cosine_sim_matrix(vecs)

    removed: set[int] = set()
    for i in range(n):
        if i in removed:
            continue
        for j in range(i + 1, n):
            if j in removed:
                continue
            if sim[i, j] >= threshold:
                drop = j if len(texts[i]) >= len(texts[j]) else i
                removed.add(drop)

    kept = [b for idx, b in enumerate(beliefs) if idx not in removed]
    print(f"  {tag} {n} → {len(kept)} ({len(removed)} removed, threshold={threshold})")
    return kept


# ── TIER 2: LLM-based dedup (Pass C-2) ───────────────────────────────────────

def _llm_dedup_batch(beliefs: list[dict], belief_type: str) -> list[dict]:
    """Send one same-type batch to GPT for deduplication. Retries once."""
    raw_input = json.dumps(beliefs, ensure_ascii=False)
    for attempt in range(2):
        try:
            response = chat_client.chat.completions.create(
                model=MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": DEDUP_SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"Deduplicate this batch of {belief_type} beliefs:\n{raw_input}"},
                ],
                timeout=120,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            result = json.loads(raw.strip())
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                for v in result.values():
                    if isinstance(v, list):
                        return v
            return beliefs
        except Exception as e:
            if attempt == 0:
                print(f"    [WARN] LLM dedup attempt 1 failed ({belief_type}): {e}")
                time.sleep(2)
            else:
                print(f"    [ERROR] LLM dedup failed ({belief_type}): {e}")
                return beliefs
    return beliefs


def _llm_dedup_group(
    beliefs: list[dict],
    belief_type: str,
    batch_size: int = 80,
) -> list[dict]:
    """
    Two-pass batched LLM dedup for one same-type group.
    Pass 1: intra-batch. Pass 2: cross-batch survivors.
    """
    n   = len(beliefs)
    tag = f"[LLM-{belief_type.upper()}]"
    if n == 0:
        return []

    n_batches = (n + batch_size - 1) // batch_size
    print(f"\n  {tag} Pass 1 — {n} beliefs in {n_batches} batches")
    pass1: list[dict] = []
    for i in range(0, n, batch_size):
        batch     = beliefs[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"    Batch {batch_num}/{n_batches} ({len(batch)}) ...", end=" ")
        deduped = _llm_dedup_batch(batch, belief_type)
        print(f"→ {len(deduped)}")
        pass1.extend(deduped)
        time.sleep(0.5)

    print(f"  {tag} After pass 1: {len(pass1)}")

    n_batches2 = (len(pass1) + batch_size - 1) // batch_size
    print(f"  {tag} Pass 2 — {len(pass1)} beliefs in {n_batches2} batches")
    pass2: list[dict] = []
    for i in range(0, len(pass1), batch_size):
        batch     = pass1[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"    Batch {batch_num}/{n_batches2} ({len(batch)}) ...", end=" ")
        deduped = _llm_dedup_batch(batch, belief_type)
        print(f"→ {len(deduped)}")
        pass2.extend(deduped)
        time.sleep(0.5)

    print(f"  {tag} After pass 2: {len(pass2)}")
    return pass2


# ── TIER 3: Local Jaccard fallback (Pass C-3) ────────────────────────────────

def _bigrams(text: str) -> set[tuple[str, str]]:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 2:
        return {(t, t) for t in tokens}
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _jaccard(a: str, b: str) -> float:
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba and not bb:
        return 1.0
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def _jaccard_dedup_group(
    beliefs: list[dict],
    belief_type: str,
    threshold: float = 0.62,
) -> list[dict]:
    """
    Local bigram Jaccard dedup for one same-type group. No API calls.
    Retains the first-seen belief in each cluster.
    """
    retained: list[dict] = []
    tag = f"[JACCARD-{belief_type.upper()}]"
    for candidate in beliefs:
        c_text = candidate.get("belief", "")
        c_cat  = candidate.get("category", "")
        is_dup = False
        for existing in retained:
            if existing.get("category", "") != c_cat:
                continue
            if _jaccard(c_text, existing.get("belief", "")) >= threshold:
                is_dup = True
                break
        if not is_dup:
            retained.append(candidate)
    print(f"  {tag} {len(beliefs)} → {len(retained)}")
    return retained


# ── ORCHESTRATOR: structural split + tier selection ───────────────────────────

def _deduplicate(
    beliefs: list[dict],
    embedding_threshold: float = 0.85,
) -> list[dict]:
    """
    Main deduplication orchestrator.

    Structural guarantee: primary and secondary beliefs are split at the
    Python level BEFORE any dedup function is called. They are recombined
    only after both groups have been independently processed. No dedup
    method can ever merge beliefs across the Schein (2010) layer boundary,
    regardless of semantic similarity.

    Tier selection (per group):
      1. Try embedding dedup (text-embedding-3-large, full pairwise).
      2. On embedding API failure → fall back to batched LLM dedup.
      3. On LLM API failure → fall back to local Jaccard dedup.

    Args:
        beliefs:             Full mixed list of extracted beliefs.
        embedding_threshold: Cosine similarity threshold for Pass C-1.

    Returns:
        Deduplicated list: primaries first, then secondaries.
    """
    primaries   = [b for b in beliefs if b.get("belief_type") == "primary"]
    secondaries = [b for b in beliefs if b.get("belief_type") == "secondary"]
    others      = [b for b in beliefs if b.get("belief_type")
                   not in ("primary", "secondary")]

    print(f"\n[DEDUP] Structural split → primary: {len(primaries)} | "
          f"secondary: {len(secondaries)}"
          + (f" | other: {len(others)}" if others else ""))

    def _process_group(group: list[dict], btype: str) -> list[dict]:
        if not group:
            return []
        # ── Tier 1: embedding ────────────────────────────────────────────────
        try:
            return _embedding_dedup_group(group, btype, embedding_threshold)
        except Exception as e:
            print(f"  [WARN] Embedding dedup failed for {btype}: {e}")
            print(f"  [FALLBACK] Trying LLM dedup ...")
        # ── Tier 2: LLM ──────────────────────────────────────────────────────
        try:
            return _llm_dedup_group(group, btype)
        except Exception as e:
            print(f"  [WARN] LLM dedup failed for {btype}: {e}")
            print(f"  [FALLBACK] Running local Jaccard dedup ...")
        # ── Tier 3: Jaccard ───────────────────────────────────────────────────
        return _jaccard_dedup_group(group, btype)

    deduped_p = _process_group(primaries,   "primary")
    deduped_s = _process_group(secondaries, "secondary")
    final     = deduped_p + deduped_s + others

    print(f"\n[DEDUP] Result → primary: {len(deduped_p)} (was {len(primaries)}) | "
          f"secondary: {len(deduped_s)} (was {len(secondaries)}) | "
          f"total: {len(final)}")
    return final


# ── EXTRACTION API CALL ───────────────────────────────────────────────────────

def _call_api(text: str, prompt_template: str, schema: dict) -> list[dict]:
    """Call OpenAI with structured output. Retries once on transient failure."""
    prompt = prompt_template.format(text=text)
    for attempt in range(2):
        try:
            response = chat_client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_schema", "json_schema": schema},
                timeout=90,
            )
            payload = json.loads(response.choices[0].message.content)
            return payload.get("beliefs", [])
        except Exception as e:
            if attempt == 0:
                print(f"    [WARN] attempt 1 failed: {e}")
                time.sleep(1)
            else:
                print(f"    [ERROR] both attempts failed: {e}")
                return []


# ── EXTRACTOR CLASS ───────────────────────────────────────────────────────────

class LLMExtractor(BaseExtractor):
    """
    Method 1 — LLM-Based Direct Extraction (primary + secondary beliefs).

    Pass A (primary)   → espoused beliefs / explicit propositions (Schein, 2010;
                         van Dijk, 1998)
    Pass B (secondary) → underlying assumptions / presupposed background beliefs
    Pass C (dedup)     → three-tier deduplication: embedding → LLM → Jaccard

    Primary and secondary beliefs are structurally separated throughout the
    deduplication stage and can never be merged.
    """

    def extract(self, text: str, source_label: str) -> list[dict]:
        """
        Run Pass A + Pass B on a single text segment.
        Both passes return a `confidence` field for downstream uniformity.
        """
        primary   = _call_api(text, PROMPT_A_TEMPLATE, PRIMARY_SCHEMA)
        secondary = _call_api(text, PROMPT_B_TEMPLATE, SECONDARY_SCHEMA)
        for b in primary + secondary:
            b["source_document"] = source_label
        return primary + secondary

    def run_dedup_only(
        self,
        input_path:  str | Path,
        output_path: str | Path,
        embedding_threshold: float = 0.85,
    ) -> list[dict]:
        """
        Re-run ONLY the deduplication pass (Pass C) on an existing extracted
        beliefs JSON. Use when Step 2 completed but dedup failed.

        Args:
            input_path:          Path to the raw beliefs JSON.
            output_path:         Destination path (can be same as input).
            embedding_threshold: Cosine threshold for embedding dedup.
        """
        input_path      = Path(input_path)
        output_path     = Path(output_path)
        checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.json"

        all_beliefs: list[dict] = json.loads(
            input_path.read_text(encoding="utf-8")
        )
        print(f"\n[run_dedup_only] Loaded {len(all_beliefs)} beliefs from {input_path}")

        print(f"\n[DEDUP] Before: {len(all_beliefs)}")
        final = _deduplicate(all_beliefs, embedding_threshold)
        print(f"[DEDUP] After : {len(final)}")

        self._print_summary(final)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Delete checkpoint now that everything succeeded
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"[CHECKPOINT] Deleted: {checkpoint_path.name}")

        print(f"\n[OUTPUT] Written: {output_path}")
        return final

    def run_pipeline(
        self,
        blog_path:   str | Path,
        posts_path:  str | Path,
        output_path: str | Path,
        seed_path:   str | Path | None = None,
        max_docs:    int = 268,
        embedding_threshold: float = 0.85,
    ) -> list[dict]:
        """
        Full Step 2 pipeline:
        1. Build corpus (blog chunks + LinkedIn posts)
        2. Load optional seed beliefs
        3. Run extract() on every document (Pass A + Pass B)
        4. Run three-tier deduplication (Pass C)
        5. Save to output_path
        """
        from utils.text_processing import build_corpus
        from extractors import load_seed_beliefs

        blog_path   = Path(blog_path)
        posts_path  = Path(posts_path)
        output_path = Path(output_path)

        # ── Checkpoint path ───────────────────────────────────────────────────────
        # The checkpoint saves all extracted beliefs to disk after every document.
        # If extraction is interrupted (timeout, crash, Ctrl+C), re-running
        # --steps 2 will resume from the last completed document rather than
        # starting over. The checkpoint is deleted after successful dedup.
        checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.json"

        corpus = build_corpus(blog_path, posts_path)
        print(f"\n[LLMExtractor] Documents prepared: {len(corpus)}")

        # ── Resume from checkpoint if it exists ───────────────────────────────
        all_beliefs: list[dict] = []
        completed_ids: set[str] = set()

        if checkpoint_path.exists():
            all_beliefs   = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            completed_ids = {b["source_id"] for b in all_beliefs if "source_id" in b}
            print(f"[CHECKPOINT] Resumed: {len(all_beliefs)} beliefs from "
                  f"{len(completed_ids)} completed documents")

        if seed_path and not completed_ids:
            seeds = load_seed_beliefs(seed_path)
            all_beliefs.extend(seeds)
            print(f"[SEED] Loaded {len(seeds)} seed beliefs")

        for i, doc in enumerate(corpus[:max_docs], start=1):
            text   = doc["text"].strip()
            source = doc["source"]
            doc_id = doc["id"]

            if not text:
                continue

            # Skip already-completed documents when resuming
            if doc_id in completed_ids:
                print(f"[{i}/{len(corpus)}] {doc_id} — skipped (checkpoint)")
                continue

            print(f"\n[{i}/{len(corpus)}] {doc_id} | {source}")
            beliefs = self.extract(text, source_label=source)
            for b in beliefs:
                b["source_id"]   = doc_id
                b["source_text"] = text
                b["meta"]        = doc.get("meta", {})
            all_beliefs.extend(beliefs)

            # Save checkpoint after every document
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(all_beliefs, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            time.sleep(0.4)

        print(f"\n[DEDUP] Before: {len(all_beliefs)}")
        final = _deduplicate(all_beliefs, embedding_threshold)
        print(f"[DEDUP] After : {len(final)}")

        self._print_summary(final)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Delete checkpoint now that everything succeeded
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"[CHECKPOINT] Deleted: {checkpoint_path.name}")

        print(f"\n[OUTPUT] Written: {output_path}")
        return final

    @staticmethod
    def _print_summary(beliefs: list[dict]) -> None:
        """Print category / type / confidence breakdown."""
        category_counts = Counter(b.get("category",    "unknown") for b in beliefs)
        type_counts     = Counter(b.get("belief_type", "unknown") for b in beliefs)
        conf_counts     = Counter(b.get("confidence",  "unknown") for b in beliefs)

        print("\n[SUMMARY] By category")
        for k, v in sorted(category_counts.items()):
            print(f"  {k:<22} {v}")
        print("\n[SUMMARY] By type")
        for k, v in sorted(type_counts.items()):
            print(f"  {k:<22} {v}")
        print("\n[SUMMARY] By confidence")
        for k, v in sorted(conf_counts.items()):
            print(f"  {k:<22} {v}")