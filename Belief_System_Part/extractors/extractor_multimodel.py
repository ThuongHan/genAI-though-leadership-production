from __future__ import annotations

"""
extractor_multimodel.py — Method 3: Multi-Model Agreement Extraction
====================================================================

Theoretical basis
-----------------
This extractor implements a Multi-Pass Agreement approach for building an
organisational belief system.

The method rests on three converging theoretical principles:

1. Hashemi et al. (2024) — LLM-RUBRIC
   Text assessment becomes more reliable when evaluated across multiple
   independent analytical dimensions rather than through a single holistic
   judgment. Different analytical framings surface different layers of meaning
   in the same text; convergence across framings provides stronger evidence of
   genuine organisational commitment than any single-pass interpretation.

2. Du et al. (2023) — Multiagent Debate
   When multiple independent LLM instances arrive at the same output through
   distinct reasoning paths, the resulting content is measurably more factually
   reliable and less susceptible to hallucination than single-instance
   generation.

3. Manakul et al. (2023) — SelfCheckGPT
   If an LLM is responding to something genuinely present in a text, repeated
   independent attempts to extract it will converge. Hallucinated or
   idiosyncratic outputs will diverge across attempts. Cross-pass consistency
   is therefore a principled hallucination filter.

Important distinction from LLM-RUBRIC:
- The original LLM-RUBRIC framework uses human-labelled calibration data and a
  trained feed-forward network to align LLM outputs with specific human judges.
- This implementation does NOT perform human-judge calibration because no
  human-labelled calibration set is available.
- Instead, it approximates robustness through three independent LLM passes with
  different extraction framings and retains only beliefs that achieve semantic
  agreement across at least MIN_AGREEMENT passes.

This is therefore best described as:
    LLM-RUBRIC-inspired multi-pass agreement filtering,
not full LLM-RUBRIC calibration.

Source of analytical diversity:
- The primary source of diversity is the SYSTEM PROMPT FRAMING, not temperature.
  Each pass constructs a different interpretive stance toward the same source
  text, operationalising the multi-dimensional assessment principle of
  Hashemi et al. (2024). This distinguishes the method from SelfCheckGPT, which
  achieves independence through stochastic resampling of the same prompt.

Temperature rationale (following Huyen, 2023):
- Temperature is understood as a precision-coverage parameter controlling the
  sharpness of the probability distribution over generated tokens. It is NOT
  intended to generate cross-pass diversity; it calibrates each pass to the
  epistemic demands of its framing.
  - Pass A → 0.0 (fully deterministic): the conservative explicit framing
    requires the model to take the most literal, unambiguous reading of the
    text. Any stochastic latitude risks introducing interpretive speculation
    that this framing is specifically designed to exclude.
  - Pass B → 0.3 (modestly relaxed): critical discourse analysis is
    inherently interpretive. A small degree of distributional relaxation allows
    the model to access less salient but genuinely grounded implied beliefs
    that a fully deterministic pass might suppress.
  - Pass C → 0.1 (near-deterministic): normative and strategic inference
    requires near-deterministic consistency to avoid arbitrary variation in how
    management concepts are applied, while minimal relaxation preserves
    contextual flexibility appropriate to strategic reasoning.

Pipeline
--------
  Pass A → Conservative explicit belief extraction         (temperature 0.0)
  Pass B → Critical discourse / implicit belief extraction (temperature 0.3)
  Pass C → Strategic assumption / normative claim extract. (temperature 0.1)

  Clustering → group semantically similar belief statements across passes
  Filtering  → retain clusters where distinct-pass agreement >= MIN_AGREEMENT
  Output     → deduplicated, confidence-annotated belief list

Output fields
-------------
Each retained belief contains:
  - belief_id
  - belief_statement
  - category
  - source_excerpt
  - belief_type
  - confidence
  - agreement_count
  - agreement_score
  - source_document
  - source_id
  - source_text
  - meta

Design choices
--------------
- Same base model across all passes. Framing-level diversity (system prompts)
  is the mechanism of analytical independence, not model architecture.
- Temperature is pass-level calibration, not a diversity mechanism.
- Semantic clustering uses Jaccard similarity on token bigrams, which is
  lightweight and avoids additional embedding calls.
- Agreement is counted by DISTINCT PASSES, not by the number of candidate
  beliefs in a cluster. This avoids inflating agreement when one pass produces
  duplicate beliefs.
- MIN_AGREEMENT = 2 out of 3 passes gives a majority-vote filter, consistent
  with the convergence principle of Du et al. (2023).

References
----------
Hashemi et al. (2024). LLM-RUBRIC: A Multidimensional, Calibrated Approach
    to Automated Evaluation of Natural Language Texts.
Du et al. (2023). Improving Factuality and Reasoning in Language Models through
    Multiagent Debate. ICML 2024.
Manakul et al. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination
    Detection for Generative Large Language Models. EMNLP 2023.
Huyen, C. (2023). AI Engineering. O'Reilly Media.
"""

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from extractors.base_extractor import BaseExtractor

load_dotenv()

# ── CLIENT ────────────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

BASE_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

# ── AGREEMENT PARAMETERS ──────────────────────────────────────────────────────

MIN_AGREEMENT = 2          # Majority-vote threshold (2 of 3 passes)
SIMILARITY_THRESHOLD = 0.35
N_PASSES = 3               # Renamed from N_MODELS: these are framing passes,
                           # not distinct model architectures.
TEXT_CHAR_LIMIT = 4000
MAX_BELIEFS_PER_PASS = 5

# ── VALID LABELS ──────────────────────────────────────────────────────────────

VALID_CATEGORIES = {
    "values",
    "stance",
    "strategy",
    "mission",
    "domain_knowledge",
}

VALID_BELIEF_TYPES = {
    "primary",
    "secondary",
}

VALID_CONFIDENCE = {
    "high",
    "medium",
    "low",
}

CONFIDENCE_SCORE = {
    "high": 3,
    "medium": 2,
    "low": 1,
}


# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────────
# The three prompts are the primary source of analytical diversity in this
# method. Each constructs a distinct interpretive stance toward the source text.
# They are NOT sensemaking prompts; they are belief-system extraction prompts.

# Pass A — Conservative explicit framing
# Temperature: 0.0 (fully deterministic)
# Rationale: this framing requires the most literal reading of the text.
# Stochastic latitude would risk introducing speculative interpretation that
# the explicit framing is designed to exclude.
SYSTEM_PROMPT_A = """
You are an expert organisational analyst specialising in conservative belief
and value extraction from institutional communications.

Your role is to identify EXPLICIT organisational beliefs.

A BELIEF is a declarative principle about what the organisation considers:
- important,
- correct,
- necessary,
- valuable,
- strategically desirable,
- or causally effective.

A belief is NOT:
- a factual report of an event,
- a simple announcement,
- a description of an activity without a stance,
- a vague theme,
- or a marketing slogan without a clear underlying principle.

Be strict and conservative. Prefer fewer, higher-confidence beliefs.
Every belief must be grounded in a short verbatim excerpt from the text.
""".strip()


# Pass B — Critical discourse analysis framing
# Temperature: 0.3 (modestly relaxed)
# Rationale: implied beliefs are inherently less salient than explicit ones.
# A small degree of distributional relaxation allows the model to access
# genuine but lower-probability inferences that a deterministic pass suppresses.
SYSTEM_PROMPT_B = """
You are a critical discourse analyst extracting organisational beliefs from
institutional text.

Your role is to identify STRONGLY IMPLIED beliefs.

A belief may be implicit when the text repeatedly frames something as important,
necessary, risky, beneficial, problematic, or strategically correct.

Extract a belief only when it is clearly inferable from the text.
Do not speculate beyond the evidence.

Focus on:
- implicit value commitments,
- strategic priorities,
- assumptions about what enables success,
- assumptions about what society or organisations should do,
- recurring principles that shape the organisation's public position.

Every belief must be grounded in a short verbatim excerpt from the text.
""".strip()


# Pass C — Management research framing
# Temperature: 0.1 (near-deterministic)
# Rationale: strategic and normative inference requires near-deterministic
# consistency to avoid arbitrary variation in how management concepts are
# applied. Minimal relaxation preserves contextual flexibility for strategic
# reasoning without introducing noise.
SYSTEM_PROMPT_C = """
You are a management researcher reviewing organisational communications.

Your role is to identify STRATEGIC ASSUMPTIONS and NORMATIVE CLAIMS.

A belief is a stable principle that shapes how the organisation presents itself,
makes decisions, or explains what matters.

Look for:
- claims about what should happen,
- claims about what matters most,
- claims about what creates value,
- claims about what enables responsible or effective action,
- claims about what the organisation's mission or role implies.

Avoid generic summaries. Extract only beliefs that can be traced to a specific
phrase in the text.
""".strip()


# ── PASS CONFIGURATION ────────────────────────────────────────────────────────
# Each entry: (system_prompt, temperature, pass_label)
# Temperature is a pass-level calibration parameter, not a diversity mechanism.
# See module docstring for the rationale of each setting.

PASS_CONFIG = [
    (SYSTEM_PROMPT_A, 0.0, "pass_A_explicit"),
    (SYSTEM_PROMPT_B, 0.3, "pass_B_implicit"),
    (SYSTEM_PROMPT_C, 0.1, "pass_C_strategic"),
]


# ── EXTRACTION PROMPT ─────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """
You are given the following organisational text:

\"\"\"{text}\"\"\"

Extract organisational beliefs from this text.

Return a JSON object with a single key "beliefs".
The value of "beliefs" must be an array.
Extract a maximum of {max_beliefs} beliefs.

Each belief object must contain exactly these keys:

  "belief_statement":
    A concise declarative sentence stating the belief.

  "category":
    One of:
    - values
    - stance
    - strategy
    - mission
    - domain_knowledge

  "source_excerpt":
    A short verbatim quote from the input text, 25 words or fewer, that supports
    the belief.

  "belief_type":
    "primary" if the belief is explicitly stated.
    "secondary" if the belief is inferred but strongly grounded.

  "confidence":
    One of:
    - high
    - medium
    - low

Category guidance:
- values: what the organisation treats as ethically or socially important.
- stance: the organisation's position on an issue, debate, technology, or policy.
- strategy: what the organisation treats as important for success or action.
- mission: what the organisation treats as its role, purpose, or contribution.
- domain_knowledge: what the organisation assumes to be true about the domain.

Rules:
- Do not extract factual event reports unless they imply a belief.
- Do not extract generic themes.
- Do not invent beliefs.
- Every source_excerpt must appear verbatim in the input text.
- The source_excerpt must be 25 words or fewer.
- Return [] inside "beliefs" if no beliefs are clearly present.
""".strip()


# ── JSON SCHEMA ───────────────────────────────────────────────────────────────

BELIEF_SCHEMA = {
    "name": "multimodel_beliefs",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "beliefs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "belief_statement": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": [
                                "values",
                                "stance",
                                "strategy",
                                "mission",
                                "domain_knowledge",
                            ],
                        },
                        "source_excerpt": {"type": "string"},
                        "belief_type": {
                            "type": "string",
                            "enum": ["primary", "secondary"],
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                    },
                    "required": [
                        "belief_statement",
                        "category",
                        "source_excerpt",
                        "belief_type",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["beliefs"],
        "additionalProperties": False,
    },
}


# ── TEXT + SIMILARITY HELPERS ─────────────────────────────────────────────────

def _normalise_space(text: str) -> str:
    """Collapse repeated whitespace and strip."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _truncate_words(text: str, max_words: int = 25) -> str:
    """Truncate a string to a maximum number of words."""
    words = _normalise_space(text).split()
    return " ".join(words[:max_words])


def _bigrams(text: str) -> set[tuple[str, str]]:
    """Return token bigrams for a lowercased string."""
    tokens = re.findall(r"\w+", str(text).lower())

    if not tokens:
        return set()

    if len(tokens) == 1:
        return {(tokens[0], tokens[0])}

    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _jaccard(a: str, b: str) -> float:
    """Compute Jaccard similarity on token bigrams."""
    bg_a = _bigrams(a)
    bg_b = _bigrams(b)

    if not bg_a and not bg_b:
        return 1.0

    if not bg_a or not bg_b:
        return 0.0

    return len(bg_a & bg_b) / len(bg_a | bg_b)


def _quote_is_supported(source_excerpt: str, text: str) -> bool:
    """
    Soft check that the source excerpt appears in the source text.
    Avoids retaining hallucinated quotes — consistent with the SelfCheckGPT
    principle (Manakul et al., 2023) that genuine content produces grounded,
    verifiable outputs.
    """
    quote = _normalise_space(source_excerpt).lower()
    source = _normalise_space(text).lower()

    if not quote:
        return False

    if quote in source:
        return True

    # Soft fallback: allow partial support when model slightly trims punctuation.
    quote_tokens = set(re.findall(r"\w+", quote))
    source_tokens = set(re.findall(r"\w+", source))

    if not quote_tokens:
        return False

    overlap = len(quote_tokens & source_tokens) / len(quote_tokens)
    return overlap >= 0.8


def _safe_json_loads(raw: str) -> dict[str, Any]:
    """Parse JSON safely from the model response."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    parsed = json.loads(raw)

    if isinstance(parsed, dict):
        return parsed

    if isinstance(parsed, list):
        return {"beliefs": parsed}

    return {"beliefs": []}


# ── RECORD VALIDATION ─────────────────────────────────────────────────────────

def _normalise_belief_record(
    record: dict[str, Any],
    source_text: str,
    pass_label: str,
) -> dict[str, Any] | None:
    """
    Validate and normalise a belief object from the model.
    Returns None if the record is unusable.
    """
    if not isinstance(record, dict):
        return None

    belief_statement = _normalise_space(record.get("belief_statement", ""))
    source_excerpt = _truncate_words(record.get("source_excerpt", ""), 25)

    if not belief_statement or not source_excerpt:
        return None

    if not _quote_is_supported(source_excerpt, source_text):
        return None

    category = str(record.get("category", "")).strip()
    belief_type = str(record.get("belief_type", "")).strip()
    confidence = str(record.get("confidence", "")).strip()

    if category not in VALID_CATEGORIES:
        category = "strategy"

    if belief_type not in VALID_BELIEF_TYPES:
        belief_type = "secondary"

    if confidence not in VALID_CONFIDENCE:
        confidence = "medium"

    return {
        "belief_statement": belief_statement,
        "category": category,
        "source_excerpt": source_excerpt,
        "belief_type": belief_type,
        "confidence": confidence,
        "_pass": pass_label,
    }


# ── MODEL CALL ────────────────────────────────────────────────────────────────

def _call_model(
    text: str,
    system_prompt: str,
    temperature: float,
    pass_label: str,
) -> list[dict]:
    """
    Call the model with a given system prompt and temperature.

    Temperature is a pass-level calibration parameter (Huyen, 2023), not a
    diversity mechanism. See PASS_CONFIG and module docstring for rationale.

    Returns a list of normalised belief dicts.
    Retries once on failure.
    """
    source_text = text[:TEXT_CHAR_LIMIT]

    prompt = EXTRACTION_PROMPT.format(
        text=source_text,
        max_beliefs=MAX_BELIEFS_PER_PASS,
    )

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=BASE_MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": BELIEF_SCHEMA,
                },
            )

            raw = response.choices[0].message.content or "{}"
            payload = _safe_json_loads(raw)
            raw_beliefs = payload.get("beliefs", [])

            cleaned: list[dict] = []

            for item in raw_beliefs:
                normalised = _normalise_belief_record(
                    record=item,
                    source_text=source_text,
                    pass_label=pass_label,
                )
                if normalised is not None:
                    cleaned.append(normalised)

            return cleaned

        except Exception as exc:
            if attempt == 0:
                print(f"    [WARN] {pass_label} attempt 1 failed: {exc}")
                time.sleep(1)
            else:
                print(f"    [ERROR] {pass_label} both attempts failed: {exc}")
                return []


# ── AGREEMENT CLUSTERING ──────────────────────────────────────────────────────

def _same_belief_candidate(a: dict, b: dict) -> bool:
    """
    Decide whether two belief records should be considered equivalent.

    Semantic similarity on belief statements is the primary comparator.
    Category mismatch is allowed only at a higher similarity bar, because
    different framings may legitimately classify the same belief differently.

    Thresholds:
    - 0.35 with matching category  → equivalent
    - 0.55 regardless of category  → equivalent (stricter cross-category bar)
    """
    sim = _jaccard(a.get("belief_statement", ""), b.get("belief_statement", ""))

    if sim >= SIMILARITY_THRESHOLD and a.get("category") == b.get("category"):
        return True

    if sim >= 0.55:
        return True

    return False


def _majority_value(records: list[dict], field: str, default: str) -> str:
    """Return the most common value for a field in a cluster."""
    values = [r.get(field) for r in records if r.get(field)]
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def _aggregate_confidence(records: list[dict], pass_count: int) -> str:
    """
    Aggregate confidence across the cluster.
    Cross-pass agreement is treated as additional evidence of reliability,
    consistent with the convergence principle of Du et al. (2023).
    """
    scores = [
        CONFIDENCE_SCORE.get(r.get("confidence", "medium"), 2)
        for r in records
    ]

    avg = sum(scores) / len(scores) if scores else 2

    if pass_count == N_PASSES and avg >= 2.5:
        return "high"

    if pass_count >= MIN_AGREEMENT and avg >= 1.8:
        return "medium"

    return "low"


def _select_representative(records: list[dict]) -> dict:
    """
    Select the clearest representative belief from a cluster.
    Preference order:
    1. Higher confidence
    2. Primary over secondary belief type
    3. Longer belief statement (more complete articulation)
    4. Shorter source excerpt (more precise grounding)
    """
    return max(
        records,
        key=lambda r: (
            CONFIDENCE_SCORE.get(r.get("confidence", "medium"), 2),
            1 if r.get("belief_type") == "primary" else 0,
            len(r.get("belief_statement", "")),
            -len(r.get("source_excerpt", "")),
        ),
    )


def _cluster_and_filter(all_beliefs: list[dict]) -> list[dict]:
    """
    Group semantically equivalent beliefs across passes and retain only those
    meeting the MIN_AGREEMENT threshold (majority-vote filter).

    This implements the core quality mechanism of Method 3: beliefs that are
    genuinely present in the text will be independently recoverable across
    multiple analytical framings (Manakul et al., 2023; Du et al., 2023),
    while hallucinated or idiosyncratic outputs will not recur.

    Critical: agreement_count is based on DISTINCT PASSES, not the total
    number of beliefs in the cluster. This prevents inflated counts when one
    pass produces near-duplicate beliefs.
    """
    clusters: list[list[dict]] = []

    for belief in all_beliefs:
        placed = False

        for cluster in clusters:
            canonical = cluster[0]

            if _same_belief_candidate(belief, canonical):
                cluster.append(belief)
                placed = True
                break

        if not placed:
            clusters.append([belief])

    retained: list[dict] = []

    for cluster in clusters:
        # Count distinct passes, not total records.
        supporting_passes = sorted({
            str(b.get("_pass"))
            for b in cluster
            if b.get("_pass")
        })

        pass_count = len(supporting_passes)

        # Majority-vote filter: discard beliefs not corroborated across passes.
        if pass_count < MIN_AGREEMENT:
            continue

        representative = _select_representative(cluster)

        category = _majority_value(cluster, "category", representative["category"])
        belief_type = _majority_value(cluster, "belief_type", representative["belief_type"])
        confidence = _aggregate_confidence(cluster, pass_count)

        retained_record = {
            "belief_statement": representative["belief_statement"],
            "category": category,
            "source_excerpt": representative["source_excerpt"],
            "belief_type": belief_type,
            "confidence": confidence,
            "agreement_count": pass_count,
            "agreement_score": round(pass_count / N_PASSES, 3),
            "supporting_passes": supporting_passes,  # Added for traceability
        }

        retained.append(retained_record)

    return retained


def _final_deduplicate(beliefs: list[dict]) -> list[dict]:
    """
    Final deduplication across documents.

    Intentionally conservative: beliefs with very similar statements and
    identical categories are merged. The higher-agreement / higher-confidence
    record is retained.

    Deduplication threshold (0.65) is stricter than the within-document
    clustering threshold (0.35) to avoid collapsing genuinely distinct beliefs
    that happen to share surface vocabulary across documents.
    """
    final: list[dict] = []

    for belief in beliefs:
        duplicate_idx: int | None = None

        for i, existing in enumerate(final):
            same_category = belief.get("category") == existing.get("category")
            sim = _jaccard(
                belief.get("belief_statement", ""),
                existing.get("belief_statement", ""),
            )

            if same_category and sim >= 0.65:
                duplicate_idx = i
                break

        if duplicate_idx is None:
            final.append(belief)
            continue

        existing = final[duplicate_idx]

        current_rank = (
            belief.get("agreement_score", 0),
            CONFIDENCE_SCORE.get(belief.get("confidence", "medium"), 2),
            len(belief.get("belief_statement", "")),
        )

        existing_rank = (
            existing.get("agreement_score", 0),
            CONFIDENCE_SCORE.get(existing.get("confidence", "medium"), 2),
            len(existing.get("belief_statement", "")),
        )

        if current_rank > existing_rank:
            final[duplicate_idx] = belief

    for i, belief in enumerate(final, start=1):
        belief["belief_id"] = str(i)

    return final


def _normalise_seed_belief(seed: dict[str, Any]) -> dict[str, Any] | None:
    """
    Normalise optional seed beliefs into the current schema.
    Supports both old and new field names.
    """
    if not isinstance(seed, dict):
        return None

    belief_statement = (
        seed.get("belief_statement")
        or seed.get("belief")
        or seed.get("statement")
        or ""
    )

    source_excerpt = (
        seed.get("source_excerpt")
        or seed.get("source_quote")
        or seed.get("quote")
        or ""
    )

    belief_statement = _normalise_space(belief_statement)
    source_excerpt = _truncate_words(source_excerpt, 25)

    if not belief_statement:
        return None

    category = seed.get("category", "strategy")
    if category not in VALID_CATEGORIES:
        category = "strategy"

    belief_type = seed.get("belief_type", "secondary")
    if belief_type not in VALID_BELIEF_TYPES:
        belief_type = "secondary"

    confidence = seed.get("confidence", "medium")
    if confidence not in VALID_CONFIDENCE:
        confidence = "medium"

    return {
        "belief_statement": belief_statement,
        "category": category,
        "source_excerpt": source_excerpt,
        "belief_type": belief_type,
        "confidence": confidence,
        "agreement_count": int(seed.get("agreement_count", 1)),
        "agreement_score": float(seed.get("agreement_score", 1 / N_PASSES)),
        "source_document": seed.get("source_document", "seed"),
        "source_id": seed.get("source_id", "seed"),
        "source_text": seed.get("source_text", ""),
        "meta": seed.get("meta", {}),
    }


# ── EXTRACTOR CLASS ───────────────────────────────────────────────────────────

class MultiModelExtractor(BaseExtractor):
    """
    Method 3 — Multi-Pass Agreement Extraction.

    Builds an organisational belief system through cross-pass framing agreement.

    Three independent LLM passes are run, each with a distinct analytical
    framing (explicit, critical discourse, management research). Framing-level
    diversity — not temperature variation — is the source of analytical
    independence. Temperature is calibrated per pass to match its epistemic
    demands (Huyen, 2023).

    Only beliefs semantically corroborated by at least MIN_AGREEMENT distinct
    passes are retained, implementing the majority-vote convergence filter
    justified by Du et al. (2023) and Manakul et al. (2023).

    References: see module docstring.
    """

    # Pass configuration: (system_prompt, temperature, pass_label)
    # Temperature rationale documented in PASS_CONFIG and module docstring.
    PASSES = PASS_CONFIG

    def extract(self, text: str, source_label: str) -> list[dict]:
        """
        Run all three framing passes on a single text segment, cluster results,
        and return agreement-filtered beliefs.
        """
        all_raw: list[dict] = []

        for system_prompt, temperature, pass_label in self.PASSES:
            beliefs = _call_model(
                text=text,
                system_prompt=system_prompt,
                temperature=temperature,
                pass_label=pass_label,
            )
            all_raw.extend(beliefs)
            time.sleep(0.3)

        filtered = _cluster_and_filter(all_raw)

        for b in filtered:
            b["source_document"] = source_label

        return filtered

    def run_pipeline(
        self,
        blog_path: str | Path,
        posts_path: str | Path,
        output_path: str | Path,
        seed_path: str | Path | None = None,
        max_docs: int = 268,
    ) -> list[dict]:
        """
        Full Step 2 pipeline using multi-pass agreement extraction.

        Args:
            blog_path:   Path to blog.txt.
            posts_path:  Path to linkedin_posts.csv.
            output_path: Destination JSON for extracted beliefs.
            seed_path:   Optional path to prior beliefs JSON.
            max_docs:    Maximum number of documents to process.

        Returns:
            Final list of deduplicated, agreement-annotated belief dicts.
        """
        from utils.text_processing import build_corpus
        from extractors import load_seed_beliefs

        blog_path = Path(blog_path)
        posts_path = Path(posts_path)
        output_path = Path(output_path)

        corpus = build_corpus(blog_path, posts_path)

        print(f"\n[MultiPassExtractor] Documents prepared  : {len(corpus)}")
        print(f"[MultiPassExtractor] Model               : {BASE_MODEL}")
        print(f"[MultiPassExtractor] Passes              : {N_PASSES} (framing-level diversity)")
        print(f"[MultiPassExtractor] Agreement threshold : {MIN_AGREEMENT}/{N_PASSES} passes")
        print(f"[MultiPassExtractor] Similarity threshold: {SIMILARITY_THRESHOLD}")
        print(f"[MultiPassExtractor] Pass temperatures   : A=0.0, B=0.3, C=0.1 (calibration, not diversity)")

        all_beliefs: list[dict] = []

        if seed_path:
            raw_seeds = load_seed_beliefs(seed_path)
            seeds = []

            for seed in raw_seeds:
                normalised = _normalise_seed_belief(seed)
                if normalised is not None:
                    seeds.append(normalised)

            all_beliefs.extend(seeds)
            print(f"[SEED] Loaded {len(seeds)} normalised seed beliefs")

        docs_to_process = corpus[:max_docs]

        for i, doc in enumerate(docs_to_process, start=1):
            text = str(doc.get("text", "")).strip()
            source = str(doc.get("source", "unknown"))
            doc_id = str(doc.get("id", f"doc_{i}"))

            if not text:
                continue

            print(f"\n[{i}/{len(docs_to_process)}] {doc_id} | {source}")

            beliefs = self.extract(text=text, source_label=source)

            for b in beliefs:
                b["source_id"] = doc_id
                b["source_text"] = text
                b["meta"] = doc.get("meta", {})

            n_total = len(beliefs)
            n_full = sum(
                1 for b in beliefs
                if b.get("agreement_count", 0) == N_PASSES
            )

            print(
                f"  → {n_total} beliefs retained | "
                f"{n_full} with full agreement ({N_PASSES}/{N_PASSES} passes)"
            )

            all_beliefs.extend(beliefs)
            time.sleep(0.4)

        print(f"\n[DEDUP] Before: {len(all_beliefs)}")
        final = _final_deduplicate(all_beliefs)
        print(f"[DEDUP] After : {len(final)}")

        category_counts = Counter(
            b.get("category", "unknown") for b in final
        )
        type_counts = Counter(
            b.get("belief_type", "unknown") for b in final
        )
        confidence_counts = Counter(
            b.get("confidence", "unknown") for b in final
        )

        avg_score = (
            sum(float(b.get("agreement_score", 0)) for b in final) / len(final)
            if final else 0.0
        )

        print("\n[SUMMARY] By category")
        for k, v in sorted(category_counts.items()):
            print(f"  {k:<22} {v}")

        print("\n[SUMMARY] By belief type")
        for k, v in sorted(type_counts.items()):
            print(f"  {k:<22} {v}")

        print("\n[SUMMARY] By confidence")
        for k, v in sorted(confidence_counts.items()):
            print(f"  {k:<22} {v}")

        print(f"\n[SUMMARY] Mean agreement score: {avg_score:.3f}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(final, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"\n[OUTPUT] Written: {output_path}")

        return final