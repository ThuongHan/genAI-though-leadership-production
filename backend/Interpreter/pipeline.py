"""
KickstartAI Pipeline — Refactored Module
==========================================
Unified interface: retrieve → interpret → generate_posts → run_pipeline_for_one_news

Usage:
  python pipeline.py                     # interactive single-article pipeline
  python pipeline.py --batch 40          # batch-process 40 articles
  python pipeline.py --batch 40 --input kickstartai_blogs.json
"""

import json
import os
import sys
import time
import argparse
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TypedDict, Literal, Optional

import numpy as np
from openai import OpenAI
# TF-IDF removed → replaced with BM25 + Hybrid (see _BaseRetriever subclasses)
from sklearn.metrics.pairwise import cosine_similarity
from jsonschema import validate, ValidationError

# ╔══════════════════════════════════════════════════════════════╗
# ║                        CONFIG                                ║
# ╚══════════════════════════════════════════════════════════════╝

CONFIG = {
    "retriever": "embedding",      # "bm25" | "embedding" | "hybrid"
    "reasoning_mode": "cot",       # "cot" | "flat"
    "top_k_beliefs": 3,
    "hybrid_alpha": 0.5,
    "auto_score": False,           # enable LLM-as-Judge auto-scoring
}

BELIEF_REPOSITORY_PATH = os.getenv("BELIEF_REPOSITORY_PATH", "Interpreter/belief_repository_sensemaking.json")
LOG_PATH = os.getenv("PIPELINE_LOG_PATH", "./logs/sessions.jsonl")

# ── JSON Schema for interpreter output validation ──
# Matches the prompt output fields (What happened / Why does it matter / etc.)
# Any field not listed here is allowed (additionalProperties=True by default in jsonschema draft-04/07).
INTERPRETER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "What happened": {
            "type": "string",
            "minLength": 10,
        },
        "Why does it matter (globally and NL)": {
            "type": "string",
            "minLength": 10,
        },
        "Why does it matter for KickstartAI": {
            "type": "string",
            "minLength": 10,
        },
        "Key stance / opinion": {
            "type": "string",
            "enum": ["Supportive", "Critical", "Neutral", "Cautious"],
        },
        "Supporting arguments": {
            "type": "array",
            "items": {"type": "string", "minLength": 10},
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": [
        "What happened",
        "Why does it matter (globally and NL)",
        "Why does it matter for KickstartAI",
        "Key stance / opinion",
        "Supporting arguments",
    ],
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                       TYPED TYPES                            ║
# ╚══════════════════════════════════════════════════════════════╝

class RAGResult(TypedDict):
    """Result of RAG retrieval: top_k beliefs with scores."""
    news_id: str
    retriever_type: str           # "bm25" | "embedding" | "hybrid"
    beliefs: list[dict]           # [{"belief_id": str, "belief_text": str, "score": float}, ...]


class InterpreterResult(TypedDict):
    """Result of LLM interpretation: parsed JSON + raw output."""
    news_id: str
    retriever_type: str           # "bm25" | "embedding" | "hybrid"
    reasoning_mode: str           # "flat" | "cot"
    schema_pass: bool             # JSON parse succeeded?
    parsed_json: Optional[dict]   # parsed interpretation JSON, or None
    raw_llm_output: str           # raw LLM response string


class PostCandidate(TypedDict):
    """A single generated LinkedIn post candidate."""
    candidate_id: str             # "A" | "B" | "C"
    text: str


# ╔══════════════════════════════════════════════════════════════╗
# ║                        CLIENT                                ║
# ╚══════════════════════════════════════════════════════════════╝

def _create_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Lazy singleton client (shared across retrievers, interpreter, generator)."""
    global _client
    if _client is None:
        _client = _create_client()
    return _client


# ╔══════════════════════════════════════════════════════════════╗
# ║                    OLD ARTICLE MODEL                         ║
# ╚══════════════════════════════════════════════════════════════╝
# NOTE: kept as a lightweight internal helper for sliding new
# news_obj dicts into the original TfidfRetriever / EmbeddingRetriever
# retrieve() signatures which expect .title and .full_text attributes.

class _Article:
    __slots__ = ("title", "full_text")
    def __init__(self, title: str, full_text: str):
        self.title = title
        self.full_text = full_text


# ╔══════════════════════════════════════════════════════════════╗
# ║                    BELIEF LOADER                             ║
# ╚══════════════════════════════════════════════════════════════╝

def load_beliefs(path: str = BELIEF_REPOSITORY_PATH) -> list[dict]:
    """Load the belief repository and normalise keys for the pipeline."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Wrap if raw is a dict with a "beliefs" key, otherwise assume list
    items: list[dict] = raw if isinstance(raw, list) else raw.get("beliefs", raw.get("items", []))

    return [
        {
            "belief_id": item.get("id", f"B{i:03d}"),
            "belief_text": item.get("belief", str(item)),
        }
        for i, item in enumerate(items)
    ]


# ╔══════════════════════════════════════════════════════════════╗
# ║              RETRIEVERS  (wrapping original logic)           ║
# ╚══════════════════════════════════════════════════════════════╝

class _BaseRetriever(ABC):
    @abstractmethod
    def _retrieve_top_k(self, article: _Article,
                        belief_dicts: list[dict], top_k: int) -> list[dict]:
        """Return top_k results as [{"belief_id", "belief_text", "score"}, ...]."""
        raise NotImplementedError


class _BM25Retriever(_BaseRetriever):
    """Okapi BM25 retriever — k1=1.5, b=0.75, implemented from scratch with numpy."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        return re.findall(r'\w+', text.lower())

    def _retrieve_top_k(self, article, belief_dicts, top_k):
        query_text = f"{article.title}\n{article.full_text}"
        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            raise ValueError("Empty query after tokenisation")

        belief_texts = [b["belief_text"] for b in belief_dicts]
        belief_tokens_list = [self._tokenize(t) for t in belief_texts]
        if not any(belief_tokens_list):
            raise ValueError("No belief texts")

        N = len(belief_tokens_list)
        avgdl = np.mean([len(bt) for bt in belief_tokens_list])

        df = {}
        for term in set(query_tokens):
            df[term] = sum(1 for bt in belief_tokens_list if term in bt)

        scores = np.zeros(N)
        for i, bt in enumerate(belief_tokens_list):
            dl = len(bt)
            if dl == 0:
                continue
            tf = {}
            for t in bt:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for term in set(query_tokens):
                n = df.get(term, 0)
                if n == 0:
                    continue
                idf = np.log((N - n + 0.5) / (n + 0.5) + 1.0)
                f = tf.get(term, 0)
                numerator = f * (self.k1 + 1.0)
                denominator = f + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                score += idf * numerator / denominator
            scores[i] = score

        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]

        return [
            {
                "belief_id": belief_dicts[i]["belief_id"],
                "belief_text": belief_dicts[i]["belief_text"],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]

class _EmbeddingRetriever(_BaseRetriever):
    """Wraps the original OpenAI Embeddings + cosine_similarity logic."""
    def __init__(self, model: str = "text-embedding-3-large"):
        self.model = model
        self._emb_client: Optional[OpenAI] = None

    @property
    def emb_client(self):
        if self._emb_client is None:
            self._emb_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._emb_client

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.emb_client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def _retrieve_top_k(self, article, belief_dicts, top_k):
        query_text = f"{article.title}\n{article.full_text[:2000]}"
        belief_texts = [b["belief_text"] for b in belief_dicts]
        all_texts = [query_text] + belief_texts
        embeddings = self._embed(all_texts)

        query_vec = np.array(embeddings[0]).reshape(1, -1)
        belief_vecs = np.array(embeddings[1:])
        sims = cosine_similarity(query_vec, belief_vecs).flatten()

        k = min(top_k, len(sims))
        top_indices = np.argsort(sims)[::-1][:k]

        return [
            {
                "belief_id": belief_dicts[i]["belief_id"],
                "belief_text": belief_dicts[i]["belief_text"],
                "score": float(sims[i]),
            }
            for i in top_indices
        ]


class _HybridRetriever(_BaseRetriever):
    """Sparse + dense hybrid: BM25 (keyword) ⊗ Embedding (semantic).

    Runs both retrievers independently on a candidate pool of top_k×3,
    then merges via min-max normalisation and α-weighted fusion.
    """
    def __init__(self, alpha: float = 0.5, embed_model: str = "text-embedding-3-large"):
        self.alpha = alpha
        self._bm25 = _BM25Retriever()
        self._emb = _EmbeddingRetriever(model=embed_model)

    @staticmethod
    def _minmax_normalise(scores: np.ndarray) -> np.ndarray:
        rng = scores.max() - scores.min()
        if rng == 0:
            return np.full_like(scores, 0.5)
        return (scores - scores.min()) / rng

    def _retrieve_top_k(self, article, belief_dicts, top_k):
        # Retrieve a larger candidate pool from each retriever
        candidate_k = min(top_k * 3, len(belief_dicts))
        bm25_candidates = self._bm25._retrieve_top_k(article, belief_dicts, candidate_k)
        emb_candidates = self._emb._retrieve_top_k(article, belief_dicts, candidate_k)

        # Collect unique beliefs from both pools
        seen_ids = set()
        all_candidates = []
        for c in bm25_candidates + emb_candidates:
            if c["belief_id"] not in seen_ids:
                seen_ids.add(c["belief_id"])
                all_candidates.append(c)

        if len(all_candidates) <= top_k:
            return all_candidates

        # Normalise scores per retriever
        bm25_scores = np.array([c.get("_bm25_score", c["score"]) for c in all_candidates])
        emb_scores = np.array([c.get("_emb_score", c["score"]) for c in all_candidates])

        # For pure candidates (only in one pool), approximate the missing score
        bm25_ids = {c["belief_id"] for c in bm25_candidates}
        emb_ids = {c["belief_id"] for c in emb_candidates}
        # Pre-compute mean scores for fallback
        bm25_mean = float(np.mean(bm25_scores)) if len(bm25_scores) > 0 else 0.0
        emb_mean = float(np.mean(emb_scores)) if len(emb_scores) > 0 else 0.0
        for i, c in enumerate(all_candidates):
            bid = c["belief_id"]
            if bid not in bm25_ids:
                bm25_scores[i] = bm25_mean
            if bid not in emb_ids:
                emb_scores[i] = emb_mean

        bm25_norm = self._minmax_normalise(bm25_scores)
        emb_norm = self._minmax_normalise(emb_scores)
        hybrid_scores = self.alpha * bm25_norm + (1.0 - self.alpha) * emb_norm

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        result = []
        for i in top_indices:
            c = all_candidates[i]
            result.append({
                "belief_id": c["belief_id"],
                "belief_text": c["belief_text"],
                "score": float(hybrid_scores[i]),
            })
        return result


# ╔══════════════════════════════════════════════════════════════╗
# ║                   PROMPT BUILDERS                            ║
# ║   (wrapping original build_cot_prompt / build_flat_prompt)   ║
# ╚══════════════════════════════════════════════════════════════╝

_MULTILINGUAL_RULE = """
[MULTILINGUAL INPUT HANDLING]
The news article may be in English, Dutch, or both.
- If Dutch: process internally in Dutch, OUTPUT ALL JSON VALUES IN ENGLISH.
- Preserve Dutch-specific context (NL policy, ecosystem, EU AI Act references).
- CRITICAL: Final JSON must always be in English, regardless of input language.
"""


def _build_cot_prompt(title: str, excerpt: str, top_beliefs: list[dict]) -> str:
    """Chain-of-Thought prompt — wraps original CoT logic."""
    beliefs_block = "\n\n".join(
        f"[Belief {b['belief_id']} | score={b['score']:.4f}]\n\"\"\"{b['belief_text']}\"\"\""
        for b in top_beliefs
    )
    return f"""
You are the Chief Strategic Interpreter for KickstartAI.
{_MULTILINGUAL_RULE}

[EXTERNAL FACT: NEWS ARTICLE]
Title: {title}
Content: {excerpt[:1500]}

[INTERNAL BRAIN: RETRIEVED KICKSTARTAI BELIEFS (top {len(top_beliefs)})]
{beliefs_block}

[CHAIN OF THOUGHT]
Step 1 (Fact Extraction): What happened? Global and NL impact?
Step 2 (Strategic Alignment): How does this align with our beliefs?
Step 3 (Stance Synthesis): Our stance + 3 supporting arguments.

[STRICT OUTPUT FORMAT — ONLY JSON, no markdown, ALL VALUES IN ENGLISH]
{{
    "What happened": "Brief summary (50-200 words, English)",
    "Why does it matter (globally and NL)": "Global + NL context (80-250 words, English)",
    "Why does it matter for KickstartAI": "Connection to belief (60-200 words, English)",
    "Key stance / opinion": "Supportive / Critical / Neutral / Cautious",
    "Supporting arguments": ["Arg 1", "Arg 2", "Arg 3"]
}}
""".strip()


def _build_flat_prompt(title: str, excerpt: str, top_beliefs: list[dict]) -> str:
    """Flat (no chain-of-thought) prompt — wraps original flat logic."""
    beliefs_block = "\n\n".join(
        f"[Belief {b['belief_id']} | score={b['score']:.4f}]\n\"\"\"{b['belief_text']}\"\"\""
        for b in top_beliefs
    )
    return f"""
You are the Chief Strategic Interpreter for KickstartAI.
{_MULTILINGUAL_RULE}

[NEWS ARTICLE] Title: {title}
Content: {excerpt[:1500]}

[INTERNAL BELIEFS (top {len(top_beliefs)})]
{beliefs_block}

[OUTPUT — ONLY JSON, all English]
{{
    "What happened": "Brief summary (50-200 words, English)",
    "Why does it matter (globally and NL)": "Global + NL context (80-250 words, English)",
    "Why does it matter for KickstartAI": "Connection to belief (60-200 words, English)",
    "Key stance / opinion": "Supportive / Critical / Neutral / Cautious",
    "Supporting arguments": ["Arg 1", "Arg 2", "Arg 3"]
}}
""".strip()


# ╔══════════════════════════════════════════════════════════════╗
# ║              GENERATE POSTS  (new functionality)             ║
# ╚══════════════════════════════════════════════════════════════╝

def _build_generator_prompt(parsed_json: dict) -> str:
    """Build a prompt that asks the LLM to write 3 LinkedIn posts
    based on the interpreter's output."""
    return f"""
You are a KickstartAI content strategist. Based on the strategic interpretation below,
write 3 LinkedIn posts that KickstartAI could publish.

[STRATEGIC INTERPRETATION]
- What happened: {parsed_json.get("What happened", "")}
- Why it matters (globally & NL): {parsed_json.get("Why does it matter (globally and NL)", "")}
- Why it matters for KickstartAI: {parsed_json.get("Why does it matter for KickstartAI", "")}
- Stance: {parsed_json.get("Key stance / opinion", "")}
- Supporting arguments: {json.dumps(parsed_json.get("Supporting arguments", []))}

[WRITING GUIDELINES]
- Each post should be 120-250 words, professional but engaging LinkedIn style.
- Each post MUST take a DIFFERENT angle (e.g. one policy-focused, one community-focused,
  one technology-focused).
- Use 2-4 relevant hashtags per post.
- Maintain a tone consistent with KickstartAI's mission: accelerating responsible,
  real-world AI adoption in the Netherlands for positive societal impact.

[OUTPUT — ONLY JSON, no markdown]
{{
    "posts": [
        {{"candidate_id": "A", "text": "Post A content here..."}},
        {{"candidate_id": "B", "text": "Post B content here..."}},
        {{"candidate_id": "C", "text": "Post C content here..."}}
    ]
}}
""".strip()


# ╔══════════════════════════════════════════════════════════════╗
# ║              UNIFIED PUBLIC API                              ║
# ╚══════════════════════════════════════════════════════════════╝

# ── retrieve ────────────────────────────────────────────────────

def retrieve(news_obj: dict,
             beliefs: list[dict],
             retriever_type: str,
             top_k: int) -> RAGResult:
    """
    Unified RAG retrieval entry point.

    Args:
        news_obj:   {"news_id": str, "title": str, "excerpt": str}
        beliefs:    [{"belief_id": str, "belief_text": str}, ...]
        retriever_type: "bm25" | "embedding" | "hybrid"
        top_k:      number of top beliefs to return

    Returns:
        RAGResult with news_id, retriever_type, and beliefs list.
    """
    # ── Wrap news_obj into the internal _Article shape ──
    article = _Article(
        title=news_obj.get("title", news_obj.get("name", "Unknown")),
        full_text=news_obj.get("excerpt", news_obj.get("full_text", "")),
    )

    # ── Select and run retriever ──
    if retriever_type == "bm25":
        retriever = _BM25Retriever()
    elif retriever_type == "embedding":
        retriever = _EmbeddingRetriever()
    elif retriever_type == "hybrid":
        retriever = _HybridRetriever()
    else:
        raise ValueError(f"Unknown retriever_type: {retriever_type}")

    top_beliefs = retriever._retrieve_top_k(article, beliefs, top_k)

    return RAGResult(
        news_id=news_obj.get("news_id", ""),
        retriever_type=retriever_type,
        beliefs=top_beliefs,
    )


# ── Schema validation helper ────────────────────────────────────

def validate_interpreter_output(obj: dict) -> tuple[bool, str | None]:
    """
    Validate parsed interpreter JSON against INTERPRETER_OUTPUT_SCHEMA.

    Returns:
        (passed: bool, error_message: str | None)
    """
    try:
        validate(instance=obj, schema=INTERPRETER_OUTPUT_SCHEMA)
        return True, None
    except ValidationError as e:
        return False, str(e)


# ── interpret ───────────────────────────────────────────────────

def interpret(news_obj: dict,
              rag_result: RAGResult,
              reasoning_mode: str) -> InterpreterResult:
    """
    Build prompt from news + retrieved beliefs, call LLM, parse JSON.

    Args:
        news_obj:       {"news_id": str, "title": str, "excerpt": str}
        rag_result:     output of retrieve()
        reasoning_mode: "cot" | "flat"

    Returns:
        InterpreterResult with schema_pass, parsed_json, raw_llm_output.
    """
    title = news_obj.get("title", news_obj.get("name", "Unknown"))
    excerpt = news_obj.get("excerpt", news_obj.get("full_text", ""))
    top_beliefs = rag_result.get("beliefs", [])

    # ── Build prompt (wrapping original CoT / Flat logic) ──
    if reasoning_mode == "cot":
        prompt = _build_cot_prompt(title, excerpt, top_beliefs)
    elif reasoning_mode == "flat":
        prompt = _build_flat_prompt(title, excerpt, top_beliefs)
    else:
        raise ValueError(f"Unknown reasoning_mode: {reasoning_mode}")

    # ── Call LLM (wrapping original step4_execute_llm logic) ──
    client = _get_client()
    raw_llm_output = ""
    parsed_json = None
    schema_pass = False

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a KickstartAI strategic analyst. "
                        "Output ONLY valid JSON. All text in English. "
                        "Stance MUST be: Supportive, Critical, Neutral, or Cautious."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        raw_llm_output = response.choices[0].message.content or ""

        # ── Parse JSON + Schema validation ──
        parsed_json = json.loads(raw_llm_output)

        passed, err_msg = validate_interpreter_output(parsed_json)
        if passed:
            schema_pass = True
        else:
            print(f"  ⚠️  Schema validation failed: {err_msg}")
            schema_pass = False
            parsed_json = None
    except json.JSONDecodeError:
        parsed_json = None
        schema_pass = False
    except Exception as exc:
        print(f"  LLM API error: {exc}")
        raw_llm_output = str(exc)
        parsed_json = None
        schema_pass = False

    return InterpreterResult(
        news_id=news_obj.get("news_id", ""),
        retriever_type=rag_result.get("retriever_type", ""),
        reasoning_mode=reasoning_mode,
        schema_pass=schema_pass,
        parsed_json=parsed_json,
        raw_llm_output=raw_llm_output,
    )


# ── generate_posts ──────────────────────────────────────────────

def generate_posts(interpreter_result: InterpreterResult) -> list[PostCandidate]:
    """
    Generate 3 LinkedIn post candidates from the interpreter output.

    Args:
        interpreter_result: output of interpret() with schema_pass=True

    Returns:
        List of 3 PostCandidate objects (candidate_id "A"/"B"/"C").
    """
    if not interpreter_result.get("schema_pass") or interpreter_result.get("parsed_json") is None:
        raise ValueError("Cannot generate posts: interpreter_result has schema_pass=False")

    parsed = interpreter_result["parsed_json"]
    prompt = _build_generator_prompt(parsed)

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a KickstartAI content strategist. "
                        "Output ONLY valid JSON with a 'posts' array."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,  # higher temperature for creative variety
        )
        raw = response.choices[0].message.content or ""
        data = json.loads(raw)
        posts_raw = data.get("posts", [])

        candidates: list[PostCandidate] = []
        for p in posts_raw[:3]:
            candidates.append(PostCandidate(
                candidate_id=p.get("candidate_id", "?"),
                text=p.get("text", ""),
            ))
        return candidates
    except Exception as exc:
        print(f"  generate_posts error: {exc}")
        return []


# ╔══════════════════════════════════════════════════════════════╗
# ║              PIPELINE ORCHESTRATOR                           ║
# ╚══════════════════════════════════════════════════════════════╝

def log_session(news_obj: dict,
                rag_result: RAGResult,
                interpreter_result: InterpreterResult,
                candidates: list[PostCandidate] | None,
                user_choice: str | None,
                log_path: str = LOG_PATH) -> None:
    """
    Write one JSON line to the session log for RQ1/RQ2/RQ3 analysis.

    Called once per news article — whether schema_pass succeeds or fails.
    """
    # Ensure log directory exists
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # ── RQ1 fields: retrieval quality (TF-IDF vs Embedding) ──
    #   retriever_type, selected_beliefs (id + text + score per belief)

    # ── RQ2 fields: interpretation quality & structural stability ──
    #   reasoning_mode, schema_pass, interpreter_parsed, interpreter_raw

    # ── RQ3 fields: failure modes in real usage ──
    #   error_type (R=retrieval / C=content / G=generation / S=schema)

    entry = {
        # ── Identifiers ──
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "news_id": news_obj.get("news_id", ""),
        "news_title": news_obj.get("title", news_obj.get("name", "Unknown")),

        # ── RQ1: Retrieval quality ──
        "retriever_type": rag_result.get("retriever_type", ""),
        "selected_beliefs": [
            {
                "belief_id": b["belief_id"],
                "belief_text": b["belief_text"],
                "score": b["score"],
            }
            for b in rag_result.get("beliefs", [])
        ],

        # ── RQ2: Interpretation quality & structural stability ──
        "reasoning_mode": interpreter_result.get("reasoning_mode", ""),
        "schema_pass": interpreter_result.get("schema_pass", False),
        "interpreter_parsed": interpreter_result.get("parsed_json"),
        "interpreter_raw": interpreter_result.get("raw_llm_output", ""),

        # ── RQ3: Generation quality & user preference ──
        "candidates": (
            [{"candidate_id": c["candidate_id"], "text": c["text"]}
             for c in candidates]
            if candidates is not None
            else None
        ),
        "user_choice": user_choice,   # "A"/"B"/"C" or None

        # ── Reserved for human/LLM annotation ──
        "human_relevance_labels": None,         # RQ1: human relevance scores
        "interpretation_quality_score": None,   # RQ2: human quality rating
        "error_type": None,                     # RQ3: "R"|"C"|"G"|"S"
        "llm_diagnosis": None,                  # optional LLM-based error analysis
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_pipeline_for_one_news(news_obj: dict,
                               beliefs: list[dict],
                               config: dict) -> None:
    """
    Full pipeline for a single news article.

    Steps:
      1. RAG retrieval
      2. LLM interpretation (fail-fast if schema_pass=False)
      3. Generate 3 LinkedIn post candidates
      4. Print candidates, let user pick A/B/C
      5. Log session (placeholder)
    """
    news_id = news_obj.get("news_id", "?")

    # ── Step 1: RAG ──
    print(f"\n{'='*60}")
    print(f"  PIPELINE: {news_obj.get('title', news_id)[:70]}")
    print(f"  Retriever: {config['retriever']} | Mode: {config['reasoning_mode']} | top_k: {config['top_k_beliefs']}")
    print(f"{'='*60}")

    rag_result = retrieve(news_obj, beliefs, config["retriever"], config["top_k_beliefs"])
    print(f"\n  📎 Top-{config['top_k_beliefs']} beliefs:")
    for i, b in enumerate(rag_result["beliefs"]):
        print(f"     {i+1}. [{b['belief_id']}] score={b['score']:.4f} — {b['belief_text'][:60]}...")

    # ── Step 2: Interpret ──
    interpreter_result = interpret(news_obj, rag_result, config["reasoning_mode"])

    if not interpreter_result["schema_pass"]:
        print(f"\n  ⚠️  schema_pass=False — aborting pipeline for this article.")
        print(f"  raw_llm_output (first 300 chars): {interpreter_result['raw_llm_output'][:300]}")
        # Log the failed session (no candidates, no user choice)
        log_session(news_obj, rag_result, interpreter_result,
                    candidates=None, user_choice=None)
        return  # Fail-Fast

    parsed = interpreter_result["parsed_json"]
    print(f"\n  ✅ Interpretation OK")
    print(f"     Stance: {parsed.get('Key stance / opinion', '?')}")
    print(f"     Summary: {parsed.get('What happened', '')[:100]}...")

    # ── Step 3: Generate Posts ──
    candidates = generate_posts(interpreter_result)
    if not candidates:
        print("  ⚠️  No candidates generated.")
        log_session(news_obj, rag_result, interpreter_result,
                    candidates=None, user_choice=None)
        return

    # ── Step 4: User selection (CLI for now, UI later) ──
    print(f"\n  ── Generated Posts ──")
    for c in candidates:
        print(f"\n  [{c['candidate_id']}] {c['text'][:200]}...")

    print(f"\n  {'─'*40}")
    user_choice = input("  Choose A / B / C (or press Enter to skip): ").strip().upper()
    if user_choice not in ("A", "B", "C"):
        user_choice = "SKIP"

    # ── Step 5: Log ──
    log_session(news_obj, rag_result, interpreter_result, candidates, user_choice)


# ╔══════════════════════════════════════════════════════════════╗
# ║              BATCH MODE  (backward-compatible wrapper)       ║
# ╚══════════════════════════════════════════════════════════════╝

def run_batch_pipeline(input_path: str,
                       output_path: str,
                       config: dict,
                       start_index: int = 0,
                       num_articles: int | None = None) -> None:
    """
    Batch-process articles through retrieve → interpret (no generate_posts
    in batch mode — keeps the original interpreter artefact format for
    backward compatibility).
    """
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f).get("articles", [])

    total = len(articles)
    if num_articles is None:
        num_articles = total
    if start_index + num_articles > total:
        num_articles = total - start_index

    print(f"\n{'='*60}")
    print(f"  BATCH: {num_articles} articles → {output_path}")
    print(f"  Retriever: {config['retriever']} | Mode: {config['reasoning_mode']} | top_k: {config['top_k_beliefs']}")
    print(f"{'='*60}\n")

    beliefs = load_beliefs()
    results, ok, fail = [], 0, 0

    # Resume support
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            old = json.load(f)
        results = old.get("results", [])
        ok = sum(1 for r in results if r.get("parsed_json") is not None)
        fail = len(results) - ok
        if results:
            print(f"📂 Resuming: {len(results)} done ({ok}✅ {fail}❌)\n")

    for i in range(len(results), num_articles):
        idx = start_index + i
        art = articles[idx]
        title = art.get("name", art.get("title", "Unknown"))[:90]

        news_obj = {
            "news_id": art.get("id", f"art-{idx:03d}"),
            "title": title,
            "excerpt": art.get("full_text", art.get("description", "")),
        }

        print(f"[{i+1}/{num_articles}] #{idx} — {title}")

        for attempt in range(3):
            try:
                rag_result = retrieve(news_obj, beliefs,
                                      config["retriever"], config["top_k_beliefs"])
                interpreter_result = interpret(news_obj, rag_result,
                                               config["reasoning_mode"])

                if interpreter_result["schema_pass"]:
                    results.append({
                        "news_id": news_obj["news_id"],
                        "title": title,
                        "parsed_json": interpreter_result["parsed_json"],
                        "rag_result": rag_result,
                        "reasoning_mode": config["reasoning_mode"],
                        "schema_pass": True,
                    })
                    ok += 1
                    stance = interpreter_result["parsed_json"].get("Key stance / opinion", "?")
                    top_belief = rag_result["beliefs"][0] if rag_result["beliefs"] else {}
                    print(f"  ✅ {stance} | belief: {top_belief.get('belief_text', '')[:50]}...")
                else:
                    if attempt < 2:
                        print(f"  ⚠️  schema_pass=False, retry {attempt+1}/3...")
                        time.sleep(5)
                        continue
                    fail += 1
                    results.append({
                        "news_id": news_obj["news_id"],
                        "title": title,
                        "parsed_json": None,
                        "schema_pass": False,
                        "raw_llm_output": interpreter_result.get("raw_llm_output", ""),
                    })
                    print(f"  ❌ schema_pass=False after 3 retries")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  ⚠️  {e} — retry {attempt+1}/3 in 5s...")
                    time.sleep(5)
                else:
                    fail += 1
                    results.append({
                        "news_id": news_obj["news_id"],
                        "title": title,
                        "parsed_json": None,
                        "schema_pass": False,
                        "error": str(e),
                    })
                    print(f"  ❌ {e}")
                    break

        # Save after each article
        out = {
            "batch_metadata": {
                "total_requested": num_articles,
                "successful": ok,
                "failed": fail,
                "start_index": start_index,
                "retriever": config["retriever"],
                "reasoning_mode": config["reasoning_mode"],
                "top_k_beliefs": config["top_k_beliefs"],
                "generated_at": datetime.now().isoformat(),
            },
            "results": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"  DONE ✅ {ok}/{num_articles} succeeded  ❌ {fail} failed")
    print(f"  📁 {os.path.abspath(output_path)}")
    print(f"{'='*60}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                        CLI                                   ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KickstartAI Pipeline (refactored)")
    parser.add_argument("--batch", type=int, default=0,
                        help="Number of articles to batch-process (0 = interactive single)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index for batch")
    parser.add_argument("--retriever", choices=["bm25", "embedding", "hybrid"],
                        default=CONFIG["retriever"])
    parser.add_argument("--mode", choices=["cot", "flat"],
                        default=CONFIG["reasoning_mode"])
    parser.add_argument("--top-k", type=int, default=CONFIG["top_k_beliefs"],
                        help="Number of top beliefs to retrieve")
    parser.add_argument("--input", default=None,
                        help="Custom input JSON file path")
    parser.add_argument("--output", default="batch_pipeline_results.json")
    parser.add_argument("--dir", default="output")
    args = parser.parse_args()

    # Merge CLI args into config
    run_config = {
        "retriever": args.retriever,
        "reasoning_mode": args.mode,
        "top_k_beliefs": args.top_k,
    }

    if args.batch > 0:
        input_path = args.input or "scanner_output.json"
        output_path = os.path.join(args.dir, args.output)
        run_batch_pipeline(
            input_path=input_path,
            output_path=output_path,
            config=run_config,
            start_index=args.start,
            num_articles=args.batch,
        )
    else:
        # ── Interactive single-article mode ──
        input_path = args.input or "Interpreter/scanner_output.json"
        with open(input_path, "r", encoding="utf-8") as f:
            articles = json.load(f).get("articles", [])
        if not articles:
            print("No articles found in input file.")
            sys.exit(1)

        art = articles[args.start]
        news_obj = {
            "news_id": art.get("id", "art-000"),
            "title": art.get("name", art.get("title", "Unknown")),
            "excerpt": art.get("full_text", art.get("description", "")),
        }
        beliefs = load_beliefs()
        run_pipeline_for_one_news(news_obj, beliefs, run_config)


# python3 -m Interpreter.pipeline 