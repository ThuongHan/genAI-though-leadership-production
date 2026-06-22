"""
KickstartAI Strategy Interpreter — Standalone Runner
=====================================================
Reads scanner_output.json + belief_repository.json,
runs RAG → CoT/Flat prompt → LLM → Generator contract output.

Usage:
  # Single article
  python run_interpreter.py

  # Batch 55 articles
  python run_interpreter.py --batch 55

  # Batch with custom range
  python run_interpreter.py --batch 20 --start 10

  # Use embedding retriever
  python run_interpreter.py --batch 55 --retriever embedding

  # Flat reasoning mode
  python run_interpreter.py --batch 55 --mode flat
"""

import json
import os
import sys
import time
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal

import numpy as np
from openai import OpenAI
# TF-IDF removed → replaced with BM25 + Hybrid (see Step 2)
from sklearn.metrics.pairwise import cosine_similarity

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

SCANNER_OUTPUT_PATH = os.getenv("SCANNER_OUTPUT_PATH", "scanner_output.json")
BELIEF_REPOSITORY_PATH = os.getenv("BELIEF_REPOSITORY_PATH", "belief_repository.json")
INTERPRETER_OUTPUT_PATH = os.getenv("INTERPRETER_OUTPUT_PATH", "interpreter_to_generator.json")
DEFAULT_INPUT_PATH = os.getenv("INTERPRETER_INPUT_PATH", "scanner_output.json")

REQUIRED_CONTRACT_KEYS = [
    "What happened",
    "Why does it matter (globally and NL)",
    "Why does it matter for KickstartAI",
    "Key stance / opinion",
    "Supporting arguments",
]

VALID_STANCES = {"Supportive", "Critical", "Neutral", "Cautious"}


@dataclass
class Article:
    title: str
    full_text: str
    raw: Dict[str, Any]


@dataclass
class RAGResult:
    belief: str
    score: float


# ═══════════════════════════════════════════════════════════
# CLIENT
# ═══════════════════════════════════════════════════════════

def create_client() -> OpenAI:
    token = os.getenv("UVA_API_TOKEN")
    if not token:
        raise RuntimeError("UVA_API_TOKEN not set. Run: $env:UVA_API_TOKEN='your-key'")
    return OpenAI(api_key=token, base_url="https://llmproxy.uva.nl/v1/")

client = create_client()

# ═══════════════════════════════════════════════════════════
# STEP 1: INGEST
# ═══════════════════════════════════════════════════════════

def ingest_data(scanner_path=None, belief_path=None, article_index=0):
    if scanner_path is None:
        for p in [os.getenv("SCANNER_OUTPUT_PATH"), "scanner_output.json"]:
            if p and os.path.exists(p):
                scanner_path = p; break
        if scanner_path is None:
            raise FileNotFoundError("scanner_output.json not found")
    with open(scanner_path, "r", encoding="utf-8") as f:
        scanner_data = json.load(f)
    articles = scanner_data.get("articles", [])
    if article_index >= len(articles):
        raise ValueError(f"Index {article_index} out of range ({len(articles)} articles)")
    article_dict = articles[article_index]

    if belief_path is None:
        for p in [os.getenv("BELIEF_REPOSITORY_PATH"), "belief_repository.json"]:
            if p and os.path.exists(p):
                belief_path = p; break
        if belief_path is None:
            raise FileNotFoundError("belief_repository.json not found")
    with open(belief_path, "r", encoding="utf-8") as f:
        beliefs = json.load(f)

    article = Article(
        title=article_dict.get("name") or article_dict.get("title", "Unknown"),
        full_text=article_dict.get("full_text") or "",
        raw=article_dict)
    return article, beliefs


def step1_ingest_data(article_index=0, scanner_path=None):
    article, beliefs = ingest_data(article_index=article_index, scanner_path=scanner_path)
    return article.raw, beliefs

# ═══════════════════════════════════════════════════════════
# STEP 2: RAG
# ═══════════════════════════════════════════════════════════

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, article: Article, beliefs: List[str]) -> RAGResult:
        raise NotImplementedError


class BM25Retriever(BaseRetriever):
    """Okapi BM25 retriever — improved keyword-based retrieval (k1=1.5, b=0.75)."""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    @staticmethod
    def _tokenize(text):
        return re.findall(r'\w+', text.lower())

    def retrieve(self, article, beliefs):
        query_text = f"{article.title}\n{article.full_text}"
        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            raise ValueError("Empty query after tokenisation")

        belief_texts = [b["belief"] if isinstance(b, dict) else str(b) for b in beliefs]
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

        best_idx = int(scores.argmax())
        return RAGResult(belief=belief_texts[best_idx], score=float(scores[best_idx]))


class EmbeddingRetriever(BaseRetriever):
    def __init__(self, model="text-embedding-3-large"):
        self.model = model
        self._client = None

    @property
    def emb_client(self):
        if self._client is None:
            token = os.getenv("UVA_API_TOKEN")
            self._client = OpenAI(api_key=token, base_url="https://llmproxy.uva.nl/v1/")
        return self._client

    def _embed(self, texts):
        resp = self.emb_client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def retrieve(self, article, beliefs):
        query_text = f"{article.title}\n{article.full_text[:2000]}"
        belief_texts = [b["belief"] if isinstance(b, dict) else str(b) for b in beliefs]
        all_texts = [query_text] + belief_texts
        embeddings = self._embed(all_texts)
        query_vec = np.array(embeddings[0]).reshape(1, -1)
        belief_vecs = np.array(embeddings[1:])
        sims = cosine_similarity(query_vec, belief_vecs).flatten()
        best_idx = int(sims.argmax())
        return RAGResult(belief=belief_texts[best_idx], score=float(sims[best_idx]))


class HybridRetriever(BaseRetriever):
    """Sparse + dense hybrid: BM25 (keyword) ⊗ Embedding (semantic)."""
    def __init__(self, alpha=0.5, embed_model="text-embedding-3-large"):
        self.alpha = alpha
        self._bm25 = BM25Retriever()
        self._emb = EmbeddingRetriever(model=embed_model)

    def retrieve(self, article, beliefs):
        bm25_result = self._bm25.retrieve(article, beliefs)
        emb_result = self._emb.retrieve(article, beliefs)
        combined = self.alpha * bm25_result.score + (1.0 - self.alpha) * emb_result.score
        if bm25_result.score >= emb_result.score:
            return RAGResult(belief=bm25_result.belief, score=combined)
        else:
            return RAGResult(belief=emb_result.belief, score=combined)


def step2_rag_retrieval(article_dict, beliefs):
    article = Article(
        title=article_dict.get("name") or article_dict.get("title", "Unknown"),
        full_text=article_dict.get("full_text", ""), raw=article_dict)
    return BM25Retriever().retrieve(article, beliefs)

# ═══════════════════════════════════════════════════════════
# STEP 3: PROMPT
# ═══════════════════════════════════════════════════════════

MULTILINGUAL_RULE = """
[MULTILINGUAL INPUT HANDLING]
The news article may be in English, Dutch, or both.
- If Dutch: process internally in Dutch, OUTPUT ALL JSON VALUES IN ENGLISH.
- Preserve Dutch-specific context (NL policy, ecosystem, EU AI Act references).
- CRITICAL: Final JSON must always be in English, regardless of input language.
"""


def build_cot_prompt(article, retrieved_belief):
    return f"""
You are the Chief Strategic Interpreter for KickstartAI.
{MULTILINGUAL_RULE}

[EXTERNAL FACT: NEWS ARTICLE]
Title: {article.title}
Content: {article.full_text[:1500]}

[INTERNAL BRAIN: RETRIEVED KICKSTARTAI BELIEF]
\"\"\"{retrieved_belief}\"\"\"

[CHAIN OF THOUGHT]
Step 1 (Fact Extraction): What happened? Global and NL impact?
Step 2 (Strategic Alignment): How does this align with our belief?
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


def build_flat_prompt(article, retrieved_belief):
    return f"""
You are the Chief Strategic Interpreter for KickstartAI.
{MULTILINGUAL_RULE}

[NEWS ARTICLE] Title: {article.title}
Content: {article.full_text[:1500]}

[INTERNAL BELIEF] \"\"\"{retrieved_belief}\"\"\"

[OUTPUT — ONLY JSON, all English]
{{
    "What happened": "Brief summary (50-200 words, English)",
    "Why does it matter (globally and NL)": "Global + NL context (80-250 words, English)",
    "Why does it matter for KickstartAI": "Connection to belief (60-200 words, English)",
    "Key stance / opinion": "Supportive / Critical / Neutral / Cautious",
    "Supporting arguments": ["Arg 1", "Arg 2", "Arg 3"]
}}
""".strip()


def step3_build_prompt(article_dict, retrieved_belief, mode="cot"):
    article = Article(
        title=article_dict.get("name") or article_dict.get("title", "Unknown"),
        full_text=article_dict.get("full_text", ""), raw=article_dict)
    if mode == "cot":
        return build_cot_prompt(article, retrieved_belief)
    elif mode == "flat":
        return build_flat_prompt(article, retrieved_belief)
    raise ValueError(f"Unknown mode: {mode}")

# ═══════════════════════════════════════════════════════════
# STEP 4: LLM + CONTRACT
# ═══════════════════════════════════════════════════════════

def validate_generator_contract(final_json):
    errors = []
    for key in REQUIRED_CONTRACT_KEYS:
        if key not in final_json:
            errors.append(f"Missing: '{key}'")
        elif not final_json[key] or (isinstance(final_json[key], str) and len(final_json[key].strip()) < 5):
            errors.append(f"Empty: '{key}'")
    if errors:
        raise ValueError("Contract violation:\n  " + "\n  ".join(errors))
    stance = final_json.get("Key stance / opinion", "")
    if stance not in VALID_STANCES:
        raise ValueError(f"Invalid stance: '{stance}'. Must be: {', '.join(sorted(VALID_STANCES))}")
    args = final_json.get("Supporting arguments", [])
    if not isinstance(args, list) or len(args) != 3:
        raise ValueError(f"Supporting arguments must be a list of exactly 3, got {len(args)}")
    for i, a in enumerate(args):
        if not isinstance(a, str) or len(a.strip()) < 10:
            errors.append(f"Arg {i+1} too short")
    if errors:
        raise ValueError("Argument errors:\n  " + "\n  ".join(errors))
    return final_json


def build_generator_payload(interpretation_json, article_metadata, matched_belief, rag_score, reasoning_mode):
    return {
        "interpretation": {
            "What happened": interpretation_json["What happened"],
            "Why does it matter (globally and NL)": interpretation_json["Why does it matter (globally and NL)"],
            "Why does it matter for KickstartAI": interpretation_json["Why does it matter for KickstartAI"],
            "Key stance / opinion": interpretation_json["Key stance / opinion"],
            "Supporting arguments": interpretation_json["Supporting arguments"],
        },
        "metadata": {
            "source_article": {
                "title": article_metadata.get("name") or article_metadata.get("title", "Unknown"),
                "url": article_metadata.get("url", ""),
                "source": article_metadata.get("source", {}).get("name", "Unknown"),
                "published_at": article_metadata.get("published_at", ""),
            },
            "interpreter": {
                "matched_belief": matched_belief,
                "rag_score": round(rag_score, 4),
                "reasoning_mode": reasoning_mode,
                "contract_version": "1.0.0",
                "generated_at": datetime.now().isoformat(),
            },
        },
    }


def step4_execute_llm(prompt_string, validate_for_generator=True):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a KickstartAI strategic analyst. Output ONLY valid JSON. All text in English. Stance MUST be: Supportive, Critical, Neutral, or Cautious."},
                {"role": "user", "content": prompt_string},
            ],
            temperature=0.3,
        )
        final_json = json.loads(response.choices[0].message.content)
        if validate_for_generator:
            final_json = validate_generator_contract(final_json)
        return final_json
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  API error: {e}")
        return None

# ═══════════════════════════════════════════════════════════
# SINGLE ARTICLE
# ═══════════════════════════════════════════════════════════

def run_interpreter(reasoning_mode="cot", retriever_mode="embedding", article_index=0, output_dir=".", save_text=False, input_path=None):
    article_dict, beliefs = step1_ingest_data(article_index=article_index, scanner_path=input_path)
    article = Article(title=article_dict.get("name", "Unknown"),
                      full_text=article_dict.get("full_text", ""), raw=article_dict)

    if retriever_mode == "embedding":
        rag_result = EmbeddingRetriever().retrieve(article, beliefs)
    elif retriever_mode == "hybrid":
        rag_result = HybridRetriever().retrieve(article, beliefs)
    else:  # "bm25"
        rag_result = BM25Retriever().retrieve(article, beliefs)

    final_prompt = step3_build_prompt(article_dict, rag_result.belief, mode=reasoning_mode)
    interpretation = step4_execute_llm(final_prompt, validate_for_generator=True)

    if not interpretation:
        return None

    return build_generator_payload(
        interpretation_json=interpretation, article_metadata=article_dict,
        matched_belief=rag_result.belief, rag_score=rag_result.score,
        reasoning_mode=reasoning_mode)

# ═══════════════════════════════════════════════════════════
# BATCH MODE
# ═══════════════════════════════════════════════════════════

def run_batch(start_index=0, num_articles=55, reasoning_mode="cot",
              retriever_mode="embedding", output_dir=".",
              output_filename="batch_55_interpretations.json",
              input_path=None):
    input_file = input_path or "scanner_output.json"
    with open(input_file, "r", encoding="utf-8") as f:
        all_articles = json.load(f).get("articles", [])
    total = len(all_articles)

    if start_index + num_articles > total:
        print(f"⚠️  Only {total} articles available, adjusting to {total - start_index}")
        num_articles = total - start_index

    print(f"\n{'='*60}")
    print(f"  BATCH: {num_articles} articles → {output_filename}")
    print(f"  Range: #{start_index} → #{start_index + num_articles - 1}")
    print(f"  Mode: {reasoning_mode} | Retriever: {retriever_mode}")
    print(f"{'='*60}\n")

    # Resume support
    out_path = os.path.join(output_dir, output_filename)
    results, ok, fail = [], 0, 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            old = json.load(f)
        results = old.get("results", [])
        ok = sum(1 for r in results if r.get("interpretation") is not None)
        fail = len(results) - ok
        if len(results) > 0:
            print(f"📂 Resuming: {len(results)} done ({ok}✅ {fail}❌)\n")

    for i in range(len(results), num_articles):
        idx = start_index + i
        title = all_articles[idx].get("name", "")[:90]

        print(f"[{i+1}/{num_articles}] #{idx} — {title}")

        for attempt in range(3):
            try:
                payload = run_interpreter(
                    reasoning_mode=reasoning_mode, retriever_mode=retriever_mode,
                    article_index=idx, output_dir=output_dir, input_path=input_file)
                if payload:
                    results.append(payload); ok += 1
                    print(f"  ✅ {payload['interpretation']['Key stance / opinion']} | "
                          f"belief: {payload['metadata']['interpreter']['matched_belief'][:50]}...")
                else:
                    if attempt < 2:
                        print(f"  ⚠️  Retry {attempt+1}/3 in 5s...")
                        time.sleep(5); continue
                    fail += 1
                    results.append({"interpretation": None, "metadata": {
                        "source_article": {"title": all_articles[idx].get("name","")},
                        "interpreter": {"error": "LLM None after 3 retries"}}})
                    print(f"  ❌ Failed")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  ⚠️  {e} — retry {attempt+1}/3 in 5s...")
                    time.sleep(5)
                else:
                    fail += 1
                    results.append({"interpretation": None, "metadata": {
                        "source_article": {"title": all_articles[idx].get("name","")},
                        "interpreter": {"error": str(e)}}})
                    print(f"  ❌ {e}")
                    break

        # Save after each article
        out = {"batch_metadata": {"total_requested": num_articles,
                "successful": ok, "failed": fail, "start_index": start_index,
                "reasoning_mode": reasoning_mode, "retriever_mode": retriever_mode,
                "contract_version": "1.0.0", "generated_at": datetime.now().isoformat()},
               "results": results}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)  # Rate limit buffer

    print(f"\n{'='*60}")
    print(f"  DONE ✅ {ok}/{num_articles} succeeded  ❌ {fail} failed")
    print(f"  📁 {os.path.abspath(out_path)}")
    print(f"{'='*60}")
    return out


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KickstartAI Interpreter")
    parser.add_argument("--batch", type=int, default=0, help="Number of articles to batch-process")
    parser.add_argument("--start", type=int, default=0, help="Start index for batch")
    parser.add_argument("--mode", choices=["cot", "flat"], default="cot")
    parser.add_argument("--retriever", choices=["bm25", "embedding", "hybrid"], default="embedding")
    parser.add_argument("--output", default="batch_55_interpretations.json")
    parser.add_argument("--dir", default=".")
    parser.add_argument("--input", default=None, help="Custom input JSON file path (default: scanner_output.json)")
    args = parser.parse_args()

    if args.batch > 0:
        run_batch(start_index=args.start, num_articles=args.batch,
                  reasoning_mode=args.mode, retriever_mode=args.retriever,
                  output_dir=args.dir, output_filename=args.output,
                  input_path=args.input)
    else:
        # Single
        payload = run_interpreter(reasoning_mode=args.mode, retriever_mode=args.retriever,
                                  article_index=args.start, output_dir=args.dir,
                                  input_path=args.input)
        if payload:
            out_path = os.path.join(args.dir, INTERPRETER_OUTPUT_PATH)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
            print(f"\n✅ Saved to {out_path}")
