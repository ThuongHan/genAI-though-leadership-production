# Interpreter — RAG Upgrade (BM25 / Embedding / Hybrid)

## What Changed

### 1. TF-IDF → BM25 (Interpreter.py + pipeline.py)
- Removed `TfidfVectorizer` / `_TfidfRetriever`
- Added `BM25Retriever` (Okapi BM25 from scratch, numpy only, k1=1.5, b=0.75)
- Added `HybridRetriever` (min-max normalize + alpha-weighted fusion of BM25 + Embedding)
- All three retrievers available via CLI: `--retriever {bm25, embedding, hybrid}`
- Default: `embedding`

### 2. New Belief Repository
- `belief_repository_sensemaking.json` — 384 beliefs (SNS-B001 to SNS-B384)
- Replaces old `belief_repository.json` (110 beliefs)
- `pipeline.py` `BELIEF_REPOSITORY_PATH` updated accordingly

### 3. Auto-Scoring (NEW: scorer.py)
- LLM-as-Judge: scores each interpretation on 5 dimensions (0-10)
- `stance_accuracy`, `argument_quality`, `belief_alignment`, `factual_precision`, `nl_perspective`
- Weighted overall score
- Standalone CLI or integrated via `pipeline.py --score`

### 4. Eval Updates (eval_retrieval.py)
- Retriever list: `["bm25", "embedding", "hybrid"]`
- Extended metrics: Hit@1/3/5, NDCG, MRR (from eval_retrieval.py in thesis)

## API Integration Notes

The web frontend's `/api/interpret` endpoint needs two extra fields in `ArticleBody`:

```python
retriever_type: str = "embedding"  # "bm25" | "embedding" | "hybrid"
reasoning_mode: str = "cot"        # "cot" | "flat"
```

And pass them through:
```python
rag    = retrieve(news_obj, beliefs, retriever_type=body.retriever_type, top_k=3)
result = interpret(news_obj, rag, reasoning_mode=body.reasoning_mode)
```

## Files to Replace in the Repo

| File | Action |
|------|--------|
| `Interpreter/Interpreter.py` | Replace |
| `Interpreter/pipeline.py` | Replace |
| `Interpreter/scorer.py` | Add (new) |
| `Interpreter/belief_repository_sensemaking.json` | Add (new) |
| `Interpreter/eval_retrieval.py` | Replace |
| `Interpreter/diagnose.py` | No change (unchanged) |
