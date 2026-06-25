# Generator

Takes the structured output of the Interpreter and produces LinkedIn posts in KickstartAI's voice. Supports few-shot retrieval, dual-judge scoring, and an iterative refinement loop.

---

## How it works

```
Interpreter output (what happened / why it matters / stance / arguments)
        │
        ▼
  [ PostGenerator ]
        ├── embed query text
        ├── retrieve k similar LinkedIn posts (few-shot)
        ├── build prompt from config template + few-shot block
        └── LLM call → parse GeneratedPosts (3 post variants)
        │
        ▼
  [ Judge panel ]  (optional — used by UI auto-refine and experiments)
        ├── J1: Claude Opus 4.8
        └── J2: GPT-5.5
        │    each scores 6 dimensions (1–5)
        ▼
  [ Refinement loop ]  (regeneration/refine.py)
        ├── collect feedback from failing dimensions
        ├── regenerate with Claude Sonnet 4.6
        └── repeat up to MAX_ITER=3 times; keep best-scoring version
```

---

## Directory structure

```
Generator/
├── post_generator.py           # Main generation class (PostGenerator)
├── prompt_builder.py           # Fills config template with event + few-shot block
├── schemas/
│   └── generator_schema.py     # Pydantic models: GeneratedPosts, LinkedInPosts
│
├── config/
│   ├── zeroshot-prompt.md          # Pre-prompt: basic zero-shot template
│   ├── post-reformulated-prompt.md # Post-prompt: reformulated/improved template
│   └── eval-prompt.md              # Judge evaluation template
│
├── utils/
│   ├── llm/
│   │   ├── registry.py         # get_llm(model_name) factory
│   │   ├── claude.py           # ClaudeLLM wrapper (LangChain)
│   │   ├── gpt.py              # GPTLLM wrapper (LangChain)
│   │   └── base.py
│   ├── embedder.py             # OpenAI embeddings for few-shot retrieval
│   ├── few_shot.py             # FewShotPost: load + retrieve similar posts
│   └── linkedin_data_processing.py
│
├── judge/
│   ├── runner.py               # Shared judge logic: prompt building, JSON parsing
│   ├── judge_opus.py           # Standalone Claude Opus judge script
│   └── judge_gpt5-5.py         # Standalone GPT-5.5 judge script
│
├── regeneration/
│   ├── refine.py               # Dual-judge refinement loop (also used by the UI)
│   └── thesis-regeneration.md  # Regeneration prompt template
│
├── data/
│   ├── LinkedIn_processed_data.json    # Few-shot post library
│   └── Interpreter_output/
│       └── 40_blog_posts.json          # 40 pre-run interpreter outputs (for experiments)
│
└── Experiments/                # Thesis experiments — see section below
    ├── data/
    │   ├── sample_61.xlsx              # 60 generated posts (4 conditions × 15 topics)
    │   └── UvA Expert Voice - Output annotation.xlsx  # Human expert annotations
    ├── experiment1/
    │   ├── generate_posts.py   # Generate the 60 posts (2×2 conditions, Sonnet)
    │   └── run_human_scores.py # Table 1 & 2: human baseline scores
    ├── experiment2/
    │   └── run.py              # Table 2: LLM-judge vs. human inter-rater agreement
    └── experiment3/
        ├── run.py              # FS-Post regeneration comparison
        └── run_costs/
            └── run.py          # Computational cost: generation vs. regeneration
```

---

## Usage

### As a module (via the API or scripts)

```python
from backend.Generator.post_generator import PostGenerator

gen = PostGenerator(
    model="claude-sonnet-4-6",                          # or any GPT model
    config_path="path/to/post-reformulated-prompt.md",  # optional override
)
result = gen.generate(
    interpreter_output={
        "what_happened":   "...",
        "why_relevance":   "...",
        "why_kickstartai": "...",
        "stance":          "...",
        "arguments":       [...],
    },
    k_posts=1,           # number of few-shot examples to retrieve
    use_few_shot=True,
    save=False,
)
post_text = result["posts"][0].content
```

### Standalone refinement loop

```bash
# From pipeline/ root — runs the dual-judge loop on example_generated/generated_posts.json
python3 -m backend.Generator.regeneration.refine
```

### Standalone judges

```bash
python3 -m backend.Generator.judge.judge_opus
python3 -m backend.Generator.judge.judge_gpt5-5
```

---

## Models

| Role | Model |
|------|-------|
| Generator | `claude-sonnet-4-6` (default) |
| Judge J1 | `claude-opus-4-8` |
| Judge J2 | `gpt-5.5` |

Configurable in [regeneration/refine.py](regeneration/refine.py) (`GENERATOR_MODEL`) and per-experiment judge dictionaries.

Available alternatives:
- Claude: `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`, `claude-opus-4-8`
- GPT: `gpt-4o`, `gpt-4.1`, `gpt-5`, `gpt-5.1`, `gpt-5.5`

---

## Evaluation dimensions

Both judges score every post on six dimensions (1–5 scale):

| Dimension | What it measures |
|-----------|-----------------|
| `tone_of_voice` | Authentic human expert voice, not robotic |
| `language_and_style` | Clear, direct, suited to LinkedIn |
| `coherence_readability` | Logical flow between ideas |
| `discourse_structure` | Absence of contrastive / from-to / this-that violations |
| `specificity` | Concrete details, not vague generalisations |
| `historical_similarity` | Matches KickstartAI's historical post style |

A post passes auto-refinement when every dimension scores ≥ 4 from every judge (`PASS_THRESHOLD = 4`).

---

## Thesis experiments

All experiments read from `Experiments/data/` and write results back there or to per-experiment `results/` folders. Run everything from the `pipeline/` root.

### Experiment 1 — Generate the sample

Produces `sample_61.xlsx`: 60 posts across a 2×2 design (zero-shot vs. few-shot × pre-prompt vs. post-prompt), 15 topics per condition, generator = Sonnet.

```bash
python3 -m backend.Generator.Experiments.experiment1.generate_posts
```

### Experiment 1 — Human baseline scores

Reads human expert annotations from `UvA Expert Voice - Output annotation.xlsx` and computes mean dimension scores and violation proportions per condition.

```bash
python3 -m backend.Generator.Experiments.experiment1.run_human_scores
```

### Experiment 2 — Inter-rater agreement

Runs both LLM judges on the annotated posts and computes exact and adjacent (±1) agreement against the human scores per dimension.

```bash
python3 -m backend.Generator.Experiments.experiment2.run
```

### Experiment 3 — Regeneration comparison

Takes the 15 FS-Post posts and runs the dual-judge refinement loop on each. Outputs original vs. regenerated post side-by-side.

```bash
python3 -m backend.Generator.Experiments.experiment3.run
```

### Experiment 3 — Computational cost

Measures wall-clock time, API call counts, token usage, and estimated USD cost for generation vs. regeneration across 15 topics.

```bash
python3 -m backend.Generator.Experiments.experiment3.run_costs.run
```
