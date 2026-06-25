# KickstartAI Thought Leadership Generator

A full-stack pipeline that monitors AI news, interprets articles through KickstartAI's belief system, and generates ready-to-publish LinkedIn posts — all from a single web UI.

---

## How it works

```
News sources (RSS / NewsAPI / arXiv / scraper)
        │
        ▼
   [ Scanner ]  — collects & scores articles by relevance to KickstartAI's focus
        │
        ▼ top-5 articles
   [ Interpreter ]  — RAG over belief repository → LLM reasoning → structured output
        │              (what happened / why it matters / KickstartAI angle / stance)
        ▼
   [ Generator ]  — few-shot + embedding similarity → 3 LinkedIn post drafts
        │              (Claude or GPT, configurable)
        ▼
   [ Judge / Refiner ]  — auto or human feedback loop → scored + refined final post
```

The **Belief System** is a pre-built knowledge base of KickstartAI's organisational beliefs (mission, strategy, values, stance, domain knowledge), extracted from internal documents using three complementary LLM methods (LLM, Sensemaking, MultiModel). It is the grounding layer the Interpreter uses at inference time.

---

## Project structure

1. The thesis experiments for the Generator are in the backend/Generator/Experiments.

```
pipeline/
├── backend/
│   ├── api.py                  # FastAPI server — all endpoints
│   ├── secrets/
│   │   └── .env                # Single API key file for all components
│   │
│   ├── Scanner/                # News collection & scoring
│   │   ├── run_scanner.py
│   │   ├── run_filter.py
│   │   └── main_scanner/
│   │       ├── settings.py     # Source list, keywords, tunables
│   │       ├── sources/        # RSS, NewsAPI, arXiv, scraper
│   │       ├── weighted_filter.py
│   │       └── ...
│   │
│   ├── Interpreter/            # Article → structured interpretation
│   │   ├── Interpreter.py      # BM25 + hybrid RAG, CoT/Flat LLM
│   │   ├── pipeline.py
│   │   ├── scorer.py
│   │   └── diagnose.py
│   │
│   ├── Generator/              # Structured interpretation → LinkedIn post
│   │   ├── post_generator.py
│   │   ├── prompt_builder.py
│   │   ├── judge/              # GPT-5.5 / Claude Opus scoring judges
│   │   ├── regeneration/       # Feedback-based refinement
│   │   ├── schemas/
│   │   ├── utils/
│   │   │   ├── llm/            # Claude + GPT wrappers & registry
│   │   │   ├── embedder.py     # OpenAI embeddings for few-shot retrieval
│   │   │   └── few_shot.py
│   │   └── Experiments/        # Thesis experiments (human scores, inter-rater agreement, cost analysis)
│   │       ├── experiment1/    # Human-annotation baseline tables
│   │       ├── experiment2/    # LLM-judge vs. human inter-rater agreement (Table 2: exact & adjacent)
│   │       └── experiment3/    # FS-Post regeneration comparison + computational cost analysis
│   │
│   └── Belief_System/          # KickstartAI belief extraction & repository
│       ├── extractors/         # LLM, Sensemaking, MultiModel extractors
│       ├── belief_system/      # Deduplication & canonical structuring
│       ├── embeddings/         # Belief embedding index
│       ├── config/settings.py
│       └── data/
│           ├── raw/            # Source documents
│           └── processed/      # Extracted & structured beliefs
│
├── frontend/                   # Svelte 5 + Vite web UI
│   └── src/
│       └── App.svelte
│
├── venv/                       # Python virtual environment
├── requirements.txt            # All Python dependencies
└── .gitignore
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- A Python virtual environment at `pipeline/venv/`

---

## Setup

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd pipeline
```

### 2. Create the virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API keys

Create `backend/secrets/.env` (this file is gitignored):

```env
# Anthropic / Claude
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
OPENAI_API_KEY=sk-...

# NewsAPI  (https://newsapi.org — free tier works)
NEWSAPI_KEY=...
```

All four components (Scanner, Interpreter, Generator, Belief System) read from this single file — no per-component `.env` files are needed.

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running the application

Open **two terminals** from the `pipeline/` directory.

**Terminal 1 — backend:**

```bash
source venv/bin/activate
uvicorn backend.api:app --reload --port 8000
```

**Terminal 2 — frontend:**

```bash
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Using the UI

The interface walks you through the pipeline in five steps:

### Step 1 — Scan

Click **Start fresh scan** to collect the latest AI news (5–10 min) or **Use cached articles** if a scan ran in the last 24 hours. The scanner hits RSS feeds, NewsAPI, arXiv, and scraped sources, then the weighted filter scores and ranks the top 5.

### Step 2 — Select an article

Browse the top-5 ranked articles. Each card shows the source, publish date, and relevance score. Click one to proceed.

### Step 3 — Interpret

The Interpreter runs BM25 + hybrid RAG over the belief repository to find the most relevant KickstartAI beliefs, then sends the article + beliefs to an LLM for structured reasoning:

- What happened
- Why it matters (globally and in the Netherlands)
- Why it matters for KickstartAI specifically
- Key stance and supporting arguments

### Step 4 — Generate posts

Three LinkedIn post drafts are generated using few-shot examples retrieved by semantic similarity. You can select which post to work with.

### Step 5 — Refine and publish

Two refinement paths:

- **Human feedback** — type what to change and get a revised draft
- **Auto-refine** — a judge LLM scores the post on six dimensions (tone, style, coherence, structure, specificity, historical similarity) and refines it automatically

Copy the final post to the clipboard when done.

---

## Running components standalone

Each component can also be run independently for development or batch processing.

### Scanner

```bash
cd backend/Scanner
python run_scanner.py          # collect articles → data/scans/scanner_output_*.json
python run_filter.py           # score and rank  → data/filter/weighted_top5_*.json
```

Configure news sources, keywords, and API keys in `main_scanner/settings.py`.

### Interpreter

```bash
cd backend/Interpreter

# Single article (index 0 from scanner_output.json)
python Interpreter.py

# Batch — 55 articles
python Interpreter.py --batch 55

# Use embedding retriever instead of BM25
python Interpreter.py --batch 55 --retriever embedding

# Flat reasoning mode (no chain-of-thought)
python Interpreter.py --batch 55 --mode flat
```

Expects `scanner_output.json` and `belief_repository.json` in the working directory, or set `SCANNER_OUTPUT_PATH` / `BELIEF_REPOSITORY_PATH` environment variables.

### Generator

```python
from Generator.post_generator import PostGenerator

gen = PostGenerator(model="claude-sonnet-4-6")   # or "gpt-5.1", "gpt-4.1", etc.
posts = gen.generate(interpreter_output, k_posts=3)
```

Available models:

- Claude: `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`, `claude-opus-4-8`
- GPT: `gpt-4o`, `gpt-4.1`, `gpt-5`, `gpt-5.1`

### Belief System (re-extraction)

Run these only when you have new source documents and want to rebuild the belief repository.

```bash
cd backend/Belief_System

# Step 1 — extract raw beliefs (choose extractor)
python main.py --extractor llm
python main.py --extractor sensemaking
python main.py --extractor multimodel

# Step 2 — deduplicate and canonicalise
python -m belief_system.structure --extractor llm
python -m belief_system.structure --extractor sensemaking
python -m belief_system.structure --extractor multimodel
```

Outputs land in `data/processed/` and `data/belief_store/`.

---

## API reference

The FastAPI backend exposes these endpoints (all under `http://localhost:8000`):

| Method | Endpoint               | Description                                                                      |
| ------ | ---------------------- | -------------------------------------------------------------------------------- |
| `GET`  | `/api/articles/cached` | Return top-5 articles from the most recent filter run                            |
| `POST` | `/api/scan`            | SSE stream: run scanner + filter, return articles. `?force=true` skips 24h cache |
| `POST` | `/api/interpret`       | Run Interpreter on a selected article; returns structured interpretation         |
| `POST` | `/api/generate`        | Generate LinkedIn post drafts from interpretation output                         |
| `POST` | `/api/refine/feedback` | Refine a post using human-provided feedback text                                 |
| `POST` | `/api/refine/auto`     | Auto-refine using judge LLM scoring loop                                         |

Scan and interpret endpoints return **Server-Sent Events** (SSE) so the UI can stream live log output.

---

## Configuration

| File                                                   | What to edit                                               |
| ------------------------------------------------------ | ---------------------------------------------------------- |
| `backend/secrets/.env`                                 | API keys                                                   |
| `backend/Scanner/main_scanner/settings.py`             | News sources, AI keywords, scoring weights, request timing |
| `backend/Generator/post_generator.py`                  | Default LLM model                                          |
| `backend/Generator/config/post-reformulated-prompt.md` | LinkedIn post generation prompt                            |
| `backend/Interpreter/Interpreter.py`                   | RAG strategy, retriever type, reasoning mode               |

---

## Tech stack

| Layer        | Technology                                                |
| ------------ | --------------------------------------------------------- |
| Frontend     | Svelte 5, Vite                                            |
| Backend      | FastAPI, Python 3.10+                                     |
| LLMs         | Anthropic Claude, OpenAI GPT (via LangChain + direct SDK) |
| Embeddings   | OpenAI `text-embedding-*`                                 |
| News sources | NewsAPI, RSS, arXiv API, Trafilatura scraper              |
| RAG          | BM25 + cosine similarity hybrid                           |
