# How to run — KickstartAI Scanner

Command reference for running the **scanner**, the **weighted filter** (3 prompting
strategies), and the **annotation-set builder** — each separately.

> On Windows, if `python` isn't on your PATH, use **`py`** instead of `python`.

---

## 0. One-time setup

```bash
pip install -r requirements.txt
```

Add your API keys: copy `main_scanner/.env.example` → `main_scanner/.env` and fill in:

```
ANTHROPIC_API_KEY=sk-ant-...      # required (Claude — default filter provider + summaries)
OPENAI_API_KEY=                   # optional (only for --provider openai)
NEWSAPI_KEY=...                   # required for NewsAPI sources (empty => those sources are skipped)
```

---

## 1. Run a scan (collect articles)

```bash
python run_scanner.py
```

- Pulls NewsAPI + RSS + arXiv + scraping, dedups, applies the source filters.
- Output: `data/scans/scanner_output_DD-MM-HHMM.json` (date-stamped — never overwrites a past run).
- Takes ~5–11 min (article fetches run in parallel, `MAX_WORKERS`).

Equivalent module form:

```bash
python -m main_scanner
```

> **Paywalled sources** (NRC, Telegraaf, FT, …) are kept as the **free preview only**,
> never the full body. Such articles are marked `"paywalled": true` in the output.

---

## 2. Filter a scan to the top 5 (`weighted_filter` — the active filter)

Automatically scores the **newest** scan in `data/scans/`. Three prompting strategies:

```bash
# zero-shot (baseline — rubric only, no examples)
python -m main_scanner.weighted_filter --strategy zero_shot

# few-shot (rubric + 2 KEEP / 2 REJECT worked examples)
python -m main_scanner.weighted_filter --strategy few_shot

# chain-of-thought (step-by-step reasoning before scoring)
python -m main_scanner.weighted_filter --strategy cot
```

Output: `data/filter/weighted_top5_<provider>_<model>_<strategy>_DD-MM-HHMM.json`
(top-5 full records + `all_ranked` = every article scored, high→low).

### Options

| Flag | Meaning | Default |
|---|---|---|
| `--strategy / -s` | `zero_shot` \| `few_shot` \| `cot` | `zero_shot` |
| `--provider / -p` | `claude` \| `openai` | `claude` |
| `--model` | override the model id | Haiku / gpt-4o-mini |
| `--excerpt / -e` | which part of the text to score: `smart` \| `head` \| `middle` \| `end` | `smart` |
| `--max-chars` | excerpt size (the cost lever) | 2000 |
| `--top-n` | how many to return | 5 |
| `<input.json>` | score a specific scan instead of the newest | newest in `data/scans/` |
| `--output / -o` | custom output path | auto-named |

Examples:

```bash
python -m main_scanner.weighted_filter --strategy few_shot --provider openai        # ~6x cheaper
python -m main_scanner.weighted_filter --excerpt head --max-chars 1000              # cheaper
python -m main_scanner.weighted_filter data/scans/scanner_output_09-06-1430.json    # a specific scan
```

Cost (Haiku, ~650 articles): ~$0.70–1.00 per run; ~6× cheaper on OpenAI.

---

## 3. (Optional) Build the human-annotation evaluation set

```bash
python build_annotation_dataset.py
```

- Pools scans, LLM-grades each article 0–10, hybrid-selects 120 good + 30 bad,
  adds AI summaries, writes a dated Excel + JSON.
- Output: `data/annotation/annotation_db_v4_DD-MM-HHMM.{json,xlsx}`.

Re-select **for free** (no re-scoring) from the cached pool:

```bash
python build_annotation_dataset.py --reuse-scores data/annotation/_scored_pool_DD-MM-HHMM.json
```

Re-render only the Excel/JSON after a schema change (no API):

```bash
python build_annotation_dataset.py --rebuild-from data/annotation/<some_db>.json
```

---

## Where things go

| What | Location |
|---|---|
| Scans | `data/scans/scanner_output_*.json` |
| Filter top-5 | `data/filter/weighted_top5_*.json` |
| Annotation set | `data/annotation/` |

## Good to know

- Edit **`main_scanner/settings.py`** to change sources, queries, keywords, `MAX_WORKERS`,
  `EXCLUDED_SOURCES`, `PAYWALLED_SOURCES` / `PAYWALL_PREVIEW_CHARS`.
- The three prompt texts (verbatim) are in **`prompts/`**.
- Archived tools (the other filters, annotation-scan scripts, legacy code) live in
  **`old files/`** — preserved, not deleted.