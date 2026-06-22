"""
FastAPI backend — KickstartAI Thought Leadership UI.

Run from the pipeline root:
    uvicorn backend.api:app --reload --port 8000
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Env & paths ────────────────────────────────────────────────────────────────
# ROOT = backend/ (where Generator, Scanner, Interpreter, Belief_System live)
ROOT = Path(__file__).parent
os.chdir(str(ROOT))          # ensure relative paths in submodules resolve from backend/
sys.path.insert(0, str(ROOT))

# Pre-load interpreter .env so pipeline.py finds UVA_API_TOKEN at import time
load_dotenv(ROOT / "Interpreter" / ".env", override=True)

SCANNER_DIR = ROOT / "Scanner"
SCANS_DIR   = SCANNER_DIR / "data" / "scans"
FILTER_DIR  = SCANNER_DIR / "data" / "filter"

# Prefer the project venv Python so scanner/filter subprocesses have the right deps.
_VENV_PYTHON = ROOT.parent / "venv" / "bin" / "python3"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="KickstartAI Thought Leadership API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.is_dir():
        return None
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def _is_recent(path: Path, hours: int = 24) -> bool:
    return (
        datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        < timedelta(hours=hours)
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _iter_proc(cmd: list, cwd: Path):
    """Run a subprocess and yield raw event dicts (not SSE-encoded)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd),
    )
    recent: list[str] = []
    async for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").strip()
        if line:
            recent.append(line)
            if len(recent) > 30:
                recent.pop(0)
            yield {"type": "log", "message": line}
    await proc.wait()
    if proc.returncode not in (None, 0):
        tail = "\n".join(recent[-10:]) if recent else "(no output captured)"
        yield {
            "type": "error",
            "message": (
                f"Process exited with code {proc.returncode}\n\n"
                f"Last output:\n{tail}"
            ),
        }


async def _stream_proc(cmd: list, cwd: Path):
    """Run a subprocess and yield stdout lines as SSE log events."""
    async for ev in _iter_proc(cmd, cwd):
        yield _sse(ev)


# ── 0. Cached articles ────────────────────────────────────────────────────────

def _build_articles(filter_path: Path) -> list:
    with open(filter_path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "id":           a.get("id", ""),
            "title":        a.get("name", ""),
            "url":          a.get("url", ""),
            "source":       a.get("source", ""),
            "published_at": a.get("published_at", ""),
            "language":     a.get("language", "en"),
            "full_text":    a.get("full_text", ""),
            "score":        round(a.get("_weighted", 0), 2),
        }
        for a in data.get("articles", [])
    ]


@app.get("/api/articles/cached")
def get_cached_articles():
    """Return top-5 articles from the most recent filter file (no LLM calls)."""
    latest_filter = _latest_file(FILTER_DIR, "weighted_top5_*.json")
    if not latest_filter:
        raise HTTPException(
            status_code=404,
            detail="No filter file found. Run a fresh scan first.",
        )
    articles = _build_articles(latest_filter)
    age_hours = (
        datetime.now() - datetime.fromtimestamp(latest_filter.stat().st_mtime)
    ).total_seconds() / 3600
    return {
        "articles":    articles,
        "source_file": latest_filter.name,
        "age_hours":   round(age_hours, 1),
    }


# ── 1. Scan ────────────────────────────────────────────────────────────────────

@app.post("/api/scan")
async def scan_and_filter(force: bool = False):
    """
    SSE stream: run scanner + weighted filter, return top-5 articles.
    Pass ?force=true to bypass the 24-hour cache and always run fresh.
    """
    async def gen():
        try:
            SCANS_DIR.mkdir(parents=True, exist_ok=True)
            FILTER_DIR.mkdir(parents=True, exist_ok=True)

            # ── scanner ────────────────────────────────────────────────────
            latest_scan = _latest_file(SCANS_DIR, "scanner_output_*.json")
            if not force and latest_scan and _is_recent(latest_scan):
                yield _sse({
                    "type": "progress", "step": "scan",
                    "message": f"Recent scan found ({latest_scan.name}) — skipping scanner.",
                })
            else:
                yield _sse({
                    "type": "progress", "step": "scan",
                    "message": "No recent scan found. Running scanner (5–10 min)…",
                })
                scanner_error: Optional[str] = None
                async for ev in _iter_proc([PYTHON, "run_scanner.py"], SCANNER_DIR):
                    if ev["type"] == "error":
                        scanner_error = ev["message"]
                    else:
                        yield _sse(ev)

                latest_scan = _latest_file(SCANS_DIR, "scanner_output_*.json")

                if scanner_error:
                    if latest_scan:
                        yield _sse({
                            "type": "progress", "step": "scan",
                            "message": (
                                f"⚠️  Scanner failed but an existing scan was found "
                                f"({latest_scan.name}) — using it instead.\n\n"
                                f"Scanner error:\n{scanner_error}"
                            ),
                        })
                    else:
                        yield _sse({"type": "error", "message": scanner_error})
                        return
                elif not latest_scan:
                    yield _sse({"type": "error",
                                "message": "Scanner produced no output file."})
                    return

            # ── filter ─────────────────────────────────────────────────────
            latest_filter = _latest_file(FILTER_DIR, "weighted_top5_*.json")
            if not force and latest_filter and _is_recent(latest_filter):
                yield _sse({
                    "type": "progress", "step": "filter",
                    "message": f"Recent filter found ({latest_filter.name}) — using it.",
                })
            else:
                yield _sse({
                    "type": "progress", "step": "filter",
                    "message": "Scoring articles with AI judges (zero-shot)…",
                })
                async for ev in _stream_proc(
                    [PYTHON, "-m", "main_scanner.weighted_filter",
                     "--strategy", "zero_shot",
                     "--model", "claude-haiku-4-5-20251001"],
                    SCANNER_DIR,
                ):
                    yield ev
                latest_filter = _latest_file(FILTER_DIR, "weighted_top5_*.json")
                if not latest_filter:
                    yield _sse({"type": "error",
                                "message": "Filter produced no output file."})
                    return

            # ── return articles ────────────────────────────────────────────
            yield _sse({"type": "done", "articles": _build_articles(latest_filter)})

        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 2. Interpret ───────────────────────────────────────────────────────────────

class ArticleBody(BaseModel):
    id: str = ""
    title: str
    full_text: str
    url: str = ""


@app.post("/api/interpret")
def interpret_article(body: ArticleBody):
    """
    Interpret a news article via embedding RAG + CoT + KickstartAI belief repository.
    FastAPI runs sync endpoints in a thread pool automatically.
    """
    from Interpreter.pipeline import (
        load_beliefs,
        retrieve,
        interpret,
        BELIEF_REPOSITORY_PATH,
    )

    beliefs  = load_beliefs(BELIEF_REPOSITORY_PATH)
    news_obj = {
        "news_id": body.id,
        "title":   body.title,
        "excerpt": body.full_text,
    }

    rag    = retrieve(news_obj, beliefs, "embedding", 3)
    result = interpret(news_obj, rag, "cot")

    if not result["schema_pass"]:
        raise HTTPException(
            status_code=422,
            detail="Interpretation failed schema validation. Raw: "
                   + result["raw_llm_output"][:300],
        )

    return {
        "interpretation":  result["parsed_json"],
        "matched_beliefs": rag["beliefs"],
        "metadata": {
            "source_article": {"title": body.title, "url": body.url}
        },
    }


# ── 3. Generate ────────────────────────────────────────────────────────────────

class GenerateBody(BaseModel):
    interpretation: dict
    metadata: dict


@app.post("/api/generate")
def generate_posts(body: GenerateBody):
    """
    Generate 3 LinkedIn posts with PostGenerator (Claude Sonnet + few-shot).
    """
    from Generator.post_generator import PostGenerator

    interp      = body.interpretation
    event_input = {
        "what_happened":   interp["What happened"],
        "why_relevance":   interp["Why does it matter (globally and NL)"],
        "why_kickstartai": interp["Why does it matter for KickstartAI"],
        "stance":          interp["Key stance / opinion"],
        "arguments":       interp["Supporting arguments"],
        "source":          body.metadata.get("source_article", {}).get("url", ""),
    }

    generator = PostGenerator(
        model="claude-sonnet-4-6",
        config_path="Generator/config/post-reformulated-prompt.md",
    )
    result = generator.generate(
        interpreter_output=event_input,
        k_posts=1,
        use_few_shot=True,
        save=False,
    )

    return {
        "posts": [
            {"post_idx": p.post_idx, "angle": p.angle, "content": p.content}
            for p in result["posts"]
        ]
    }


# ── 4. Refine — user feedback ──────────────────────────────────────────────────

class FeedbackBody(BaseModel):
    post_content: str
    feedback: str


@app.post("/api/refine/feedback")
def refine_with_feedback(body: FeedbackBody):
    """
    Refine a post once using user-provided feedback.
    Mirrors option 2 in interact-claude.py.
    """
    from Generator.utils.embedder import Embedder
    from Generator.utils.few_shot import FewShotPost
    from Generator.utils.llm.claude import ClaudeLLM
    from Generator.regeneration.refine import build_regen_prompt

    embedder = Embedder()
    few_shot = FewShotPost()
    prompt   = build_regen_prompt(
        body.post_content,
        f"### User feedback\n\n{body.feedback}",
        few_shot,
        embedder,
    )
    refined = ClaudeLLM("claude-sonnet-4-6").invoke(prompt).content.strip()
    return {"refined_post": refined}


# ── 5. Refine — auto judge loop ────────────────────────────────────────────────

class AutoRefineBody(BaseModel):
    post_content: str


_PASS_THRESHOLD = 4
_DIMENSIONS = [
    "tone_of_voice",
    "language_and_style",
    "coherence_readability",
    "discourse_structure",
    "specificity",
    "historical_similarity",
]


@app.post("/api/refine/auto")
async def auto_refine(body: AutoRefineBody):
    """
    SSE stream: dual-judge auto-refinement loop (Opus + GPT-5), up to 3 iterations.
    Mirrors option 1 in Generator/Interaction/interact-claude.py.
    """
    async def gen():
        try:
            from Generator.utils.embedder import Embedder
            from Generator.utils.few_shot import FewShotPost
            from Generator.utils.llm.claude import ClaudeLLM
            from Generator.utils.llm.gpt import GPTLLM
            from Generator.judge.runner import (
                _build_prompt as build_eval_prompt,
                _extract_json as extract_json,
            )
            from Generator.regeneration.refine import build_feedback, build_regen_prompt

            embedder = Embedder()
            few_shot = FewShotPost()
            current  = body.post_content
            history  = []  # list of (iteration, post, evals, avg)

            # blocking helpers — run in thread pool via asyncio.to_thread
            def _evaluate(text: str) -> dict:
                embedding = embedder.embed_text(text)
                refs      = few_shot.get_similar_posts(embedding, top_k=1)
                prompt    = build_eval_prompt(text, refs)
                results   = {}
                for name, model_id in [("Opus", "claude-opus-4-8"), ("GPT-5.5", "gpt-5.5")]:
                    llm = ClaudeLLM(model_id) if "claude" in model_id else GPTLLM(model_id)
                    for attempt in range(3):
                        resp = llm.invoke(prompt)
                        try:
                            results[name] = extract_json(resp.content)
                            break
                        except Exception:
                            if attempt == 2:
                                raise
                return results

            def _regenerate(post: str, feedback: str) -> str:
                prompt = build_regen_prompt(post, feedback, few_shot, embedder)
                return ClaudeLLM("claude-sonnet-4-6").invoke(prompt).content.strip()

            def _avg(evals: dict) -> float:
                sc = [d["score"] for ev in evals.values()
                      for d in ev.get("dimensions", [])]
                return round(sum(sc) / len(sc), 2) if sc else 0.0

            def _all_pass(evals: dict) -> bool:
                return all(
                    d["score"] >= _PASS_THRESHOLD
                    for ev in evals.values()
                    for d in ev.get("dimensions", [])
                )

            def _payload(evals: dict) -> dict:
                out = {}
                for name, ev in evals.items():
                    dims = ev.get("dimensions", [])
                    out[name] = {
                        "dimensions": dims,
                        "avg": round(
                            sum(d["score"] for d in dims) / max(len(dims), 1), 2
                        ),
                    }
                return out

            # ── iteration 1: evaluate initial post ────────────────────────
            yield _sse({"type": "progress",
                        "message": "Evaluating initial post with both judges…"})
            evals = await asyncio.to_thread(_evaluate, current)
            avg   = _avg(evals)
            history.append((1, current, evals, avg))

            yield _sse({
                "type": "evaluation", "iteration": 1,
                "post": current, "evaluations": _payload(evals), "avg": avg,
            })

            MAX_ITER = 3
            for iteration in range(1, MAX_ITER + 1):
                if _all_pass(evals):
                    yield _sse({"type": "progress",
                                "message": "All dimensions passed — stopping early."})
                    break
                if iteration == MAX_ITER:
                    yield _sse({"type": "progress",
                                "message": "Maximum iterations reached."})
                    break

                yield _sse({
                    "type": "progress",
                    "message": f"Refining post (iteration {iteration}/{MAX_ITER - 1})…",
                })
                feedback = build_feedback(evals)
                current  = await asyncio.to_thread(_regenerate, current, feedback)

                yield _sse({"type": "progress",
                            "message": "Re-evaluating refined post…"})
                evals = await asyncio.to_thread(_evaluate, current)
                avg   = _avg(evals)
                history.append((iteration + 1, current, evals, avg))

                yield _sse({
                    "type": "evaluation", "iteration": iteration + 1,
                    "post": current, "evaluations": _payload(evals), "avg": avg,
                })

            best_iter, best_post, _, best_avg = max(history, key=lambda x: x[3])
            yield _sse({
                "type": "done",
                "final_post":     best_post,
                "final_avg":      best_avg,
                "best_iteration": best_iter,
            })

        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# dd