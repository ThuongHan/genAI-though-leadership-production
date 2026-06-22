"""
weighted_filter.py — rubric-weighted top-5 selector.

For every article: send the title + an excerpt of the full text to the LLM,
get a 0-RUBRIC_MAX score on each of the 4 rubric dimensions, compute a
mission-weighted average, and return the highest-scoring articles.

Prompting strategy is selectable via --strategy: zero_shot (default),
few_shot (2 KEEP + 2 REJECT worked examples), or cot (chain-of-thought).

Design (per project decisions):
  - Pure weighted average — NO veto thresholds
  - Pure top-5 by score   — NO per-tag diversity cap
  - Top-5 selection is plain sorting (no AI) — the LLM is used ONLY for scoring
  - Scoring runs concurrently (thread pool) — LLM_SCORE_WORKERS at a time
  - Excerpt strategy is user-selectable via --excerpt

Excerpt strategies (--excerpt), each bounded by --max-chars:
  smart  : first + middle + end chunks of full_text   (default)
  head   : first part only                            (intro-focused / cheapest)
  middle : middle part only                           (skip intro & conclusion)
  end    : end part only                              (conclusion-focused)

Difference from the other two filters:
  - basic_filter.py    : titles only, LLM picks 5 directly (no per-dim scores)
  - filter_articles.py : TF-IDF prefilter + 6-dim scoring + hard filter + diversity
  - weighted_filter.py : 5-dim scoring on every article + weighted average + pure top-5

Usage:
    python -m main_scanner.weighted_filter
    python -m main_scanner.weighted_filter --excerpt head --max-chars 1000
    python -m main_scanner.weighted_filter --provider openai --model gpt-4o-mini
    python -m main_scanner.weighted_filter data/scans/scanner_output.json -o data/filter/wf.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Auto-load API keys from main_scanner/.env (sibling of this file).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / "secrets" / ".env")
except ImportError:
    pass

# Make absolute imports work whether run as a script (play button) or module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main_scanner.llm_providers import DEFAULT_MODELS, build_llm_client
from main_scanner.paths import latest_scan, timestamped_path


# ---------------------------------------------------------------------------
# Tunables — edit these
# ---------------------------------------------------------------------------

# Mission weights. Relevance + audience_fit dominate (they capture "is this a
# KickstartAI article" most directly). Need not sum to 1 — the average
# normalises by their total.
WEIGHTS: dict = {
    "relevance":       0.35,
    "audience_fit":    0.29,
    "trustworthiness": 0.18,
    "specificity":     0.18,
}

FINAL_TOP_N: int = 5
EXCERPT_MAX_CHARS: int = 2000     # excerpt budget (cost lever — lower = cheaper)
LLM_MAX_TOKENS: int = 100         # response cap (just 5 numbers)
LLM_SCORE_WORKERS: int = 3        # concurrent LLM scoring calls (lower if rate-limited)

# Per-dimension score scale. ADJUSTABLE — change this one number and the prompt,
# clamping, and weighted average all follow. 0-10 gives 11 levels per dimension
# (vs 4 on a 0-3 scale), so far fewer ties between articles.
RUBRIC_MAX: int = 10

RUBRIC_DIMS: tuple = ("relevance", "trustworthiness", "specificity", "audience_fit")

# Tiebreak order when weighted scores are equal (all high-to-low):
# weighted -> relevance -> trustworthiness -> audience_fit -> specificity
TIEBREAK_ORDER: tuple = ("relevance", "trustworthiness", "audience_fit", "specificity")


# ---------------------------------------------------------------------------
# Prompting strategies
# ---------------------------------------------------------------------------

STRATEGIES: tuple = ("zero_shot", "few_shot", "cot")
COT_MAX_TOKENS: int = 400          # CoT needs room for the reasoning text

# Four worked examples (2 KEEP + 2 REJECT) drawn from KickstartAI's human
# annotations (dataset_KickstartAI_annotations.xlsx), with the original 0-3
# scores converted to the 0-10 rubric and the annotator's own "Why". They show
# the weighting in action — e.g. a credible, specific but off-topic article still
# scores low because relevance + audience_fit are 0. (Assumes RUBRIC_MAX = 10.)
_FEWSHOT_BLOCK: str = """\
CALIBRATION EXAMPLES — real KickstartAI human judgements (0-10). Each shows the
title, source, a short content excerpt, the scores, and the annotator's reasoning.
Match this style:

EXAMPLE 1 (KEEP):
Title: Establishing AI and data sovereignty in the age of autonomous systems
Source: MIT Technology Review  Tag: news
Excerpt: "Enterprises are increasingly prioritizing AI and data sovereignty - establishing
independent control over their proprietary data and AI models rather than relying on third-party
cloud providers. 70% of global executives surveyed believe a sovereign data and AI platform is
necessary for success, echoing a broader trend toward nations and companies building their own
AI infrastructure."
Scores: {"relevance": 10, "trustworthiness": 10, "specificity": 7, "audience_fit": 10}
Why: "Spot on for audience, industry" — enterprise AI & data sovereignty, squarely on theme.

EXAMPLE 2 (KEEP):
Title: Europe's cloud dependency is a political risk, not just a technical one
Source: The Next Web  Tag: news
Excerpt: "Europe's reliance on US cloud providers and semiconductor manufacturers for AI
infrastructure creates significant political and data-sovereignty risks beyond mere technical
concerns. US hyperscalers (AWS, Azure, GCP) control 70% of Europe's cloud market, and although
the EU has invested 43 billion euro via the Chips Act, US legal mechanisms like the CLOUD Act can
override contractual agreements."
Scores: {"relevance": 10, "trustworthiness": 10, "specificity": 7, "audience_fit": 10}
Why: "Good to comment on overall urgency for AI sovereignty and AI capabilities" — EU digital
sovereignty. (Note: the source is a generic outlet, but the CONTENT is on-theme, so it stays high.)

EXAMPLE 3 (REJECT):
Title: Who trusts Sam Altman?
Source: TechCrunch  Tag: news
Excerpt: "Sam Altman's credibility is being scrutinised in a California federal court case brought
by Elon Musk seeking to shut down OpenAI's for-profit structure. Musk's lawyers are challenging
Altman's truthfulness, citing his incomplete disclosure to Congress about his economic interest in
OpenAI and the 2023 incident when OpenAI's board briefly fired him."
Scores: {"relevance": 0, "trustworthiness": 10, "specificity": 3, "audience_fit": 0}
Why: "We don't comment on tech-industry figures unless it is about research or important AI
developments" — a credible source, but tech-figure commentary with no NL/EU enterprise-adoption angle.

EXAMPLE 4 (REJECT):
Title: Google DeepMind to open its first AI campus in Seoul
Source: The Next Web  Tag: news
Excerpt: "Google DeepMind will establish its first-ever AI campus in Seoul, South Korea, expected
to open by 2026, following a meeting between CEO Demis Hassabis and President Lee Jae Myung. The
memorandum of understanding covers joint AI research, skills development, and responsible AI use,
as part of South Korea's push to become a top-three AI power."
Scores: {"relevance": 0, "trustworthiness": 3, "specificity": 7, "audience_fit": 0}
Why: "Not interested in company news" — company expansion with the wrong geography and no NL/EU
adoption angle."""


def _criteria_block(scale: int) -> str:
    """Task description + the four scoring criteria (shared by ALL strategies)."""
    mid = scale // 2
    hi = scale - 1
    return f"""\
You are a content strategist at KickstartAI, a Dutch non-profit that accelerates
the PRACTICAL ADOPTION of AI in large Dutch enterprises and public organisations
(partners: KLM, ING, Ahold Delhaize, NS). You curate developments for Dutch
business and public-sector leaders. Rate ONE article on FOUR dimensions, each on
a 0-{scale} scale.

WHAT KICKSTARTAI WANTS (these make an article RELEVANT and a strong AUDIENCE FIT):
  - Practical AI ADOPTION and IMPLEMENTATION in organisations — not theory or hype
  - Responsible deployment, AI governance, the EU AI Act and concrete regulation
    that affects how companies actually deploy AI
  - The AI landscape specifically in the NETHERLANDS and EUROPE
  - Where AI is APPLIED IN PRACTICE — large enterprises, public sector, healthcare,
    finance, retail, aviation, transport, logistics, agriculture, infrastructure
  - What companies, governments, universities and research institutes are LEARNING
    about implementation; recurring BARRIERS (data, governance, skills, trust,
    regulation, infrastructure, evaluation, scaling)
  - REPORTS, case studies, achievements and challenges that give usable evidence
  - Developments affecting KickstartAI's partners, community or the Dutch AI ecosystem
  - Global developments ONLY when they clearly affect European adoption, regulation,
    enterprise implementation, or the Dutch ecosystem
  - Technical AI research is in scope but secondary (useful for a technical audience)

WHAT TO AVOID (score these LOW on relevance and audience_fit):
  - US-only or India-only news with no clear EU / NL link
  - Generic AI hype, futurist speculation, opinion pieces without substance
  - Product launches, funding rounds, executive hires/moves with no adoption substance
  - Consumer-tech AI (gadgets, apps, chatbots aimed at end-consumers)
  - Profiles of or gossip about AI industry figures
  - EU regulation discussed only in broad/abstract AI terms, not tied to industry

BE STRICT and USE THE FULL 0-{scale} RANGE. Most articles are NOT highly relevant
to this narrow focus. Reserve {hi}-{scale} for articles squarely on the themes
above, give about {mid} to partial fits, and 0-2 to off-topic items. Do NOT
cluster everything in the middle.

THE FOUR CRITERIA — what each one means and what to weigh:

relevance (0-{scale}) — How squarely the article sits in KickstartAI's focus.
    Weigh: Is it about PRACTICAL AI adoption / implementation / responsible
    deployment in organisations? Is the geography the NETHERLANDS or EUROPE — or a
    global story with a CLEAR EU/NL link? Penalise US/India-only news, generic hype,
    product launches, company PR, and broad AI regulation not tied to industry.
    0 = off-topic or wrong geography   {mid} = related but partial / weak EU link
    {scale} = squarely on a core theme (NL/EU enterprise adoption, governance, applied use)

trustworthiness (0-{scale}) — Source credibility + verifiability.
    Weigh: Is the source a research institute, university, government or EU body,
    primary report, or an established enterprise-tech outlet (e.g. MIT Technology
    Review, Stanford AI Index)? Penalise press-release wires, content farms, SEO
    blogs, and promotional or anonymous sources.
    0 = dubious / promotional / press release   {mid} = credible outlet
    {scale} = official, research, or primary source

specificity (0-{scale}) — Concrete, evidence-backed and actionable vs. vague.
    Weigh: Does it name organisations, give data, outcomes, case studies, or real
    implementation detail — or is it generic claims and buzzwords? Reports and
    case studies with evidence score high; vague think-pieces score low.
    0 = vague / buzzwords   {mid} = some specifics or data
    {scale} = concrete data, named organisations, real case studies

audience_fit (0-{scale}) — Usefulness for Dutch business & public-sector leaders
    moving AI from experimentation to real-world impact.
    Weigh: Would a decision-maker at a large NL/EU organisation find this useful for
    adoption, strategy, governance, or knowledge-sharing? Technical AI research is a
    SECONDARY (technical) audience — useful, but not the core readership.
    0 = wrong audience   {mid} = somewhat useful   {scale} = directly useful to their decisions"""


def _output_instruction(scale: int, strategy: str) -> str:
    """The closing instruction — differs for chain-of-thought."""
    if strategy == "cot":
        return f"""\
Before scoring, reason step by step (briefly) through these questions:
  1. GEOGRAPHY & EU/NL LINK — is this the Netherlands/Europe, or global with a clear EU link?
  2. ADOPTION vs HYPE — real implementation/evidence, or hype / PR / a product launch?
  3. SOURCE — research, government, primary, or an established outlet? Or promotional/wire?
  4. SPECIFICITY — concrete data, named organisations, case studies — or vague?
  5. AUDIENCE — useful to Dutch enterprise & public-sector leaders?
Then give the four scores, consistent with your reasoning.

Reply with ONE JSON object and nothing else:
{{"reasoning": "<2-4 sentences working through the questions above>", "relevance": <0-{scale}>, "trustworthiness": <0-{scale}>, "specificity": <0-{scale}>, "audience_fit": <0-{scale}>}}"""
    return f"""\
Reply with ONE JSON object and nothing else:
{{"relevance": <0-{scale}>, "trustworthiness": <0-{scale}>, "specificity": <0-{scale}>, "audience_fit": <0-{scale}>}}"""


def _build_system_prompt(scale: int, strategy: str = "zero_shot") -> str:
    """Compose the system prompt for the chosen strategy.

    zero_shot : criteria + output instruction
    few_shot  : criteria + 2 KEEP / 2 REJECT worked examples + output instruction
    cot       : criteria + step-by-step reasoning + output instruction (with reasoning)
    """
    parts = [_criteria_block(scale)]
    if strategy == "few_shot":
        parts.append(_FEWSHOT_BLOCK)
    parts.append(_output_instruction(scale, strategy))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Excerpting
# ---------------------------------------------------------------------------

def make_excerpt(full_text: str, strategy: str, max_chars: int = EXCERPT_MAX_CHARS) -> str:
    """Return the slice of full_text to send to the LLM per the chosen strategy.

    head/middle/end take `max_chars` from that single region.
    smart takes three ~max_chars/3 chunks from start + middle + end.
    """
    text = (full_text or "").strip()
    if not text or len(text) <= max_chars:
        return text

    if strategy == "head":
        return text[:max_chars]
    if strategy == "end":
        return text[-max_chars:]
    if strategy == "middle":
        lo = (len(text) - max_chars) // 2
        return text[lo:lo + max_chars]

    # smart: first + middle + end (equal thirds)
    chunk = max_chars // 3
    start = text[:chunk]
    mid_lo = (len(text) - chunk) // 2
    middle = text[mid_lo:mid_lo + chunk]
    end = text[-chunk:]
    return f"{start}\n[...]\n{middle}\n[...]\n{end}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _clamp_score(value: Any) -> int:
    """Clamp a raw LLM value into 0..RUBRIC_MAX."""
    try:
        return max(0, min(RUBRIC_MAX, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def _parse_scores(text: str) -> dict[str, int]:
    """Extract the 4 rubric scores from the LLM's JSON response.

    Robust to leading text (e.g. the CoT JSON carries a `reasoning` field first):
    the regex grabs the JSON object and we read only the dimensions we need.
    """
    fallback = {d: 0 for d in RUBRIC_DIMS}
    if not text:
        return fallback
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return fallback
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return fallback
    return {dim: _clamp_score(d.get(dim)) for dim in RUBRIC_DIMS}


def _parse_reasoning(text: str) -> str:
    """Extract the `reasoning` field from a CoT response (empty if absent)."""
    if not text:
        return ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return ""
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return ""
    return str(d.get("reasoning", "")).strip()


def weighted_score(scores: dict[str, int]) -> float:
    """Mission-weighted average of the 4 rubric scores (0-RUBRIC_MAX scale)."""
    total_w = sum(WEIGHTS.values())
    if total_w == 0:
        return 0.0
    return sum(scores.get(dim, 0) * w for dim, w in WEIGHTS.items()) / total_w


def score_article(client, article: dict, excerpt_strategy: str, max_chars: int,
                  system: str | None = None, max_tokens: int = LLM_MAX_TOKENS,
                  capture_reasoning: bool = False) -> dict:
    """Score one article; returns it enriched with _scores + _weighted.

    `excerpt_strategy` is the smart/head/middle/end excerpt choice; the prompting
    strategy is already baked into `system` (defaults to the zero-shot prompt).
    CoT runs set capture_reasoning=True.
    """
    if system is None:
        system = _build_system_prompt(RUBRIC_MAX, "zero_shot")
    title = article.get("name", "")
    source = article.get("source", "")
    excerpt = make_excerpt(article.get("full_text", ""), excerpt_strategy, max_chars)
    user = f"Title: {title}\nSource: {source}\n\nContent:\n{excerpt}"
    reasoning = ""
    try:
        raw = client.complete(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        scores = _parse_scores(raw)
        if capture_reasoning:
            reasoning = _parse_reasoning(raw)
    except Exception as exc:  # graceful — a failed article scores 0
        scores = {d: 0 for d in RUBRIC_DIMS}
        article = {**article, "_score_error": str(exc)}
    result = {**article, "_scores": scores, "_weighted": round(weighted_score(scores), 4)}
    if reasoning:
        result["_reasoning"] = reasoning
    return result


def score_all(client, articles: list[dict], excerpt_strategy: str, max_chars: int,
              system: str | None = None, max_tokens: int = LLM_MAX_TOKENS,
              capture_reasoning: bool = False) -> list[dict]:
    """Score every article concurrently (thread pool), preserving order.

    `system` defaults to the zero-shot prompt (so existing callers like
    build_annotation_dataset keep working unchanged).
    """
    if system is None:
        system = _build_system_prompt(RUBRIC_MAX, "zero_shot")
    total = len(articles)
    results: list[dict] = [None] * total
    done = 0

    def _work(pair):
        i, a = pair
        return i, score_article(client, a, excerpt_strategy, max_chars,
                                system, max_tokens, capture_reasoning)

    workers = max(1, LLM_SCORE_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, enriched in ex.map(_work, enumerate(articles)):
            results[i] = enriched
            done += 1
            print(f"  scored {done}/{total}", end="\r", flush=True)
    print(f"  Done -- scored {total} articles.{' ' * 20}")
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_top(picks: list[dict]) -> None:
    print("\n" + "=" * 80)
    print(f"TOP {len(picks)} ARTICLES (mission-weighted average of 4 rubrics)")
    print("=" * 80)
    for rank, a in enumerate(picks, 1):
        s = a["_scores"]
        print(f"\n{rank}. [{a['_weighted']:.2f}] {a.get('name', '')}")
        print(f"   source: {a.get('source')}  |  tag: {a.get('tag')}  |  lang: {a.get('language')}")
        print(f"   url: {a.get('url')}")
        print(f"   scores: rel={s['relevance']} trust={s['trustworthiness']} "
              f"spec={s['specificity']} aud={s['audience_fit']}")


def _ranked_summary(scored: list[dict]) -> list[dict]:
    """Lightweight, fully-ranked view of every scored article (no full_text)."""
    out = []
    for rank, a in enumerate(scored, 1):
        row = {
            "rank": rank,
            "weighted": a["_weighted"],
            "scores": a["_scores"],
            "name": a.get("name", ""),
            "source": a.get("source", ""),
            "tag": a.get("tag", ""),
            "language": a.get("language", ""),
            "url": a.get("url", ""),
        }
        # Surface scoring failures (e.g. a 429 that exhausted retries) so a
        # silently-zeroed article is visible rather than mistaken for low quality.
        if a.get("_score_error"):
            row["score_error"] = a["_score_error"]
        out.append(row)
    return out


def write_output(picks: list[dict], all_scored: list[dict], path: str, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "filter": "weighted_filter.py — 4-rubric mission-weighted top-5",
        **meta,
        "score_scale": f"0-{RUBRIC_MAX}",
        "weights": WEIGHTS,
        "articles": picks,                       # top-N, full records
        "all_ranked": _ranked_summary(all_scored),  # every article, high->low, lightweight
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nWrote top {len(picks)} + full ranked list of {len(all_scored)} -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_articles(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["articles"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Scanner output JSON (default: newest scanner_output*.json in data/scans/)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output path (default: data/filter/weighted_top5_<DD-MM-HHMM>.json)",
    )
    parser.add_argument(
        "--provider", "-p", choices=["claude", "openai"], default="claude",
        help="LLM provider (default: claude)",
    )
    parser.add_argument("--model", default=None, help="Override model id")
    parser.add_argument(
        "--strategy", "-s", choices=list(STRATEGIES), default="zero_shot",
        help="Prompting strategy: zero_shot (default), few_shot (2 KEEP + 2 REJECT "
             "worked examples), cot (chain-of-thought reasoning before scoring)",
    )
    parser.add_argument(
        "--excerpt", "-e", choices=["smart", "head", "middle", "end"], default="smart",
        help="Which part of full_text to score on (default: smart = start+middle+end)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=EXCERPT_MAX_CHARS,
        help=f"Excerpt char budget — the cost lever (default: {EXCERPT_MAX_CHARS})",
    )
    parser.add_argument(
        "--top-n", type=int, default=FINAL_TOP_N,
        help=f"How many articles to return (default: {FINAL_TOP_N})",
    )
    args = parser.parse_args()

    # Default input: newest scan in data/scans/.
    input_path = args.input or latest_scan()
    if not input_path or not Path(input_path).exists():
        print(f"ERROR: no scan file found ({input_path or 'data/scans/scanner_output*.json'})")
        sys.exit(1)

    env_key = "ANTHROPIC_API_KEY" if args.provider == "claude" else "OPENAI_API_KEY"
    if not os.environ.get(env_key, ""):
        print(f"ERROR: {env_key} not set — required for {args.provider} provider")
        sys.exit(1)

    articles = load_articles(input_path)
    model = args.model or DEFAULT_MODELS[args.provider]

    # Default output encodes provider + model + strategy so the thesis matrix of
    # runs doesn't overwrite (e.g. weighted_top5_claude_haiku_few_shot_DD-MM-HHMM.json).
    model_short = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:20]
    default_out = f"data/filter/weighted_top5_{args.provider}_{model_short}_{args.strategy}.json"
    output_path = args.output or timestamped_path(default_out)

    # Build the strategy-specific system prompt once; CoT needs more output room.
    system = _build_system_prompt(RUBRIC_MAX, args.strategy)
    max_tokens = COT_MAX_TOKENS if args.strategy == "cot" else LLM_MAX_TOKENS
    capture_reasoning = args.strategy == "cot"

    print(f"Loaded {len(articles)} articles from {input_path}")
    print(f"Provider: {args.provider} ({model})  |  strategy: {args.strategy}  |  "
          f"excerpt: {args.excerpt} (<= {args.max_chars} chars)  |  "
          f"scoring all {len(articles)} on 4 rubrics ...")

    client = build_llm_client(args.provider, model=model)
    scored = score_all(client, articles, args.excerpt, args.max_chars,
                       system=system, max_tokens=max_tokens,
                       capture_reasoning=capture_reasoning)

    # Sort: weighted score first, then the tiebreak dimensions (all high->low).
    # No veto, no diversity cap — pure ranking (just sorting, no AI).
    scored.sort(
        key=lambda a: (a["_weighted"], *(a["_scores"].get(d, 0) for d in TIEBREAK_ORDER)),
        reverse=True,
    )
    picks = scored[:args.top_n]

    print_top(picks)
    write_output(picks, scored, output_path, meta={
        "provider": args.provider,
        "model": model,
        "strategy": args.strategy,
        "excerpt_strategy": args.excerpt,
        "excerpt_max_chars": args.max_chars,
        "input_article_count": len(articles),
        "top_n": args.top_n,
    })


if __name__ == "__main__":
    main()