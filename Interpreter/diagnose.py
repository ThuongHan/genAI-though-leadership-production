"""
diagnose.py — Self-Diagnosis & Refinement for Interpreter
==========================================================
Analyzes why rejected posts failed, traces root causes back to the
Interpreter output, and generates one-shot refinement guidance for
re-running interpret() with improved results.

Does NOT touch generate_posts() or any Generator code.
"""

import json
import os
from typing import Optional

from openai import OpenAI

# ╔══════════════════════════════════════════════════════════════╗
# ║                        CLIENT                                ║
# ╚══════════════════════════════════════════════════════════════╝

def _get_client() -> OpenAI:
    token = os.getenv("UVA_API_TOKEN")
    if not token:
        raise RuntimeError("UVA_API_TOKEN not set")
    return OpenAI(api_key=token, base_url="https://llmproxy.uva.nl/v1/")


# ╔══════════════════════════════════════════════════════════════╗
# ║                  DIAGNOSE (internal, LLM-powered)            ║
# ╚══════════════════════════════════════════════════════════════╝

_DIAGNOSE_SYSTEM_PROMPT = """You are a quality auditor for an AI strategy interpretation pipeline.
Your job: compare an ACCEPTED LinkedIn post against REJECTED ones,
and trace quality differences back to the upstream Interpreter output.

The pipeline is:
  News → RAG (belief retrieval) → Interpreter (LLM) → Generator (LLM) → Posts

You can ONLY change the Interpreter. The Generator is a black box you cannot touch.
So you must identify what the Interpreter should have produced differently
so that the Generator would have written better rejected posts.

Analyze across these dimensions:
1. **Stance accuracy** — did the interpreter pick the right stance? If the accepted
   post feels more authentic, was its stance better aligned with the belief?
2. **Argument quality** — are the rejected posts' arguments vague, unbalanced, or
   lacking specificity? Did the interpreter's Supporting arguments lack concrete
   details, data points, or NL-specific context?
3. **Belief alignment** — did the interpreter's "Why does it matter for KickstartAI"
   section fail to connect meaningfully to the retrieved belief?
4. **Factual precision** — did the interpreter miss or misrepresent key facts
   from the news article that would have grounded the posts better?
5. **NL perspective** — did the interpreter adequately address the Dutch context,
   or did it stay too generic/global?

Output ONLY valid JSON, no markdown."""


def diagnose_rejection(
    accepted_post: dict,        # {"candidate_id": "A", "text": "..."}
    rejected_posts: list[dict], # [{"candidate_id": "B", "text": "..."}, ...]
    parsed_json: dict,          # Interpreter output
    news_title: str = "",
) -> dict:
    """
    LLM-powered diagnosis: compare accepted vs rejected posts, trace issues
    back to the Interpreter output.

    Args:
        accepted_post:  the post the user chose
        rejected_posts: the posts the user rejected (1–2 items)
        parsed_json:    the Interpreter's parsed_json (5 fields)
        news_title:     for context

    Returns:
        diagnosis dict:
        {
            "interpreter_issues": [
                {
                    "dimension": "stance" | "arguments" | "belief_alignment"
                                | "factual_precision" | "nl_perspective",
                    "severity": "high" | "medium" | "low",
                    "finding": "具体问题描述",
                    "evidence": "accepted vs rejected 对比证据"
                }
            ],
            "generator_issues": [...],   # issues NOT caused by interpreter
            "overall_assessment": "一句话总结",
        }
    """
    client = _get_client()

    # Build comparison text
    accepted_text = f"[ACCEPTED — Post {accepted_post.get('candidate_id', '?')}]\n{accepted_post.get('text', '')}"

    rejected_text = "\n\n".join(
        f"[REJECTED — Post {r.get('candidate_id', '?')}]\n{r.get('text', '')}"
        for r in rejected_posts
    )

    interpreter_text = json.dumps(parsed_json, indent=2, ensure_ascii=False)

    user_prompt = f"""
News article: {news_title}

=== ACCEPTED POST ===
{accepted_text}

=== REJECTED POSTS ===
{rejected_text}

=== INTERPRETER OUTPUT (what the Generator received) ===
{interpreter_text}

Please diagnose why the rejected posts were rejected, tracing root causes
back to the Interpreter output. Distinguish between:

- **interpreter_issues**: problems originating from the Interpreter's
  stance, arguments, belief alignment, factual precision, or NL perspective.
  These are things the Interpreter COULD fix in a re-run.

- **generator_issues**: problems originating from the Generator's writing
  style, structure, angle selection, or expression. These are things the
  Interpreter CANNOT fix.

For each interpreter issue, provide:
  - dimension: one of [stance, arguments, belief_alignment, factual_precision, nl_perspective]
  - severity: high / medium / low
  - finding: concise description of the problem
  - evidence: specific contrast between accepted and rejected posts that reveals this

JSON only:
{{
    "interpreter_issues": [...],
    "generator_issues": [...],
    "overall_assessment": "one-sentence summary"
}}
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _DIAGNOSE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content or "{}")
    except Exception as e:
        print(f"  ⚠️  diagnose_rejection error: {e}")
        return {
            "interpreter_issues": [],
            "generator_issues": [],
            "overall_assessment": f"Diagnosis failed: {e}",
        }


# ╔══════════════════════════════════════════════════════════════╗
# ║              BUILD REFINEMENT GUIDANCE                       ║
# ╚══════════════════════════════════════════════════════════════╝

def build_refinement_guidance(diagnosis: dict) -> str | None:
    """
    Convert diagnosis into a one-shot guidance string that can be
    appended to the interpreter prompt for a re-run.

    Only includes interpreter_issues (not generator_issues).

    Args:
        diagnosis: output of diagnose_rejection()

    Returns:
        Guidance string (1-3 sentences, actionable), or None if no
        interpreter issues found.
    """
    interpreter_issues = diagnosis.get("interpreter_issues", [])
    if not interpreter_issues:
        return None

    # Sort by severity: high > medium > low
    severity_order = {"high": 0, "medium": 1, "low": 2}
    sorted_issues = sorted(interpreter_issues,
                           key=lambda x: severity_order.get(x.get("severity", "low"), 2))

    # Build concise, actionable guidance lines
    guidance_lines = []
    for issue in sorted_issues[:3]:  # top 3 at most
        dim = issue.get("dimension", "unknown")
        finding = issue.get("finding", "")

        if dim == "stance":
            guidance_lines.append(f"Stance: {finding}")
        elif dim == "arguments":
            guidance_lines.append(f"Supporting arguments should be more specific: {finding}")
        elif dim == "belief_alignment":
            guidance_lines.append(f"Better connect to the KickstartAI belief: {finding}")
        elif dim == "factual_precision":
            guidance_lines.append(f"Ground claims in article facts: {finding}")
        elif dim == "nl_perspective":
            guidance_lines.append(f"Strengthen Dutch/NL context: {finding}")

    if not guidance_lines:
        return None

    guidance = (
        "[REFINEMENT GUIDANCE — previous round was rejected because:]\n"
        + "\n".join(f"  - {line}" for line in guidance_lines)
        + "\n\nPlease address these points in your interpretation this time."
    )
    return guidance


# ╔══════════════════════════════════════════════════════════════╗
# ║              REFINEMENT OPTIONS (user-facing)                ║
# ╚══════════════════════════════════════════════════════════════╝

def format_refinement_options(diagnosis: dict) -> list[dict]:
    """
    Turn interpreter issues into user-friendly refinement choices.

    Args:
        diagnosis: output of diagnose_rejection()

    Returns:
        List of options for the user to pick from:
        [{"key": "1", "label": "加强论据，引用具体数据", "guidance": "..."}, ...]
    """
    interpreter_issues = diagnosis.get("interpreter_issues", [])
    if not interpreter_issues:
        return []

    options = []
    issue_to_key = {
        "arguments":         ("1", "加强论据，引用原文具体数据"),
        "nl_perspective":    ("2", "增加荷兰本地视角"),
        "stance":            ("3", "调整立场/态度"),
        "belief_alignment":  ("4", "更紧密关联 KickstartAI 信念"),
        "factual_precision": ("5", "修正事实，更精准引用原文"),
    }

    seen_dims = set()
    for issue in interpreter_issues:
        dim = issue.get("dimension", "")
        if dim in seen_dims or dim not in issue_to_key:
            continue
        seen_dims.add(dim)
        key, label = issue_to_key[dim]
        options.append({
            "key": key,
            "label": label,
            "dimension": dim,
            "finding": issue.get("finding", ""),
        })

    return options[:4]  # max 4 options
