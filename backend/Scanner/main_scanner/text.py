"""Keyword filtering, language detection, consent-wall + paywall detection."""

from __future__ import annotations

from urllib.parse import urlparse

from .settings import AI_KEYWORDS_EN, AI_KEYWORDS_NL, PAYWALLED_SOURCES

_ALL_KEYWORDS = [k.lower() for k in AI_KEYWORDS_EN + AI_KEYWORDS_NL]

_NL_KEYWORDS = [k.lower() for k in AI_KEYWORDS_NL] + [
    " de ", " het ", " van ", " een ", " dat ", " zijn ", " wordt ", " heeft ",
]

# Any single match from this list is enough to flag a consent/cookie wall page
_CONSENT_MARKERS = [
    "before you continue to google",
    "non-personalized content is influenced by",
    "we use cookies and data to\n- deliver and maintain",
    "select “more options” to see additional information, including details about managing your privacy",
    "manage your privacy settings",
]


def detect_language(title: str, full_text: str) -> str:
    """Return 'nl' if Dutch keywords dominate, else 'en'."""
    sample = (title + " " + full_text[:400]).lower()
    hits = sum(1 for kw in _NL_KEYWORDS if kw in sample)
    return "nl" if hits >= 2 else "en"


def is_ai_related(text: str) -> bool:
    """Return True if the text contains at least one AI-related keyword."""
    lower = text.lower()
    return any(kw in lower for kw in _ALL_KEYWORDS)


def is_consent_wall(text: str) -> bool:
    """Return True if the text appears to be a cookie/consent prompt rather than article content."""
    if not text:
        return False
    sample = text[:800].lower()
    return any(m in sample for m in _CONSENT_MARKERS)


# ---------------------------------------------------------------------------
# Paywall handling — keep only the freely-visible preview of subscription sources
# ---------------------------------------------------------------------------

# Markers that indicate a subscription/paywall stub rather than free content.
_PAYWALL_MARKERS = [
    "abonnee", "word abonnee", "lees verder met", "log in om verder te lezen",
    "subscribe to read", "subscribe to continue", "create a free account",
    "sign in to read", "this article is for subscribers", "to continue reading",
    "registreer", "al abonnee",
]


def domain_of(url: str) -> str:
    """Return the bare domain of a URL (lowercased, 'www.' stripped)."""
    netloc = urlparse(url or "").netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def is_paywalled_source(url: str) -> bool:
    """True if the URL's domain is a known subscription source (settings list)."""
    d = domain_of(url)
    return any(d == h or d.endswith("." + h) for h in PAYWALLED_SOURCES)


def looks_paywalled(text: str) -> bool:
    """Runtime catch for UNLISTED paywalls: a short body that also contains
    subscribe-wall markers. The marker requirement protects genuinely-short
    FREE articles (which carry no such markers) from being mis-flagged."""
    if not text:
        return False
    low = text[:1200].lower()
    return len(text) < 1200 and any(m in low for m in _PAYWALL_MARKERS)


def free_preview(text: str, max_chars: int) -> str:
    """Return only the freely-visible preview: the first `max_chars` of the
    article, cut at a sentence boundary where possible (the lede). Shorter
    text is returned whole — `max_chars` is a cap, not a minimum."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind("\n"))
    if end >= max_chars * 0.5:           # only snap back if a boundary is reasonably near the cap
        cut = cut[:end + 1]
    return cut.strip()
